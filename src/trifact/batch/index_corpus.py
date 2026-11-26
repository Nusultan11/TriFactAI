"""Batch job index_corpus: bulk-load task/vectors into a versioned Qdrant collection (ADR-0013, ADR-0014).

Build order (qdrant-at-scale.md):
1. Create ``corpus_chunks__<embedder>@<ver>`` with HNSW disabled (``m=0``, ``indexing_threshold=0``)
   and payload indexes (``lang``, ``source``, ``document_id``, ``entity_ids``) BEFORE bulk upload.
2. Stream ``task/vectors`` Parquet shards with idempotent ``.INDEXED.json`` shard markers.
3. Optional entity enrichment from ``data/task/annotations`` (never required, ADR-0004).
4. Enable HNSW + quantization and verify point count + approximate vs. exact recall.
5. Switch alias ``corpus_chunks`` ONLY when ``--switch-alias`` is explicitly passed (post-gate).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from qdrant_client import models

from trifact.core.params import ROOT, load_params
from trifact.core.schemas import Chunk, Lang, SourceType
from trifact.lake.stats import ResourceMonitor
from trifact.retrieval.embedder import estimate_vector_bytes
from trifact.retrieval.store import (
    CORPUS_ALIAS,
    CorpusIndexConfig,
    QdrantStore,
    versioned_collection_name,
)

log = logging.getLogger("trifact.batch.index_corpus")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_entity_ids_map(annotations_dir: Path | None) -> dict[str, list[str]]:
    """Optional entity QID enrichment map ``chunk_id -> sorted unique QIDs`` (ADR-0004)."""
    if annotations_dir is None or not annotations_dir.is_dir():
        return {}
    out: dict[str, set[str]] = defaultdict(set)
    for p in sorted(annotations_dir.rglob("*.parquet")):
        if "_work" in p.parts:
            continue
        tbl = pq.read_table(p, columns=["chunk_id", "wikidata_qid"])
        for cid, qid in zip(tbl["chunk_id"].to_pylist(), tbl["wikidata_qid"].to_pylist(), strict=True):
            if qid:
                out[str(cid)].add(str(qid))
    return {cid: sorted(qids) for cid, qids in out.items()}


def index_vector_shards(
    store: QdrantStore,
    vectors_dir: Path,
    collection_name: str,
    cfg: CorpusIndexConfig,
    *,
    annotations_dir: Path | None = None,
    batch_size: int = 512,
    recreate: bool = False,
    enable_hnsw_after: bool = True,
    switch_alias: bool = False,
    alias_name: str = CORPUS_ALIAS,
) -> dict[str, Any]:
    """Resumable bulk upload of ``task/vectors`` Parquet shards into Qdrant."""
    shard_files = sorted(p for p in vectors_dir.rglob("*.parquet") if "_work" not in p.parts)
    if not shard_files:
        raise FileNotFoundError(f"No vector Parquet shards found in {vectors_dir}")

    # Infer dense dim from first non-empty shard
    first_tbl = pq.read_table(shard_files[0], columns=["dense"])
    if first_tbl.num_rows > 0:
        first_vec = first_tbl["dense"][0].as_py()
        if first_vec:
            cfg.dim = len(first_vec)

    t0 = time.perf_counter()
    store.create_corpus_collection(collection_name, cfg, disable_hnsw_for_bulk=True, recreate=recreate)
    ent_map = load_entity_ids_map(annotations_dir)

    marker_dir = vectors_dir / ".index_markers" / collection_name
    marker_dir.mkdir(parents=True, exist_ok=True)

    uploaded_points = 0
    skipped_shards = 0
    processed_shards = 0

    with ResourceMonitor() as mon:
        for shard_path in shard_files:
            rel = shard_path.relative_to(vectors_dir)
            shard_sha = _sha256_file(shard_path)
            marker_path = marker_dir / (str(rel).replace("/", "__").replace("\\", "__") + ".INDEXED.json")
            if not recreate and marker_path.is_file():
                try:
                    m = json.loads(marker_path.read_text(encoding="utf-8"))
                    if m.get("shard_sha256") == shard_sha and m.get("collection") == collection_name:
                        skipped_shards += 1
                        uploaded_points += int(m.get("points", 0))
                        continue
                except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    log.debug("Ignoring unreadable index marker %s: %s", marker_path, exc)

            pf = pq.ParquetFile(shard_path)
            shard_pts = 0
            for batch in pf.iter_batches(batch_size=batch_size):
                d = batch.to_pydict()
                n = len(d["chunk_id"])
                chunks: list[Chunk] = []
                dense_vecs: list[list[float]] = []
                sparse_vecs: list[dict[int, float]] = []
                emb_ver = str(d["embedder_version"][0]) if n and "embedder_version" in d else ""

                for i in range(n):
                    lang_raw = str(d["lang"][i] or "en")
                    try:
                        lang = Lang(lang_raw)
                    except ValueError:
                        lang = Lang.EN
                    chunks.append(
                        Chunk(
                            document_id=str(d["doc_id"][i]),
                            chunk_id=str(d["chunk_id"][i]),
                            ordinal=int(d["ordinal"][i]),
                            language=lang,
                            char_start=int(d["char_start"][i]),
                            char_end=int(d["char_end"][i]),
                            text=str(d["text"][i] or ""),
                            source_type=SourceType.CORPUS,
                            corpus_source=str(d["source"][i] or "corpus"),
                            token_count=int(d["n_tokens"][i]),
                        )
                    )
                    dense_vecs.append([float(x) for x in d["dense"][i]])
                    s_idx = d["sparse_indices"][i] or []
                    s_val = d["sparse_values"][i] or []
                    sparse_vecs.append({int(k): float(v) for k, v in zip(s_idx, s_val, strict=True)})

                shard_pts += store.upsert_chunks(
                    collection_name,
                    chunks,
                    dense_vecs,
                    sparse_vecs,
                    entity_ids_by_chunk=ent_map if ent_map else None,
                    embedder_version=emb_ver,
                    batch_size=batch_size,
                )

            marker_path.write_text(
                json.dumps(
                    {
                        "collection": collection_name,
                        "shard": str(rel),
                        "shard_sha256": shard_sha,
                        "points": shard_pts,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            uploaded_points += shard_pts
            processed_shards += 1

        upload_s = time.perf_counter() - t0
        hnsw_t0 = time.perf_counter()
        if enable_hnsw_after:
            store.enable_hnsw(collection_name, cfg)
        hnsw_s = time.perf_counter() - hnsw_t0

    verified_count = store.count(collection_name)
    alias_res = store.switch_alias(alias_name, collection_name) if switch_alias else None

    return {
        "collection": collection_name,
        "dim": cfg.dim,
        "quantization": cfg.quantization,
        "rescore": cfg.rescore,
        "oversampling": cfg.oversampling,
        "hnsw": {"m": cfg.hnsw_m, "ef_construct": cfg.hnsw_ef_construct, "on_disk": cfg.hnsw_on_disk},
        "shards_total": len(shard_files),
        "shards_processed": processed_shards,
        "shards_skipped": skipped_shards,
        "uploaded_points": uploaded_points,
        "verified_count": verified_count,
        "upload_seconds": round(upload_s, 3),
        "hnsw_enable_seconds": round(hnsw_s, 3),
        "points_per_s": round(uploaded_points / max(upload_s, 1e-6), 2),
        "alias_switch": alias_res,
        "memory_estimate": estimate_vector_bytes(verified_count, cfg.dim, cfg.quantization),
        "resources": mon.stats(uploaded_points, 0, 1).model_dump(),
    }


def benchmark_ann_vs_exact(
    store: QdrantStore,
    collection_name: str,
    query_vectors: np.ndarray,
    *,
    top_k: int = 10,
    candidate_k: int = 100,
    rescore: bool = True,
    oversampling: float = 3.0,
    hnsw_ef: int = 128,
) -> dict[str, Any]:
    """Compare approximate quantized search vs. ``exact=true`` over identical vectors & queries (RET-S)."""
    n_q = int(query_vectors.shape[0])
    if n_q == 0:
        return {"n_queries": 0, "recall@10": 0.0, "recall@100": 0.0}

    exact_top10: list[set[str]] = []
    exact_top100: list[set[str]] = []
    exact_lat_ms: list[float] = []

    for i in range(n_q):
        q = query_vectors[i].tolist()
        t0 = time.perf_counter()
        pts = store.client.query_points(
            collection_name=collection_name,
            query=q,
            using="dense",
            limit=candidate_k,
            search_params=models.SearchParams(exact=True),
            with_payload=False,
        ).points
        exact_lat_ms.append((time.perf_counter() - t0) * 1000.0)
        ids = [str(pt.id) for pt in pts]
        exact_top10.append(set(ids[:top_k]))
        exact_top100.append(set(ids[:candidate_k]))

    ann_rec10: list[float] = []
    ann_rec100: list[float] = []
    ann_lat_ms: list[float] = []

    t_start = time.perf_counter()
    for i in range(n_q):
        q = query_vectors[i].tolist()
        t0 = time.perf_counter()
        pts = store.client.query_points(
            collection_name=collection_name,
            query=q,
            using="dense",
            limit=candidate_k,
            search_params=models.SearchParams(
                exact=False,
                hnsw_ef=hnsw_ef,
                quantization=models.QuantizationSearchParams(rescore=rescore, oversampling=oversampling),
            ),
            with_payload=False,
        ).points
        ann_lat_ms.append((time.perf_counter() - t0) * 1000.0)
        ids = [str(pt.id) for pt in pts]
        ann_rec10.append(len(exact_top10[i] & set(ids[:top_k])) / max(len(exact_top10[i]), 1))
        ann_rec100.append(len(exact_top100[i] & set(ids[:candidate_k])) / max(len(exact_top100[i]), 1))
    total_ann_s = max(time.perf_counter() - t_start, 1e-6)

    ann_sorted = sorted(ann_lat_ms)
    ex_sorted = sorted(exact_lat_ms)
    p50_idx = int(0.50 * (n_q - 1))
    p95_idx = int(0.95 * (n_q - 1))

    return {
        "n_queries": n_q,
        "rescore": rescore,
        "oversampling": oversampling,
        "hnsw_ef": hnsw_ef,
        "ann_vs_exact_recall@10": round(statistics.fmean(ann_rec10), 6),
        "ann_vs_exact_recall@100": round(statistics.fmean(ann_rec100), 6),
        "ann_p50_ms": round(ann_sorted[p50_idx], 3),
        "ann_p95_ms": round(ann_sorted[p95_idx], 3),
        "ann_qps": round(n_q / total_ann_s, 2),
        "exact_p50_ms": round(ex_sorted[p50_idx], 3),
        "exact_p95_ms": round(ex_sorted[p95_idx], 3),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vectors", type=Path, default=ROOT / "data" / "task" / "vectors")
    ap.add_argument("--annotations", type=Path, default=None)
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument(
        "--qdrant-path", default=None, help="Local embedded Qdrant path (for self-contained cloud runs)"
    )
    ap.add_argument("--in-memory", action="store_true")
    ap.add_argument("--embedder-name", default=None)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--quantization", choices=["none", "scalar", "binary", "turbo"], default=None)
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--switch-alias", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "index.json")
    args = ap.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    params = load_params()
    emb_name = args.embedder_name or params["retrieval"]["embedder"]
    cfg = CorpusIndexConfig.from_params(params["retrieval"]["corpus_index"])
    if args.quantization is not None:
        cfg.quantization = args.quantization

    collection = versioned_collection_name(CORPUS_ALIAS, emb_name, args.version)
    store = QdrantStore(url=args.qdrant_url, path=args.qdrant_path, in_memory=args.in_memory)
    report = index_vector_shards(
        store,
        args.vectors,
        collection,
        cfg,
        annotations_dir=args.annotations,
        recreate=args.recreate,
        switch_alias=args.switch_alias,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    log.info(
        "Indexed %d points into %s (%.1f pts/s)", report["verified_count"], collection, report["points_per_s"]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
