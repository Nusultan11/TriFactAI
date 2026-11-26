"""Sharded, idempotent corpus embedding batch job (ADR-0014, ADR-0025, ROADMAP 6.2).

Reads ``data/task/chunks`` Parquet shards and writes fp16 dense + sparse vectors to
``data/task/vectors`` with cryptographic shard-resume markers:
- Full runs enforce cloud-only execution (ADR-0025) unless ``--allow-local-pilot`` is passed.
- Limited pilot runs (``--limit-chunks N``) emit ``.PILOT.json``, NEVER ``.SUCCESS.json``.
- Resumed full shards are skipped ONLY when input SHA-256, model/config fingerprint, output
  SHA-256, and row count all match.
- Optional ``--apply-downstream-filter`` enforces Stage 2 EDA rules (``curation.downstream``):
  drops ``kk`` chunks with Kazakh-letter share < 0.05 and masks checksum-valid IIN / email.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from trifact.batch.annotate_corpus import discover_chunk_shards, is_cloud_runtime
from trifact.core.params import ROOT, load_params
from trifact.lake.stats import ResourceMonitor, git_sha
from trifact.retrieval.embedder import HashEmbedder, MultilingualEmbedder

log = logging.getLogger("trifact.batch.embed_corpus")

_KK_LETTERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_IIN_RE = re.compile(r"\b(\d{12})\b")

VECTOR_SCHEMA = pa.schema(
    [
        ("chunk_id", pa.string()),
        ("doc_id", pa.string()),
        ("source", pa.string()),
        ("lang", pa.string()),
        ("title", pa.string()),
        ("ordinal", pa.int32()),
        ("char_start", pa.int32()),
        ("char_end", pa.int32()),
        ("n_tokens", pa.int32()),
        ("text", pa.string()),
        ("dense", pa.list_(pa.float16())),
        ("sparse_indices", pa.list_(pa.int32())),
        ("sparse_values", pa.list_(pa.float32())),
        ("embedder_version", pa.string()),
    ]
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _valid_iin(digits: str) -> bool:
    if len(digits) != 12 or not digits.isdigit():
        return False
    d = [int(c) for c in digits]
    w1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    c = sum(a * b for a, b in zip(d[:11], w1, strict=True)) % 11
    if c == 10:
        w2 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2]
        c = sum(a * b for a, b in zip(d[:11], w2, strict=True)) % 11
    return c == d[11]


def mask_pii(text: str) -> str:
    """Mask emails and checksum-valid 12-digit Kazakh IINs (curation.downstream)."""
    out = _EMAIL_RE.sub("[EMAIL]", text)
    return _IIN_RE.sub(lambda m: "[IIN]" if _valid_iin(m.group(1)) else m.group(0), out)


def kk_letter_share(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    kk_cnt = sum(1 for ch in letters if ch in _KK_LETTERS)
    return kk_cnt / len(letters)


def passes_downstream_filter(text: str, lang: str, min_kk_letter_share: float = 0.05) -> bool:
    if lang == "kk" and kk_letter_share(text) < min_kk_letter_share:
        return False
    return bool(text.strip())


def compute_embed_fingerprint(
    input_path: Path,
    embedder_version: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    input_sha = _sha256_file(input_path)
    cfg_str = json.dumps(config, sort_keys=True, ensure_ascii=False)
    combined = hashlib.sha256(f"{input_sha}|{embedder_version}|{cfg_str}".encode()).hexdigest()
    return {
        "combined": combined,
        "input_sha256": input_sha,
        "embedder_version": embedder_version,
        "config": config,
    }


def _shard_is_valid(out_parquet: Path, marker_path: Path, expected_fp: str) -> bool:
    if not out_parquet.is_file() or not marker_path.is_file():
        return False
    try:
        meta = json.loads(marker_path.read_text(encoding="utf-8"))
        if meta.get("status") != "success" or meta.get("pilot"):
            return False
        if meta.get("fingerprint") != expected_fp:
            return False
        if _sha256_file(out_parquet) != meta.get("output_sha256"):
            return False
        pf = pq.ParquetFile(out_parquet)
        return int(pf.metadata.num_rows) == int(meta.get("rows", -1))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def embed_shard(
    in_path: Path,
    out_parquet: Path,
    embedder: Any,
    *,
    contextual_header: bool = False,
    apply_downstream_filter: bool = False,
    min_kk_letter_share: float = 0.05,
    batch_size: int = 256,
    limit_chunks: int | None = None,
) -> dict[str, Any]:
    """Encode one chunk shard into fp16 dense + sparse Parquet atomically."""
    t0 = time.perf_counter()
    pf = pq.ParquetFile(in_path)
    available_cols = set(pf.schema_arrow.names)
    read_cols = [
        c
        for c in (
            "chunk_id",
            "doc_id",
            "source",
            "lang",
            "title",
            "ordinal",
            "char_start",
            "char_end",
            "n_tokens",
            "text",
        )
        if c in available_cols
    ]

    tmp_path = out_parquet.with_suffix(".parquet.tmp")
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    if tmp_path.exists():
        tmp_path.unlink()

    rows_in = 0
    rows_out = 0
    dropped_downstream = 0
    writer: pq.ParquetWriter | None = None

    try:
        for batch in pf.iter_batches(batch_size=batch_size, columns=read_cols):
            if limit_chunks is not None and rows_in >= limit_chunks:
                break
            d = batch.to_pydict()
            n = len(d["chunk_id"])
            if limit_chunks is not None and rows_in + n > limit_chunks:
                take = limit_chunks - rows_in
                d = {k: v[:take] for k, v in d.items()}
                n = take
            rows_in += n

            kept_idx: list[int] = []
            texts_to_embed: list[str] = []
            clean_texts: list[str] = []

            for i in range(n):
                raw_text = str(d["text"][i] or "")
                lang = str(d["lang"][i] or "en")
                if apply_downstream_filter:
                    if not passes_downstream_filter(raw_text, lang, min_kk_letter_share):
                        dropped_downstream += 1
                        continue
                    raw_text = mask_pii(raw_text)
                title = str(d["title"][i] or "") if "title" in d else ""
                emb_text = (
                    f"{title.strip()}\n{raw_text}".strip()
                    if (contextual_header and title.strip())
                    else raw_text
                )
                kept_idx.append(i)
                clean_texts.append(raw_text)
                texts_to_embed.append(emb_text)

            if not kept_idx:
                continue

            if hasattr(embedder, "embed_both"):
                dense_np, sparse_list = embedder.embed_both(texts_to_embed, is_query=False)
                dense_f16 = np.asarray(dense_np, dtype=np.float16)
            else:
                dense_f16 = np.asarray(embedder.embed_dense(texts_to_embed), dtype=np.float16)
                sparse_list = embedder.embed_sparse(texts_to_embed)

            sp_idx = [[int(k) for k in sv] for sv in sparse_list]
            sp_val = [[float(v) for v in sv.values()] for sv in sparse_list]

            tbl = pa.table(
                {
                    "chunk_id": [str(d["chunk_id"][i]) for i in kept_idx],
                    "doc_id": [str(d["doc_id"][i]) for i in kept_idx],
                    "source": [str(d["source"][i]) if "source" in d else "corpus" for i in kept_idx],
                    "lang": [str(d["lang"][i]) for i in kept_idx],
                    "title": [str(d["title"][i] or "") if "title" in d else "" for i in kept_idx],
                    "ordinal": [int(d["ordinal"][i]) if "ordinal" in d else 0 for i in kept_idx],
                    "char_start": [int(d["char_start"][i]) if "char_start" in d else 0 for i in kept_idx],
                    "char_end": [
                        int(d["char_end"][i]) if "char_end" in d else len(clean_texts[j])
                        for j, i in enumerate(kept_idx)
                    ],
                    "n_tokens": [int(d["n_tokens"][i]) if "n_tokens" in d else 0 for i in kept_idx],
                    "text": clean_texts,
                    "dense": pa.array(dense_f16.tolist(), type=pa.list_(pa.float16())),
                    "sparse_indices": pa.array(sp_idx, type=pa.list_(pa.int32())),
                    "sparse_values": pa.array(sp_val, type=pa.list_(pa.float32())),
                    "embedder_version": [str(embedder.version)] * len(kept_idx),
                },
                schema=VECTOR_SCHEMA,
            )
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, VECTOR_SCHEMA, compression="zstd")
            writer.write_table(tbl)
            rows_out += len(kept_idx)

        if writer is None:
            # Empty shard (all filtered) -> write valid empty parquet
            empty_tbl = pa.Table.from_pylist([], schema=VECTOR_SCHEMA)
            pq.write_table(empty_tbl, tmp_path, compression="zstd")
        else:
            writer.close()
            writer = None
        tmp_path.replace(out_parquet)
    except BaseException:
        if writer is not None:
            writer.close()
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    dt = max(time.perf_counter() - t0, 1e-6)
    return {
        "rows_in": rows_in,
        "rows_out": rows_out,
        "dropped_downstream": dropped_downstream,
        "seconds": round(dt, 3),
        "chunks_per_s": round(rows_out / dt, 2),
        "output_sha256": _sha256_file(out_parquet),
    }


def run_embed_corpus(
    input_dir: Path,
    output_dir: Path,
    embedder: Any,
    *,
    shard_spec: str | None = None,
    contextual_header: bool = False,
    apply_downstream_filter: bool = False,
    min_kk_letter_share: float = 0.05,
    batch_size: int = 256,
    limit_chunks: int | None = None,
    allow_local_pilot: bool = False,
) -> dict[str, Any]:
    """Run sharded idempotent embedding over ``input_dir`` -> ``output_dir``."""
    if limit_chunks is None and not allow_local_pilot and not is_cloud_runtime():
        raise RuntimeError(
            "Full-corpus embed_corpus must run on an authorized cloud runtime (ADR-0025). "
            "Pass --limit-chunks N or --allow-local-pilot for bounded local tests."
        )

    shards = discover_chunk_shards(input_dir, shard_spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "contextual_header": contextual_header,
        "apply_downstream_filter": apply_downstream_filter,
        "min_kk_letter_share": min_kk_letter_share,
    }

    t0 = time.perf_counter()
    skipped = 0
    processed = 0
    total_rows = 0
    shard_reports: list[dict[str, Any]] = []

    with ResourceMonitor() as mon:
        for in_path in shards:
            rel = in_path.relative_to(input_dir)
            out_parquet = output_dir / rel
            success_marker = out_parquet.with_suffix(".SUCCESS.json")
            pilot_marker = out_parquet.with_suffix(".PILOT.json")
            fp = compute_embed_fingerprint(in_path, str(embedder.version), cfg)

            if limit_chunks is None and _shard_is_valid(out_parquet, success_marker, fp["combined"]):
                skipped += 1
                meta = json.loads(success_marker.read_text(encoding="utf-8"))
                total_rows += int(meta.get("rows", 0))
                shard_reports.append(
                    {"shard": str(rel), "status": "skipped", "rows": int(meta.get("rows", 0))}
                )
                continue

            res = embed_shard(
                in_path,
                out_parquet,
                embedder,
                contextual_header=contextual_header,
                apply_downstream_filter=apply_downstream_filter,
                min_kk_letter_share=min_kk_letter_share,
                batch_size=batch_size,
                limit_chunks=limit_chunks,
            )
            processed += 1
            total_rows += res["rows_out"]

            marker_payload = {
                "status": "pilot" if limit_chunks is not None else "success",
                "pilot": limit_chunks is not None,
                "limit_chunks": limit_chunks,
                "fingerprint": fp["combined"],
                "input_sha256": fp["input_sha256"],
                "embedder_version": str(embedder.version),
                "output_sha256": res["output_sha256"],
                "rows": res["rows_out"],
                "dropped_downstream": res["dropped_downstream"],
                "seconds": res["seconds"],
                "chunks_per_s": res["chunks_per_s"],
                "git_sha": git_sha(ROOT),
            }
            target_marker = pilot_marker if limit_chunks is not None else success_marker
            target_marker.write_text(
                json.dumps(marker_payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            shard_reports.append({"shard": str(rel), "status": marker_payload["status"], **res})

    dt = max(time.perf_counter() - t0, 1e-6)
    summary = {
        "embedder_version": str(embedder.version),
        "shard_spec": shard_spec or "0/1",
        "pilot": limit_chunks is not None,
        "shards_total": len(shards),
        "shards_processed": processed,
        "shards_skipped": skipped,
        "total_rows": total_rows,
        "seconds": round(dt, 2),
        "chunks_per_s": round(total_rows / dt, 2),
        "truncation": embedder.truncation_stats.to_dict() if hasattr(embedder, "truncation_stats") else None,
        "resources": mon.stats(total_rows, 0, 1).model_dump(),
        "shards": shard_reports,
    }
    manifest_name = "embed_pilot_manifest.json" if limit_chunks is not None else "embed_manifest.json"
    (output_dir / manifest_name).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=ROOT / "data" / "task" / "chunks")
    ap.add_argument("--output", type=Path, default=ROOT / "data" / "task" / "vectors")
    ap.add_argument("--model", default=None)
    ap.add_argument("--mrl-dim", type=int, default=None)
    ap.add_argument("--shard", default=None, help="Shard slice i/N")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--contextual-header", action="store_true")
    ap.add_argument("--apply-downstream-filter", action="store_true")
    ap.add_argument("--limit-chunks", type=int, default=None)
    ap.add_argument(
        "--hash-embedder", action="store_true", help="Use deterministic HashEmbedder (tests/smoke only)"
    )
    ap.add_argument("--allow-local-pilot", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    params = load_params()
    if args.hash_embedder:
        embedder: Any = HashEmbedder(dim=args.mrl_dim or 64)
    else:
        model_name = args.model or params["retrieval"]["embedder"]
        embedder = MultilingualEmbedder(
            model_name,
            max_length=args.max_length,
            batch_size=args.batch_size,
            mrl_dim=args.mrl_dim,
        )

    min_kk = float(params["curation"]["downstream"]["kk_min_kk_letter_share"])
    summary = run_embed_corpus(
        args.input,
        args.output,
        embedder,
        shard_spec=args.shard,
        contextual_header=args.contextual_header,
        apply_downstream_filter=args.apply_downstream_filter,
        min_kk_letter_share=min_kk,
        batch_size=args.batch_size,
        limit_chunks=args.limit_chunks,
        allow_local_pilot=args.allow_local_pilot,
    )
    log.info(
        "embed_corpus finished: %d rows (%d processed, %d skipped) in %.1fs (%.1f chunks/s)",
        summary["total_rows"],
        summary["shards_processed"],
        summary["shards_skipped"],
        summary["seconds"],
        summary["chunks_per_s"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
