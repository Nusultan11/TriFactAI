"""Sharded, idempotent corpus annotation batch job (ADR-0014, ROADMAP 5.6).

Runs the champion NER model + rule-based DATE extractor + Wikidata Entity Linker over
``data/task/chunks`` shards and persists calibrated entity mentions to ``data/task/annotations``.

Usage:
    python -m trifact.batch.annotate_corpus \\
        --input data/task/chunks --output data/task/annotations \\
        --model models/ner/champion --backend onnx \\
        --kb-db data/task/kb/kb_index.sqlite --shard 0/4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from trifact.core.params import ROOT, load_params
from trifact.core.schemas import Chunk, EntityMention, EntityType, Lang, SourceType
from trifact.extraction.dates import RuleBasedDateExtractor
from trifact.extraction.linking import InMemoryKBIndex, SQLiteKBIndex, WikidataEntityLinker
from trifact.extraction.ner import CombinedExtractor, OnnxNERExtractor, TokenClassificationExtractor
from trifact.lake.stats import ResourceMonitor, git_sha

log = logging.getLogger("trifact.batch.annotate_corpus")

ANNOTATION_SCHEMA = pa.schema(
    [
        ("chunk_id", pa.string()),
        ("doc_id", pa.string()),
        ("source", pa.string()),
        ("lang", pa.string()),
        ("surface", pa.string()),
        ("type", pa.string()),
        ("char_start", pa.int32()),
        ("char_end", pa.int32()),
        ("score", pa.float32()),
        ("wikidata_qid", pa.string()),
        ("model_version", pa.string()),
    ]
)


def parse_shard_spec(spec: str | None) -> tuple[int, int]:
    """Parse ``i/N`` shard specification (0 <= i < N). ``None`` processes all shards (0/1)."""
    if not spec:
        return 0, 1
    parts = spec.strip().split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid --shard spec {spec!r}; expected 'i/N' with 0 <= i < N")
    idx, total = int(parts[0]), int(parts[1])
    if total <= 0 or not (0 <= idx < total):
        raise ValueError(f"Invalid --shard bounds {idx}/{total}; require 0 <= i < N and N > 0")
    return idx, total


def discover_chunk_shards(input_dir: Path, shard_spec: str | None = None) -> list[Path]:
    """Return sorted Parquet shard paths assigned to ``shard_spec``."""
    all_files = sorted(p for p in input_dir.rglob("*.parquet") if "_work" not in p.parts)
    idx, total = parse_shard_spec(shard_spec)
    return [p for i, p in enumerate(all_files) if i % total == idx]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_mention_offsets(mention: EntityMention, chunk_text: str) -> bool:
    """Ensure mention offsets are strictly in-bounds and match ``chunk_text``."""
    n = len(chunk_text)
    if not (0 <= mention.char_start < mention.char_end <= n):
        return False
    if chunk_text[mention.char_start : mention.char_end] != mention.surface:
        return False
    return 0.0 <= mention.score <= 1.0


def annotate_shard(
    in_parquet: Path,
    out_parquet: Path,
    extractor: Any,
    linker: WikidataEntityLinker | None = None,
    *,
    batch_size: int = 64,
    limit_chunks: int | None = None,
    compression: str = "zstd",
) -> dict[str, Any]:
    """Annotate one input chunk Parquet shard atomically with ``_SUCCESS`` idempotency marker."""
    marker = out_parquet.with_suffix(out_parquet.suffix + ".SUCCESS.json")
    if marker.is_file() and out_parquet.is_file() and out_parquet.stat().st_size > 0:
        prev = json.loads(marker.read_text(encoding="utf-8"))
        prev["skipped"] = True
        return prev

    t0 = time.perf_counter()
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_parquet.with_suffix(".parquet.tmp")

    pf = pq.ParquetFile(in_parquet)
    cols_available = set(pf.schema_arrow.names)
    read_cols = [
        c
        for c in ("chunk_id", "doc_id", "source", "lang", "ordinal", "char_start", "char_end", "text")
        if c in cols_available
    ]

    n_chunks = 0
    n_mentions = 0
    n_linked = 0
    n_linkable_mentions = 0
    n_invalid_dropped = 0
    by_type: Counter[str] = Counter()
    by_lang: Counter[str] = Counter()

    writer = pq.ParquetWriter(tmp_path, ANNOTATION_SCHEMA, compression=compression)
    try:
        for rb in pf.iter_batches(batch_size=batch_size, columns=read_cols):
            d = rb.to_pydict()
            m_len = len(d["chunk_id"])
            if limit_chunks is not None:
                rem = limit_chunks - n_chunks
                if rem <= 0:
                    break
                m_len = min(m_len, rem)

            chunks: list[Chunk] = []
            meta_by_chunk: dict[str, tuple[str, str, str, str]] = {}
            for i in range(m_len):
                cid = str(d["chunk_id"][i])
                did = str(d["doc_id"][i]) if "doc_id" in d else cid.split("#", 1)[0]
                src = str(d["source"][i]) if "source" in d else "corpus"
                lg_raw = str(d["lang"][i]) if "lang" in d else "en"
                lg = Lang(lg_raw) if lg_raw in {"kk", "ru", "en"} else Lang.EN
                txt = str(d["text"][i] or "")
                ch = Chunk(
                    document_id=did,
                    chunk_id=cid,
                    ordinal=int(d["ordinal"][i]) if "ordinal" in d else 0,
                    language=lg,
                    char_start=int(d["char_start"][i]) if "char_start" in d else 0,
                    char_end=int(d["char_end"][i]) if "char_end" in d else len(txt),
                    text=txt,
                    source_type=SourceType.CORPUS,
                    corpus_source=src,
                )
                chunks.append(ch)
                meta_by_chunk[cid] = (did, src, lg.value, txt)

            n_chunks += len(chunks)
            mentions: Sequence[EntityMention] = extractor.extract(chunks)
            if linker is not None and mentions:
                mentions = linker.link(mentions, chunks)

            out_cols: dict[str, list[Any]] = {name: [] for name in ANNOTATION_SCHEMA.names}
            for m in mentions:
                did, src, lg_val, txt = meta_by_chunk[m.chunk_id]
                if not validate_mention_offsets(m, txt):
                    n_invalid_dropped += 1
                    continue
                mt = m.type.value if isinstance(m.type, EntityType) else str(m.type)
                out_cols["chunk_id"].append(m.chunk_id)
                out_cols["doc_id"].append(did)
                out_cols["source"].append(src)
                out_cols["lang"].append(lg_val)
                out_cols["surface"].append(m.surface)
                out_cols["type"].append(mt)
                out_cols["char_start"].append(int(m.char_start))
                out_cols["char_end"].append(int(m.char_end))
                out_cols["score"].append(float(m.score))
                out_cols["wikidata_qid"].append(m.wikidata_qid)
                out_cols["model_version"].append(m.model_version)
                n_mentions += 1
                by_type[mt] += 1
                by_lang[lg_val] += 1
                if mt != EntityType.DATE.value:
                    n_linkable_mentions += 1
                    if m.wikidata_qid is not None:
                        n_linked += 1

            if out_cols["chunk_id"]:
                writer.write_table(pa.table(out_cols, schema=ANNOTATION_SCHEMA))

            if limit_chunks is not None and n_chunks >= limit_chunks:
                break
    finally:
        writer.close()

    os.replace(tmp_path, out_parquet)
    secs = max(time.perf_counter() - t0, 1e-6)
    nil_rate = round(1.0 - (n_linked / n_linkable_mentions), 4) if n_linkable_mentions > 0 else 0.0
    receipt: dict[str, Any] = {
        "input": str(in_parquet),
        "output": str(out_parquet),
        "sha256": _sha256_file(out_parquet),
        "bytes": out_parquet.stat().st_size,
        "chunks": n_chunks,
        "mentions": n_mentions,
        "linkable_mentions": n_linkable_mentions,
        "linked_mentions": n_linked,
        "nil_rate": nil_rate,
        "invalid_dropped": n_invalid_dropped,
        "by_type": dict(by_type),
        "by_lang": dict(by_lang),
        "model_version": getattr(extractor, "version", "unknown"),
        "seconds": round(secs, 2),
        "chunks_per_s": round(n_chunks / secs, 2),
        "skipped": False,
    }
    marker.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return receipt


def run_annotate_corpus(
    input_dir: Path,
    output_dir: Path,
    extractor: Any,
    linker: WikidataEntityLinker | None = None,
    *,
    shard_spec: str | None = None,
    batch_size: int = 64,
    limit_chunks_per_shard: int | None = None,
    max_shards: int | None = None,
) -> dict[str, Any]:
    """Run sharded corpus annotation over ``input_dir`` -> ``output_dir``."""
    shards = discover_chunk_shards(input_dir, shard_spec=shard_spec)
    if max_shards is not None:
        shards = shards[:max_shards]

    receipts: list[dict[str, Any]] = []
    with ResourceMonitor() as mon:
        for fp in shards:
            rel = fp.relative_to(input_dir)
            out_fp = output_dir / rel
            rec = annotate_shard(
                fp,
                out_fp,
                extractor,
                linker,
                batch_size=batch_size,
                limit_chunks=limit_chunks_per_shard,
            )
            receipts.append(rec)
            log.info(
                "Shard %s: chunks=%d mentions=%d nil_rate=%.3f (%.1f chunks/s, skipped=%s)",
                rel.as_posix(),
                rec["chunks"],
                rec["mentions"],
                rec["nil_rate"],
                rec["chunks_per_s"],
                rec["skipped"],
            )

    tot_chunks = sum(r["chunks"] for r in receipts)
    tot_mentions = sum(r["mentions"] for r in receipts)
    tot_linkable = sum(r["linkable_mentions"] for r in receipts)
    tot_linked = sum(r["linked_mentions"] for r in receipts)
    agg_type: Counter[str] = Counter()
    agg_lang: Counter[str] = Counter()
    for r in receipts:
        agg_type.update(r["by_type"])
        agg_lang.update(r["by_lang"])

    summary = {
        "git_sha": git_sha(ROOT),
        "shard_spec": shard_spec or "0/1",
        "shards_processed": len(receipts),
        "shards_skipped": sum(1 for r in receipts if r["skipped"]),
        "total_chunks": tot_chunks,
        "total_mentions": tot_mentions,
        "linkable_mentions": tot_linkable,
        "linked_mentions": tot_linked,
        "nil_rate": round(1.0 - (tot_linked / tot_linkable), 4) if tot_linkable else 0.0,
        "by_type": dict(agg_type),
        "by_lang": dict(agg_lang),
        "wall_seconds": round(mon.seconds, 2),
        "chunks_per_s": round(tot_chunks / max(mon.seconds, 1e-6), 2),
        "peak_rss_mb": round(mon.peak_rss_mb, 1),
        "model_version": getattr(extractor, "version", "unknown"),
        "receipts": receipts,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_name = f"summary_{shard_spec.replace('/', '_of_')}.json" if shard_spec else "summary.json"
    (output_dir / summary_name).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=ROOT / "data" / "task" / "chunks")
    ap.add_argument("--output", type=Path, default=ROOT / "data" / "task" / "annotations")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "ner")
    ap.add_argument("--backend", choices=["hf", "onnx"], default="onnx")
    ap.add_argument("--kb-db", type=Path, default=None, help="SQLite KB index path for entity linking")
    ap.add_argument("--shard", type=str, default=None, help="Shard spec i/N (0 <= i < N)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit-chunks", type=int, default=None)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--nil-threshold", type=float, default=0.35)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    params = load_params()

    if args.backend == "onnx":
        ner_ext: Any = OnnxNERExtractor(args.model, batch_size=args.batch_size)
    else:
        ner_ext = TokenClassificationExtractor(args.model, batch_size=args.batch_size, fp16=args.fp16)

    combined = CombinedExtractor(ner_ext, RuleBasedDateExtractor(include_relative=True))

    linker: WikidataEntityLinker | None = None
    kb_idx: SQLiteKBIndex | InMemoryKBIndex | None = None
    if args.kb_db and args.kb_db.is_file():
        kb_idx = SQLiteKBIndex(args.kb_db)
        linker = WikidataEntityLinker(
            kb_idx,
            candidates_k=int(params["linking"]["candidates_k"]),
            nil_threshold=args.nil_threshold,
        )

    try:
        summary = run_annotate_corpus(
            args.input,
            args.output,
            combined,
            linker,
            shard_spec=args.shard,
            batch_size=args.batch_size,
            limit_chunks_per_shard=args.limit_chunks,
            max_shards=args.max_shards,
        )
    finally:
        if isinstance(kb_idx, SQLiteKBIndex):
            kb_idx.close()

    log.info(
        "Annotation complete: %d chunks, %d mentions, %.1f chunks/s, peak RSS %.1f MB",
        summary["total_chunks"],
        summary["total_mentions"],
        summary["chunks_per_s"],
        summary["peak_rss_mb"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
