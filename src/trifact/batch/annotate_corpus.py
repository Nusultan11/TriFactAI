"""Sharded, idempotent corpus annotation batch job (ADR-0014, ADR-0025, ROADMAP 5.6).

Runs the eligible champion NER model (PyTorch fp16/fp32 by default; never rejected NER-5 int8)
+ rule-based DATE extractor + Wikidata Entity Linker over ``data/task/chunks`` shards and
persists calibrated entity mentions to ``data/task/annotations``.

Safety & resume invariants:
- Default backend is ``hf`` (eligible champion). Rejected ONNX artifacts are refused.
- Entity Linking (``--kb-db``) and calibration (``calibration.json``) are required by default;
  omission requires explicit opt-out flags (``--no-linking``, ``--allow-uncalibrated``).
- Full corpus runs enforce cloud-only execution (ADR-0025) unless ``--allow-local-pilot`` is passed.
- Limited pilot runs (``--limit-chunks N``) NEVER emit a full ``.SUCCESS.json`` marker and cannot
  cause a subsequent full run to skip the shard.
- Resumed full shards are skipped ONLY when input SHA-256, model fingerprint, KB fingerprint,
  calibration fingerprint, output SHA-256, and row count all match.
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


def is_cloud_runtime() -> bool:
    """Return True when executing on an authorized cloud runtime (Colab, Kaggle, or explicit flag)."""
    if os.environ.get("TRIFACT_CLOUD_RUNTIME", "").lower() in {"1", "true", "yes"}:
        return True
    if os.environ.get("COLAB_RELEASE_TAG") or Path("/content").is_dir():
        return True
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle").is_dir())


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


def _path_fingerprint(path: Path | None) -> str:
    """Fast, deterministic content fingerprint for a file or model directory."""
    if path is None or not path.exists():
        return "none"
    if path.is_file():
        return _sha256_file(path)
    # Directory (e.g. HF model dir): hash key config/calibration/weight sizes & mtimes/digests
    h = hashlib.sha256()
    for name in ("model_meta.json", "calibration.json", "config.json", "ner_onnx_meta.json"):
        p = path / name
        if p.is_file():
            h.update(name.encode("utf-8"))
            h.update(_sha256_file(p).encode("utf-8"))
    for weight_name in ("model.safetensors", "pytorch_model.bin", "model.onnx"):
        wp = path / weight_name
        if wp.is_file():
            st = wp.stat()
            h.update(f"{weight_name}:{st.st_size}".encode())
            # Hash first + last 1 MB for fast tamper/change detection
            with wp.open("rb") as f:
                h.update(f.read(1 << 20))
                if st.st_size > (1 << 20):
                    f.seek(max(0, st.st_size - (1 << 20)))
                    h.update(f.read(1 << 20))
    return h.hexdigest()


def compute_shard_fingerprint(
    in_parquet: Path,
    model_version: str,
    model_fingerprint: str = "unknown",
    kb_fingerprint: str = "none",
    calibration_fingerprint: str = "none",
    config_params: dict[str, Any] | None = None,
) -> str:
    """Compute SHA-256 fingerprint binding input shard + model + KB + calibration + config."""
    payload = {
        "input_sha256": _sha256_file(in_parquet),
        "model_version": model_version,
        "model_fingerprint": model_fingerprint,
        "kb_fingerprint": kb_fingerprint,
        "calibration_fingerprint": calibration_fingerprint,
        "config": config_params or {},
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_mention_offsets(mention: EntityMention, chunk_text: str) -> bool:
    """Ensure mention offsets are strictly in-bounds and match ``chunk_text``."""
    n = len(chunk_text)
    if not (0 <= mention.char_start < mention.char_end <= n):
        return False
    if chunk_text[mention.char_start : mention.char_end] != mention.surface:
        return False
    return 0.0 <= mention.score <= 1.0


def _can_reuse_completed_shard(
    in_parquet: Path,
    out_parquet: Path,
    marker: Path,
    expected_fingerprint: str,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Verify marker integrity, fingerprint match, output checksum, and completeness before skipping."""
    if not marker.is_file():
        return False, "no_marker", None
    if not out_parquet.is_file() or out_parquet.stat().st_size == 0:
        return False, "missing_or_empty_output", None
    try:
        prev = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "corrupted_marker", None
    if prev.get("partial", False):
        return False, "partial_pilot_marker", None
    if prev.get("fingerprint") != expected_fingerprint:
        return False, "stale_fingerprint", None
    expected_rows = pq.ParquetFile(in_parquet).metadata.num_rows
    if prev.get("chunks") != expected_rows:
        return False, "incomplete_chunk_count", None
    actual_sha = _sha256_file(out_parquet)
    if prev.get("sha256") != actual_sha:
        return False, "corrupted_output_sha256", None
    return True, "ok", prev


def annotate_shard(
    in_parquet: Path,
    out_parquet: Path,
    extractor: Any,
    linker: WikidataEntityLinker | None = None,
    *,
    batch_size: int = 64,
    limit_chunks: int | None = None,
    compression: str = "zstd",
    model_fingerprint: str = "unknown",
    kb_fingerprint: str = "none",
    calibration_fingerprint: str = "none",
    config_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Annotate one input chunk Parquet shard atomically with verified resume & pilot protection."""
    is_partial = limit_chunks is not None
    model_version = getattr(extractor, "version", "unknown")
    expected_fp = compute_shard_fingerprint(
        in_parquet,
        model_version=model_version,
        model_fingerprint=model_fingerprint,
        kb_fingerprint=kb_fingerprint,
        calibration_fingerprint=calibration_fingerprint,
        config_params=config_params,
    )
    success_marker = out_parquet.with_suffix(out_parquet.suffix + ".SUCCESS.json")
    pilot_marker = out_parquet.with_suffix(out_parquet.suffix + ".PILOT.json")

    if not is_partial:
        reusable, reason, prev = _can_reuse_completed_shard(
            in_parquet, out_parquet, success_marker, expected_fp
        )
        if reusable and prev is not None:
            prev["skipped"] = True
            prev["resume_check"] = "verified"
            return prev
        if reason != "no_marker":
            log.warning("Re-running shard %s because resume check failed: %s", in_parquet.name, reason)

    t0 = time.perf_counter()
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_parquet.with_suffix(".parquet.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    pf = pq.ParquetFile(in_parquet)
    total_input_rows = pf.metadata.num_rows
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
    actual_partial = is_partial or (n_chunks < total_input_rows)
    receipt: dict[str, Any] = {
        "input": str(in_parquet),
        "output": str(out_parquet),
        "fingerprint": expected_fp,
        "sha256": _sha256_file(out_parquet),
        "bytes": out_parquet.stat().st_size,
        "chunks": n_chunks,
        "total_input_chunks": total_input_rows,
        "partial": actual_partial,
        "limit_chunks": limit_chunks,
        "mentions": n_mentions,
        "linkable_mentions": n_linkable_mentions,
        "linked_mentions": n_linked,
        "nil_rate": nil_rate,
        "invalid_dropped": n_invalid_dropped,
        "by_type": dict(by_type),
        "by_lang": dict(by_lang),
        "model_version": model_version,
        "linking_enabled": linker is not None,
        "seconds": round(secs, 2),
        "chunks_per_s": round(n_chunks / secs, 2),
        "skipped": False,
    }
    if actual_partial:
        # Never write .SUCCESS.json for a limited pilot run; remove stale success marker if present
        success_marker.unlink(missing_ok=True)
        pilot_marker.write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )
    else:
        pilot_marker.unlink(missing_ok=True)
        success_marker.write_text(
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
    model_fingerprint: str = "unknown",
    kb_fingerprint: str = "none",
    calibration_fingerprint: str = "none",
    config_params: dict[str, Any] | None = None,
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
                model_fingerprint=model_fingerprint,
                kb_fingerprint=kb_fingerprint,
                calibration_fingerprint=calibration_fingerprint,
                config_params=config_params,
            )
            receipts.append(rec)
            log.info(
                "Shard %s: chunks=%d mentions=%d nil_rate=%.3f (%.1f chunks/s, partial=%s, skipped=%s)",
                rel.as_posix(),
                rec["chunks"],
                rec["mentions"],
                rec["nil_rate"],
                rec["chunks_per_s"],
                rec["partial"],
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
        "partial": any(r["partial"] for r in receipts) or max_shards is not None,
        "total_chunks": tot_chunks,
        "total_mentions": tot_mentions,
        "linkable_mentions": tot_linkable,
        "linked_mentions": tot_linked,
        "nil_rate": round(1.0 - (tot_linked / tot_linkable), 4) if tot_linkable else 0.0,
        "by_type": dict(agg_type),
        "by_lang": dict(agg_lang),
        "linking_enabled": linker is not None,
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
    ap.add_argument("--backend", choices=["hf", "onnx"], default="hf")
    ap.add_argument("--kb-db", type=Path, default=ROOT / "data" / "task" / "kb" / "kb_index.sqlite")
    ap.add_argument("--no-linking", action="store_true", help="Explicitly disable Wikidata Entity Linking")
    ap.add_argument(
        "--allow-uncalibrated", action="store_true", help="Allow running without calibration.json"
    )
    ap.add_argument(
        "--allow-unverified-onnx", action="store_true", help="Allow loading ONNX model without gate pass"
    )
    ap.add_argument(
        "--allow-local-pilot", action="store_true", help="Allow local bounded execution outside cloud"
    )
    ap.add_argument("--shard", type=str, default=None, help="Shard spec i/N (0 <= i < N)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit-chunks", type=int, default=None)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--nil-threshold", type=float, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    params = load_params()

    if not is_cloud_runtime() and not args.allow_local_pilot:
        raise RuntimeError(
            "Refusing to run corpus annotation on local PC (ADR-0025 cloud-only compute policy). "
            "Run on Colab/Kaggle or pass --allow-local-pilot for bounded local validation."
        )

    if args.backend == "onnx":
        ner_ext: Any = OnnxNERExtractor(
            args.model,
            batch_size=args.batch_size,
            allow_unverified=args.allow_unverified_onnx,
        )
    else:
        ner_ext = TokenClassificationExtractor(
            args.model,
            batch_size=args.batch_size,
            fp16=args.fp16,
            require_calibrated=not args.allow_uncalibrated,
        )

    combined = CombinedExtractor(ner_ext, RuleBasedDateExtractor(include_relative=True))

    nil_thr = float(
        args.nil_threshold if args.nil_threshold is not None else params["linking"].get("nil_threshold", 0.35)
    )
    linker: WikidataEntityLinker | None = None
    kb_idx: SQLiteKBIndex | InMemoryKBIndex | None = None
    kb_fp = "none"
    if not args.no_linking:
        if not args.kb_db or not args.kb_db.is_file():
            raise FileNotFoundError(
                f"Wikidata KB index not found at {args.kb_db}. Build it via "
                "'python -m trifact.extraction.linking build-index' or pass --no-linking explicitly."
            )
        kb_idx = SQLiteKBIndex(args.kb_db)
        kb_fp = _path_fingerprint(args.kb_db)
        linker = WikidataEntityLinker(
            kb_idx,
            candidates_k=int(params["linking"]["candidates_k"]),
            nil_threshold=nil_thr,
        )

    model_fp = _path_fingerprint(args.model)
    cal_file = args.model / "calibration.json"
    cal_fp = _path_fingerprint(cal_file) if cal_file.is_file() else "none"
    cfg_params = {
        "backend": args.backend,
        "fp16": args.fp16,
        "linking_enabled": linker is not None,
        "nil_threshold": nil_thr,
        "candidates_k": int(params["linking"]["candidates_k"]),
    }

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
            model_fingerprint=model_fp,
            kb_fingerprint=kb_fp,
            calibration_fingerprint=cal_fp,
            config_params=cfg_params,
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
