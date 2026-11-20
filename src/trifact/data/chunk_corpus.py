"""Stage chunk_corpus: curated corpus documents -> retrieval chunks (curated/corpus -> task/chunks).

python -m trifact.data.chunk_corpus [--profile full|pilot] [--workers N]

Every document is cut by `trifact.ingestion.chunker` into chunks of up to `chunking.max_tokens` tokens of
the retrieval tokenizer (XLM-R sentencepiece, the BGE-M3 vocabulary; pinned in
`curation.stats.tokenizers.xlmr`) with `chunking.overlap_tokens` of sentence overlap. Rows:
`chunk_id (<doc_id>#<k>), doc_id, source, lang, domain, title, url, ordinal, char_start, char_end, n_tokens,
text`. One work unit per corpus shard, resumable and fingerprinted like the curation stages (ADR-0014).
The `pilot` profile keeps the first chunks up to `scale.pilot.corpus_chunks` spread over all shards.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa

from trifact.core.params import ROOT, load_params
from trifact.data.curation import (
    StageDir,
    Unit,
    UnitWriter,
    dataset_input,
    fingerprint,
    iter_batches,
    licenses_of,
    run_units,
    stage_report,
    sum_by_lang,
    write_report,
)
from trifact.ingestion.chunker import chunk_many, hf_counter
from trifact.lake.layout import LakeConfig
from trifact.lake.manifest import DATASET_NAME, DatasetManifest, ShardEntry
from trifact.lake.stats import ResourceMonitor, git_sha, log_mlflow

log = logging.getLogger("trifact.chunk_corpus")
CODE = [Path(__file__), ROOT / "src/trifact/ingestion/chunker.py", ROOT / "src/trifact/ingestion/segment.py"]
SCHEMA = pa.schema(
    [
        ("chunk_id", pa.string()), ("doc_id", pa.string()), ("source", pa.string()), ("lang", pa.string()),
        ("domain", pa.string()), ("title", pa.string()), ("url", pa.string()), ("ordinal", pa.int32()),
        ("char_start", pa.int32()), ("char_end", pa.int32()), ("n_tokens", pa.int32()), ("text", pa.string()),
    ]
)  # fmt: skip
_CTX: dict[str, Any] = {}


def init_worker(ctx: dict[str, Any]) -> None:
    _CTX.clear()
    _CTX.update(ctx)
    _CTX["count"] = hf_counter(ctx["tokenizer"])


def chunk_unit(u: Unit) -> dict[str, Any]:
    out = Path(_CTX["out"])
    w = UnitWriter(out / u.source, u.idx, "part", _CTX["lake"], SCHEMA)
    count, mx, ov = _CTX["count"], _CTX["max_tokens"], _CTX["overlap_tokens"]
    docs = chunks = tokens = 0
    hist = [0] * 8  # chunks per document: 1, 2, 3-4, 5-8, 9-16, 17-32, 33-64, 65+
    try:
        cols = ["doc_id", "source", "lang", "domain", "title", "url", "text"]
        for b in iter_batches(Path(_CTX["root"]) / u.path, cols, u.limit):
            rows: dict[str, list[Any]] = {n: [] for n in SCHEMA.names}
            recs = pl.from_arrow(b).to_dicts()  # type: ignore[union-attr]
            all_spans = chunk_many([r["text"] or "" for r in recs], count, mx, ov)
            for r, spans in zip(recs, all_spans, strict=True):
                docs += 1
                hist[min(7, max(0, math.ceil(math.log2(max(len(spans), 1)))))] += 1
                for k, s in enumerate(spans):
                    rows["chunk_id"].append(f"{r['doc_id']}#{k}")
                    for c in ("doc_id", "source", "lang", "domain", "title", "url"):
                        rows[c].append(r[c])
                    rows["ordinal"].append(k)
                    rows["char_start"].append(s.start)
                    rows["char_end"].append(s.end)
                    rows["n_tokens"].append(s.tokens)
                    rows["text"].append(r["text"][s.start : s.end])
                    tokens += s.tokens
                chunks += len(spans)
            t = pa.table(rows, schema=SCHEMA)
            for (lang,), part in pl.from_arrow(t).group_by("lang"):  # type: ignore[union-attr]
                w.write(str(lang), part.to_arrow())
        shards = w.close()
    except BaseException:
        w.abort()
        raise
    for s in shards:
        s["path"] = f"{u.source}/{s['path']}"
    return {"shards": shards, "rows_in": {u.lang: docs}, "chunks": {u.lang: chunks}, "tokens": {u.lang: tokens},
            "chunks_per_doc_hist": hist}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=["pilot", "full"])
    ap.add_argument("--workers", type=int)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    params = load_params()
    profile = args.profile or params["scale"]["profile"]
    root = (ROOT / params["lake"]["root"]).resolve()
    corpus = root / "curated" / "corpus"
    out = root / "task" / "chunks"
    m_in = DatasetManifest.read(corpus)
    parts = sorted((s for s in m_in.shards if s.table == "part"), key=lambda s: s.path)
    cap = params["scale"][profile]["corpus_chunks"] if profile == "pilot" else None
    # pilot: about cap / 3 chunks per shard is plenty (most documents give 1-3 chunks)
    per_shard = None if cap is None else max(1, math.ceil(cap / max(len(parts), 1) / 3))
    units = [
        Unit(i, s.path.split("/", 1)[0], f"curated/corpus/{s.path}", s.lang, s.rows, "", per_shard)
        for i, s in enumerate(parts)
    ]
    tok = params["curation"]["stats"]["tokenizers"]["xlmr"]
    from huggingface_hub import hf_hub_download

    tok_path = hf_hub_download(tok["hf_id"], tok["file"], revision=tok["revision"],
                               cache_dir=str(ROOT / params["curation"]["langid"]["cache"] / "hf"))  # fmt: skip
    stage_params = {
        "chunking": params["chunking"],
        "tokenizer": tok,
        "profile": profile,
        "docs_per_shard": per_shard,
    }
    fp = fingerprint("chunk_corpus", stage_params, CODE, [corpus / DATASET_NAME])
    sd = StageDir(out, fp)
    if sd.complete():
        log.info("%s: up to date", out)
        return 0
    sd.prepare()
    workers = args.workers or params["curation"]["workers"]
    lake = LakeConfig.from_params(params["lake"], root)
    ctx = {"root": str(root), "out": str(out), "lake": lake, "tokenizer": tok_path,
           "max_tokens": params["chunking"]["max_tokens"], "overlap_tokens": params["chunking"]["overlap_tokens"]}  # fmt: skip
    t0 = time.perf_counter()
    with ResourceMonitor() as mon:
        results = run_units(units, sd, chunk_unit, workers, init_worker, (ctx,))
    res = [results[u.idx] for u in units]
    shards = [ShardEntry.model_validate(s) for r in res for s in r["shards"]]
    chunks, tokens = sum_by_lang(res, "chunks"), sum_by_lang(res, "tokens")
    hist = [sum(r["chunks_per_doc_hist"][i] for r in res) for i in range(8)]
    rows_in = sum_by_lang(res, "rows_in")
    m = DatasetManifest(
        name="task/chunks", layer="task", stage="chunk_corpus", source=m_in.source,
        description=f"Retrieval chunks of the curated corpus: up to {params['chunking']['max_tokens']} XLM-R tokens, "
        f"{params['chunking']['overlap_tokens']} tokens of sentence overlap, exact character offsets into the document.",
        code_version=git_sha(ROOT), fingerprint=fp, params=stage_params,
        inputs=[dataset_input(root, corpus / DATASET_NAME, m_in.rows_of())], licenses=licenses_of([m_in]),
        rows_in=rows_in, shards=shards, stats=mon.stats(sum(rows_in.values()), 0, workers),
        metrics={"chunks_by_lang": chunks, "tokens_by_lang": tokens,
                 "mean_tokens_per_chunk": {k: round(tokens[k] / max(chunks[k], 1), 1) for k in chunks},
                 "chunks_per_doc_hist": dict(zip(["1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65+"], hist, strict=True))},
        partial=profile != "full",
    )  # fmt: skip
    sd.finish(m)
    write_report(ROOT / "reports" / "chunks.json", stage_report(m))
    log_mlflow("task", "chunk_corpus", {**stage_params, "fingerprint": fp}, {
        "docs": sum(rows_in.values()), "chunks": sum(chunks.values()), "seconds": round(time.perf_counter() - t0, 1),
        **{f"chunks_{k}": v for k, v in chunks.items()}})  # fmt: skip
    print("chunks", chunks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
