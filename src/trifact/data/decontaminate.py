"""Stage decontaminate: drop corpus documents sharing word 13-grams with eval candidates (curated/dedup -> curated/corpus).

python -m trifact.data.decontaminate [--dedup DIR] [--out DIR] [--workers N]

Eval candidates (`params.yaml: curation.decontamination.sources`) are every split of the sources
that stage 3 builds eval sets from (QA contexts and questions, NLI pairs, NER sentences, MIRACL
queries and positive passages): the frozen test split does not exist yet and is cut by source
document later (ADR-0007), so any of their texts may end up in a test set.

1. Eval grams: eval texts are normalized like the corpus (`trifact.ingestion.normalize`) and
   hashed into word `ngram`-grams (`trifact.data.hashing`); texts shorter than `ngram` words are
   skipped (counted in the report). Result: sorted unique gram hashes with the first eval source.
2. Matches (parallel, one unit per shard): each document's grams are looked up by binary search;
   (row, gram) hits are kept in the work directory.
3. Template grams: an eval gram found in more than `max_gram_docs` corpus documents is template
   text (licence lines, navigation, common phrases), not leakage, and is ignored; the list is
   reported.
4. Filter (parallel): in a document with >= `min_matches` remaining grams the lines (paragraphs)
   holding a gram are removed, so one eval sentence does not cost a whole article; the document
   is dropped instead when a gram spans lines (still found after the removal) or more than
   `max_removed_share` of its characters would go. Every such document is logged in table
   `contaminated` (eval sources, grams, action, lines and characters removed); edited documents
   get new `text`, `n_chars`, `n_words` and `decontaminated_lines` > 0, their `quality` signals
   stay those of the clean stage.
5. Post-check: the output is scanned again with the same grams and must have zero hits.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

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
    write_report,
)
from trifact.data.hashing import ngram_hashes, word_hashes
from trifact.ingestion.normalize import normalize_series
from trifact.lake.layout import LakeConfig, source_dir
from trifact.lake.manifest import DATASET_NAME, MANIFEST_NAME, DatasetManifest, ShardEntry, SourceManifest
from trifact.lake.stats import ResourceMonitor, git_sha, log_mlflow

log = logging.getLogger("trifact.decontaminate")
CODE = [
    Path(__file__),
    ROOT / "src/trifact/data/hashing.py",
    ROOT / "src/trifact/data/curation.py",
    ROOT / "src/trifact/ingestion/normalize.py",
]
TOP_TEMPLATES = 30


# ----------------------------------------------------------------------------- eval texts


def eval_texts(root: Path, name: str, spec: dict[str, Any]) -> pl.DataFrame:
    """Columns `eval_source`, `eval_id`, `text` of one eval-candidate source (all splits)."""
    base = source_dir(root, "raw", name)
    m = SourceManifest.read(base)
    frames = []
    if "passages_from" in spec:  # MIRACL: queries + positive passages from the corpus source
        topics = [str(base / s.path) for s in m.shards if s.table == "topics"]
        qrels = [str(base / s.path) for s in m.shards if s.table == "qrels"]
        frames.append(
            pl.scan_parquet(topics).select(eval_id=pl.col("qid").cast(pl.String), text=pl.col("query"))
        )
        pos = (
            pl.scan_parquet(qrels)
            .filter(pl.col("relevance").cast(pl.Int32) > 0)
            .select("docid")
            .unique()
            .collect()["docid"]
        )
        cbase = source_dir(root, "raw", spec["passages_from"])
        cm = SourceManifest.read(cbase)
        corpus = [str(cbase / s.path) for s in cm.shards if s.table == "part"]
        frames.append(
            pl.scan_parquet(corpus)
            .filter(pl.col("docid").is_in(pos.implode()))
            .select(eval_id=pl.col("docid"), text=pl.col("text"))
        )
    else:
        parts = [str(base / s.path) for s in m.shards if s.table == "part"]
        lf = pl.scan_parquet(parts, extra_columns="ignore", missing_columns="insert")
        rid = pl.concat_str(
            pl.col("_split").fill_null(""), pl.int_range(pl.len()).cast(pl.String), separator=":"
        )
        if "tokens" in spec:
            frames.append(lf.select(eval_id=rid, text=pl.col(spec["tokens"]).list.join(" ")))
        for col in spec.get("columns", []):
            frames.append(lf.select(eval_id=rid, text=pl.col(col)))
    df = pl.concat([f.collect() for f in frames]).filter(pl.col("text").is_not_null())
    return df.unique("text", keep="first", maintain_order=True).with_columns(eval_source=pl.lit(name))


def eval_grams(root: Path, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Sorted unique gram hashes, index of the first eval source of each, source names, statistics."""
    n = cfg["ngram"]
    names = sorted(cfg["sources"])
    all_h, all_s = [], []
    stats: dict[str, Any] = {}
    for si, name in enumerate(names):
        df = eval_texts(root, name, cfg["sources"][name])
        text = normalize_series(df["text"])
        words, counts = word_hashes(text)
        ok = counts >= n
        grams, per_doc = ngram_hashes(words, counts, n)
        keep = np.repeat(ok, per_doc)
        all_h.append(grams[keep])
        all_s.append(np.full(int(keep.sum()), si, dtype=np.uint8))
        stats[name] = {"texts": df.height, "texts_long_enough": int(ok.sum()), "grams": int(keep.sum())}
        log.info("eval %s: %s", name, stats[name])
    h = np.concatenate(all_h)
    s = np.concatenate(all_s)
    order = np.lexsort((s, h))
    h, s = h[order], s[order]
    first = np.ones(len(h), dtype=bool)
    first[1:] = h[1:] != h[:-1]
    return h[first], s[first], names, stats


# ----------------------------------------------------------------------------- workers

_CTX: dict[str, Any] = {}


def init_worker(ctx: dict[str, Any]) -> None:
    _CTX.clear()
    _CTX.update(ctx)
    gram_dir = Path(ctx["grams"])
    _CTX["grams"] = np.load(gram_dir / "grams.npy", mmap_mode="r")
    if (gram_dir / "ignore.npy").is_file():
        active = np.ones(len(_CTX["grams"]), dtype=bool)
        active[np.load(gram_dir / "ignore.npy")] = False
        _CTX["active"] = active
        _CTX["gram_src"] = np.load(gram_dir / "gram_src.npy", mmap_mode="r")


def doc_hits(texts: pl.Series, grams: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(document row, gram index) of every distinct eval gram found in each document."""
    words, counts = word_hashes(texts)
    h, per_doc = ngram_hashes(words, counts, n)
    doc = np.repeat(np.arange(len(counts), dtype=np.int64), per_doc)
    long_enough = np.repeat(counts >= n, per_doc)  # a shorter document is one partial shingle
    h, doc = h[long_enough], doc[long_enough]
    if len(grams) == 0 or len(h) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    pos = np.searchsorted(grams, h)
    pos[pos == len(grams)] = 0
    hit = grams[pos] == h
    pairs = np.unique((doc[hit] << 32) | pos[hit].astype(np.int64))
    return pairs >> 32, pairs & 0xFFFFFFFF


def hits_path(work: Path, idx: int) -> Path:
    return work / "hits" / f"hits-{idx:05d}.parquet"


def match_unit(u: Unit) -> dict[str, Any]:
    out = hits_path(Path(_CTX["work"]), u.idx)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, grams = [], []
    pos = 0
    for b in iter_batches(Path(_CTX["input"]) / u.path, ["text"], None):
        r, g = doc_hits(pl.Series(b.column("text")), _CTX["grams"], _CTX["ngram"])
        rows.append(r + pos)
        grams.append(g)
        pos += b.num_rows
    if pos != u.rows:
        raise ValueError(f"{u.path}: {pos} rows, expected {u.rows}")
    df = (
        pl.DataFrame({"row": np.concatenate(rows), "gram": np.concatenate(grams)})
        if rows
        else pl.DataFrame({"row": [], "gram": []})
    )
    df = df.cast(pl.Int64).with_columns(unit=pl.lit(u.idx, dtype=pl.Int64))
    tmp = out.with_name(out.name + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(out)
    return {"rows": u.rows, "hits": df.height, "bytes_in": (Path(_CTX["input"]) / u.path).stat().st_size}


def active_hits(texts: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    """Like `doc_hits`, without template grams."""
    r, g = doc_hits(texts, _CTX["grams"], _CTX["ngram"])
    keep = _CTX["active"][g] if g.size else np.zeros(0, dtype=bool)
    return r[keep], g[keep]


def strip_lines(texts: list[str]) -> tuple[list[str | None], list[int]]:
    """Remove lines holding an eval gram; None = drop the document (gram across lines or too much removed)."""
    lines = [t.split("\n") for t in texts]
    flat = pl.Series([ln for doc in lines for ln in doc], dtype=pl.String)
    hit_rows, _ = active_hits(flat)
    hit = np.zeros(len(flat), dtype=bool)
    hit[hit_rows] = True
    new: list[str | None] = []
    n_removed: list[int] = []
    pos = 0
    for doc in lines:
        h = hit[pos : pos + len(doc)]
        pos += len(doc)
        kept = [ln for ln, x in zip(doc, h, strict=True) if not x]
        removed_chars = sum(len(ln) for ln, x in zip(doc, h, strict=True) if x)
        total = max(sum(len(ln) for ln in doc), 1)
        n_removed.append(int(h.sum()))
        text = "\n".join(kept).strip("\n")
        new.append(text if text and removed_chars / total <= _CTX["max_removed_share"] else None)
    check = [i for i, t in enumerate(new) if t is not None]
    if check:
        still, _ = active_hits(pl.Series([new[i] for i in check], dtype=pl.String))
        for j in np.unique(still):
            new[check[j]] = None
    return new, n_removed


def filter_unit(u: Unit) -> dict[str, Any]:
    names: list[str] = _CTX["names"]
    hits = pl.read_parquet(hits_path(Path(_CTX["work"]), u.idx))
    if hits.height:
        hits = hits.filter(pl.Series(_CTX["active"][hits["gram"].to_numpy()]))
    src = (
        np.asarray(_CTX["gram_src"])[hits["gram"].to_numpy()] if hits.height else np.zeros(0, dtype=np.uint8)
    )
    per_doc = (
        hits.with_columns(
            eval_source=pl.Series(np.array(names, dtype=object)[src] if src.size else [], dtype=pl.String)
        )
        .group_by("row")
        .agg(matches=pl.len(), eval_sources=pl.col("eval_source").unique().sort())
        .filter(pl.col("matches") >= _CTX["min_matches"])
    )
    lake: LakeConfig = _CTX["lake"]
    out = Path(_CTX["out"])
    parts = UnitWriter(out / u.source, u.idx, "part", lake)
    bad = UnitWriter(out / u.source, u.idx, "contaminated", lake)
    pos = 0
    counts = {"dropped": 0, "lines_removed": 0}
    by_eval: dict[str, dict[str, int]] = {}
    try:
        for b in iter_batches(Path(_CTX["input"]) / u.path, None, None):
            df = pl.from_arrow(b)
            assert isinstance(df, pl.DataFrame)
            df = df.with_row_index("row", offset=pos).with_columns(pl.col("row").cast(pl.Int64))
            pos += b.num_rows
            joined = df.join(per_doc, on="row", how="left", maintain_order="left")
            is_bad = joined["matches"].is_not_null()
            clean = (
                joined.filter(~is_bad)
                .drop("matches", "eval_sources")
                .with_columns(decontaminated_lines=pl.lit(0, dtype=pl.Int32))
            )
            c = joined.filter(is_bad)
            if c.height:
                new, n_lines = strip_lines(c["text"].to_list())
                action = ["dropped" if t is None else "lines_removed" for t in new]
                c = c.with_columns(
                    new_text=pl.Series(new, dtype=pl.String),
                    decontaminated_lines=pl.Series(n_lines, dtype=pl.Int32),
                    action=pl.Series(action, dtype=pl.String),
                )
                fixed = (
                    c.filter(pl.col("action") == "lines_removed")
                    .with_columns(
                        text=pl.col("new_text"),
                        n_chars=pl.col("new_text").str.len_chars().cast(pl.Int32),
                        n_words=pl.col("new_text").str.count_matches(r"[\p{L}\p{N}]+").cast(pl.Int32),
                    )
                    .select(clean.columns)
                )
                clean = pl.concat([clean, fixed]).sort("row")
                log_rows = c.select(
                    "doc_id", "source", "lang", "n_chars", "matches", "eval_sources", "action", "decontaminated_lines",
                    chars_removed=pl.col("n_chars") - pl.col("new_text").str.len_chars().fill_null(0).cast(pl.Int32),
                )  # fmt: skip
                bad.write(u.lang, log_rows.to_arrow())
                for act, srcs in zip(action, c["eval_sources"].to_list(), strict=True):
                    counts[act] += 1
                    for s in srcs:
                        d = by_eval.setdefault(s, {"dropped": 0, "lines_removed": 0})
                        d[act] += 1
            parts.write(u.lang, clean.drop("row").to_arrow())
    except BaseException:
        parts.abort()
        bad.abort()
        raise
    shards = parts.close() + bad.close()
    for s in shards:
        s["path"] = f"{u.source}/{s['path']}"
    return {"shards": shards, "source": u.source, "lang": u.lang, "rows": u.rows, "counts": counts,
            "by_eval": by_eval}  # fmt: skip


# ----------------------------------------------------------------------------- driver


def plan(base: Path, m: DatasetManifest) -> list[Unit]:
    units = []
    for s in sorted((s for s in m.shards if s.table == "part"), key=lambda s: s.path):
        units.append(Unit(len(units), s.path.split("/", 1)[0], s.path, s.lang, s.rows, ""))
    return units


def run(params: dict[str, Any], root: Path, dedup: Path, out: Path, workers: int) -> DatasetManifest | None:
    cfg = params["curation"]["decontamination"]
    m_in = DatasetManifest.read(dedup)
    units = plan(dedup, m_in)
    eval_manifests = []
    for name, spec in sorted(cfg["sources"].items()):
        eval_manifests.append(source_dir(root, "raw", name) / MANIFEST_NAME)
        if "passages_from" in spec:
            eval_manifests.append(source_dir(root, "raw", spec["passages_from"]) / MANIFEST_NAME)
    fp = fingerprint("decontaminate", cfg, CODE, [dedup / DATASET_NAME, *eval_manifests])
    sd = StageDir(out, fp)
    if sd.complete():
        log.info("%s: up to date (fingerprint %s)", out, fp)
        return None
    sd.prepare()
    lake = LakeConfig.from_params(params["lake"], root)
    ctx: dict[str, Any] = {"input": str(dedup), "work": str(sd.work), "grams": str(sd.work), "out": str(out),
                           "lake": lake, "ngram": cfg["ngram"], "min_matches": cfg["min_matches"],
                           "max_removed_share": cfg["max_removed_share"]}  # fmt: skip
    with ResourceMonitor() as mon:
        stats_path = sd.work / "eval.json"
        if not stats_path.is_file():
            g, gs, names, ev_stats = eval_grams(root, cfg)
            np.save(sd.work / "grams.npy", g)
            np.save(sd.work / "gram_src.npy", gs)
            stats_path.write_text(json.dumps({"names": names, "stats": ev_stats}), encoding="utf-8")
        ev = json.loads(stats_path.read_text(encoding="utf-8"))
        ctx["names"] = ev["names"]
        match_sd = StageDir(sd.work / "hits", fp)
        match_sd.work.mkdir(parents=True, exist_ok=True)
        match_res = run_units(units, match_sd, match_unit, workers, init_worker, (ctx,))
        hit_files = [hits_path(sd.work, u.idx) for u in units]
        per_gram = (
            pl.scan_parquet(hit_files)
            .group_by("gram")
            .agg(docs=pl.len())
            .sort("docs", descending=True)
            .collect()
        )
        templates = per_gram.filter(pl.col("docs") > cfg["max_gram_docs"])
        np.save(sd.work / "ignore.npy", templates["gram"].to_numpy())
        results = run_units(units, sd, filter_unit, workers, init_worker, (ctx,))
        # post-check: the written output must not contain any non-template eval gram
        res = [results[u.idx] for u in units]
        shards = [ShardEntry.model_validate(s) for r in res for s in r["shards"]]
        post_units = [
            Unit(i, "", s.path, s.lang, s.rows, "")
            for i, s in enumerate(x for x in shards if x.table == "part")
        ]
        post_ctx = {**ctx, "input": str(out), "work": str(sd.work / "post")}
        post_sd = StageDir(sd.work / "post" / "hits", fp)
        post_sd.work.mkdir(parents=True, exist_ok=True)
        run_units(post_units, post_sd, match_unit, workers, init_worker, (post_ctx,))
        post = pl.scan_parquet([hits_path(sd.work / "post", u.idx) for u in post_units]).collect()
        leaked = post.filter(~pl.col("gram").is_in(templates["gram"].implode()))
        post_docs = leaked.select("unit", "row").n_unique() if leaked.height else 0
    if post_docs:
        raise RuntimeError(f"post-check: {post_docs} output documents still share eval grams")
    rows_in: dict[str, int] = {}
    drops: dict[str, int] = {}
    by_source: dict[str, dict[str, int]] = {}
    by_eval: dict[str, dict[str, int]] = {}
    for r in res:
        rows_in[r["lang"]] = rows_in.get(r["lang"], 0) + r["rows"]
        drops[r["lang"]] = drops.get(r["lang"], 0) + r["counts"]["dropped"]
        bs = by_source.setdefault(r["source"], {"in": 0, "dropped": 0, "lines_removed": 0})
        bs["in"] += r["rows"]
        for k in ("dropped", "lines_removed"):
            bs[k] += r["counts"][k]
        for k, v in r["by_eval"].items():
            d = by_eval.setdefault(k, {"dropped": 0, "lines_removed": 0})
            for a, n in v.items():
                d[a] += n
    names = ev["names"]
    gram_src = np.load(sd.work / "gram_src.npy")
    top = [
        {"docs": d, "eval_source": names[int(gram_src[g])]}
        for g, d in templates.head(TOP_TEMPLATES).select("gram", "docs").iter_rows()
    ]
    total_in = sum(rows_in.values())
    manifest = DatasetManifest(
        name="curated/corpus",
        layer="curated",
        stage="decontaminate",
        source=m_in.source,
        description=(
            f"Deduplicated corpus without documents sharing word {cfg['ngram']}-grams with any split of the eval "
            f"candidates ({', '.join(names)}); table `part` per `<source>/<lang>`, removed documents with "
            "matching eval sources in table `contaminated`."
        ),
        code_version=git_sha(ROOT),
        fingerprint=fp,
        params=cfg,
        inputs=[dataset_input(dedup.parent, dedup / DATASET_NAME, m_in.rows_of())]
        + [dataset_input(root, p, SourceManifest.read(p.parent).rows) for p in eval_manifests],
        licenses=licenses_of([m_in]),
        rows_in=dict(sorted(rows_in.items())),
        drops={"contaminated": dict(sorted(drops.items()))},
        shards=shards,
        stats=mon.stats(total_in, sum(r["bytes_in"] for r in match_res.values()), workers),
        metrics={
            "removed_share": round(sum(drops.values()) / max(total_in, 1), 6),
            "modified_share": round(
                sum(b["lines_removed"] for b in by_source.values()) / max(total_in, 1), 6
            ),
            "by_source": by_source,
            "by_eval_source": dict(sorted(by_eval.items())),
            "eval": ev["stats"],
            "eval_grams": len(np.load(sd.work / "grams.npy", mmap_mode="r")),
            "grams_matched": per_gram.height,
            "template_grams": templates.height,
            "top_template_grams": top,
            "post_check": {
                "documents": sum(s.rows for s in shards if s.table == "part"),
                "leaked_documents": 0,
            },
        },
        partial=m_in.partial,
    )
    sd.finish(manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dedup", type=Path, help="curated/dedup directory (default <lake root>/curated/dedup)")
    ap.add_argument("--out", type=Path, help="output directory (default <lake root>/curated/corpus)")
    ap.add_argument("--report", type=Path, help="metrics JSON (default reports/decontamination.json)")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    params = load_params()
    root = (ROOT / params["lake"]["root"]).resolve()
    dedup = (args.dedup or root / "curated" / "dedup").resolve()
    out = (args.out or root / "curated" / "corpus").resolve()
    workers = args.workers or params["curation"]["workers"]
    m = run(params, root, dedup, out, workers) or DatasetManifest.read(out)
    rep_path = args.report or (
        ROOT / "reports/decontamination.json" if args.out is None else out.parent / "decontamination.json"
    )
    write_report(rep_path, stage_report(m))
    print(f"decontaminate: in {sum(m.rows_in.values())} out {sum(m.rows_of().values())} removed {m.drops} "
          f"by eval {m.metrics['by_eval_source']}")  # fmt: skip
    if m.stats and not m.partial:
        log_mlflow("curation", "decontaminate", {**m.params, "fingerprint": m.fingerprint,
                   "code_version": m.code_version}, {
            "docs_in": sum(m.rows_in.values()), "docs_out": sum(m.rows_of().values()),
            "removed_share": m.metrics["removed_share"], "template_grams": m.metrics["template_grams"],
            "seconds": m.stats.seconds, "docs_per_s": m.stats.rows_per_s, "peak_rss_mb": m.stats.peak_rss_mb,
            "modified_share": m.metrics["modified_share"],
            **{f"dropped_{k}": v["dropped"] for k, v in m.metrics["by_eval_source"].items()},
        })  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
