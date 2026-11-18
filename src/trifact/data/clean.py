"""Stage clean: normalize text, detect language per document/paragraph, apply quality filters (raw -> clean).

python -m trifact.data.clean [--profile full|pilot] [--sources a,b] [--out DIR] [--workers N]

Per corpus source (`params.yaml: curation.sources`), two passes over the raw shards:
1. Boilerplate lines: the first `curation.boilerplate.sample_docs` documents of every shard are
   normalized and their lines (<= `max_line_chars`) counted by document frequency; a line found in
   >= `min_doc_share` of the sample and >= `min_docs` documents is boilerplate (menus, share
   buttons, cookie notices, wiki maintenance templates). The list is kept as `boilerplate.parquet`.
2. Documents: normalization (`trifact.ingestion.normalize`) -> boilerplate lines removed from
   multi-line documents -> language of every paragraph and of the document
   (`trifact.ingestion.langdetect`, model `curation.langid.model`) -> quality signals and rules
   (`trifact.data.quality`). Kept documents go to `<lang>/part-*.parquet` under the detected
   language; dropped ones go to `<src_lang>/dropped-*.parquet` with the reason and the first
   `dropped_text_chars` characters, for audit and manual review.

The `pilot` profile reads `scale.pilot.corpus_docs` documents per source language, spread over all
shards (all domains), and marks the dataset `partial`.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
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
    merge_counts,
    run_units,
    stage_report,
    sum_by_lang,
    write_report,
    write_single,
)
from trifact.data.quality import REASONS, SIGNALS, apply_rules, signals
from trifact.ingestion.langdetect import LangConfig, LangId, detect, model_path
from trifact.ingestion.normalize import normalize_series
from trifact.lake.layout import LakeConfig, source_dir
from trifact.lake.manifest import DATASET_NAME, MANIFEST_NAME, DatasetManifest, ShardEntry, SourceManifest
from trifact.lake.stats import ResourceMonitor, git_sha, log_mlflow

log = logging.getLogger("trifact.clean")
CODE = [
    Path(__file__),
    ROOT / "src/trifact/data/quality.py",
    ROOT / "src/trifact/data/curation.py",
    ROOT / "src/trifact/ingestion/normalize.py",
    ROOT / "src/trifact/ingestion/langdetect.py",
]
# raw column -> documents column, per corpus source
COLUMNS = {
    "wikipedia": {"id": "id", "title": "title", "url": "url"},
    "fineweb2_kk": {"id": "id", "url": "url", "date": "date"},
    "multidomain_kk": {"id": "id"},
}
QUALITY_FIELDS = [s for s in SIGNALS if s not in ("n_chars", "n_words")] + ["boilerplate_lines"]


# ----------------------------------------------------------------------------- planning


def domains_of(m: SourceManifest, spec: dict[str, str] | str) -> dict[str, str]:
    """Shard path -> domain. A dict spec maps input file names to domains; shards follow files in order."""
    shards = [s for s in m.shards if s.table == "part"]
    if isinstance(spec, str):
        return {s.path: spec for s in shards}
    out: dict[str, str] = {}
    for lang in sorted({s.lang for s in shards}):
        files = [(f.path, f.rows) for f in m.inputs if f.lang == lang and f.table == "part"]
        it = iter(files)
        name, left = next(it)
        for s in (s for s in shards if s.lang == lang):
            while left == 0:
                name, left = next(it)
            if s.rows > left:
                raise ValueError(f"{m.name}: shard {s.path} spans input files, domain is ambiguous")
            out[s.path] = spec[Path(name).name]
            left -= s.rows
    return out


def plan(root: Path, source: str, cur: dict[str, Any], limit_docs: int | None) -> list[Unit]:
    m = SourceManifest.read(source_dir(root, "raw", source))
    dom = domains_of(m, cur["domains"][source])
    shards = [s for s in m.shards if s.table == "part"]
    per_lang = {lang: sum(1 for s in shards if s.lang == lang) for lang in {s.lang for s in shards}}
    units = []
    for i, s in enumerate(shards):
        limit = None if limit_docs is None else min(s.rows, math.ceil(limit_docs / per_lang[s.lang]))
        units.append(Unit(i, source, f"raw/{source}/{s.path}", s.lang, s.rows, dom[s.path], limit))
    return units


# ----------------------------------------------------------------------------- worker side

_CTX: dict[str, Any] = {}


def init_worker(cfg: dict[str, Any]) -> None:
    logging.basicConfig(level=logging.WARNING)
    _CTX.clear()
    _CTX.update(cfg)
    if "model" in cfg:
        _CTX["lid"] = LangId(Path(cfg["model"]), cfg["model_name"])
        bp = cfg.get("boilerplate")
        _CTX["bp"] = pl.read_parquet(bp)["hash"] if bp else pl.Series("hash", [], dtype=pl.UInt64)


def docs_frame(batch: pa.RecordBatch, unit: Unit) -> pl.DataFrame:
    df = pl.from_arrow(batch)
    assert isinstance(df, pl.DataFrame)
    cols = COLUMNS[unit.source]
    get = {k: pl.col(v).cast(pl.String) for k, v in cols.items()}
    return df.select(
        doc_id=pl.concat_str(pl.lit(f"{unit.source}/{unit.lang}/"), get["id"]),
        source=pl.lit(unit.source),
        domain=pl.lit(unit.domain),
        src_lang=pl.lit(unit.lang),
        title=get.get("title", pl.lit(None, dtype=pl.String)),
        url=get.get("url", pl.lit(None, dtype=pl.String)),
        date=get.get("date", pl.lit(None, dtype=pl.String)),
        text=pl.col("text").cast(pl.String).fill_null(""),
    )


def line_hashes(text: pl.Series, max_chars: int) -> pl.DataFrame:
    """(doc, hash) of distinct non-empty lines up to `max_chars` characters."""
    return (
        pl.DataFrame({"text": text})
        .with_row_index("doc")
        .select("doc", line=pl.col("text").str.split("\n"))
        .explode("line")
        .filter((pl.col("line").str.len_bytes() > 0) & (pl.col("line").str.len_chars() <= max_chars))
        .with_columns(hash=pl.col("line").hash(seed=0))
        .unique(["doc", "hash"])
    )


def count_lines(unit: Unit) -> dict[str, Any]:
    """Pass 1: document frequency of short lines in the first `sample_docs` documents of a shard."""
    bcfg = _CTX["bp_cfg"]
    limit = min(x for x in (unit.limit, bcfg["sample_docs"]) if x is not None)
    frames, docs = [], 0
    for batch in iter_batches(_CTX["root"] / unit.path, ["text"], limit):
        text = normalize_series(pl.from_arrow(batch.column(0)).alias("text"))  # type: ignore[union-attr]
        lh = line_hashes(text, bcfg["max_line_chars"])
        frames.append(lh.group_by("hash").agg(docs=pl.len(), line=pl.col("line").first()))
        docs += len(text)
    out = (
        pl.concat(frames).group_by("hash").agg(pl.col("docs").sum(), pl.col("line").first())
        if frames
        else None
    )
    path = Path(_CTX["work"]) / f"lines-{unit.idx:05d}.parquet"
    if out is not None:
        out.filter(pl.col("docs") >= 2).write_parquet(path)
    return {"docs": docs, "lines": str(path) if out is not None else None}


def strip_boilerplate(df: pl.DataFrame, bp: pl.Series, max_chars: int) -> pl.DataFrame:
    """Remove boilerplate lines from documents with >= 2 non-empty lines; adds `boilerplate_lines`."""
    if bp.is_empty():
        return df.with_columns(boilerplate_lines=pl.lit(0, dtype=pl.Int16))
    lines = (
        df.select(pl.col("text").str.split("\n"))
        .with_row_index("doc")
        .explode("text")
        .with_row_index("pos")
        .with_columns(
            bp=(pl.col("text").str.len_chars() <= max_chars)
            & pl.col("text").hash(seed=0).is_in(bp.implode()),
            nonempty=pl.col("text").str.len_bytes() > 0,
        )
        .with_columns(multi=pl.col("nonempty").sum().over("doc") >= 2)
        .with_columns(drop=pl.col("bp") & pl.col("multi") & pl.col("nonempty"))
    )
    per_doc = lines.group_by("doc").agg(n_bp=pl.col("drop").sum())
    hit = per_doc.filter(pl.col("n_bp") > 0)["doc"]
    rebuilt = (
        lines.filter(pl.col("doc").is_in(hit.implode()) & ~pl.col("drop"))
        .sort("pos")
        .group_by("doc", maintain_order=True)
        .agg(new=pl.col("text").str.join("\n"))
        .with_columns(pl.col("new").str.replace_all("\n{3,}", "\n\n").str.strip_chars())
    )
    return (
        df.with_row_index("doc")
        .join(per_doc, on="doc", how="left")
        .join(rebuilt, on="doc", how="left")
        .with_columns(
            text=pl.when(pl.col("n_bp") > 0).then(pl.col("new").fill_null("")).otherwise(pl.col("text")),
            boilerplate_lines=pl.col("n_bp").fill_null(0).cast(pl.Int16),
        )
        .sort("doc")
        .drop("doc", "n_bp", "new")
    )


def process_unit(unit: Unit) -> dict[str, Any]:
    """Pass 2 over one raw shard."""
    lid: LangId = _CTX["lid"]
    lcfg = LangConfig(_CTX["min_para_chars"], _CTX["max_para_chars"])
    lake: LakeConfig = _CTX["lake"]
    out = Path(_CTX["out"])
    kept_w = UnitWriter(out, unit.idx, "part", lake)
    drop_w = UnitWriter(out, unit.idx, "dropped", lake)
    rows_in = drops = 0
    fails: dict[str, int] = {}
    reasons: dict[str, int] = {}
    kept_lang: dict[str, int] = {}
    homoglyph = 0
    cols = ["text", *COLUMNS[unit.source].values()]
    try:
        for batch in iter_batches(_CTX["root"] / unit.path, list(dict.fromkeys(cols)), unit.limit):
            df = docs_frame(batch, unit)
            rows_in += df.height
            raw = df["text"]
            df = df.with_columns(text=normalize_series(raw), title=normalize_series(df["title"]))
            homoglyph += int((df["text"] != raw).sum())
            df = strip_boilerplate(df, _CTX["bp"], _CTX["bp_cfg"]["max_line_chars"])
            df = pl.concat([df, detect(df["text"], lid, lcfg)], how="horizontal")
            df = apply_rules(signals(df), _CTX["quality"])
            for r in REASONS:
                n = int(df[f"fail_{r}"].sum())
                if n:
                    fails[r] = fails.get(r, 0) + n
            for r, n in df.group_by("drop_reason").len().iter_rows():
                if r is not None:
                    reasons[r] = reasons.get(r, 0) + n
            base = [
                "doc_id", "source", "domain", "lang", "src_lang", "lang_score", "lang_conf", "para_lang",
                "title", "url", "date",
            ]  # fmt: skip
            quality = pl.struct(QUALITY_FIELDS).alias("quality")
            kept = df.filter(pl.col("drop_reason").is_null()).select(
                *base, "text", "n_chars", "n_words", quality
            )
            for (lang,), part in kept.group_by("lang"):
                kept_w.write(str(lang), part.to_arrow())
                kept_lang[str(lang)] = kept_lang.get(str(lang), 0) + part.height
            dropped = df.filter(pl.col("drop_reason").is_not_null()).select(
                *base,
                pl.col("text").str.slice(0, _CTX["dropped_text_chars"]),
                "n_chars",
                "n_words",
                quality,
                "drop_reason",
            )
            drops += dropped.height
            drop_w.write(unit.lang, dropped.to_arrow())
        shards = kept_w.close() + drop_w.close()
    except BaseException:
        kept_w.abort()
        drop_w.abort()
        raise
    lang = unit.lang
    return {
        "shards": shards,
        "rows_in": {lang: rows_in},
        "drops": {r: {lang: n} for r, n in reasons.items()},
        "fails": {r: {lang: n} for r, n in fails.items()},
        "kept": kept_lang,
        "moves": {f"{lang}->{k}": {lang: n} for k, n in kept_lang.items()},
        "changed_by_normalization": {lang: homoglyph},
        "domain": unit.domain,
        "bytes_in": (Path(_CTX["root"]) / unit.path).stat().st_size,
    }


# ----------------------------------------------------------------------------- driver


def build_boilerplate(lines: list[str], sample_docs: int, cfg: dict[str, Any]) -> pl.DataFrame:
    if not lines:
        return pl.DataFrame(schema={"hash": pl.UInt64, "line": pl.String, "docs": pl.Int64})
    need = max(cfg["min_docs"], math.ceil(cfg["min_doc_share"] * sample_docs))
    return (
        pl.scan_parquet(lines)
        .group_by("hash")
        .agg(pl.col("docs").sum(), pl.col("line").first())
        .filter(pl.col("docs") >= need)
        .sort("docs", descending=True)
        .collect()
        .select("hash", "line", pl.col("docs").cast(pl.Int64))
    )


def clean_source(
    source: str, params: dict[str, Any], root: Path, out_root: Path, profile: str, workers: int
) -> DatasetManifest | None:
    cur = params["curation"]
    limit = params["scale"][profile].get("corpus_docs")
    units = plan(root, source, cur, limit)
    raw_dir = source_dir(root, "raw", source)
    spec = cur["langid"]
    quality = {
        **cur["quality"],
        "default": {"min_lang_score": spec["min_score"], **cur["quality"]["default"]},
    }
    stage_params = {
        "langid": {k: spec[k] for k in ("model", "min_para_chars", "max_para_chars")},
        "langid_model": spec["models"][spec["model"]],
        "quality": quality,
        "boilerplate": cur["boilerplate"],
        "dropped_text_chars": cur["dropped_text_chars"],
        "domains": cur["domains"][source],
        "profile": profile,
        "corpus_docs": limit,
    }
    fp = fingerprint("clean", stage_params, CODE, [raw_dir / MANIFEST_NAME])
    out = out_root / source
    sd = StageDir(out, fp)
    if sd.complete():
        log.info("%s: up to date (fingerprint %s)", out, fp)
        return None
    sd.prepare()
    lake = LakeConfig.from_params(params["lake"], root)
    bp_path = sd.work / "boilerplate.parquet"
    with ResourceMonitor() as mon:
        if not bp_path.is_file():
            bp_sd = StageDir(sd.work / "lines", fp)
            bp_sd.work.mkdir(parents=True, exist_ok=True)
            ctx = {"root": root, "work": str(bp_sd.work), "bp_cfg": cur["boilerplate"]}
            res = run_units(units, bp_sd, count_lines, workers, init_worker, (ctx,))
            sample = sum(r["docs"] for r in res.values())
            bp = build_boilerplate(
                [r["lines"] for r in res.values() if r["lines"]], sample, cur["boilerplate"]
            )
            bp.write_parquet(bp_path)
            log.info("%s: %d boilerplate lines from a %d-document sample", source, bp.height, sample)
        model = model_path(spec["model"], spec["models"][spec["model"]], ROOT / spec["cache"])
        ctx = {
            "root": root, "out": str(out), "lake": lake, "model": str(model), "model_name": spec["model"],
            "min_para_chars": spec["min_para_chars"], "max_para_chars": spec["max_para_chars"],
            "boilerplate": str(bp_path), "bp_cfg": cur["boilerplate"], "quality": quality,
            "dropped_text_chars": cur["dropped_text_chars"],
        }  # fmt: skip
        results = run_units(units, sd, process_unit, workers, init_worker, (ctx,))
    res = [results[u.idx] for u in units]
    shards = [ShardEntry.model_validate(s) for r in res for s in r["shards"]]
    bp = pl.read_parquet(bp_path)
    shards.append(write_single(out, "boilerplate.parquet", bp.to_arrow(), "multi", "boilerplate", lake))
    rows_in = sum_by_lang(res, "rows_in")
    kept = sum_by_lang(res, "kept")
    by_domain: dict[str, dict[str, int]] = {}
    for r in res:
        d = by_domain.setdefault(r["domain"], {"in": 0, "kept": 0})
        d["in"] += sum(r["rows_in"].values())
        d["kept"] += sum(r["kept"].values())
    m_raw = SourceManifest.read(raw_dir)
    total_in = sum(rows_in.values())
    manifest = DatasetManifest(
        name=f"clean/{source}",
        layer="clean",
        stage="clean",
        source=source,
        description=(
            f"Normalized, language-identified and quality-filtered documents of `{source}` "
            "(table `part`); dropped documents with reasons (table `dropped`, text truncated); "
            "boilerplate line list (table `boilerplate`)."
        ),
        code_version=git_sha(ROOT),
        fingerprint=fp,
        params=stage_params,
        inputs=[dataset_input(root, raw_dir / MANIFEST_NAME, m_raw.rows)],
        licenses=licenses_of([m_raw]),
        rows_in=rows_in,
        drops=merge_counts(res, "drops"),
        shards=shards,
        stats=mon.stats(total_in, sum(r["bytes_in"] for r in res), workers),
        metrics={
            "kept_by_lang": kept,
            "kept_share": round(sum(kept.values()) / max(total_in, 1), 4),
            "fails_by_rule": merge_counts(res, "fails"),
            "language_moves": {k: sum(v.values()) for k, v in merge_counts(res, "moves").items()},
            "changed_by_normalization": sum_by_lang(res, "changed_by_normalization"),
            "boilerplate_lines": bp.height,
            "by_domain": by_domain,
        },
        partial=limit is not None,
    )
    sd.finish(manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=["pilot", "full"], default="full")
    ap.add_argument("--sources", help="comma-separated subset of curation.sources")
    ap.add_argument("--root", type=Path, help="lake root with the raw layer (default params lake.root)")
    ap.add_argument("--out", type=Path, help="output directory (default <root>/clean)")
    ap.add_argument("--workers", type=int)
    ap.add_argument(
        "--report", type=Path, help="metrics JSON (default reports/clean.json for the lake output)"
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    params = load_params()
    root = (args.root or ROOT / params["lake"]["root"]).resolve()
    out_root = (args.out or root / "clean").resolve()
    cur = params["curation"]
    sources = args.sources.split(",") if args.sources else cur["sources"]
    workers = args.workers or cur["workers"]
    for source in sources:
        m = clean_source(source, params, root, out_root, args.profile, workers)
        if m is None:
            continue
        st = m.stats
        kept = m.metrics["kept_by_lang"]
        print(
            f"{source}: in {m.rows_in} kept {kept} ({m.metrics['kept_share']:.1%}) drops "
            f"{ {r: sum(v.values()) for r, v in m.drops.items()} } {st.seconds if st else '-'} s"
        )
        if st and not m.partial:
            log_mlflow("curation", f"clean-{source}", {**m.params, "fingerprint": m.fingerprint,
                       "code_version": m.code_version}, {
                "docs_in": sum(m.rows_in.values()), "docs_kept": sum(kept.values()),
                "kept_share": m.metrics["kept_share"], "seconds": st.seconds, "docs_per_s": st.rows_per_s,
                "mb_per_s": st.mb_per_s, "peak_rss_mb": st.peak_rss_mb,
                **{f"kept_{k}": v for k, v in kept.items()},
                **{f"drop_{r}": sum(v.values()) for r, v in m.drops.items()},
            })  # fmt: skip
    done = {s: stage_report(DatasetManifest.read(out_root / s)) for s in cur["sources"]
            if (out_root / s / DATASET_NAME).is_file()}  # fmt: skip
    rep_path = args.report or (ROOT / "reports/clean.json" if args.out is None else out_root / "clean.json")
    write_report(rep_path, done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
