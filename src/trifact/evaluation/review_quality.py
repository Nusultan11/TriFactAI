"""Manual review of the clean-stage quality filters (ROADMAP 2.3, stage 2 done-criterion).

python -m trifact.evaluation.review_quality sample [--clean DIR]   # data/review/quality/sample.parquet
python -m trifact.evaluation.review_quality score  [--clean DIR]   # reports/quality_review.json (+ MLflow)

Per target language: `kept` documents (table `part`, detected language) and `dropped` documents
(table `dropped`, detected language if it is a target one, else the source partition language),
drawn with a seeded hash of `doc_id` and stratified by domain (kept) and drop reason (dropped)
with a floor per stratum. A reviewer reads the first `TEXT_CHARS` characters and labels every
document in `labels.csv` (`doc_id,label,note`): `good` = natural text in that language that is
worth training and retrieval on, `bad` = boilerplate, lists/tables without prose, broken
encoding, machine garbage, wrong language. Scores:

* precision of the kept set (share `good`), per stratum and weighted by stratum sizes;
* per drop reason the share of `bad` (correct drops);
* estimated share of good documents the filters lose (`good_lost`), weighted by reason sizes;
* Wilson 95% intervals for every share.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import polars as pl

from trifact.core.params import ROOT, load_params
from trifact.data.curation import write_report
from trifact.data.quality import TARGET_LANGS
from trifact.lake.manifest import DatasetManifest
from trifact.lake.stats import git_sha, log_mlflow

REVIEW = ROOT / "data" / "review" / "quality"
TEXT_CHARS = 600
KEPT_PER_LANG = 350
DROPPED_PER_LANG = 150
MIN_PER_STRATUM = 10


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def frames(clean: Path, sources: list[str]) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    kept, dropped = [], []
    for s in sources:
        m = DatasetManifest.read(clean / s)
        kept += [str(clean / s / x.path) for x in m.shards if x.table == "part"]
        dropped += [str(clean / s / x.path) for x in m.shards if x.table == "dropped"]
    cols = ["doc_id", "source", "domain", "lang", "src_lang", "text"]
    k = pl.scan_parquet(kept).select(*cols, drop_reason=pl.lit(None, dtype=pl.String), status=pl.lit("kept"))
    d = pl.scan_parquet(dropped).select(*cols, "drop_reason", status=pl.lit("dropped"))
    rl = pl.when(pl.col("lang").is_in(TARGET_LANGS)).then("lang").otherwise("src_lang").alias("review_lang")
    return k.with_columns(rl, stratum=pl.col("domain")), d.with_columns(rl, stratum=pl.col("drop_reason"))


def allocate(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Proportional allocation with a floor of MIN_PER_STRATUM (capped by the stratum size)."""
    alloc = {k: min(n, MIN_PER_STRATUM) for k, n in sizes.items()}
    left = total - sum(alloc.values())
    rest = {k: n - alloc[k] for k, n in sizes.items() if n > alloc[k]}
    mass = sum(rest.values())
    for k, n in rest.items():
        alloc[k] += min(n, int(left * n / mass)) if mass else 0
    short = total - sum(alloc.values())
    for k in sorted(sizes, key=lambda x: -sizes[x]):  # remainder of the rounding to the largest strata
        add = min(short, sizes[k] - alloc[k])
        alloc[k] += add
        short -= add
    return alloc


def draw(lf: pl.LazyFrame, per_lang: int, seed: int) -> pl.DataFrame:
    sizes = lf.group_by("review_lang", "stratum").len().collect(engine="streaming")
    out = []
    for lang in TARGET_LANGS:
        s = {
            r["stratum"]: r["len"] for r in sizes.filter(pl.col("review_lang") == lang).iter_rows(named=True)
        }
        for stratum, n in allocate(s, per_lang).items():
            if n == 0:
                continue
            out.append(
                lf.filter((pl.col("review_lang") == lang) & (pl.col("stratum") == stratum))
                .with_columns(h=pl.col("doc_id").hash(seed=seed))
                .sort("h")
                .head(n)
                .drop("h")
                .with_columns(pl.col("text").str.slice(0, TEXT_CHARS), stratum_size=pl.lit(s[stratum]))
                .collect(engine="streaming")
            )
    return pl.concat(out)


def score(sample: pl.DataFrame, labels: pl.DataFrame) -> dict[str, Any]:
    df = sample.join(labels, on="doc_id", how="left")
    missing = df.filter(pl.col("label").is_null()).height
    if missing:
        raise ValueError(f"{missing} sampled documents have no label in labels.csv")
    bad_labels = set(df["label"].unique()) - {"good", "bad"}
    if bad_labels:
        raise ValueError(f"unknown labels {bad_labels}")
    out: dict[str, Any] = {"reviewed": df.height, "by_lang": {}}
    for lang in TARGET_LANGS:
        d = df.filter(pl.col("review_lang") == lang)
        res: dict[str, Any] = {"reviewed": d.height}
        for status, positive in (("kept", "good"), ("dropped", "bad")):
            part = d.filter(pl.col("status") == status)
            strata = (
                part.group_by("stratum")
                .agg(n=pl.len(), k=(pl.col("label") == positive).sum(), size=pl.col("stratum_size").first())
                .sort("stratum")
            )
            rows = {}
            for r in strata.iter_rows(named=True):
                rows[r["stratum"]] = {"n": r["n"], f"{positive}": r["k"], "share": round(r["k"] / r["n"], 4),
                                      "ci95": wilson(r["k"], r["n"]), "population": r["size"]}  # fmt: skip
            pop = sum(v["population"] for v in rows.values())
            weighted = sum(v["share"] * v["population"] for v in rows.values()) / pop if pop else 0.0
            n, k = part.height, int((part["label"] == positive).sum())
            res[status] = {"n": n, positive: k, "share": round(k / n, 4) if n else None, "ci95": wilson(k, n),
                           "weighted_share": round(weighted, 4), "population": pop, "by_stratum": rows}  # fmt: skip
        kept_pop, kept_good = res["kept"]["population"], res["kept"]["weighted_share"]
        lost = sum(v["population"] * (1 - v["share"]) for v in res["dropped"]["by_stratum"].values())
        good_total = kept_pop * kept_good + lost
        res["good_lost"] = round(lost / good_total, 4) if good_total else None
        out["by_lang"][lang] = res
    notes = df.filter(pl.col("note").is_not_null() & (pl.col("note") != ""))
    out["notes"] = notes.group_by("note").len().sort("len", descending=True).head(30).to_dicts()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sample", "sheet", "score"])
    ap.add_argument("--clean", type=Path, help="clean layer directory (default <lake root>/clean)")
    ap.add_argument("--review-dir", type=Path, default=REVIEW, help="sample / labels directory")
    ap.add_argument("--reviewer", default="Claude (LLM), not a human annotator",
                    help="who labelled labels.csv; a human review must name the person")  # fmt: skip
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    params = load_params()
    clean = (args.clean or ROOT / params["lake"]["root"] / "clean").resolve()
    review = args.review_dir.resolve()
    if args.cmd == "sheet":  # blank CSV for human reviewers: doc_id,label,note + the text to read
        sample = pl.read_parquet(review / "sample.parquet")
        sheet = sample.select(
            "doc_id", "review_lang", "status", "stratum", "text", label=pl.lit(""), note=pl.lit("")
        )
        out = review / "to_label.csv"
        if out.exists():
            print(f"{out} exists, kept")
            return 0
        sheet.write_csv(out)
        print(f"{sheet.height} documents -> {out}; save the filled sheet as labels.csv (doc_id,label,note)")
        return 0
    if args.cmd == "sample":
        kept, dropped = frames(clean, params["curation"]["sources"])
        sample = pl.concat(
            [draw(kept, KEPT_PER_LANG, params["seed"]), draw(dropped, DROPPED_PER_LANG, params["seed"])]
        ).sort("review_lang", "status", "stratum", "doc_id")
        review.mkdir(parents=True, exist_ok=True)
        sample.write_parquet(review / "sample.parquet")
        print(sample.group_by("review_lang", "status").len().sort("review_lang", "status"))
        return 0
    sample = pl.read_parquet(review / "sample.parquet")
    labels = pl.read_csv(review / "labels.csv", schema_overrides={"note": pl.String}).select(
        "doc_id", "label", "note"
    )
    res = score(sample, labels)
    res["clean"] = {s: DatasetManifest.read(clean / s).fingerprint for s in params["curation"]["sources"]}
    res["reviewer"] = args.reviewer
    res["human"] = "LLM" not in args.reviewer
    name = "quality_review.json" if review == REVIEW.resolve() else f"quality_review_{review.name}.json"
    write_report(ROOT / "reports" / name, res)
    for lang, r in res["by_lang"].items():
        print(lang, "kept good", r["kept"]["weighted_share"], r["kept"]["ci95"], "dropped bad", r["dropped"]["share"],
              r["dropped"]["ci95"], "good lost", r["good_lost"])  # fmt: skip
    metrics = {}
    for lang, r in res["by_lang"].items():
        metrics[f"{lang}_kept_precision"] = r["kept"]["weighted_share"]
        metrics[f"{lang}_dropped_correct"] = r["dropped"]["weighted_share"]
        metrics[f"{lang}_good_lost"] = r["good_lost"] or 0.0
    log_mlflow(
        "curation", "quality_review", {"reviewed": res["reviewed"], "code_version": git_sha(ROOT)}, metrics
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
