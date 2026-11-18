"""Compare language-id models (lid.176, GlotLID) on a manually checked paragraph sample (ROADMAP 2.2).

python -m trifact.evaluation.eval_langid sample   # build data/review/langid/sample.parquet (seeded)
python -m trifact.evaluation.eval_langid score    # metrics -> reports/langid.json (+ MLflow run)

The sample holds normalized paragraphs from every corpus domain (kk/ru/en) and from Kyrgyz and
Tatar Wikipedia (close languages that lid models confuse with Kazakh). The default gold label is
the language of the source; every paragraph where a candidate model disagrees with it, or scores
below `min_score`, is read by a reviewer and labelled in `labels.csv` (`para_id,gold,note`), which
overrides the default. Gold labels outside kk/ru/en count as `other`, the class the pipeline drops.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import psutil
import pyarrow.parquet as pq

from trifact.core.params import ROOT, load_params
from trifact.ingestion.langdetect import LangId, model_path
from trifact.ingestion.normalize import normalize_series
from trifact.lake.stats import git_sha, log_mlflow

REVIEW = ROOT / "data" / "review" / "langid"
TARGET = ("kk", "ru", "en")
# group -> (parquet glob relative to ROOT, source language, paragraphs)
GROUPS: dict[str, tuple[str, str, int]] = {
    "wikipedia_kk": ("data/raw/wikipedia/kk/part-*.parquet", "kk", 150),
    "fineweb2_kk": ("data/raw/fineweb2_kk/kk/part-*.parquet", "kk", 150),
    "multidomain_cc100": ("data/raw/multidomain_kk/kk/part-000[01]?.parquet", "kk", 100),
    "multidomain_books": ("data/raw/multidomain_kk/kk/part-0002[01].parquet", "kk", 50),
    "multidomain_news": ("data/raw/multidomain_kk/kk/part-0002[2-7].parquet", "kk", 150),
    "multidomain_leipzig": ("data/raw/multidomain_kk/kk/part-0002[89].parquet", "kk", 50),
    "multidomain_oscar": ("data/raw/multidomain_kk/kk/part-0003[01].parquet", "kk", 50),
    "wikipedia_ru": ("data/raw/wikipedia/ru/part-*.parquet", "ru", 300),
    "wikipedia_en": ("data/raw/wikipedia/en/part-*.parquet", "en", 300),
    "wikipedia_ky": ("data/.cache/langid_eval/20231101.ky/*.parquet", "ky", 250),
    "wikipedia_tt": ("data/.cache/langid_eval/20231101.tt/*.parquet", "tt", 250),
}
CLOSE_SOURCE = (  # pinned like the raw Wikipedia source; only used to build this sample
    "wikimedia/wikipedia@b04c8d1ceb2f5cd4588862100d08de323dccfbaa: "
    "20231101.ky/train-00000-of-00001.parquet, 20231101.tt/train-00000-of-00002.parquet"
)


def _paragraphs(path: Path, rows: int, min_letters: int, max_chars: int) -> list[str]:
    pf = pq.ParquetFile(path)
    parts, size = [], 0
    for batch in pf.iter_batches(batch_size=256, columns=["text"]):
        parts.append(pl.from_arrow(batch.column(0)).alias("text"))  # type: ignore[union-attr]
        size += batch.nbytes
        if sum(len(p) for p in parts) >= rows or size >= 32 << 20:
            break
    text = normalize_series(pl.concat(parts))
    lines = (
        pl.DataFrame({"line": text.str.split("\n")})
        .explode("line")
        .filter(pl.col("line").str.count_matches(r"\p{L}") >= min_letters)
    )
    return [s[:max_chars] for s in lines["line"].to_list()]


def build_sample(seed: int, min_letters: int, max_chars: int) -> pl.DataFrame:
    rng = random.Random(seed)
    rows = []
    for group, (glob, lang, n) in GROUPS.items():
        files = sorted(ROOT.glob(glob))
        if not files:
            raise FileNotFoundError(f"{group}: nothing matches {glob}")
        picked = rng.sample(files, min(3, len(files)))
        pool = [p for f in picked for p in _paragraphs(f, 4000, min_letters, max_chars)]
        for i, text in enumerate(rng.sample(pool, n)):
            rows.append({"para_id": f"{group}-{i:04d}", "group": group, "src_lang": lang, "text": text})
    return pl.DataFrame(rows)


def load_models(params: dict[str, Any], names: list[str]) -> dict[str, LangId]:
    spec = params["curation"]["langid"]
    cache = ROOT / spec["cache"]
    return {n: LangId(model_path(n, spec["models"][n], cache), n) for n in names}


def predict_all(sample: pl.DataFrame, models: dict[str, LangId]) -> tuple[pl.DataFrame, dict[str, Any]]:
    perf: dict[str, Any] = {}
    texts = sample["text"].to_list()
    cols = []
    for name, m in models.items():
        t0 = time.perf_counter()
        langs, scores = m.predict(texts)
        dt = time.perf_counter() - t0
        perf[name] = {
            "paragraphs_per_s": round(len(texts) / dt, 1),
            "mb_per_s": round(sum(len(t.encode()) for t in texts) / dt / 1e6, 2),
        }
        cols += [pl.Series(f"pred_{name}", langs), pl.Series(f"score_{name}", scores)]
    return sample.with_columns(cols), perf


def to_review(pred: pl.DataFrame, names: list[str], min_score: float) -> pl.DataFrame:
    cond = pl.lit(False)
    for n in names:
        cond = cond | (pl.col(f"pred_{n}") != pl.col("src_lang")) | (pl.col(f"score_{n}") < min_score)
    return pred.filter(cond)


def decide(pred: pl.Expr, score: pl.Expr, thr: float) -> pl.Expr:
    return pl.when(pred.is_in(TARGET) & (score >= thr)).then(pred).otherwise(pl.lit("other"))


def prf(df: pl.DataFrame, gold: str, pred: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    f1s = []
    for c in (*TARGET, "other"):
        tp = df.filter((pl.col(gold) == c) & (pl.col(pred) == c)).height
        fp = df.filter((pl.col(gold) != c) & (pl.col(pred) == c)).height
        fn = df.filter((pl.col(gold) == c) & (pl.col(pred) != c)).height
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[c] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "support": tp + fn}
        f1s.append(f)
    out["macro_f1"] = round(sum(f1s) / len(f1s), 4)
    return out


def score(params: dict[str, Any]) -> dict[str, Any]:
    spec = params["curation"]["langid"]
    names = list(spec["candidates"])
    sample = pl.read_parquet(REVIEW / "sample.parquet")
    labels = pl.read_csv(REVIEW / "labels.csv", schema_overrides={"note": pl.String})
    df = (
        sample.join(labels.select("para_id", "gold"), on="para_id", how="left")
        .with_columns(gold_fine=pl.coalesce("gold", "src_lang"))
        .filter(pl.col("gold_fine") != "skip")
        .with_columns(
            gold=pl.when(pl.col("gold_fine").is_in(TARGET)).then("gold_fine").otherwise(pl.lit("other"))
        )
    )
    proc = psutil.Process()
    models = {}
    rss = {}
    for n in names:
        r0 = proc.memory_info().rss
        models.update(load_models(params, [n]))
        rss[n] = round((proc.memory_info().rss - r0) / 2**20, 1)
    pred, perf = predict_all(df.select("para_id", "group", "src_lang", "text"), models)
    df = df.join(pred.drop("group", "src_lang", "text"), on="para_id")
    report: dict[str, Any] = {
        "sample": {
            "paragraphs": df.height,
            "reviewed": labels.height,
            "by_gold": dict(df.group_by("gold_fine").len().sort("gold_fine").iter_rows()),
        },
        "close_language_source": CLOSE_SOURCE,
        "min_score": spec["min_score"],
        "models": {},
    }
    grid = [round(0.05 * i, 2) for i in range(20)]
    for n in names:
        d = df.with_columns(dec=decide(pl.col(f"pred_{n}"), pl.col(f"score_{n}"), spec["min_score"]))
        at = prf(d, "gold", "dec")
        sweep = {}
        for thr in grid:
            dd = df.with_columns(dec=decide(pl.col(f"pred_{n}"), pl.col(f"score_{n}"), thr))
            sweep[str(thr)] = prf(dd, "gold", "dec")["macro_f1"]
        best = max(sweep, key=lambda k: sweep[k])
        confusion = (
            d.group_by("gold_fine", "dec")
            .len()
            .sort("gold_fine", "dec")
            .pivot(on="dec", index="gold_fine", values="len")
            .fill_null(0)
        )
        errors = (
            d.filter(pl.col("gold") != pl.col("dec"))
            .group_by("gold_fine", f"pred_{n}")
            .len()
            .sort("len", descending=True)
        )
        report["models"][n] = {
            "at_min_score": at,
            "threshold_sweep_macro_f1": sweep,
            "best_threshold": float(best),
            "best_macro_f1": sweep[best],
            "close_language_false_accept": {
                g: d.filter((pl.col("gold_fine") == g) & (pl.col("dec") != "other")).height
                / max(d.filter(pl.col("gold_fine") == g).height, 1)
                for g in ("ky", "tt")
            },
            "confusion": confusion.to_dicts(),
            "top_errors": errors.head(10).to_dicts(),
            "rss_mb": rss[n],
            **perf[n],
        }
    report["chosen"] = spec["model"]
    report["bootstrap_macro_f1_diff"] = bootstrap_diff(
        df,
        spec["model"],
        [n for n in names if n != spec["model"]],
        spec["min_score"],
        int(params["evaluation"]["bootstrap_samples"]),
        params["seed"],
    )
    return report


def _macro_f1_np(gold: np.ndarray, pred: np.ndarray, k: int) -> float:
    f1 = []
    for c in range(k):
        tp = np.sum((gold == c) & (pred == c))
        fp = np.sum((gold != c) & (pred == c))
        fn = np.sum((gold == c) & (pred != c))
        f1.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(f1))


def bootstrap_diff(
    df: pl.DataFrame, chosen: str, others: list[str], thr: float, n: int, seed: int
) -> dict[str, dict[str, float]]:
    """Paired bootstrap of macro-F1(chosen) - macro-F1(other) over paragraphs (95% interval)."""
    classes = [*TARGET, "other"]
    idx = {c: i for i, c in enumerate(classes)}
    gold = df["gold"].replace_strict(idx).to_numpy()
    dec = {
        m: df.select(decide(pl.col(f"pred_{m}"), pl.col(f"score_{m}"), thr))
        .to_series()
        .replace_strict(idx)
        .to_numpy()
        for m in [chosen, *others]
    }
    rng = np.random.default_rng(seed)
    out = {}
    for o in others:
        diffs = []
        for _ in range(n):
            s = rng.integers(0, len(gold), len(gold))
            diffs.append(_macro_f1_np(gold[s], dec[chosen][s], 4) - _macro_f1_np(gold[s], dec[o][s], 4))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out[o] = {
            "mean": round(float(np.mean(diffs)), 4),
            "ci95_low": round(float(lo), 4),
            "ci95_high": round(float(hi), 4),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["sample", "score"])
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "langid.json")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    params = load_params()
    spec = params["curation"]["langid"]
    if args.command == "sample":
        REVIEW.mkdir(parents=True, exist_ok=True)
        sample = build_sample(params["seed"], spec["min_para_chars"], spec["max_para_chars"])
        sample.write_parquet(REVIEW / "sample.parquet")
        pred, _ = predict_all(sample, load_models(params, list(spec["candidates"])))
        review = to_review(pred, list(spec["candidates"]), spec["min_score"])
        review.write_parquet(REVIEW / "to_review.parquet")
        print(f"sample {sample.height} paragraphs, {review.height} to review")
        return 0
    report = score(params)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print("bootstrap", report["bootstrap_macro_f1_diff"])
    for n, r in report["models"].items():
        print(
            n,
            "macro_f1@min_score",
            r["at_min_score"]["macro_f1"],
            "best",
            r["best_threshold"],
            r["best_macro_f1"],
            {c: r["at_min_score"][c]["f1"] for c in (*TARGET, "other")},
            r["close_language_false_accept"],
            f"{r['paragraphs_per_s']} para/s, {r['rss_mb']} MB",
        )
    metrics = {
        f"{n}_{k}": v
        for n, r in report["models"].items()
        for k, v in (
            [
                ("macro_f1", r["at_min_score"]["macro_f1"]),
                ("best_macro_f1", r["best_macro_f1"]),
                ("paragraphs_per_s", r["paragraphs_per_s"]),
                ("rss_mb", r["rss_mb"]),
            ]
            + [(f"f1_{c}", r["at_min_score"][c]["f1"]) for c in (*TARGET, "other")]
        )
    }
    log_mlflow(
        "curation",
        "eval_langid",
        {"candidates": spec["candidates"], "min_score": spec["min_score"], "git_sha": git_sha(ROOT)},
        metrics,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
