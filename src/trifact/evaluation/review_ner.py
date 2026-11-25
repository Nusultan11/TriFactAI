"""Human noise estimate of silver NER (ADR-0015) from the review sheet `data/review/ner_silver/labels.csv`.

python -m trifact.evaluation.review_ner score [--reviewer NAME]   # reports/ner_silver_review.json (+ MLflow)

The sheet (`build_ner`: 200 silver sentences per language, seeded hash of `sent_id`) lists per sentence the
number of marked entities and, filled by a person, the counts of entities with a wrong type, wrong boundaries,
marked spans that are not a PER/ORG/LOC entity, and missed entities. Per language:

* precision = correct marked entities / marked entities, correct = marked - wrong_type - wrong_span - not_entity;
* recall    = correct / (correct + missed + wrong_type + wrong_span): a wrongly typed or bounded entity is also
  an entity the silver did not get right;
* error shares by kind; Wilson 95% intervals over entities (an approximation: entities of one sentence are not
  independent) and a sentence-level bootstrap interval for precision and recall.

Every row must name its reviewer; a reviewer that names a model is refused (ADR-0022: model labels are not human
labels). `reports/ner_silver_review.json` records `human: true` only for a human review.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Any

import numpy as np
import polars as pl

from trifact.core.params import ROOT, load_params
from trifact.data.curation import write_report
from trifact.evaluation.review_quality import wilson
from trifact.lake.stats import git_sha, log_mlflow

REVIEW = ROOT / "data" / "review" / "ner_silver"
COUNTS = ["wrong_type", "wrong_span", "not_entity", "missed"]
MODEL_NAMES = re.compile(r"llm|claude|gpt|qwen|machine|auto|model", re.IGNORECASE)


def load(path: Any, *, calibrated: bool = False) -> pl.DataFrame:
    df = pl.read_csv(path, infer_schema_length=0, encoding="utf8-lossy")
    missing = [c for c in ["sent_id", "lang", "entities", *COUNTS, "reviewer"] if c not in df.columns]
    if missing:
        raise ValueError(f"labels.csv lacks columns {missing}")
    df = df.with_columns(pl.col(["entities", *COUNTS]).str.strip_chars().replace("", None).cast(pl.Int64))
    if calibrated:
        import json

        cal_ids = set(
            json.loads((ROOT / "data" / "review" / "calibration_sample.json").read_text(encoding="utf-8"))[
                "ids"
            ]["ner"]
        )
        df = df.filter(pl.col("reviewer").fill_null("").str.strip_chars() != "")
        missing_cal = cal_ids - set(df["sent_id"].to_list())
        if missing_cal:
            raise ValueError(f"{len(missing_cal)} ADR-0026 calibration NER sentences have no reviewer")
    else:
        todo = df.filter(pl.col("reviewer").fill_null("").str.strip_chars() == "")
        if todo.height:
            raise ValueError(f"{todo.height} sentences have no reviewer")
    models = df.filter(pl.col("reviewer").str.contains(MODEL_NAMES.pattern))
    if models.height:
        raise ValueError(
            f"{models.height} rows are labelled by a model ({models['reviewer'][0]}), not by a person"
        )
    return df.with_columns(pl.col(COUNTS).fill_null(0))


def prf(d: pl.DataFrame) -> tuple[float, float]:
    marked = int(d["entities"].sum())
    wrong = int(d.select(pl.sum_horizontal("wrong_type", "wrong_span", "not_entity")).sum().item())
    correct = marked - wrong
    gold = correct + int(d.select(pl.sum_horizontal("missed", "wrong_type", "wrong_span")).sum().item())
    return (correct / marked if marked else 0.0), (correct / gold if gold else 0.0)


def score(df: pl.DataFrame, samples: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"sentences": df.height, "by_lang": {}}
    for (lang,), d in sorted(df.group_by("lang"), key=lambda x: str(x[0][0])):
        marked = int(d["entities"].sum())
        wrong = {c: int(d[c].sum()) for c in COUNTS}
        p, r = prf(d)
        boot = []
        for _ in range(samples):
            idx = rng.integers(0, d.height, d.height)
            boot.append(prf(d[idx]))
        bp, br = np.array(boot).T
        correct = marked - wrong["wrong_type"] - wrong["wrong_span"] - wrong["not_entity"]
        out["by_lang"][str(lang)] = {
            "sentences": d.height, "marked": marked, "errors": wrong,
            "precision": round(p, 4), "precision_ci95_wilson": wilson(correct, marked),
            "precision_ci95_bootstrap": [round(float(np.quantile(bp, 0.025)), 4), round(float(np.quantile(bp, 0.975)), 4)],
            "recall": round(r, 4),
            "recall_ci95_bootstrap": [round(float(np.quantile(br, 0.025)), 4), round(float(np.quantile(br, 0.975)), 4)],
            "missed_per_sentence": round(wrong["missed"] / d.height, 3),
            "sentences_without_errors": round(float((d.select(pl.sum_horizontal(COUNTS)).to_series() == 0).mean()), 4),
        }  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["score"])
    ap.add_argument("--labels", default=str(REVIEW / "labels.csv"))
    ap.add_argument("--calibrated", action="store_true", help="ADR-0026 calibrated human audit mode")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    params = load_params()
    df = load(args.labels, calibrated=args.calibrated)
    res = score(df, int(params["evaluation"]["bootstrap_samples"]), int(params["seed"]))
    res["reviewers"] = sorted(df["reviewer"].unique().to_list())
    res["human"] = True
    if args.calibrated:
        from trifact.evaluation.review_quality import cohen_kappa

        res["protocol"] = "ADR-0026"
        res["human_audit_n"] = df.height
        mfile = REVIEW / "machine_labels.csv"
        if mfile.is_file():
            m = pl.read_csv(mfile, infer_schema_length=10000)
            res["machine_audit_n"] = m.height
            j = df.with_columns(human_has_err=(pl.sum_horizontal(COUNTS) > 0).cast(pl.String)).join(
                m.select("sent_id", judge_has_err=(pl.col("machine_errors") > 0).cast(pl.String)),
                on="sent_id",
            )
            hl, jl = j["human_has_err"].to_list(), j["judge_has_err"].to_list()
            res["judge_calibration"] = {
                "n": j.height,
                "error_presence_agreement": round(
                    sum(1 for x, y in zip(hl, jl, strict=True) if x == y) / j.height, 4
                )
                if j.height
                else None,
                "cohen_kappa": cohen_kappa(hl, jl),
            }
    if "assist" in df.columns:
        res["assist"] = dict(df.group_by("assist").len().iter_rows())
    write_report(ROOT / "reports" / "ner_silver_review.json", res)
    metrics = {}
    for lang, r in res["by_lang"].items():
        metrics[f"{lang}_precision"], metrics[f"{lang}_recall"] = r["precision"], r["recall"]
        print(
            lang,
            "precision",
            r["precision"],
            r["precision_ci95_bootstrap"],
            "recall",
            r["recall"],
            r["recall_ci95_bootstrap"],
        )
    log_mlflow("review", "ner_silver_review", {"human": True, "code_version": git_sha(ROOT)}, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
