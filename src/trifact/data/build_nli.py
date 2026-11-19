"""Stage build_nli: XNLI ru/en + Kazakh XNLI by machine translation with filters (ADR-0015).

python -m trifact.data.build_nli [--scores DIR]

Rows: `pair_id, lang, origin (gold|mt), split, premise, hypothesis, label (0 entailment, 1 neutral,
2 contradiction), qc` in tables `train`, `validation`, `test` of `data/task/nli/<lang>/`.

* en / ru: XNLI as published (ru train is the XNLI machine translation of MultiNLI, ru validation/test
  are human translations), split assignments from `task/splits` (`drop` = leakage, ADR-0007).
* kk: the en premise and hypothesis translated by NLLB (`task/mt/xnli_kk`). A pair is kept when both
  sides pass `params.yaml: nli.kk_filters`: back-translation chrF, translation detected as Kazakh,
  length ratio, no copy. Label consistency (`--scores`, NLI model predictions on the en and kk pair
  from the Kaggle job `nli_score`): a pair whose kk prediction disagrees with both the gold label and
  the en prediction is dropped as `label_flip` (the translation probably changed the meaning).
* kk validation/test are machine translations: they are eval candidates only after the human check
  of stage 3.5 (`needs_review = true`), never frozen unreviewed (ADR-0015).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl

from trifact.core.params import ROOT, load_params
from trifact.data.curation import fingerprint, write_report
from trifact.data.task_common import (
    DatasetWriter,
    by_lang,
    dataset_input_of,
    finish,
    inherited,
    raw_input,
    scan_raw,
)
from trifact.data.translations import MT_LICENSE
from trifact.lake.layout import LakeConfig
from trifact.lake.stats import ResourceMonitor, log_mlflow

log = logging.getLogger("trifact.build_nli")
LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}


def xnli(root: Path) -> pl.DataFrame:
    df = scan_raw(root, "xnli").select("lang", "_split", "premise", "hypothesis", "label").collect()
    df = df.with_columns(row=pl.int_range(pl.len()).over("lang", "_split"))
    df = df.with_columns(example_id=pl.concat_str(pl.col("_split"), pl.col("row"), separator="/"))
    asg = (
        pl.scan_parquet(str(root / "task/splits/*/assignments-*.parquet"))
        .filter(pl.col("source") == "xnli")
        .select("lang", "example_id", "split")
        .collect()
    )
    return df.join(asg, on=["lang", "example_id"], how="left").drop("row", "_split")


def kk_pairs(root: Path, en: pl.DataFrame, filt: dict[str, Any]) -> tuple[pl.DataFrame, dict[str, Any]]:
    sent = pl.read_parquet(
        root / "external/kaggle/translate_xnli_kk/sentences_all.parquet"
    )  # id, text sent to the MT job
    mt = pl.scan_parquet(str(root / "task/mt/xnli_kk/kk/translations-*.parquet")).collect()
    mt = mt.join(sent.rename({"text": "src"}), on="id", how="inner")
    ok = (
        (pl.col("chrf_back") >= filt["min_chrf_back"])
        & (pl.col("mt_lang") == "kk")
        & (pl.col("mt_lang_score") >= filt["min_lang_score"])
        & pl.col("len_ratio").is_between(filt["min_len_ratio"], filt["max_len_ratio"])
        & ~pl.col("copy")
        & (pl.col("translation").str.strip_chars() != "")
    )
    mt = mt.with_columns(ok=ok)
    side = mt.select("src", "translation", "ok", "chrf_back")
    p = en.join(
        side.rename({"src": "premise", "translation": "premise_kk", "ok": "ok_p", "chrf_back": "chrf_p"}),
        on="premise",
        how="left",
    )
    p = p.join(
        side.rename(
            {"src": "hypothesis", "translation": "hypothesis_kk", "ok": "ok_h", "chrf_back": "chrf_h"}
        ),
        on="hypothesis",
        how="left",
    )
    p = p.with_columns(
        reason=pl.when(pl.col("premise_kk").is_null() | pl.col("hypothesis_kk").is_null()).then(pl.lit("not_translated"))
        .when(~pl.col("ok_p")).then(pl.lit("premise_qc"))
        .when(~pl.col("ok_h")).then(pl.lit("hypothesis_qc"))
        .otherwise(None)
    )  # fmt: skip
    stats = {"translated_sentences": mt.height, "sentences_passing_qc": int(mt["ok"].sum()),
             "drops": {str(k): int(v) for k, v in p.group_by("reason").len().iter_rows() if k is not None}}  # fmt: skip
    return p, stats


def apply_scores(p: pl.DataFrame, scores: Path | None) -> tuple[pl.DataFrame, dict[str, Any]]:
    if scores is None or not any(scores.glob("out/*.parquet")):
        return p.with_columns(pred_en=pl.lit(None, dtype=pl.Int64), pred_kk=pl.lit(None, dtype=pl.Int64)), {
            "label_consistency": "not run"
        }
    s = pl.concat([pl.read_parquet(f) for f in sorted(scores.glob("out/*.parquet"))])
    name_to_label = {v: k for k, v in LABELS.items()}
    pcols = [c for c in s.columns if c.startswith("p_")]
    s = s.with_columns(pred=pl.concat_list(pcols).list.arg_max()).with_columns(
        pred=pl.col("pred").replace_strict(
            {i: name_to_label[c[2:]] for i, c in enumerate(pcols)}, return_dtype=pl.Int64
        )
    )
    en_s = s.filter(pl.col("id").str.starts_with("en/")).select(
        example_id=pl.col("id").str.slice(3), pred_en="pred"
    )
    kk_s = s.filter(pl.col("id").str.starts_with("kk/")).select(
        example_id=pl.col("id").str.slice(3), pred_kk="pred"
    )
    p = p.join(en_s, on="example_id", how="left").join(kk_s, on="example_id", how="left")
    flip = (pl.col("pred_kk") != pl.col("label")) & (pl.col("pred_kk") != pl.col("pred_en"))
    p = p.with_columns(
        reason=pl.when(pl.col("reason").is_null() & flip)
        .then(pl.lit("label_flip"))
        .otherwise(pl.col("reason"))
    )
    scored = p.filter(pl.col("pred_kk").is_not_null())
    acc = scored.group_by("split").agg(
        n=pl.len(), acc_en=(pl.col("pred_en") == pl.col("label")).mean().round(4),
        acc_kk=(pl.col("pred_kk") == pl.col("label")).mean().round(4),
        agree_en_kk=(pl.col("pred_en") == pl.col("pred_kk")).mean().round(4),
    ).sort("split")  # fmt: skip
    return p, {"label_consistency": acc.to_dicts()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, help="Kaggle nli_score output directory")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    params = load_params()
    root = (ROOT / params["lake"]["root"]).resolve()
    out = root / "task" / "nli"
    filt = params["nli"]["kk_filters"]
    scores = args.scores.resolve() if args.scores else None
    stage_params = {"kk_filters": filt, "label_consistency": scores is not None}
    inputs_m = [
        root / "raw/xnli/source.yaml",
        root / "task/splits/dataset.yaml",
        root / "task/mt/xnli_kk/dataset.yaml",
    ]
    extra = sorted(scores.glob("out/*.parquet")) if scores else []
    fp = fingerprint("build_nli", stage_params, [Path(__file__)], [*inputs_m, *extra])
    t0 = time.perf_counter()
    with ResourceMonitor() as mon:
        x = xnli(root)
        en = x.filter(pl.col("lang") == "en")
        kk, kstats = kk_pairs(root, en, filt)
        kk, cons = apply_scores(kk, scores)
        rows = [
            x.filter(pl.col("split") != "drop").select(
                pair_id=pl.concat_str(pl.lit("xnli"), "lang", "example_id", separator="/"), lang="lang",
                origin=pl.when(pl.col("lang") == "ru").then(pl.when(pl.col("split") == "train").then(pl.lit("mt_published")).otherwise(pl.lit("gold"))).otherwise(pl.lit("gold")),
                split="split", premise="premise", hypothesis="hypothesis", label="label", needs_review=pl.lit(False),
            ),
            kk.filter(pl.col("reason").is_null() & (pl.col("split") != "drop")).select(
                pair_id=pl.concat_str(pl.lit("xnli/kk"), "example_id", separator="/"), lang=pl.lit("kk"), origin=pl.lit("mt"),
                split="split", premise="premise_kk", hypothesis="hypothesis_kk", label="label",
                needs_review=pl.col("split") != "train",
            ),
        ]  # fmt: skip
        df = pl.concat(rows)
        if out.exists():
            shutil.rmtree(out)
        w = DatasetWriter(out, LakeConfig.from_params(params["lake"], root))
        for split in ("train", "validation", "test"):
            w.write(split, df.filter(pl.col("split") == split).drop("split").sort("lang", "pair_id"))
        shards = w.close()
    bal = df.group_by("lang", "split", "label").len().sort("lang", "split", "label").to_dicts()
    metrics: dict[str, Any] = {"kk_translation": kstats, **cons, "class_balance": bal,
                               "by_lang_split": df.group_by("lang", "split").len().sort("lang", "split").to_dicts(),
                               "kk_drops_by_split": kk.group_by("split", "reason").len().sort("split", "reason").to_dicts()}  # fmt: skip
    finish(
        out, name="task/nli", layer="task", stage="build_nli", source="xnli",
        description="NLI pairs: XNLI en/ru and Kazakh machine translation of XNLI en (NLLB) filtered by round-trip "
        "chrF, language id, length ratio and label consistency; kk validation/test need human review before eval.",
        params=stage_params, inputs=[raw_input(root, "xnli"), dataset_input_of(root, root / "task/splits"),
                                     dataset_input_of(root, root / "task/mt/xnli_kk")],
        licenses=inherited(root, ["xnli"]) + [MT_LICENSE], shards=shards, rows_in=by_lang(x),
        drops={"kk_" + str(k): {"kk": int(v)} for k, v in kk.group_by("reason").len().iter_rows() if k is not None},
        metrics=metrics, stats=mon.stats(df.height, 0, 1), fingerprint=fp,
    )  # fmt: skip
    write_report(ROOT / "reports" / "nli_data.json", metrics)
    log_mlflow("task", "build_nli", {**stage_params, "fingerprint": fp}, {
        "pairs": df.height, "seconds": round(time.perf_counter() - t0, 1),
        **{f"pairs_{k}": v for k, v in by_lang(df).items()}})  # fmt: skip
    print(df.group_by("lang", "split").len().sort("lang", "split"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
