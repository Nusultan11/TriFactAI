"""Stage split: leakage-safe train/validation/test assignment of every gold source (ADR-0007).

python -m trifact.data.split

Per gold source one row per example in `data/task/splits/<lang>/assignments-*.parquet`:
`source, lang, config, example_id, orig_split, split, doc_key, text_key, leak`.

* Official test splits are kept when they carry labels (comparability with published numbers);
  without one, the official validation split becomes the test split (SQuAD v2, MIRACL dev) and validation is cut from train by `doc_key` hash
  (`params.yaml: split.val`). Sources without any split (KazQAD `nq-*`) are split by `doc_key`
  into `1 - val - test / val / test`; the parallel kk and en `nq-*` share ids, so they share splits.
  SberQuAD's official train and validation share contexts and its test has no answers: its labelled
  examples are pooled and re-split by `doc_key`.
* `doc_key` is the source document (Wikipedia title, NEREL document, XNLI premise, KazNERD sentence)
  so that all examples of one document stay in one split.
* Leakage: a train/validation example whose `doc_key` or normalized text (`text_key`) also occurs in
  the test split of the same task (any source, any language-agnostic exact match) gets `leak = true`
  and the split `drop`; validation is checked against test, train against validation and test.
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
    assign_split,
    by_lang,
    finish,
    inherited,
    key_expr,
    raw_input,
    scan_raw,
)
from trifact.lake.layout import LakeConfig
from trifact.lake.stats import ResourceMonitor, log_mlflow

log = logging.getLogger("trifact.split")
TASKS = {  # source -> task: leakage is checked between splits of the same task
    "kaznerd": "ner", "wikiann": "ner", "wikineural": "ner", "nerel": "ner",
    "squad_v2": "qa", "sberquad": "qa", "kazqad": "qa", "xnli": "nli", "miracl": "retrieval",
}  # fmt: skip
SOURCES = list(TASKS)


def _tokens_text(col: str = "tokens") -> pl.Expr:
    return pl.col(col).list.join(" ")


def examples(root: Path, source: str) -> pl.DataFrame:
    """source, lang, config, example_id, orig_split, doc_key, text."""
    lf = scan_raw(root, source, "topics" if source == "miracl" else "part")
    if source in ("kaznerd", "wikiann", "wikineural"):
        df = lf.select("lang", "_config", "_split", "tokens").collect()
        df = df.with_row_index("row").with_columns(
            example_id=pl.concat_str(
                pl.col("_config").fill_null(""), pl.col("_split"), pl.col("row"), separator="/"
            ),
            text=_tokens_text(),
        )
        df = df.with_columns(doc_key=key_expr("text"))
    elif source == "nerel":
        df = lf.select("lang", "_config", "_split", "id", "text").collect()
        # ids restart in every split: the document key is its normalized opening text
        df = df.with_columns(
            example_id=pl.concat_str(pl.col("_split"), pl.col("id").cast(pl.String), separator="/"),
            doc_key=key_expr("text").str.slice(0, 200),
        )
    elif source in ("squad_v2", "sberquad", "kazqad"):
        df = lf.select("lang", "_config", "_split", "id", "title", "context", "question").collect()
        if source == "sberquad":
            # the official test has no answers, and 12 of its ids repeat train ids: it is left out entirely
            df = df.filter(pl.col("_split") != "test")
        df = df.with_columns(
            example_id=pl.concat_str(
                pl.col("_config").fill_null(""), pl.col("id").cast(pl.String), separator="/"
            ),
            doc_key=pl.concat_str(
                pl.col("title").fill_null(""), key_expr("context").str.slice(0, 64), separator="|"
            ),
            text=pl.col("question"),
            ctx_key=key_expr("context").hash(seed=0).cast(pl.String),
        )
        if source == "kazqad":  # nq-* are parallel by id: one key per id, title of the en side is not shared
            df = df.with_columns(
                example_id=pl.when(pl.col("_config").str.starts_with("nq-"))
                .then(pl.concat_str(pl.lit("nq/"), pl.col("id")))
                .otherwise(pl.col("example_id")),
                doc_key=pl.when(pl.col("_config").str.starts_with("nq-"))
                .then(pl.concat_str(pl.lit("nq/"), pl.col("id")))
                .otherwise(pl.col("doc_key")),
            )
    elif source == "xnli":
        df = lf.select("lang", "_config", "_split", "premise", "hypothesis").collect()
        # en and ru rows are aligned by position within a split (HF XNLI); id = split/row in that split
        df = df.with_columns(row=pl.int_range(pl.len()).over("lang", "_split")).with_columns(
            example_id=pl.concat_str(pl.col("_split"), pl.col("row"), separator="/"),
            text=pl.concat_str("premise", "hypothesis", separator=" || "),
        )
        en_premise = df.filter(pl.col("lang") == "en").select("example_id", doc_key=key_expr("premise"))
        df = df.join(en_premise, on="example_id", how="left")
    elif source == "miracl":
        df = lf.select("lang", "_config", "_split", "qid", "query").collect()
        df = df.filter(pl.col("_split").is_in(["train", "validation"]))  # test-a/b have no qrels
        df = df.with_columns(
            example_id=pl.col("qid"),
            doc_key=pl.concat_str("lang", "qid", separator="/"),
            text=pl.col("query"),
        )
    else:
        raise ValueError(source)
    if "ctx_key" not in df.columns:
        df = df.with_columns(ctx_key=pl.lit("", dtype=pl.String))
    return df.select(
        source=pl.lit(source), lang="lang", config=pl.col("_config").cast(pl.String), example_id="example_id",
        orig_split=pl.col("_split"), doc_key="doc_key", text_key=key_expr("text"), ctx_key="ctx_key",
    )  # fmt: skip


def assign(df: pl.DataFrame, val: float, test: float, seed: int) -> pl.DataFrame:
    src = df["source"][0]
    splits = set(df["orig_split"].unique().to_list())
    if src == "sberquad":
        # official train and validation share contexts (split by question): pool the labelled
        # examples and re-split by document; the official test has no answers
        lab = df.filter(pl.col("orig_split").is_in(["train", "validation"]))
        rest = df.filter(~pl.col("orig_split").is_in(["train", "validation"])).with_columns(
            split=pl.lit("drop")
        )
        return pl.concat([lab.with_columns(split=assign_split(lab, "doc_key", val, test, seed)), rest])
    if src in ("squad_v2", "miracl") and "test" not in splits:
        # test := official validation; validation := doc-hash slice of train
        tr = df.filter(pl.col("orig_split") == "train")
        cut = assign_split(tr, "doc_key", val / (1 - test), 0.0, seed)
        tr = tr.with_columns(split=cut)
        te = df.filter(pl.col("orig_split") == "validation").with_columns(split=pl.lit("test"))
        rest = df.filter(~pl.col("orig_split").is_in(["train", "validation"])).with_columns(
            split=pl.lit("drop")
        )
        return pl.concat([tr, te, rest])
    if src == "kazqad":
        nq = df.filter(pl.col("config").str.starts_with("nq-"))
        nq = nq.with_columns(split=assign_split(nq, "doc_key", val, test, seed))
        rest = df.filter(~pl.col("config").str.starts_with("nq-")).with_columns(split=pl.col("orig_split"))
        return pl.concat([rest, nq])
    return df.with_columns(split=pl.col("orig_split"))


def mark_leaks(df: pl.DataFrame) -> pl.DataFrame:
    """Drop train/validation examples sharing a doc_key, text_key or ctx_key with a later split of the same task.

    ctx_key (QA: hash of the normalized context) catches contexts reused under different documents keys: the
    NQ part of KazQAD reuses one passage for many question ids (9 196 en contexts), and keys by id let 2 257 en
    and 1 206 kk test contexts into train. Only train/validation rows are dropped, test stays as it is."""
    out = []
    for _task, g in df.group_by("task", maintain_order=True):
        test = g.filter(pl.col("split") == "test")
        val = g.filter(pl.col("split") == "validation")
        bad_doc_val = set(test["doc_key"]) - {""}
        bad_txt_val = set(test["text_key"]) - {""}
        bad_ctx_val = set(test["ctx_key"]) - {""}
        bad_doc_tr = bad_doc_val | set(val["doc_key"])
        bad_txt_tr = bad_txt_val | set(val["text_key"])
        bad_ctx_tr = bad_ctx_val | (set(val["ctx_key"]) - {""})
        leak = (
            (pl.col("split") == "validation")
            & (pl.col("doc_key").is_in(list(bad_doc_val)) | pl.col("text_key").is_in(list(bad_txt_val))
               | pl.col("ctx_key").is_in(list(bad_ctx_val)))
        ) | (
            (pl.col("split") == "train")
            & (pl.col("doc_key").is_in(list(bad_doc_tr)) | pl.col("text_key").is_in(list(bad_txt_tr))
               | pl.col("ctx_key").is_in(list(bad_ctx_tr)))
        )  # fmt: skip
        out.append(g.with_columns(leak=leak))
    df = pl.concat(out)
    return df.with_columns(split=pl.when(pl.col("leak")).then(pl.lit("drop")).otherwise(pl.col("split")))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    params = load_params()
    root = (ROOT / params["lake"]["root"]).resolve()
    out = args.out or root / "task" / "splits"
    sp = params["split"]
    stage_params = {"split": sp, "seed": params["seed"], "sources": SOURCES}
    manifests = [root / "raw" / s / "source.yaml" for s in SOURCES]
    fp = fingerprint(
        "split", stage_params, [Path(__file__), ROOT / "src/trifact/data/task_common.py"], manifests
    )
    t0 = time.perf_counter()
    with ResourceMonitor() as mon:
        frames = []
        for s in SOURCES:
            df = assign(examples(root, s), sp["val"], sp["test"], params["seed"])
            frames.append(df.with_columns(task=pl.lit(TASKS[s])))
            log.info("%s: %d examples", s, df.height)
        df = mark_leaks(pl.concat(frames))
        if out.exists():
            shutil.rmtree(out)
        w = DatasetWriter(out, LakeConfig.from_params(params["lake"], root))
        w.write("assignments", df.sort("source", "lang", "example_id"))
        shards = w.close()
    summary = (
        df.group_by("task", "source", "lang", "split")
        .len()
        .sort("task", "source", "lang", "split")
        .to_dicts()
    )
    leaks = (
        df.filter(pl.col("leak"))
        .group_by("source", "lang", "orig_split")
        .len()
        .sort("source", "lang")
        .to_dicts()
    )
    metrics: dict[str, Any] = {"by_split": summary, "leaks_dropped": leaks}
    finish(
        out, name="task/splits", layer="task", stage="split", source="+".join(SOURCES),
        description="Split assignment (train/validation/test/drop) of every gold example, by source document, "
        "with exact-duplicate leakage between splits removed (ADR-0007).",
        params=stage_params, inputs=[raw_input(root, s) for s in SOURCES], licenses=inherited(root, SOURCES),
        shards=shards, rows_in=by_lang(df), metrics=metrics, stats=mon.stats(df.height, 0, 1), fingerprint=fp,
        drops={"leak": {str(k): int(v) for k, v in df.filter(pl.col("leak")).group_by("lang").len().iter_rows()}},
    )  # fmt: skip
    write_report(ROOT / "reports" / "splits.json", metrics)
    log_mlflow("task", "split", {**stage_params, "fingerprint": fp}, {"examples": df.height,
               "leaks_dropped": int(df["leak"].sum()), "seconds": round(time.perf_counter() - t0, 1)})  # fmt: skip
    for r in leaks:
        print("leak dropped:", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
