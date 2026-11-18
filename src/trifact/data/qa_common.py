"""Gold extractive QA (SQuAD v2, SberQuAD, KazQAD) with split assignments and BM25 distractor contexts."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from trifact.data.bm25 import BM25
from trifact.data.task_common import scan_raw

QA_SOURCES = ("squad_v2", "sberquad", "kazqad")


def load_qa(root: Path, source: str, splits: tuple[str, ...]) -> pl.DataFrame:
    """example_id, lang, config, split, title, question, context, answers (list), answerable, doc_id."""
    asg = (
        pl.scan_parquet(str(root / "task/splits/*/assignments-*.parquet"))
        .filter(pl.col("source") == source)
        .select("lang", "example_id", "split")
        .collect()
    )
    df = (
        scan_raw(root, source)
        .select("lang", "_config", "id", "title", "context", "question", "answers")
        .collect()
    )
    df = df.with_columns(
        example_id=pl.when(pl.col("_config").str.starts_with("nq-"))
        .then(pl.concat_str(pl.lit("nq/"), "id"))
        .otherwise(
            pl.concat_str(pl.col("_config").fill_null(""), pl.col("id").cast(pl.String), separator="/")
        ),
        answers=pl.col("answers").struct.field("text").list.eval(pl.element().filter(pl.element() != "")),
    ).with_columns(answerable=pl.col("answers").list.len() > 0)
    df = df.join(asg, on=["lang", "example_id"], how="left").filter(pl.col("split").is_in(list(splits)))
    return df.with_columns(
        config=pl.col("_config"),
        doc_id=pl.concat_str(
            pl.lit(f"{source}/"), "lang", pl.col("context").hash(seed=0).cast(pl.String), separator="/"
        ),
    ).drop("_config", "id")


def bm25_negatives(df: pl.DataFrame, k: int, exclude_answer: bool = False) -> pl.Series:
    """Up to `k` BM25 contexts per question from the same language pool: never the gold context, a context with
    the same title, or (`exclude_answer`) a context containing one of the answers."""
    out: list[list[str]] = [[] for _ in range(df.height)]
    df = df.with_row_index("_r")
    for (_lang,), g in df.group_by("lang"):
        ctx = g.unique("doc_id").select("doc_id", "title", "context")
        bm = BM25(ctx["context"].to_list())
        idx, _ = bm.search(g["question"].to_list(), k + 5)
        ids, titles, texts = ctx["doc_id"].to_list(), ctx["title"].to_list(), ctx["context"].to_list()
        for r, row, pos, title, answers in zip(
            g["_r"], idx, g["doc_id"], g["title"], g["answers"], strict=True
        ):
            keep = []
            for j in row:
                if j < 0 or ids[j] == pos or titles[j] == title:
                    continue
                if exclude_answer and any(a and a in texts[j] for a in answers or []):
                    continue
                keep.append(ids[j])
                if len(keep) == k:
                    break
            out[r] = keep
    return pl.Series("neg_ids", out, dtype=pl.List(pl.String))
