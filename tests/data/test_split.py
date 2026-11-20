from __future__ import annotations

import polars as pl

from trifact.data.split import mark_leaks


def test_mark_leaks_drops_train_rows_sharing_a_test_context() -> None:
    df = pl.DataFrame(
        {
            "task": ["qa"] * 4,
            "split": ["test", "train", "train", "validation"],
            "doc_key": ["nq/1", "nq/2", "nq/3", "nq/4"],
            "text_key": ["q one", "q two", "q three", "q four"],
            "ctx_key": ["c1", "c1", "c2", "c1"],
        }
    )
    got = mark_leaks(df)["split"].to_list()
    assert got == ["test", "drop", "train", "drop"]  # test kept, train/validation on the test context dropped


def test_mark_leaks_ignores_empty_context_keys() -> None:
    df = pl.DataFrame(
        {
            "task": ["ner"] * 2,
            "split": ["test", "train"],
            "doc_key": ["a", "b"],
            "text_key": ["x", "y"],
            "ctx_key": ["", ""],
        }
    )
    assert mark_leaks(df)["split"].to_list() == ["test", "train"]
