from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from trifact.data import decontaminate as dc
from trifact.data.hashing import ngram_hashes, word_hashes

EVAL = "one two three four five six"


def grams_of(text: str, n: int) -> np.ndarray:
    w, c = word_hashes(pl.Series([text]))
    g, _ = ngram_hashes(w, c, n)
    return np.unique(g)


@pytest.fixture
def ctx() -> None:
    grams = grams_of(EVAL, 3)
    dc._CTX.clear()
    dc._CTX.update(grams=grams, ngram=3, active=np.ones(len(grams), dtype=bool), max_removed_share=0.5)


def test_doc_hits_finds_eval_grams_case_and_punctuation_insensitive(ctx: None) -> None:
    rows, grams = dc.doc_hits(
        pl.Series(["alpha beta gamma", "x One, TWO three! y", "two"]), dc._CTX["grams"], 3
    )
    assert rows.tolist() == [1]
    assert len(grams) == 1


def test_strip_lines_removes_line_or_drops_document(ctx: None) -> None:
    long_ok = "a clean first line of the document text\n" * 3
    texts = [
        long_ok + "leak: one two three here",  # short leaked line -> line removed
        "one two three four five six",  # all text leaked -> dropped
        "one two\nthree four",  # gram spans the line break -> dropped
        "nothing to see in this text at all",
    ]
    new, removed = dc.strip_lines(texts)
    assert new[0] == long_ok.strip("\n") and removed[0] == 1
    assert new[1] is None
    assert new[2] is None
    assert new[3] == texts[3] and removed[3] == 0


def test_template_grams_are_ignored(ctx: None) -> None:
    dc._CTX["active"][:] = False
    new, removed = dc.strip_lines(["x one two three y"])
    assert new == ["x one two three y"] and removed == [0]
