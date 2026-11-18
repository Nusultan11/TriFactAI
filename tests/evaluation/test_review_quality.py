from __future__ import annotations

import polars as pl
import pytest

from trifact.evaluation.review_quality import allocate, score, wilson


def test_allocate_floor_and_total() -> None:
    a = allocate({"big": 1000, "mid": 100, "tiny": 3}, 50)
    assert a["tiny"] == 3 and sum(a.values()) == 50 and a["big"] > a["mid"] >= 10
    assert allocate({"x": 5}, 50) == {"x": 5}


def test_wilson_bounds() -> None:
    lo, hi = wilson(9, 10)
    assert 0.5 < lo < 0.9 < hi <= 1.0
    assert wilson(0, 0) == [0.0, 1.0]


def sample_frame() -> pl.DataFrame:
    rows = []
    for lang in ("kk", "ru", "en"):
        for i in range(4):
            rows.append((f"{lang}k{i}", lang, "kept", "web", 100))
        for i in range(2):
            rows.append((f"{lang}d{i}", lang, "dropped", "too_short", 10))
    return pl.DataFrame(
        rows, schema=["doc_id", "review_lang", "status", "stratum", "stratum_size"], orient="row"
    )


def test_score_shares_and_good_lost() -> None:
    s = sample_frame()
    labels = s.select(
        "doc_id",
        label=pl.when(pl.col("doc_id") == "kkk0")
        .then(pl.lit("bad"))
        .when(pl.col("doc_id") == "kkd0")
        .then(pl.lit("good"))
        .when(pl.col("status") == "kept")
        .then(pl.lit("good"))
        .otherwise(pl.lit("bad")),
        note=pl.lit(""),
    )
    r = score(s, labels)["by_lang"]["kk"]
    assert r["kept"]["share"] == 0.75 and r["dropped"]["share"] == 0.5
    # 100 kept docs * 0.75 good = 75; 10 dropped * 0.5 good = 5 -> 5 / 80 lost
    assert r["good_lost"] == pytest.approx(5 / 80, abs=1e-4)


def test_score_requires_every_label() -> None:
    s = sample_frame()
    with pytest.raises(ValueError, match="no label"):
        score(s, s.head(3).select("doc_id", label=pl.lit("good"), note=pl.lit("")))
