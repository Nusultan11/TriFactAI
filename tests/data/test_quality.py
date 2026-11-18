from __future__ import annotations

import polars as pl
import pytest

from trifact.data.quality import REASONS, apply_rules, dup_ngram_ratio, signals, thresholds

CFG = {
    "default": {
        "min_lang_score": 0.65, "min_lang_conf": 0.0, "min_chars": 100, "min_words": 15, "min_alpha_ratio": 0.6,
        "max_replacement_ratio": 0.001, "max_dup_line_ratio": 0.3, "max_dup_ngram_ratio": 0.3,
        "min_mean_word_len": 3.0, "max_mean_word_len": 12.0, "max_bullet_ratio": 0.9, "max_ellipsis_ratio": 0.3,
        "min_stopword_ratio": 0.02, "stopword_min_words": 10,
    },
    "languages": {"kk": {"min_stopword_ratio": 0.01}},
    "domains": {
        "wiki": {"min_lang_score": 0.5, "languages": {"kk": {"min_stopword_ratio": 0.0}}},
        "cc100": {"min_chars": 40, "min_words": 5},
    },
}  # fmt: skip

EN = (
    "The city is located on the bank of the river and it was founded in the twelfth century. "
    "It is known for its old churches, which were built by local masters, and for the market."
)


def frame(rows: list[dict]) -> pl.DataFrame:
    base = {"lang": "en", "lang_score": 1.0, "lang_conf": 1.0, "domain": "web"}
    return pl.DataFrame([{**base, **r} for r in rows])


def test_threshold_resolution_order() -> None:
    assert thresholds(CFG, "web", "en")["min_stopword_ratio"] == 0.02
    assert thresholds(CFG, "web", "kk")["min_stopword_ratio"] == 0.01  # language
    assert thresholds(CFG, "wiki", "kk")["min_stopword_ratio"] == 0.0  # domain x language
    assert thresholds(CFG, "wiki", "en")["min_lang_score"] == 0.5  # domain
    assert thresholds(CFG, "cc100", "kk")["min_chars"] == 40
    assert "languages" not in thresholds(CFG, "wiki", "kk")


def test_signals_basic() -> None:
    s = signals(
        frame([{"text": "a b\na b\nc"}, {"text": "• one\n• two\n• three\nfour"}, {"text": "– Сәлем…"}])
    )
    assert s["n_lines"].to_list() == [3, 4, 1]
    assert s["dup_line_ratio"][0] == pytest.approx(1 / 3)
    assert s["bullet_ratio"][1] == pytest.approx(0.75)
    assert s["bullet_ratio"][2] == 0.0  # dialogue dash is not a bullet; single-line doc
    assert s["ellipsis_ratio"][2] == 0.0  # fewer than MIN_LINES_RATIO lines


def test_dup_ngram_ratio() -> None:
    assert dup_ngram_ratio(["a b c d e"]) == [0.0]
    assert dup_ngram_ratio([" ".join(["x y z w v"] * 4)])[0] == pytest.approx(11 / 16)
    assert dup_ngram_ratio([""]) == [0.0]


def test_rules_first_reason_and_all_fails() -> None:
    df = frame(
        [
            {"text": EN},
            {"text": EN, "lang": "de"},
            {"text": "Too short text."},
            {"text": EN, "lang_score": 0.55},
            {"text": EN, "lang_score": 0.55, "domain": "wiki"},  # wiki allows 0.5
            {"text": "\n".join([EN] * 4)},  # duplicated lines and n-grams
            {"text": "Short sentence of the cc100 corpus, kept as is.", "domain": "cc100"},
        ]
    )
    out = apply_rules(signals(df), CFG)
    assert out["drop_reason"].to_list() == [
        None,
        "lang_other",
        "too_short",
        "lang_low_score",
        None,
        "dup_lines",
        None,
    ]
    row = out.row(5, named=True)
    assert row["fail_dup_lines"] and row["fail_dup_ngrams"]
    assert set(REASONS) == {c.removeprefix("fail_") for c in out.columns if c.startswith("fail_")}


def test_stopwords_by_language() -> None:
    kk = (
        "Алматы облысының аумағында көптеген өзендер ағады, олардың ішінде ең ірісі Іле өзені болып табылады. "
        "Өзен Қытайдан басталып Балқаш көліне құяды, ұзындығы мыңнан астам шақырым."
    )  # 1 function word ("болып") of 24 words
    stub = "Іле Алатауы ұлттық паркі Алматы облысы Талғар Еңбекшіқазақ Қарасай аудандары аумағында орналасқан"
    rows = [{"text": t, "lang": "kk", "domain": d} for t in (kk, stub) for d in ("web", "wiki")]
    out = apply_rules(
        signals(frame(rows)), {**CFG, "default": {**CFG["default"], "min_chars": 50, "min_words": 5}}
    )
    assert out["stopword_ratio"][0] == pytest.approx(1 / 24)
    assert out["drop_reason"].to_list() == [None, None, "few_stopwords", None]  # wiki kk stubs are allowed
