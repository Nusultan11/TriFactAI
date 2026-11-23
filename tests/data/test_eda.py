from __future__ import annotations

import numpy as np
import polars as pl

from trifact.data.eda import (
    accumulate,
    combine,
    hist,
    iin_valid,
    luhn,
    merge,
    quantiles_from_hist,
    text_profile,
)
from trifact.data.task_eda import ner, retrieval


def test_merge_and_quantiles() -> None:
    a = {"n": 1, "h": [1, 2], "d": {"x": 1}, "parts": [["a", 1]]}
    b = {"n": 2, "h": [3, 4], "d": {"x": 2, "y": 1}, "parts": [["b", 2]]}
    assert merge(a, b) == {"n": 3, "h": [4, 6], "d": {"x": 3, "y": 1}, "parts": [["a", 1], ["b", 2]]}
    q = quantiles_from_hist(hist(np.arange(1, 1001)))
    assert 400 <= q["p50"] <= 520 and q["p99"] >= 850  # log bins: ~6% resolution


def test_text_profile_scripts_encoding_pii() -> None:
    s = pl.Series(
        ["Қазақ тілі әдемі", "Plain english text", "bad � char", "mail me a@b.kz +7 701 123 45 67", ""]
    )
    p = text_profile(s)
    assert p["docs"] == 5 and p["empty"] == 1
    assert p["script"]["docs_with_kk_letters"] == 1 and p["script"]["docs_mostly_latin"] >= 2
    assert p["encoding"]["docs_replacement_char"] == 1
    assert p["pii"]["docs_email"] == 1 and p["pii"]["docs_phone"] == 1


def test_iin_and_luhn() -> None:
    assert luhn("4111 1111 1111 1111") and not luhn("4111 1111 1111 1112")
    assert not iin_valid("000000000000")


def test_incremental_accumulate_matches_combine() -> None:
    import copy

    parts = [
        {
            "key": ("src", "kk", "part"),
            "res": {"rows": 2, "text": {"text": {"docs": 2, "extremes": {"a": []}}}},
        },
        {
            "key": ("src", "kk", "part"),
            "res": {"rows": 3, "text": {"text": {"docs": 3, "extremes": {"a": []}}}},
        },
    ]
    acc: dict[tuple[str, str, str], dict] = {}
    for p in copy.deepcopy(parts):
        accumulate(acc, p)
    assert acc == combine(copy.deepcopy(parts))


def test_task_ner_profiles_all_shards_and_matches_test_surfaces(tmp_path) -> None:
    base = tmp_path / "task/ner/en"
    base.mkdir(parents=True)
    for table, index, tokens, tags in (
        ("gold_test", 0, ["Paris"], ["B-LOC"]),
        ("silver_train", 0, ["Paris", "today"], ["B-LOC", "O"]),
        ("silver_train", 1, ["No", "match", "today"], ["O", "O", "O"]),
    ):
        pl.DataFrame({"source": ["test"], "lang": ["en"], "tokens": [tokens], "tags": [tags]}).write_parquet(
            base / f"{table}-{index:05d}.parquet"
        )
    report = ner(tmp_path)
    assert report["by_table_source_lang"][1]["sentences"] == 2
    assert report["by_table_source_lang"][1]["p95_len"] == 3
    assert report["entity_overlap_train_test"]["silver_train/en"] == 1


def test_task_retrieval_aggregates_full_set_and_samples_bm25(tmp_path) -> None:
    base = tmp_path / "task/retrieval"
    (base / "en").mkdir(parents=True)
    (base / "multi").mkdir()
    pl.DataFrame(
        {
            "source": ["test"],
            "origin": ["gold"],
            "direction": ["en-en"],
            "split": ["train"],
            "query_id": ["q1"],
            "pair_id": ["p1"],
            "query": ["Paris article"],
            "doc_id": ["d1"],
            "neg_ids": [["d2"]],
        }
    ).write_parquet(base / "en/pairs-00000.parquet")
    pl.DataFrame({"doc_id": ["d1", "d2"], "text": ["Paris article", "Unrelated evidence"]}).write_parquet(
        base / "multi/passages-00000.parquet"
    )
    report = retrieval(tmp_path, sample=1, seed=42)
    assert report["train"][0]["n"] == 1
    assert report["bm25_ease"]["test"]["positive_rank1_share"] == 1
