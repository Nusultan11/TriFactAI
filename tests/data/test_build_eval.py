from __future__ import annotations

import polars as pl

from trifact.data.build_eval import human_done, nli_cluster, query_cluster


def test_query_cluster_joins_a_query_with_its_translations() -> None:
    ids = [
        "miracl/ru/123",
        "miracl/ru/123/mt/kk",
        "miracl/ru/123/mt/en",
        "kazqad/kk/nq/-55",
        "kazqad/en/nq/-55",
        "kazqad/kk-en/nq/-55",
        "kazqad/en-kk/nq/-55",
        "kazqad/kk/kazqad/kzh01#7_1",
        "kazqad/kk/kazqad/kzh01#7_1/mt/ru",
    ]
    got = pl.DataFrame({"q": ids}).select(query_cluster(pl.col("q"))).to_series().to_list()
    assert got == ["miracl/ru/123"] * 3 + ["kazqad/nq/-55"] * 4 + ["kazqad/kk/kazqad/kzh01#7_1"] * 2


def test_human_done_ignores_blank_and_model_reviewers() -> None:
    sheets = pl.DataFrame(
        {
            "item_id": ["a", "b", "c", "d", "e"],
            "verdict": ["ok", "fix", "ok", "", "drop"],
            "reviewer": ["Айгерим", "  ", "Claude (LLM)", "Айгерим", "Qwen judge"],
        }
    )
    assert human_done(sheets)["item_id"].to_list() == ["a"]


def test_nli_cluster_is_the_english_premise() -> None:
    en = pl.DataFrame(
        {
            "pair_id": ["xnli/en/test/0", "xnli/en/test/1", "xnli/en/test/2"],
            "premise": ["P one", "P one", "P two"],
        }
    )
    kk = pl.DataFrame({"pair_id": ["xnli/kk/test/0", "xnli/kk/test/1", "xnli/kk/test/2", "xnli/kk/test/9"]})
    c = nli_cluster(kk, en).to_list()
    assert c[0] == c[1] != c[2]
    assert c[3] is None
