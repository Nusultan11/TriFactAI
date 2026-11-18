from __future__ import annotations

import polars as pl

from trifact.data.bm25 import BM25
from trifact.data.build_sft import mix
from trifact.data.synthetic import copied, lang_key, parse_json
from trifact.data.task_common import assign_split, norm_key, stable_hash
from trifact.data.translations import chrf


def test_bm25_ranks_relevant_first() -> None:
    docs = ["Astana is the capital of Kazakhstan", "Moscow is a large city", "The cat sat on the mat"]
    idx, sc = BM25(docs).search(["capital of Kazakhstan", "cat mat", "zzz"], 2)
    assert idx[0][0] == 0 and idx[1][0] == 2 and idx[2][0] == -1 and sc[0][0] > sc[0][1]


def test_chrf_bounds() -> None:
    assert chrf("the same text", "the same text") == 100.0
    assert chrf("", "x") == 0.0
    assert 0 < chrf("the same text", "the sane test") < 100


def test_split_by_document_is_stable_and_grouped() -> None:
    df = pl.DataFrame({"doc": [f"d{i // 3}" for i in range(3000)]})
    s1, s2 = assign_split(df, "doc", 0.1, 0.1, 42), assign_split(df, "doc", 0.1, 0.1, 42)
    assert s1.to_list() == s2.to_list()
    assert df.with_columns(s=s1).group_by("doc").agg(pl.col("s").n_unique())["s"].max() == 1
    share = (s1 == "test").mean()
    assert 0.05 < share < 0.15
    assert stable_hash("a") == stable_hash("a") != stable_hash("b")
    assert norm_key("Hello,  World!") == "hello world"


def test_teacher_json_parsing_and_copy() -> None:
    d = parse_json('Sure: {"k": "Алматы қайда?", "ru": "Где Алматы?", "en": "Where is Almaty?"} done')
    assert d is not None and lang_key(d, "kk") == "Алматы қайда?" and lang_key(d, "en") == "Where is Almaty?"
    assert parse_json("no json") is None
    assert copied(
        "Almaty is the largest city in Kazakhstan", "Almaty is the largest city in Kazakhstan. It is old."
    )
    assert not copied("Which city is the largest?", "Almaty is the largest city in Kazakhstan.")


def test_sft_mix_shares_and_kk_floor() -> None:
    rows = []
    for t, n in (("answerable", 600), ("cross_lingual", 400), ("unanswerable", 300)):
        for i in range(n):
            rows.append({"example_id": f"{t}{i}", "type": t, "lang": "kk" if i % 2 else "en"})
    out, st = mix(
        pl.DataFrame(rows), {"answerable": 0.6, "cross_lingual": 0.25, "unanswerable": 0.15}, 0.3, None, 1
    )
    shares = out.group_by("type").len().with_columns(s=pl.col("len") / out.height)
    got = dict(shares.select("type", "s").iter_rows())
    assert abs(got["answerable"] - 0.6) < 0.01 and abs(got["unanswerable"] - 0.15) < 0.01
    assert (out["lang"] == "kk").mean() >= 0.3
    assert st["total_feasible"] == 1000
