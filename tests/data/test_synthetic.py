import json
from types import SimpleNamespace

import polars as pl

from trifact.data import synthetic


def test_query_length_bounds_follow_the_variant(tmp_path, monkeypatch) -> None:
    (tmp_path / "out").mkdir()
    pl.DataFrame(
        {
            "id": ["query/example"],
            "chunk_id": ["example"],
            "lang": ["en"],
            "source": ["test"],
            "text": ["The past connects several topics of science."],
        }
    ).write_parquet(tmp_path / "meta.parquet")
    answer = {"question": {"en": "past history"}, "keywords": {"en": "past research"}}
    pl.DataFrame({"id": ["query/example"], "text": [json.dumps(answer)]}).write_parquet(
        tmp_path / "out" / "gen-00000.parquet"
    )
    monkeypatch.setattr(
        synthetic, "lid", lambda: SimpleNamespace(predict=lambda texts: (["en"] * len(texts), None))
    )

    kept, dropped = synthetic.import_queries([tmp_path], top_k=1)

    assert kept["type"].to_list() == ["keywords"]
    assert dropped.filter(pl.col("q_lang") == "en").select("type", "reason").rows() == [
        ("question", "length")
    ]
