import json
from types import SimpleNamespace

import polars as pl
import pytest

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


def test_finalize_qa_scores_fails_on_missing_or_incomplete_scores(tmp_path) -> None:
    kept = pl.DataFrame(
        {
            "id": ["qa/1", "qa/2", "qa/u"],
            "lang": ["en", "en", "en"],
            "mode": ["answerable", "answerable", "unanswerable"],
            "question": ["Q1?", "Q2?", "QU?"],
            "answer": ["Fact one [c1].", "Fact two [c2].", synthetic.REFUSAL["en"]],
            "evidence_ids": [["c1"], ["c2"], ["c3"]],
            "evidence_texts": [["Fact one is true."], ["Fact two is true."], ["Other text."]],
        }
    )
    scores_dir = tmp_path / "scores"
    with pytest.raises(FileNotFoundError):
        synthetic.finalize_qa_scores(kept, scores_dir, thr=0.7)

    (scores_dir / "out").mkdir(parents=True)
    pl.DataFrame({"id": ["qa/1@c0@w0"], "p_entailment": [0.95]}).write_parquet(
        scores_dir / "out" / "part-0.parquet"
    )
    with pytest.raises(ValueError, match="Incomplete NLI scores"):
        synthetic.finalize_qa_scores(kept, scores_dir, thr=0.7)

    pl.DataFrame({"id": ["qa/1@c0@w0", "qa/2@c0@w0"], "p_entailment": [0.95, 0.35]}).write_parquet(
        scores_dir / "out" / "part-0.parquet"
    )
    final_kept, dropped = synthetic.finalize_qa_scores(kept, scores_dir, thr=0.7)
    assert final_kept["id"].to_list() == ["qa/1", "qa/u"]
    assert dropped.select("id", "reason").rows() == [("qa/2", "nli")]

    expected = tmp_path / "pairs.parquet"
    pl.DataFrame({"id": ["qa/1@c0@w0", "qa/1@c1@w0", "qa/2@c0@w0"]}).write_parquet(expected)
    with pytest.raises(ValueError, match="1 missing, 0 unexpected"):
        synthetic.finalize_qa_scores(kept, scores_dir, thr=0.7, expected_pairs=expected)
