"""Unit tests for machine pre-labeling recovery/sanitization and review HTML generation."""

from __future__ import annotations

import json
from pathlib import Path

from trifact.evaluation.prelabel import _sanitize_ner_judge, parse_json
from trifact.evaluation.review_app import build_hub


def test_parse_json_prefers_last_top_level_and_recovers_truncated() -> None:
    # Revised JSON after initial draft
    revised = (
        '{"wrong_type": [], "wrong_span": [], "not_entity": [], "missed": [], "reason": "initial"}\n'
        "Revised JSON:\n"
        '{"wrong_type": [], "wrong_span": [], "not_entity": [], "missed": [{"text": "Cyrus", "type": "PER"}], "reason": "Cyrus'
    )
    d = parse_json(revised)
    assert d is not None
    assert d["missed"] == [{"text": "Cyrus", "type": "PER"}]

    # Truncated eval verdict inside reason
    trunc_eval = (
        '{"verdict": "fix", "fixed_text": "Жаңа сұрақ", "reason": "The original translation was cut off'
    )
    e = parse_json(trunc_eval)
    assert e is not None
    assert e["verdict"] == "fix"
    assert e["fixed_text"] == "Жаңа сұрақ"
    assert e.get("_truncated") is True

    # Truncated NER inside missed array
    trunc_ner = (
        '{"wrong_type": [], "wrong_span": [], "not_entity": ["[.|LOC]"], '
        '"missed": [{"text": "Астана", "type": "LOC"}, {"text": "truncated'
    )
    n = parse_json(trunc_ner)
    assert n is not None
    assert n["not_entity"] == ["[.|LOC]"]
    assert n["missed"] == [{"text": "Астана", "type": "LOC"}]
    assert n.get("_truncated") is True


def test_sanitize_ner_judge_enforces_core_schema_and_no_double_counting() -> None:
    sent = "Cyrus visited [Babylonia|LOC] and [.|LOC] in 1959."
    raw = {
        "wrong_type": [
            {"text": "Babylonia", "type": "LOC"},  # same type -> drop
            {"text": "Babylonia", "type": "EVENT"},  # non-core -> reclassify to not_entity
        ],
        "wrong_span": [
            {"text": "[.|LOC]", "correct": "."},  # already consumed by not_entity below -> drop
        ],
        "not_entity": ["[.|LOC]"],
        "missed": [
            {"text": "Cyrus", "type": "PER"},  # valid
            {"text": "1959", "type": "DATE"},  # non-core -> drop
            {"text": "Babylonia", "type": "LOC"},  # already marked -> drop
            {"text": "NonExistent", "type": "PER"},  # not in sentence -> drop
        ],
    }
    wt, ws, ne, mi, san = _sanitize_ner_judge(sent, raw)
    assert wt == []
    assert ws == []
    assert ne == [".", "Babylonia"]
    assert mi == [{"text": "Cyrus", "type": "PER"}]
    assert san >= 5
    # Never exceeds marked entities (2)
    assert len(wt) + len(ws) + len(ne) <= 2


def test_build_hub_generates_consolidated_index(tmp_path: Path) -> None:
    hub = build_hub(
        tmp_path,
        {
            "quality": "quality_human/review.html",
            "ner": "ner_silver/review.html",
            "eval": "eval_mt/review.html",
        },
        {
            "quality": {
                "items": 1500,
                "judged": 1500,
                "uncertainty": {"low": 1000, "medium": 400, "high": 100},
            },
            "ner": {"items": 600, "judged": 600, "uncertainty": {"low": 50, "medium": 270, "high": 280}},
            "eval": {
                "items": 3500,
                "judged": 3500,
                "uncertainty": {"low": 1600, "medium": 600, "high": 1300},
            },
        },
    )
    assert hub.is_file()
    html = hub.read_text(encoding="utf-8")
    assert "quality_human/review.html" in html
    assert "ner_silver/review.html" in html
    assert "eval_mt/review.html" in html
    assert "не являются человеческой проверкой" in html
    assert json.dumps("ok")  # sanity
