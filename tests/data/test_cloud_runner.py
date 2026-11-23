from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

RUNNER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "cloud_run_stages_1_3.py"
_SPEC = importlib.util.spec_from_file_location("cloud_run_stages_1_3", RUNNER_PATH)
assert _SPEC and _SPEC.loader
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def test_validate_eval_leakage_rejects_unintended_overlap(tmp_path) -> None:
    p = tmp_path / "eval_leakage.json"
    with pytest.raises(RuntimeError, match="Missing leakage report"):
        runner.validate_eval_leakage(p)

    clean = {
        "sets": {
            "ner:sentence": {"items": 10, "overlap": 0},
            "qa:query": {"items": 10, "overlap": 0},
            "qa:passage": {"items": 10, "overlap": 0},
            "retrieval:query": {"items": 10, "overlap": 0},
            "retrieval:passage": {"items": 10, "overlap": 25},
            "nli:nli": {"items": 10, "overlap": 0},
        }
    }
    p.write_text(json.dumps(clean), encoding="utf-8")
    assert runner.validate_eval_leakage(p)["sets"]["retrieval:passage"]["overlap"] == 25

    clean["sets"]["retrieval:query"]["overlap"] = 1
    p.write_text(json.dumps(clean), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Nonzero unintended evaluation leakage"):
        runner.validate_eval_leakage(p)


def test_validate_nli_scores_and_rss_ceiling(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="Missing required Kaggle NLI QA score parquets"):
        runner.validate_nli_scores_dir(tmp_path / "missing")

    runner.validate_rss_ceiling({"peak_rss_mb": 4096}, "eda@curated", 10.0)
    with pytest.raises(RuntimeError, match="exceeds 10.0 GiB ceiling"):
        runner.validate_rss_ceiling({"peak_rss_mb": 11000}, "eda@curated", 10.0)
