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
            "retrieval:passage": {"items": 10, "overlap": 0},
            "retrieval:miracl_passage": {"items": 10, "overlap": 0, "allowed_shared_miracl": 5},
            "nli:nli": {"items": 10, "overlap": 0},
        }
    }
    p.write_text(json.dumps(clean), encoding="utf-8")
    assert runner.validate_eval_leakage(p)["sets"]["retrieval:miracl_passage"]["allowed_shared_miracl"] == 5

    clean["sets"]["retrieval:miracl_passage"]["overlap"] = 1
    p.write_text(json.dumps(clean), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Nonzero unintended evaluation leakage"):
        runner.validate_eval_leakage(p)


def test_validate_nli_scores_and_rss_ceiling(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="Missing required Kaggle NLI QA score parquets"):
        runner.validate_nli_scores_dir(tmp_path / "missing")

    runner.validate_rss_ceiling({"peak_rss_mb": 4096}, "eda@curated", 10.0)
    with pytest.raises(RuntimeError, match="exceeds 10.0 GiB ceiling"):
        runner.validate_rss_ceiling({"peak_rss_mb": 11000}, "eda@curated", 10.0)


def test_release_layer_inputs_drops_only_that_layer(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    cache = tmp_path / ".dvc/cache/files/md5"

    def obj(md5: str, data: bytes) -> Path:
        p = cache / md5[:2] / md5[2:]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p

    listing = obj("aa11.dir", json.dumps([{"md5": "bb22", "relpath": "x.parquet"}]).encode())
    member = obj("bb22", b"12345")
    kept = obj("cc33", b"keep")
    (tmp_path / "data/clean/corpus").mkdir(parents=True)
    (tmp_path / "data/clean/corpus/x.parquet").write_bytes(b"12345")
    (tmp_path / "data/curated").mkdir(parents=True)
    lock = {
        "stages": {
            "clean": {"outs": [{"path": "data/clean/corpus", "md5": "aa11.dir"}]},
            "dedup": {"outs": [{"path": "data/curated/corpus", "md5": "cc33"}]},
        }
    }
    (tmp_path / "dvc.lock").write_text(json.dumps(lock), encoding="utf-8")

    size = listing.stat().st_size
    assert runner.release_layer_inputs("clean") == 5 + size
    assert not listing.exists() and not member.exists() and kept.exists()
    assert not (tmp_path / "data/clean/corpus").exists() and (tmp_path / "data/curated").exists()
