from __future__ import annotations

import json
from pathlib import Path

import pytest

from trifact.core.params import ROOT
from trifact.evaluation.eval_ner import (
    compare_ner_systems,
    evaluate_ner_predictions,
    seqeval_strict_reference,
    span_error_taxonomy,
    strict_entity_metrics,
    unseen_entity_metrics,
    verify_frozen_eval,
)
from trifact.evaluation.gate import (
    evaluate_ner5_optimization_gate,
    evaluate_ner_gate,
    register_and_promote_mlflow,
    rollback_mlflow_alias,
)
from trifact.evaluation.stats import (
    bootstrap_ci,
    cluster_bootstrap_ci,
    cluster_paired_bootstrap,
    equal_mass_ece,
    holm_bonferroni,
    multi_seed_cluster_bootstrap,
    precision_coverage_curve,
    select_annotation_threshold,
)


def test_seqeval_strict_parity_on_reference_cases() -> None:
    y_true = [
        ["B-PER", "I-PER", "O", "B-LOC", "O", "B-ORG", "I-ORG"],
        ["B-PER", "B-PER", "O", "B-LOC", "I-LOC"],
        ["O", "O", "O"],
        ["B-ORG", "I-ORG", "B-LOC"],
    ]
    y_pred = [
        ["B-PER", "I-PER", "O", "B-ORG", "O", "B-ORG", "O"],  # 1 TP, 1 type err, 1 boundary err
        ["B-PER", "I-PER", "O", "B-LOC", "I-LOC"],  # 2 gold PER merged into 1 pred PER; 1 LOC TP
        ["O", "B-PER", "O"],  # 1 spurious
        ["I-ORG", "I-ORG", "B-LOC"],  # illegal I-ORG start (matches seqeval strict IOB2)
    ]
    ours = strict_entity_metrics(y_true, y_pred, labels=("PER", "ORG", "LOC"))["micro"]
    ref = seqeval_strict_reference(y_true, y_pred)
    assert ours["precision"] == pytest.approx(ref["precision"], abs=1e-6)
    assert ours["recall"] == pytest.approx(ref["recall"], abs=1e-6)
    assert ours["f1"] == pytest.approx(ref["f1"], abs=1e-6)


def test_span_error_taxonomy_and_unseen_f1() -> None:
    tokens = [
        ["Астана", "қаласында", "Алматы", "университеті", ".", "Абай"],
        ["John", "works", "at", "Google"],
    ]
    y_true = [
        ["B-LOC", "O", "B-ORG", "I-ORG", "O", "B-PER"],
        ["B-PER", "O", "O", "B-ORG"],
    ]
    y_pred = [
        ["B-ORG", "O", "B-ORG", "O", "B-LOC", "O"],  # metonymy, nested boundary, punct spurious, missed PER
        ["B-PER", "O", "O", "B-ORG"],  # both exact
    ]
    tax = span_error_taxonomy(y_true, y_pred, tokens_list=tokens, langs=["kk", "en"])
    assert tax["counts"]["strict_correct"] == 2
    assert tax["counts"]["type_error"] == 1
    assert tax["counts"]["boundary_error"] == 1
    assert tax["counts"]["spurious"] == 1
    assert tax["counts"]["missed"] == 1
    assert tax["error_categories"]["org_loc_metonymy"] == 1
    assert tax["error_categories"]["nested_or_compound_span"] == 1
    assert tax["error_categories"]["punctuation_only_spurious"] == 1
    assert tax["scores"]["partial_f1"] > tax["scores"]["strict_f1"]

    seen = {"john", "астана"}
    un = unseen_entity_metrics(y_true, y_pred, tokens, seen)
    assert un["seen"]["tp"] == 1  # John
    assert un["unseen"]["tp"] == 1  # Google


def test_cluster_bootstrap_and_multi_seed_stats() -> None:
    lo, hi = bootstrap_ci([0.8, 0.85, 0.9, 0.82, 0.88], n=300, ci=0.95, seed=42)
    assert 0.8 <= lo <= hi <= 0.9

    items_a = [0.0, 0.0, 1.0, 1.0] * 10
    items_b = [1.0, 1.0, 1.0, 1.0] * 10
    clusters = [f"doc_{i // 2}" for i in range(len(items_a))]
    mean_fn = lambda xs: sum(xs) / len(xs)

    single = cluster_bootstrap_ci(items_a, clusters, mean_fn, n=300, seed=42)
    assert single["n_clusters"] == 20
    assert single["point"] == pytest.approx(0.5)

    paired = cluster_paired_bootstrap(items_a, items_b, clusters, mean_fn, n=300, seed=42)
    assert paired["diff"] == pytest.approx(0.5)
    assert paired["significant"] is True
    assert paired["ci_low"] > 0.0

    ms = multi_seed_cluster_bootstrap(
        [items_a, items_a, items_a], [items_b, items_b, items_b], clusters, mean_fn, n=200
    )
    assert ms["significant"] is True
    assert ms["n_seeds_a"] == 3

    hb = holm_bonferroni({"kk": 0.005, "ru": 0.04, "en": 0.25}, alpha=0.05)
    assert hb["kk"]["reject"] is True
    assert hb["en"]["reject"] is False


def test_calibration_ece_and_threshold_selection() -> None:
    probs = [0.95, 0.92, 0.88, 0.82, 0.60, 0.55, 0.30, 0.20]
    labels = [1, 1, 1, 1, 1, 0, 0, 0]
    cal = equal_mass_ece(probs, labels, n_bins=4)
    assert 0.0 <= cal["ece"] <= 1.0
    assert 0.0 <= cal["brier"] <= 1.0
    curve = precision_coverage_curve(probs, labels)
    chosen = select_annotation_threshold(curve, min_precision=0.85, min_coverage=0.5)
    assert chosen["precision"] >= 0.85
    assert chosen["coverage"] >= 0.5


def test_frozen_eval_integrity_check(tmp_path: Path) -> None:
    real_eval = ROOT / "data" / "eval" / "ner"
    if (real_eval / "frozen.json").is_file():
        m = verify_frozen_eval(real_eval)
        assert m["status"] == "frozen"

    # Tampered shard must raise ValueError
    (tmp_path / "kk").mkdir()
    (tmp_path / "kk" / "test-00000.parquet").write_bytes(b"corrupted")
    (tmp_path / "frozen.json").write_text(
        json.dumps({"status": "frozen", "shards": {"kk/test-00000.parquet": "0" * 64}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sha256_mismatch"):
        verify_frozen_eval(tmp_path)


def test_ner_promotion_gate_and_mlflow_rollback(tmp_path: Path) -> None:
    promo = {"ner_kk_f1_min_gain": 0.03, "ner_other_max_drop": 0.01}

    passing = {
        "kk": {"diff": 0.045, "ci_low": 0.025, "ci_high": 0.065},
        "ru": {"diff": 0.002, "ci_low": -0.005, "ci_high": 0.009},
        "en": {"diff": -0.001, "ci_low": -0.008, "ci_high": 0.006},
    }
    v_pass = evaluate_ner_gate(passing, promo, "ner-2", "ner-1")
    assert v_pass["passed"] is True
    assert v_pass["decision"] == "promote"

    failing_kk = {**passing, "kk": {"diff": 0.02, "ci_low": -0.002, "ci_high": 0.04}}
    v_fail = evaluate_ner_gate(failing_kk, promo, "ner-2", "ner-1")
    assert v_fail["passed"] is False
    assert v_fail["decision"] == "reject_keep_champion"

    failing_ru = {**passing, "ru": {"diff": -0.015, "ci_low": -0.022, "ci_high": -0.008}}
    assert evaluate_ner_gate(failing_ru, promo)["passed"] is False

    # NER-5 optimization gate
    assert (
        evaluate_ner5_optimization_gate(
            {"kk": 0.85, "ru": 0.88, "en": 0.90}, {"kk": 0.847, "ru": 0.878, "en": 0.899}
        )["passed"]
        is True
    )
    assert (
        evaluate_ner5_optimization_gate(
            {"kk": 0.85, "ru": 0.88, "en": 0.90}, {"kk": 0.840, "ru": 0.878, "en": 0.899}
        )["passed"]
        is False
    )

    # MLflow registry + rollback test on isolated local SQLite DB
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    r1 = register_and_promote_mlflow(
        "trifact-ner-test",
        "file:///tmp/m1",
        v_fail,
        tracking_uri=tracking_uri,
        bootstrap_champion_if_none=True,
    )
    assert r1["champion"] == "1"
    r2 = register_and_promote_mlflow("trifact-ner-test", "file:///tmp/m2", v_pass, tracking_uri=tracking_uri)
    assert r2["promoted"] is True
    assert r2["champion"] == "2"
    assert r2["previous_champion"] == "1"
    # Rollback to v1
    rb = rollback_mlflow_alias("trifact-ner-test", "1", tracking_uri=tracking_uri)
    assert rb["version"] == "1"


def test_evaluate_and_compare_end_to_end(tmp_path: Path) -> None:
    rows = [
        {
            "sent_id": f"s{i}",
            "doc_key": f"d{i // 2}",
            "source": "kaznerd",
            "lang": "kk",
            "tokens": ["Астана", "қала"],
            "tags": ["B-LOC", "O"],
        }
        for i in range(10)
    ]
    base_preds = [["O", "O"] for _ in rows]
    cand_preds = [["B-LOC", "O"] for _ in rows]
    per_ex = tmp_path / "per_example.jsonl"
    rep = evaluate_ner_predictions(rows, cand_preds, bootstrap_samples=100, per_example_out=per_ex)
    assert rep["by_lang"]["kk"]["micro"]["f1"] == 1.0
    assert per_ex.is_file() and len(per_ex.read_text(encoding="utf-8").splitlines()) == 10

    comp = compare_ner_systems(
        rows,
        [base_preds, base_preds, base_preds],
        [cand_preds, cand_preds, cand_preds],
        bootstrap_samples=100,
    )
    assert comp["paired_diff_by_lang"]["kk"]["significant"] is True
