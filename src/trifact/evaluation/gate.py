"""Promotion gate and MLflow Model Registry alias management (ADR-0009, ADR-0022).

Evaluates challenger vs. champion using ``params.yaml: promotion`` thresholds and
paired cluster bootstrap CIs. Promotes only models that satisfy all gate rules and
supports atomic rollback via MLflow alias reassignment.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from trifact.core.params import ROOT, load_params

log = logging.getLogger("trifact.gate")


def evaluate_ner_gate(
    paired_diff_by_lang: dict[str, dict[str, Any]],
    promotion_cfg: dict[str, Any],
    candidate_name: str = "challenger",
    baseline_name: str = "champion",
) -> dict[str, Any]:
    """Check Extraction promotion gate against ``params.yaml: promotion``.

    Rules (architecture §10, ADR-0009, ADR-0022):
    - ``kk`` strict entity F1 gain >= ``ner_kk_f1_min_gain`` AND 95% CI lower bound > 0.
    - ``ru`` and ``en`` strict entity F1 drop <= ``ner_other_max_drop`` (diff >= -max_drop).
    """
    kk_min_gain = float(promotion_cfg["ner_kk_f1_min_gain"])
    other_max_drop = float(promotion_cfg["ner_other_max_drop"])

    checks: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []

    kk = paired_diff_by_lang.get("kk")
    if kk is None:
        checks["kk"] = {"passed": False, "reason": "missing kk comparison"}
        reasons.append("missing kk comparison")
    else:
        diff = float(kk["diff"])
        ci_lo = float(kk["ci_low"])
        ci_hi = float(kk["ci_high"])
        gain_ok = diff >= kk_min_gain
        sig_ok = ci_lo > 0.0
        ok = gain_ok and sig_ok
        checks["kk"] = {
            "passed": ok,
            "diff": round(diff, 6),
            "ci_low": round(ci_lo, 6),
            "ci_high": round(ci_hi, 6),
            "min_gain": kk_min_gain,
            "gain_ok": gain_ok,
            "ci_positive": sig_ok,
        }
        if not ok:
            reasons.append(
                f"kk gain {diff:+.4f} CI [{ci_lo:+.4f}, {ci_hi:+.4f}] failed min_gain={kk_min_gain:.4f} & CI>0"
            )

    for lg in ("ru", "en"):
        entry = paired_diff_by_lang.get(lg)
        if entry is None:
            checks[lg] = {"passed": False, "reason": f"missing {lg} comparison"}
            reasons.append(f"missing {lg} comparison")
            continue
        diff = float(entry["diff"])
        ci_lo = float(entry["ci_low"])
        ci_hi = float(entry["ci_high"])
        ok = diff >= -other_max_drop and ci_lo >= -other_max_drop
        checks[lg] = {
            "passed": ok,
            "diff": round(diff, 6),
            "ci_low": round(ci_lo, 6),
            "ci_high": round(ci_hi, 6),
            "max_drop": other_max_drop,
        }
        if not ok:
            reasons.append(
                f"{lg} diff {diff:+.4f} CI [{ci_lo:+.4f}, {ci_hi:+.4f}] exceeded max_drop={other_max_drop:.4f}"
            )

    passed = all(c["passed"] for c in checks.values())
    return {
        "component": "ner",
        "candidate": candidate_name,
        "baseline": baseline_name,
        "passed": passed,
        "decision": "promote" if passed else "reject_keep_champion",
        "checks": checks,
        "reasons": reasons,
    }


def evaluate_ner5_optimization_gate(
    base_f1_by_lang: dict[str, float],
    opt_f1_by_lang: dict[str, float],
    max_drop_pp: float = 0.5,
) -> dict[str, Any]:
    """Verify NER-5 optimization (vocab trim + ONNX int8) loses at most ``max_drop_pp`` F1 points."""
    max_drop = max_drop_pp / 100.0
    checks: dict[str, dict[str, Any]] = {}
    for lg in ("kk", "ru", "en"):
        if lg not in base_f1_by_lang or lg not in opt_f1_by_lang:
            checks[lg] = {"passed": False, "reason": "missing"}
            continue
        drop = float(base_f1_by_lang[lg]) - float(opt_f1_by_lang[lg])
        checks[lg] = {
            "base_f1": round(float(base_f1_by_lang[lg]), 6),
            "opt_f1": round(float(opt_f1_by_lang[lg]), 6),
            "drop": round(drop, 6),
            "max_drop": max_drop,
            "passed": drop <= max_drop,
        }
    return {
        "component": "ner5_optimization",
        "max_drop_pp": max_drop_pp,
        "passed": all(c["passed"] for c in checks.values()),
        "checks": checks,
    }


def register_and_promote_mlflow(
    model_name: str,
    model_uri: str,
    gate_verdict: dict[str, Any],
    tracking_uri: str | None = None,
    tags: dict[str, str] | None = None,
    bootstrap_champion_if_none: bool = False,
) -> dict[str, Any]:
    """Register model version in MLflow Model Registry as ``challenger``, and move ``champion``
    alias only when ``gate_verdict['passed']`` is True (or bootstrapping the very first baseline).

    Returns registry action record including ``previous_champion`` for atomic rollback.
    """
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    try:
        client.get_registered_model(model_name)
    except MlflowException:
        client.create_registered_model(model_name)

    prev_champion: str | None = None
    try:
        mv_prev = client.get_model_version_by_alias(model_name, "champion")
        prev_champion = str(mv_prev.version)
    except MlflowException:
        prev_champion = None

    mv = client.create_model_version(name=model_name, source=model_uri, tags=tags or {})
    version = str(mv.version)
    client.set_registered_model_alias(model_name, "challenger", version)

    promoted = False
    if gate_verdict.get("passed") or (bootstrap_champion_if_none and prev_champion is None):
        client.set_registered_model_alias(model_name, "champion", version)
        promoted = True

    return {
        "model_name": model_name,
        "version": version,
        "challenger": version,
        "promoted": promoted,
        "champion": version if promoted else prev_champion,
        "previous_champion": prev_champion,
        "gate_passed": bool(gate_verdict.get("passed")),
    }


def rollback_mlflow_alias(
    model_name: str,
    target_version: str,
    alias: str = "champion",
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Atomically reassign ``alias`` (default ``champion``) to ``target_version`` (ADR-0009)."""
    import mlflow
    from mlflow.tracking import MlflowClient

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    client.set_registered_model_alias(model_name, alias, str(target_version))
    mv = client.get_model_version_by_alias(model_name, alias)
    return {"model_name": model_name, "alias": alias, "version": str(mv.version)}


def evaluate_retrieval_gate(
    comparison: dict[str, Any],
    candidate_report: dict[str, Any],
    promotion_cfg: dict[str, Any],
    candidate_name: str = "challenger",
    baseline_name: str = "champion",
    component: str = "retrieval",
) -> dict[str, Any]:
    """Check Retrieval / Reranker promotion gate against ``params.yaml: promotion``.

    Rules (architecture §10, ADR-0009, ADR-0022):
    - Macro 9-direction mean nDCG@10 gain > 0 with 95% cluster paired bootstrap CI lower bound > 0.
    - When ``retrieval_no_direction_worse`` is True: no statistically significant regression on any
      of the 9 directions (no direction with 95% CI upper bound < 0.0).
    - Cross-lingual fairness: ``worst_direction_gap <= worst_direction_max_gap`` (or documented when
      evaluated against cross-lingual pool difficulty).
    """
    no_dir_worse = bool(promotion_cfg.get("retrieval_no_direction_worse", True))
    max_gap = float(promotion_cfg.get("worst_direction_max_gap", 0.10))

    reasons: list[str] = []
    macro = comparison.get("macro") or {}
    macro_diff = float(macro.get("diff", 0.0))
    macro_ci_lo = float(macro.get("ci_low", 0.0))
    macro_ci_hi = float(macro.get("ci_high", 0.0))
    macro_ok = macro_diff > 0.0 and macro_ci_lo > 0.0
    if not macro_ok:
        reasons.append(
            f"macro nDCG@10 diff {macro_diff:+.4f} CI [{macro_ci_lo:+.4f}, {macro_ci_hi:+.4f}] failed diff>0 & CI>0"
        )

    dir_checks: dict[str, dict[str, Any]] = {}
    by_dir = comparison.get("by_direction") or {}
    regressed_dirs: list[str] = []
    for d, entry in sorted(by_dir.items()):
        diff = float(entry["diff"])
        ci_lo = float(entry["ci_low"])
        ci_hi = float(entry["ci_high"])
        # Statistically significant regression: entire 95% CI is strictly below 0
        sig_regression = ci_hi < 0.0
        ok = (not sig_regression) if no_dir_worse else True
        dir_checks[d] = {
            "diff": round(diff, 6),
            "ci_low": round(ci_lo, 6),
            "ci_high": round(ci_hi, 6),
            "significant_regression": sig_regression,
            "passed": ok,
        }
        if not ok:
            regressed_dirs.append(d)
            reasons.append(f"direction {d} regressed: diff {diff:+.4f} CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    dirs_ok = len(regressed_dirs) == 0 and len(by_dir) >= 9
    if len(by_dir) < 9:
        reasons.append(f"expected 9 directions in comparison, got {len(by_dir)}")

    worst_gap = float(candidate_report.get("worst_direction_gap", 1.0))
    gap_ok = worst_gap <= max_gap
    if not gap_ok:
        reasons.append(f"worst_direction_gap {worst_gap:.4f} exceeded worst_direction_max_gap={max_gap:.4f}")

    passed = macro_ok and dirs_ok and gap_ok
    return {
        "component": component,
        "candidate": candidate_name,
        "baseline": baseline_name,
        "passed": passed,
        "decision": "promote" if passed else "reject_keep_champion",
        "checks": {
            "macro_ndcg10": {
                "passed": macro_ok,
                "diff": round(macro_diff, 6),
                "ci_low": round(macro_ci_lo, 6),
                "ci_high": round(macro_ci_hi, 6),
            },
            "no_direction_worse": {
                "passed": dirs_ok,
                "n_directions": len(by_dir),
                "regressed": regressed_dirs,
                "by_direction": dir_checks,
            },
            "worst_direction_gap": {
                "passed": gap_ok,
                "gap": round(worst_gap, 6),
                "max_gap": max_gap,
                "worst_direction": candidate_report.get("worst_direction", ""),
            },
        },
        "reasons": reasons,
    }


def promote_retrieval_with_qdrant(
    model_name: str,
    model_uri: str,
    gate_verdict: dict[str, Any],
    *,
    store: Any | None = None,
    qdrant_alias: str | None = None,
    target_collection: str | None = None,
    tracking_uri: str | None = None,
    tags: dict[str, str] | None = None,
    bootstrap_champion_if_none: bool = False,
) -> dict[str, Any]:
    """Promote retrieval model in MLflow AND switch Qdrant collection alias atomically (ADR-0009, ADR-0013)."""
    res = register_and_promote_mlflow(
        model_name=model_name,
        model_uri=model_uri,
        gate_verdict=gate_verdict,
        tracking_uri=tracking_uri,
        tags=tags,
        bootstrap_champion_if_none=bootstrap_champion_if_none,
    )
    if res["promoted"] and store is not None and qdrant_alias and target_collection:
        res["qdrant_alias"] = store.switch_alias(qdrant_alias, target_collection)
    return res


def rollback_retrieval_with_qdrant(
    model_name: str,
    target_version: str,
    *,
    store: Any | None = None,
    qdrant_alias: str | None = None,
    previous_collection: str | None = None,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Atomically roll back both MLflow ``champion`` alias and Qdrant collection alias."""
    res = rollback_mlflow_alias(model_name, target_version, alias="champion", tracking_uri=tracking_uri)
    if store is not None and qdrant_alias and previous_collection:
        res["qdrant_alias"] = store.switch_alias(qdrant_alias, previous_collection)
    return res


def evaluate_generation_gate(
    comparison: dict[str, Any],
    candidate_report: dict[str, Any],
    promotion_cfg: dict[str, Any],
    *,
    regression_report: dict[str, Any] | None = None,
    candidate_name: str = "challenger",
    baseline_name: str = "champion",
) -> dict[str, Any]:
    """Check Generation promotion gate against ``params.yaml: promotion`` (ADR-0009, ADR-0022).

    Rules (architecture §10, ROADMAP 7):
    - Per-language ``faithfulness >= promotion.faithfulness_min`` (0.85).
    - Per-language ``answer_f1`` AND ``faithfulness`` beat best baseline with 95% cluster paired
      bootstrap CI lower bound > 0.
    - Adapter regression suite (ROADMAP 7.3) must pass when provided.
    """
    faith_min = float(promotion_cfg.get("faithfulness_min", 0.85))
    by_lang_cmp = comparison.get("by_lang") or {}
    by_lang_cand = candidate_report.get("by_lang") or {}

    checks: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []

    for lg in ("kk", "ru", "en"):
        cmp_lg = by_lang_cmp.get(lg)
        cand_lg = by_lang_cand.get(lg)
        if cmp_lg is None or cand_lg is None:
            checks[lg] = {"passed": False, "reason": f"missing {lg} metrics or comparison"}
            reasons.append(f"missing {lg} metrics or comparison")
            continue

        faith_val = cand_lg.get("faithfulness")
        faith_floor_ok = faith_val is not None and float(faith_val) >= faith_min

        f1_c = cmp_lg.get("answer_f1") or {}
        f1_diff = float(f1_c.get("diff", 0.0))
        f1_lo = float(f1_c.get("ci_low", 0.0))
        f1_hi = float(f1_c.get("ci_high", 0.0))
        f1_beats = f1_diff > 0.0 and f1_lo > 0.0

        fa_c = cmp_lg.get("faithfulness") or {}
        fa_diff = float(fa_c.get("diff", 0.0))
        fa_lo = float(fa_c.get("ci_low", 0.0))
        fa_hi = float(fa_c.get("ci_high", 0.0))
        fa_beats = fa_diff > 0.0 and fa_lo > 0.0

        ok = faith_floor_ok and f1_beats and fa_beats
        checks[lg] = {
            "passed": ok,
            "faithfulness": faith_val,
            "faithfulness_min": faith_min,
            "faithfulness_floor_ok": faith_floor_ok,
            "answer_f1_diff": round(f1_diff, 6),
            "answer_f1_ci": [round(f1_lo, 6), round(f1_hi, 6)],
            "answer_f1_beats_baseline": f1_beats,
            "faithfulness_diff": round(fa_diff, 6),
            "faithfulness_ci": [round(fa_lo, 6), round(fa_hi, 6)],
            "faithfulness_beats_baseline": fa_beats,
        }
        if not faith_floor_ok:
            reasons.append(f"{lg} faithfulness {faith_val} < faithfulness_min={faith_min:.2f}")
        if not f1_beats:
            reasons.append(
                f"{lg} answer_f1 diff {f1_diff:+.4f} CI [{f1_lo:+.4f}, {f1_hi:+.4f}] did not beat baseline"
            )
        if not fa_beats:
            reasons.append(
                f"{lg} faithfulness diff {fa_diff:+.4f} CI [{fa_lo:+.4f}, {fa_hi:+.4f}] did not beat baseline"
            )

    reg_ok = True
    if regression_report is not None:
        reg_ok = bool(regression_report.get("passed", False))
        if not reg_ok:
            reasons.extend(regression_report.get("reasons") or ["adapter regression suite failed"])

    passed = all(c["passed"] for c in checks.values()) and reg_ok
    return {
        "component": "generation",
        "candidate": candidate_name,
        "baseline": baseline_name,
        "passed": passed,
        "decision": "promote" if passed else "reject_keep_champion",
        "checks": {
            "by_lang": checks,
            "adapter_regressions": {"passed": reg_ok, "detail": regression_report},
        },
        "reasons": reasons,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--component", choices=["ner", "retrieval", "reranker", "generation"], default="ner")
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "ner.json")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "gate.json")
    ap.add_argument("--candidate", default="challenger")
    ap.add_argument("--baseline", default="champion")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    params = load_params()
    if not args.report.is_file():
        raise FileNotFoundError(f"Report not found: {args.report}")
    rep = json.loads(args.report.read_text(encoding="utf-8"))
    if args.component == "ner":
        comp = rep.get("comparison_vs_baseline", {}).get("paired_diff_by_lang", {})
        verdict = evaluate_ner_gate(comp, params["promotion"], args.candidate, args.baseline)
    elif args.component == "generation":
        comp = rep.get("comparison_vs_baseline", {})
        reg = rep.get("adapter_regressions")
        verdict = evaluate_generation_gate(
            comp,
            rep,
            params["promotion"],
            regression_report=reg,
            candidate_name=args.candidate,
            baseline_name=args.baseline,
        )
    else:
        comp = rep.get("comparison_vs_baseline", {})
        verdict = evaluate_retrieval_gate(
            comp, rep, params["promotion"], args.candidate, args.baseline, component=args.component
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    log.info("Gate decision (%s): %s", args.component, verdict["decision"])
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
