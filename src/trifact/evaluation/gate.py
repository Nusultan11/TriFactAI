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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--component", choices=["ner"], default="ner")
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
    comp = rep.get("comparison_vs_baseline", {}).get("paired_diff_by_lang", {})
    verdict = evaluate_ner_gate(comp, params["promotion"], args.candidate, args.baseline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    log.info("Gate decision (%s): %s", args.component, verdict["decision"])
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
