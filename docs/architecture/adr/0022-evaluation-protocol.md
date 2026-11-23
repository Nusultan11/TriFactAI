# 22. Evaluation Protocol: Cluster Paired Bootstrap, Claim Faithfulness, LLM Judges, and Contamination

Date: 2026-11-21

## Status

Accepted

Amends [9. Model Registry and Metric-Driven Promotion Gate](0009-model-registry-and-promotion-gate.md)

## Context

ADR-0009 mandated bootstrap CI or ≥3 seeds without specifying statistical methodology. Reviewing `trifact-evaluation` surfaced five gaps:
- **Dependent examples:** QA questions share contexts, retrieval queries share translations across directions, and NER sentences share documents; item-level bootstrap yields artificially narrow CIs.
- **Undefined `faithfulness`:** Required by the LLM gate (`01-architecture.md §10`) without a reproducible definition.
- **Uncalibrated LLM judges:** Exhibit position/length/self-preference bias and weaker `kk` accuracy.
- **Pretraining contamination:** Eval suites derived from Wikipedia/MIRACL/SQuAD may appear in LLM/embedder pretraining (ADR-0019 protects only TriFact train sets).
- **Multiple comparisons:** 9 directions × multiple metrics × 3 languages inflate false positives without a pre-registered decision rule.

## Decision

- **Cluster paired bootstrap:** Evaluate both systems on identical resampled clusters: source document/context (QA, NLI premise), source query with all translations (retrieval), document (NER, `docproc`). Recompute nonlinear metrics (entity F1, nDCG) from scratch on each resample (`params.yaml: evaluation`).
- **Multi-seed variance:** Run ≥3 seeds for trained models; draw a random seed per system inside each bootstrap resample so CIs reflect both data and training variance.
- **Pre-registered test family:** Use the component's `params.yaml: promotion` gate rule; apply Holm–Bonferroni or publish all CIs for exploratory multi-slice claims.
- **Faithfulness:** Fraction of answer claims (`trifact.verification.claims`) marked `supported` by the calibrated verifier (Stage 8) against cited passages. Prior to Stage 8, use `VER-0` with identical 3-sentence windows and manually audit `verification.faithfulness_audit_claims` claims/language, reporting human agreement alongside faithfulness. Track refusals separately via refusal P/R.
- **LLM-as-a-judge:** Permitted only after per-language human calibration ($\kappa$ and class bias recorded) using rubric + reference, position-swapped pairwise prompts, `temperature=0`, and pinned model revision. Never replace measurable gate metrics with an LLM judge.
- **Contamination check:** Report non-Wikipedia and `real_docs` slices separately for LLM and retrieval; gains confined to Wikipedia are flagged as likely pretraining contamination.
- **Per-example traces & per-language reporting:** Log per-example outputs and e2e stage traces to MLflow; always report per language/direction (averages are supplementary).

## Consequences

CIs widen and spurious wins disappear. Eval suites require cluster keys (add source-query link to retrieval `queries` in Stage 6 with backup). Extract `bootstrap_ci` from `eval_docproc` into a shared evaluation module in Stage 5. Human verifier audits and judge calibration are scheduled in Stages 7–9.
