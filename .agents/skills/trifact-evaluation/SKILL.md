---
name: trifact-evaluation
description: Evaluation, statistical significance, and error analysis for TriFact components and end-to-end runs — metric selection and pitfalls (CER, entity F1, nDCG, answer F1, faithfulness, citations, refusal, ECE, latency, cost), frozen-test hygiene and pretraining contamination, paired/cluster bootstrap, multi-seed variance, multiple comparisons, power, LLM-as-a-judge calibration, slicing, root-cause analysis, and promotion gate decisions. Use when comparing models, claiming gains, building benchmarks, writing eval_* or gate code, or explaining low metrics.
---

# TriFact AI — Evaluation & Error Analysis

## Scope

- Load whenever claiming a model improvement, writing `trifact.evaluation.*` or `gate`, building benchmarks, or diagnosing low metrics.
- Cloud CLI execution for heavy eval runs & MLflow persistence → [cloud-compute.md](../trifact-mlops/references/cloud-compute.md).

## Owner Documents

- Suites, significance, gate, ablation → [01-architecture.md §10](../../../docs/architecture/docs/01-architecture.md), [ADR-0007](../../../docs/architecture/adr/0007-leakage-safe-splits.md), [ADR-0009](../../../docs/architecture/adr/0009-model-registry-and-promotion-gate.md), [ADR-0022](../../../docs/architecture/adr/0022-evaluation-protocol.md).
- Parameters → `params.yaml`: `evaluation`, `promotion`. Reference bootstrap code → `trifact.evaluation.eval_docproc` (`bootstrap_ci`, `aggregate`).
- Rule: AI/LLM checks never substitute for mandatory human review (`build_eval`, `PROGRESS.md` blockers).

## Metric Matrix

| Component | Primary Metrics | Pitfalls |
|---|---|---|
| Documents | CER/WER, reading order, heading F1, cell F1 / TEDS | Canonicalize identically; CER misses paragraph swaps |
| NER | `seqeval` strict IOB2 entity P/R/F1 (`lang × type`), unseen-F1 | Default `seqeval` is lenient; always report macro alongside micro |
| Entity Linking | `accuracy@1`, candidate `recall@k`, NIL rate | Candidate recall is the ceiling — never omit |
| Retrieval | `Recall@5/10`, `nDCG@10`, `MRR` (9 directions), candidate `recall@100` | MRR sees only rank-1; closed pools bias novel systems (`judged@10`) |
| Generation | Answer EM/F1 (+ chrF for `kk`), faithfulness, citation P/R, refusal P/R, language | Normalize per language (NFC/case/punct); token F1 penalizes `kk` morphology |
| Verification | Verdict P/R, equal-mass ECE, Brier, risk–coverage | ECE depends on binning — pair with reliability diagram |
| System | Stage p50/p95 latency, RPS, peak RAM/VRAM, GPU-h / USD cost | Measure p95 under target concurrency, not single requests |

Always report per language and direction; macro averages are supplementary, never replacements.

## Frozen-Test Hygiene & Contamination

- Never tune hyperparameters or select checkpoints on frozen test sets (`data/eval/*/MANIFEST.yaml` SHA-256). Use validation only.
- Verify zero train ↔ eval leakage by document, context, and normalized text across translations (ADR-0007).
- **Pretraining contamination check:** Report non-Wikipedia subset and `real_docs` separately; gains restricted to Wikipedia indicate pretraining leakage (ADR-0022).
- Compare systems on identical example IDs, decoding settings, and metric code revision.

## Statistical Significance

An improvement without statistical significance is not an improvement ([AGENTS.md §6](../../../AGENTS.md)). Formulas and sample-size planning → [references/statistics.md](references/statistics.md).

- **Paired bootstrap** on identical resampled indices → 95% CI of $\Delta = M(B) - M(A)$.
- **Cluster bootstrap** whenever examples are dependent (multiple QA per context, translations of one query, sentences in one NER doc).
- **Multi-seed variance:** ≥3 seeds for trained models; sample seed jointly with clusters inside bootstrap.
- **Multiple comparisons:** Apply the pre-registered gate rule or Holm–Bonferroni; publish all CIs, not only winners.

## LLM-as-a-Judge & Human Evaluation

- LLM judges exhibit position, verbosity, and self-preference bias and are weaker on `kk`.
- Mandatory protocol: rubric + reference answer, position-swapped pairwise comparison, `temperature=0`, pinned model revision, and **prior calibration against human labels** ($\kappa$ and class bias per language). Uncalibrated LLM judges are diagnostic only — never gate criteria.
- Human eval: stratified by language/direction, written guidelines with examples, double-annotated subset ($\kappa$ or $\alpha$), adjudicated disagreements, sample size planned for target CI half-width.

## Slicing, Error Analysis & Champion Selection

- **Mandatory slices:** language, direction, source, origin (`human`/`mt`/`synthetic`), length, document kind, question type, answerability, entity frequency. Rank slices by `size × gap`; report worst slice explicitly.
- **Error analysis workflow** (traces → stratified failure sample → open coding → taxonomy with CI → falsifiable root-cause hypothesis → gated intervention → regression test) → [references/error-analysis.md](references/error-analysis.md).
- **Champion selection:** Check component gate (`params.yaml: promotion`). For quality vs. latency/memory trade-offs, pick on the Pareto front within NFRs. Non-significant gain → keep champion and record negative result in `PROGRESS.md`.

## Done Criteria

- Per-language/direction metrics with paired cluster bootstrap CI; test set untouched during selection.
- Per-example outputs/traces and error-analysis taxonomy with next-step hypothesis logged to MLflow.

## Sources

- Koehn (EMNLP 2004); Berg-Kirkpatrick et al. (EMNLP 2012); Dror et al. (ACL 2018); Card et al. (arXiv:2010.06595).
- Bouthillier et al. (arXiv:2103.03098); Zheng et al. (arXiv:2306.05685); Ribeiro et al., *CheckList* (arXiv:2005.04118).
