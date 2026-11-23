# 15. Silver and Synthetic Data: Generation, Filters, and License Lineage

Date: 2026-11-18

## Status

Accepted

Amended by [20. Kaggle Batch GPU, NLLB MT, and llama.cpp Teacher](0020-kaggle-batch-gpu-mt-and-teacher.md)

## Context

Gold training data is scarce for Kazakh (KazNERD ≈ 113k sentences, WikiANN `kk` 3k, zero `kk` NLI, almost no cross-lingual `kk ↔ ru/en` retrieval pairs). Million-scale training requires silver and synthetic data, which introduce noise and inherit upstream license restrictions.

## Decision

- **Silver NER:** Wikipedia internal links → Wikidata QID → `P31`/`P279` class (`PER`/`ORG`/`LOC`), targeting 5–10M sentences across KK/RU/EN. Estimate noise against gold and human review; train via silver → gold curriculum.
- **Synthetic teacher (`Qwen3.5-35B-A3B`):** generate chunk queries (3–5M), grounded QA with `[chunk_id]` incl. cross-lingual & unanswerable (1–2M), and `kk` XNLI translation (400k; MT execution amended by ADR-0020).
- **Mandatory filters:** round-trip retrieval (generated query retrieves source chunk in top-k), NLI consistency against cited context, MinHash dedup, length bounds, and answer language verification.
- **Train-only rule:** Never place unreviewed synthetic or silver rows in eval; decontaminate train against eval (ADR-0007).
- **License lineage:** List all input licenses in `dataset.yaml`. Any dataset using NC or `research_only` inputs propagates `release: false` and is excluded from released weights.

## Consequences

Training volume scales by orders of magnitude, bounded by filter quality and compute budget. Calibrate filter thresholds on human review, log per-filter drop rates in `card.md`, and approve cloud spend prior to generation.

## Implementation Notes (2026-11-21)

Stage 3 execution details (measurements in `docs/PROGRESS.md`):
- **Silver entity typing:** assign closest `P279` closure class among `P31` classes; `PER > ORG > LOC` resolves ties only (pure priority misclassified countries/cities as ORG via "sovereign state"; fix lifted ORG agreement with gold from `0.32–0.60` to `0.79–0.93`).
- **QA NLI filter:** score each claim (text before each citation group) against 3-sentence windows of cited passages; pass if the weakest claim's best window has `entailment >= teacher.filters.min_nli_entail` (full-passage NLI rejected verbatim-supported claims).
- **Prompt citations:** use numeric passage IDs `[1]…[k]` mapped back to `chunk_id` on import (long chunk IDs truncated 16.5% of responses at `max_tokens`).
- Teacher runs on `llama.cpp` (ADR-0020), not `vLLM`.
