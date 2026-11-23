---
name: trifact-verification
description: Cloud-first evidence verification for TriFact answers in KK/RU/EN — atomic claims and citations, sentence-window NLI entailment, cross-lingual pairs, rule-based number and date checks, MiniCheck-style and LLM-judge alternatives, perturbed RAG training pairs, per-language temperature calibration (ECE, Brier), risk-coverage thresholds, ADR-0008 refusal policy, cloud NLI training and batch scoring. Use for stage 8 work in trifact.verification, train_nli, eval_verification, synthetic NLI filters, or refusal threshold tuning.
---

# TriFact AI — Evidence Verification

## Scope

- Load for VER-0…VER-2, claim splitting, NLI training/scoring, probability calibration, refusal policy tuning, and synthetic QA NLI filtering (`teacher.filters.min_nli_entail`).
- End-to-end RAG metrics → `trifact-evaluation`. RAG prompt/flow → `trifact-rag-pipeline`.
- Cloud CLI execution & persistence → [cloud-compute.md](../trifact-mlops/references/cloud-compute.md).

## Owner Documents

| Topic | Owner |
|---|---|
| Verdict & refusal policy | [ADR-0008](../../../docs/architecture/adr/0008-evidence-verification-and-refusal.md), [01-architecture.md §7.5](../../../docs/architecture/docs/01-architecture.md) |
| Faithfulness audit protocol | [ADR-0022](../../../docs/architecture/adr/0022-evaluation-protocol.md), `params.yaml: verification.faithfulness_audit_claims` |
| Windowed NLI synthetic filter | [ADR-0015](../../../docs/architecture/adr/0015-silver-and-synthetic-data.md) |
| Models, params, XNLI caveats | [models.md §5](../../../docs/research/models.md), [PROGRESS](../../../docs/PROGRESS.md), `params.yaml` (`verification`, `nli`, `promotion`) |

## Cloud Execution

- **Training (`train_nli`, VER-2):** Run on **Colab L4 (bf16)** preferred, or **Kaggle 2× T4 (fp16 AMP, 1 process/GPU)** as secondary option. Pull only `data/task/nli` and `data/eval/nli`. Persist checkpoints and MLflow snapshots to Drive.
- **Batch NLI scoring (`nli_score`):** Run sharded on Colab L4 (`trifact.batch.colab nli_score`) or Kaggle 2× T4 (`trifact.batch.kaggle run nli_score`). Verify quota/budget first and stop session on completion.

## Verification Pipeline

1. **Claim splitting:** Baseline = sentence + citation group. Alternative = LLM atomic decomposition (FActScore) with pronoun decontextualization. Benchmark splitter granularity against human decomposition per language.
2. **Citation check:** Every cited ID must belong to supplied passages; invalid citations are automatically unverified regardless of NLI.
3. **Sentence-window premise (mandatory):** NLI fails on long full passages (Stage 3 measurement). Slice each cited passage into 2–3 sentence windows and take the max entailment score across windows (SummaC-ZS); return the winning window as `quote`.
4. **Multiple citations:** `supported` if any cited window supports the claim (`max`); `contradicted` if strongly contradicted with no supporting citation.
5. **Number & date rule check:** NLI is insensitive to numeric swaps (`1991` vs. `1992`). Cross-check numbers and dates (`trifact.extraction.dates`) between claim and winning window; mismatch revokes `supported`.
6. **Cross-lingual pairs:** Evaluate `kk` claims against `ru`/`en` evidence as a dedicated slice.

## Model Options & VER-2 Data

| Option | Pros | Cons | Role |
|---|---|---|---|
| `mDeBERTa-v3-base-xnli` (VER-0) | Fast, 3-class | No `kk` in train, 512 ctx | Baseline |
| `bge-m3-zeroshot-v2.0` (VER-1) | 8192 ctx, multilingual | Binary, slower | Long-premise challenger |
| Fine-tuned VER-2 (XNLI `ru`/`en` + `kk`-MT + RAG pairs) | Target languages & domain | MT noise, XNLI artifacts | Primary candidate |
| MiniCheck-style claim–doc classifier | Trained on LLM error types | English-trained | Data-perturbation blueprint |
| LLM judge (Qwen3.5) | Flexible, explanatory | High latency, weaker calibration | Offline labeling / edge audits |

- **VER-2 training data:** XNLI `ru`/`en` + filtered `kk` MT (evaluate separately on the hard subset where hypothesis-only baseline fails) + RAG `(window, claim)` pairs with structural perturbations (entity/number/date swaps, negation, role reversal).
- **Honest domain test:** Human-labeled 300–500 claims/language from real system answers. Use LLM-AggreFact only as an external English check (license forbids training).

## Calibration & Thresholds

- Fit **per-language temperature scaling** on held-out dev (`kk` shifts most). Report equal-mass ECE, reliability diagram, and Brier score (never accept ECE without a diagram).
- Pick decision thresholds on the **risk–coverage curve** to satisfy `promotion.verification_contradicted_precision_min` and `promotion.verification_ece_max` while maximizing coverage (revise starter thresholds on Stage 8 human labels).
- Evaluate answer-level policy (`verification.nei_ratio_threshold`) by: unverified claim rate before/after, false-removal rate of true claims, and answers converted to refusal.

## Common Failures

| Symptom | Cause | Fix |
|---|---|---|
| Verbatim-supported claim gets NEI | Full-passage premise | Slice into 2–3 sentence windows, take max |
| `supported` despite wrong year/number | NLI numeric blindness | Rule-based number/date check |
| `kk` claims over-classified as NEI | Missing `kk` train + calibration shift | Fine-tune VER-2 with `kk`, per-language temperature |
| High XNLI score, low answer score | Domain shift + hypothesis artifacts | Add perturbed RAG pairs, test on human answer labels |
| Compound claim half-true | Coarse sentence split | Atomic claim decomposition |

## Done Criteria

- Verifier reduces unsupported claims without significant recall loss (ROADMAP Stage 8).
- `contradicted` precision and ECE pass gate per language on human-labeled system claims.
- Batched NLI latency per answer (`claims × windows`) logged to MLflow `verification`.

## Sources

- Laban et al., *SummaC* (arXiv:2111.09525); Tang et al., *MiniCheck / LLM-AggreFact* (arXiv:2404.10774).
- Zha et al., *AlignScore* (arXiv:2305.16739); Min et al., *FActScore* (arXiv:2305.14251); Choi et al. (arXiv:2102.05169).
- Guo et al. (arXiv:1706.04599); Nixon et al. (arXiv:1904.10685); Geifman & El-Yaniv (arXiv:1705.08500); Poliak et al. (arXiv:1805.01042).
