---
name: trifact-llm-finetuning
description: Cloud-first LLM engineering for grounded KK/RU/EN QA with Qwen3.5 in TriFact — compatibility checks (transformers, PEFT, llama.cpp, vLLM), zero-shot baselines, chat template and non-thinking mode, SFT data selection, cloud Ampere+ bf16 LoRA per ADR-0012, adapter regression evaluation, GGUF quantization with multilingual imatrix, AWQ, serving speed (TTFT, tokens/s). Use for stage 7 work in train_llm, eval_generation, generation.llm_client, or debugging ungrounded answers, bad citations, or over-refusal.
---

# TriFact AI — LLM Fine-Tuning & Quantization

## Scope

- Load for GEN-0…GEN-3, GEN-Q, `train_llm`, `eval_generation`, `generation.llm_client`, SFT curation, quantization, and serving benchmarks.
- Context & prompt design → `trifact-rag-pipeline`. General training/HPO → `trifact-training-hpo`. Synthetic teacher generation → `trifact-data-pipeline`.
- Cloud CLI execution & Drive persistence → [cloud-compute.md](../trifact-mlops/references/cloud-compute.md).

## Owner Documents

| Topic | Owner |
|---|---|
| Model family, bf16 LoRA, serving | [ADR-0012](../../../docs/architecture/adr/0012-qwen35-lora-generation.md) (supersedes [ADR-0005](../../../docs/architecture/adr/0005-qwen3-qlora-local-gguf-serving.md)) |
| Experiments, gate, model facts | [01-architecture.md §7.4, §10](../../../docs/architecture/docs/01-architecture.md), [models.md §4, §6.2](../../../docs/research/models.md) |
| Synthetic teacher & cloud policy | [ADR-0015](../../../docs/architecture/adr/0015-silver-and-synthetic-data.md), [ADR-0020](../../../docs/architecture/adr/0020-kaggle-batch-gpu-mt-and-teacher.md), [ADR-0025](../../../docs/architecture/adr/0025-cloud-only-computing.md) |
| Parameters | `params.yaml`: `llm`, `sft`, `teacher`, `cloud.budget_usd` |

## Cloud Compute Constraints (Mandatory)

- **bf16 LoRA only (ADR-0012):** Never use QLoRA for Qwen3.5. Requires **Ampere+ GPU** (Colab L4 24 GB preferred when available/permitted, or approved cloud A100/H100).
- **Kaggle T4 incompatibility:** T4 (sm75) lacks native `bf16` — never run Qwen3.5 bf16 LoRA on Kaggle T4.
- **Budget guardrail:** Verify `params.yaml: cloud.budget_usd` before launching GEN-1…GEN-3. Without approved paid budget, execute Step 0 compatibility, GEN-0 zero-shot baselines, GEN-Q quantization, and serving speed benchmarks only (ROADMAP 7.0).
- **Persistence:** Pull only `data/task/sft` and `data/eval/qa`; save LoRA checkpoints (adapter + optimizer + step) and MLflow SQLite snapshots to Drive; stop the cloud runtime immediately after verification.

## Step 0 — Compatibility Check (ROADMAP 7.1)

Run on cloud GPU before any training:
1. Verify `transformers` loads `qwen3_5`, `apply_chat_template(..., enable_thinking=False)` matches serving format, and tokenization round-trips cleanly.
2. Inspect actual linear layer names via `model.named_modules()` (hybrid Gated DeltaNet layers differ from standard attention `q_proj`/`k_proj`). Reconcile `llm.lora.target_modules` and log trainable parameter count.
3. Merge test adapter → GGUF with current `llama.cpp`; verify greedy generation matches `transformers` on probe prompts.
4. Verify `vLLM >= 0.17` loads base + LoRA adapter.
5. Record any version/module discrepancy in `models.md` with date and source.

## Baselines (GEN-0) & SFT Curation

- Evaluate all candidates (`Qwen3.5-0.8B/2B/4B/9B`, control `Qwen3-1.7B`, eval-only NC `KazLLM-8B` / `Sherkala-8B`) with identical prompt, context, and decoding on frozen `data/eval/qa`.
- Track per language & direction: answer F1/EM (+ chrF for `kk`), faithfulness ([ADR-0022](../../../docs/architecture/adr/0022-evaluation-protocol.md), with human audit of `verification.faithfulness_audit_claims`), citation P/R, refusal P/R, answer language accuracy. Best GEN-0 per language is the bar LoRA must beat.
- **SFT data:** Exact serving prompt + assistant response; compute loss **only on assistant tokens**. Enforce `sft.mix` and `scale.min_kk_share`. Use topically close distractors for unanswerable examples (prevents topic-mismatch shortcuts). Filter by NLI support, valid citations, question MinHash dedup, embedding-cluster quotas for diversity, and `llm.max_seq_len`. Exclude frozen eval questions (`build_sft`).

## LoRA Configuration

| Choice | Rule |
|---|---|
| Target modules | All linear layers (attention, Gated DeltaNet projections, MLP) — attention-only underperforms |
| Learning rate | ~10× full fine-tune LR (`llm.lr` start, refine with short dev sweep); largely rank-invariant under $\alpha/r$ scaling |
| Rank | Sweep rank for 0.3–2M SFT scale (small ranks saturate); `rsLoRA` ($\alpha/\sqrt{r}$) optional |
| Epochs & early stop | 1–3 epochs; stop by **generative dev metric**, never validation loss |
| Packing | Use padding-free / explicit `position_ids` masks so examples never cross-attend |
| Memory | `bf16` + gradient checkpointing; choose PEFT+TRL vs. Unsloth by Step 0 compatibility and speed |

## Adapter Evaluation & Regressions

- Compare generatively against GEN-0 on identical test and decoding.
- Check regressions explicitly: over-refusal on answerable queries, wrong answer language, malformed/hallucinated citation IDs, repetition loops (especially `kk`), cross-lingual degradation, and general multilingual forgetting (small context-free probe set).
- **Gate (ADR-0009):** `faithfulness >= promotion.faithfulness_min`, and both `answer F1` and `faithfulness` beat best baseline per language with CI. If not, keep baseline for serving and log negative result in `PROGRESS.md`.

## Quantization (GEN-Q) & Serving

1. Merge adapter in `bf16` → convert to GGUF.
2. Build `llama-imatrix` on in-domain **multilingual (`kk` + `ru` + `en`)** SFT dev prompts/answers (omitting `kk` degrades `kk` disproportionately at low bit widths).
3. Benchmark `Q4_K_M`, `Q5_K_M`, `Q8_0` (llama.cpp) and `AWQ` (vLLM): measure KL divergence to `bf16` (`llama-perplexity --kl-divergence`) and per-language task metrics.
4. Serving checks: match training chat template; pass `-kvu` (unified KV cache) for parallel slots on hybrid Qwen3.5 in `llama.cpp`; log TTFT, prefill vs. decode `tokens/s` (1 slot and N slots), and peak VRAM/RAM. Test speculative decoding (`0.8B` draft) on `kk` separately.

## Done Criteria

- Step 0 verified and logged; GEN-0 candidates logged in MLflow `generation`.
- Adapter passes promotion gate or negative result recorded in `PROGRESS.md`.
- Selected quant has KL vs. `bf16`, per-language metrics, TTFT, and `tokens/s` logged; cloud runtime stopped.

## Sources

- Hu et al., *LoRA* (arXiv:2106.09685); Kalajdzievski, *rsLoRA* (arXiv:2312.03732); Biderman et al. (arXiv:2405.09673).
- Thinking Machines Lab, *LoRA Without Regret* (2025); Zhou et al., *LIMA* (arXiv:2305.11206); Lin et al., *AWQ* (arXiv:2306.00978); `ggml-org/llama.cpp`.
