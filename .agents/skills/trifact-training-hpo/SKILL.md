---
name: trifact-training-hpo
description: Cloud-first training methodology and Optuna HPO for TriFact models (NER, embedders, rerankers, NLI, LoRA) — experiment design, validation discipline, cloud GPU precision (Colab L4 bf16, Kaggle 2x T4 fp16), checkpoints and resumability on Drive, seeds, learning curves, Optuna samplers/pruners, ablations, cost and quota logging. Use when implementing or running trifact.training jobs, tuning hyperparameters, planning HPO, or debugging divergence, overfitting, or baseline misses.
---

# TriFact AI — Cloud Training & HPO

Domain recipes live in `trifact-ner-linking`, `trifact-retrieval`, `trifact-verification`, and `trifact-llm-finetuning`. This skill defines shared cloud training and HPO discipline.

## Owner Documents

- Cloud CLI, Drive checkpoints & MLflow snapshots → [cloud-compute.md](../trifact-mlops/references/cloud-compute.md)
- Cloud compute & budget policy → [AGENTS.md §3](../../../AGENTS.md), [ADR-0020](../../../docs/architecture/adr/0020-kaggle-batch-gpu-mt-and-teacher.md), [ADR-0025](../../../docs/architecture/adr/0025-cloud-only-computing.md), `params.yaml: cloud.budget_usd`
- Gate & significance → [01-architecture.md §10–§11](../../../docs/architecture/docs/01-architecture.md), [ADR-0009](../../../docs/architecture/adr/0009-model-registry-and-promotion-gate.md), [ADR-0022](../../../docs/architecture/adr/0022-evaluation-protocol.md)
- Starting hyperparameters → `params.yaml` (`ner`, `retrieval`, `llm`, `verification`)

## Pre-Training Design Checklist

1. State hypothesis and target metric/language/direction.
2. Establish a strong, cheap baseline (zero-shot, prior champion, or rule) before training.
3. Change one variable per comparison; ablations are strictly leave-one-out.
4. Select hyperparameters and checkpoints **only on validation**. Open frozen test once for the promotion gate.
5. Verify budget (`cloud.budget_usd`), GPU quota, and runtime limits. Run `scale.profile: pilot` before full training.

## Cloud Compute & Precision Matrix

| Environment | Precision | Role & Guardrails |
|---|---|---|
| **Local PC** | CPU / unit only | Code, tests, tiny collator/shape checks. **Never** run ML training or HPO locally. |
| **Colab L4 (24 GB, sm89)** *(preferred)* | `bf16` | Primary training, HPO, bf16 LoRA (ADR-0012), large encoders. Persist checkpoints to Drive; stop session when done. |
| **Kaggle 2× T4 (15 GB × 2, sm75)** *(secondary)* | **`fp16` only** (no `bf16`) | Encoder fine-tuning & HPO within 30 GPU-h/week free quota (12 h/session). Use `fp16` AMP + GradScaler and 1 process/GPU. Never run bf16 LoRA here. |
| **Approved Cloud GPU (A100/H100)** | `bf16` | Full-scale LLM/embedder runs only when `cloud.budget_usd > 0` is explicitly approved. |

### Mandatory Pre-Run Sanity Checks (Before Full Cloud Run)

1. Overfit a 64–100 example batch to near-zero loss (catches broken labels/masks).
2. Decode one collated batch visually to verify tokenization, special tokens, and `-100` masks.
3. Run the evaluation script on a known baseline and verify expected score.

## Starting Search Ranges

| Architecture | Optimizer & Schedule | Starting Zone |
|---|---|---|
| Encoder fine-tune (NER, NLI, cross-encoder) | AdamW, linear decay, 6–10% warmup, wd 0.01 | `lr` `1e-5…5e-5` (base), `5e-6…2e-5` (large), 3–5 epochs; optional layer-wise LR decay |
| Bi-encoder (embedder) | InfoNCE / MNRL with in-batch + hard negatives | Large effective batch via `CachedMNRL`/GradCache, `lr` `1e-5…3e-5`, 1–2 epochs |
| LoRA LLM | See `trifact-llm-finetuning` | `lr` ≈ 10× full fine-tune LR |

Fix BERT/XLM-R instability with lower LR, warmup, more steps, and Adam bias correction — never by cherry-picking lucky seeds.

## Seeds, Checkpoints & Cloud Recovery

- **Seeds:** Minimum 3 seeds (`ner.seeds`). Fix Python, NumPy, PyTorch, DataLoader shuffle, and Optuna sampler seeds. Report mean ± std and paired cluster bootstrap CI (`trifact-evaluation`).
- **Checkpointing:** Save `(model, optimizer, scheduler, step, best_metric, rng_state)` to `/content/drive/MyDrive/TrifactAI/workspace-data/checkpoints/<exp>/<run>/` at regular intervals. Keep `best` (by target dev metric, not loss) and `last`.
- **Resumability:** Training scripts must support `--resume <checkpoint_dir>` so preempted Colab/Kaggle sessions continue without losing work.
- **MLflow:** Log `samples/s` or `tokens/s`, peak VRAM (`torch.cuda.max_memory_allocated`), peak RAM, GPU model, wall time, GPU-hours, and USD cost. Snapshot SQLite DB to Drive after each run.

## Optuna HPO Protocol

- **Search priority:** Tune `lr` (log scale) first, then warmup, epochs, batch size, and regularization. Keep architectural/data changes as separate ablations.
- **Samplers:**
  - Default (100–1000 trials): `TPESampler(multivariate=True, seed=...)`
  - Small budget (10–40 trials, continuous): `GPSampler`
  - Multi-objective (quality vs. latency/VRAM): `NSGAIISampler` / Pareto front under NFR constraints
- **Pruners:** `MedianPruner` or `HyperbandPruner` on intermediate dev metric (ensure total trials > `n_startup_trials × brackets`); `WilcoxonPruner` when evaluating across example/direction slices.
- **Persistence:** Store Optuna studies in Drive-backed `JournalStorage` or SQLite RDB with fixed `study_name` so interrupted cloud sessions resume seamlessly. Log each trial as a nested MLflow run.
- **Promotion from HPO:** Run HPO on pilot/dev slice → retrain top-3 configs on full data with ≥3 seeds → pick winner by mean + CI. Note: LR transfers across data scales; epoch count does not.

## Failure Diagnostics

| Symptom | Check | Action |
|---|---|---|
| `NaN` loss in `fp16` | Grad norm spike, overflow | Enable GradScaler, clip grad norm 1.0, lower LR, or switch to Colab L4 `bf16` |
| Random-level metric | Label alignment, masks, shuffled IDs | Run 64-example overfit test |
| Dev improves, test flat | Leakage or dev overfitting | Audit leakage, reduce HPO iterations |
| Large train/dev gap | Overfitting | Weight decay, dropout, fewer epochs, check learning curve (10/30/100% data) |
| `kk` lags `ru`/`en` | Low `kk` share, fertility truncation | Raise `kk` mix share, check token length distribution |
| CUDA OOM | Batch, sequence length, activations | Gradient checkpointing, accumulation, length bucketing, 8-bit AdamW |

## Done Criteria

- Baseline present, ≥3 seeds run, CI computed; config selected on dev, test touched once.
- Optuna study resumable and reproducible (storage, seed, `study_name`); trials logged in MLflow.
- Compute profile, GPU-hours, and cost logged; checkpoints persisted on Drive; runtime stopped; negative results recorded in `PROGRESS.md`.

## Sources

- Akiba et al., *Optuna* (arXiv:1907.10902); Li et al., *Hyperband* (arXiv:1603.06560), *ASHA* (arXiv:1810.05934).
- Google Research, *Deep Learning Tuning Playbook*; Mosbach et al. (arXiv:2006.04884); Dodge et al. (arXiv:2002.06305); Bouthillier et al. (arXiv:2103.03098); Gao et al., *GradCache* (arXiv:2101.06983).
