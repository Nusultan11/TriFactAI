# 25. Cloud-Only Computing and Stage 1–3 Recovery

Date: 2026-11-22

## Status

Accepted. Environment qualification status is tracked in [docs/PROGRESS.md](../../PROGRESS.md).

Amends [14. Sharded Idempotent Batch Jobs](0014-sharded-batch-jobs.md), [20. Kaggle Batch GPU, NLLB MT, and llama.cpp Teacher](0020-kaggle-batch-gpu-mt-and-teacher.md), and [24. Google Drive for Data and Backups](0024-google-drive-storage.md).

## Context

The 220+ GiB DVC object store exceeds a single Colab runtime's disk, cloud sessions are ephemeral, and GPU availability depends on account quota and budget. The local workstation serves as a development, Git, and CLI control station rather than an ML compute host.

## Decision

- **Cloud execution:** Run primary GPU workloads on Google Colab (NVIDIA L4 preferred when available and permitted by quota/budget) via official Google Colab CLI (`colab`) or [notebooks/cloud_stages.ipynb](../../../notebooks/cloud_stages.ipynb); run secondary GPU workloads on private Kaggle kernels (2× T4 fp16) via Kaggle CLI; run CPU DVC stages on cloud CPU runtimes using the same DVC graph. Record compute profile and host in every run.
- **Clean cloud bootstrap:** Mount private Drive, check out a pinned 40-char Git SHA with `uv.lock`, configure the local DVC remote explicitly pointing to Drive `dvc` with `verify=true`, and materialize only required stage inputs. Shard stages exceeding runtime RAM or disk.
- **Bounded Colab GPU runner (`trifact.batch.colab`):** Execute one bounded input shard per invocation with input/code/lockfile fingerprinting, ID reconciliation, and output SHA-256 verification. Reuse completed shards; retry interrupted ones (losing at most the active shard); store attempt logs in Drive `workspace-data/colab`.
- **MLflow persistence:** Run SQLite on VM disk while writing artifacts and atomic integrity-checked DB snapshots to Drive `workspace-data/mlflow` after each job (use a shared remote MLflow server when available for central Registry).
- **DVC & Git backups:** Publish DVC pointers only after Drive object sync/verification; snapshot Git history to Drive `backups` via `scripts/backup.py --git-only --keep 0`; verify clean-VM model/artifact restore via checksums.
- **Budget guardrail:** Enforce `params.yaml: cloud.budget_usd`; never start paid workloads without approved budget; log actual GPU type and quota usage per run.

## Consequences

Historical Kaggle outputs and DVC pointers preserve their original paths and provenance. Full-scale Colab L4 runs depend on clean recovery verification, L4 pilot qualification, and approved compute budget; eval freeze remains gated on human review sheets and ROADMAP criteria.
