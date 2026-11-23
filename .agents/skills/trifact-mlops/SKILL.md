---
name: trifact-mlops
description: Cloud-first MLOps for TriFact — Google Colab CLI and Kaggle CLI workflows, reproducible DVC stages with Google Drive persistence, MLflow tracking/snapshots and Model Registry aliases (champion/challenger) with Qdrant rollback, ML test pyramid, RAM/VRAM/throughput profiling, resumable sharded jobs, serving optimization, and drift monitoring. Use when orchestrating cloud compute, making stages reproducible, recovering artifacts, registering/promoting models, adding ML tests, profiling, or designing alerts.
---

# TriFact AI — MLOps & Cloud Operations

Global git, autonomy, and commit rules live in [AGENTS.md](../../../AGENTS.md). This skill owns cloud compute execution, reproducibility, ML testing, profiling, and observability.

## Owner Documents

| Topic | Owner |
|---|---|
| Cloud CLI (Colab, Kaggle), Drive, DVC, MLflow recovery | [references/cloud-compute.md](references/cloud-compute.md) |
| Cloud compute & Drive storage decisions | [ADR-0020](../../../docs/architecture/adr/0020-kaggle-batch-gpu-mt-and-teacher.md), [ADR-0024](../../../docs/architecture/adr/0024-google-drive-storage.md), [ADR-0025](../../../docs/architecture/adr/0025-cloud-only-computing.md) |
| Offline compute, MLOps, observability, deployment | [01-architecture.md §9, §11–§13](../../../docs/architecture/docs/01-architecture.md) |
| Registry & promotion gate | [ADR-0009](../../../docs/architecture/adr/0009-model-registry-and-promotion-gate.md) |
| DVC remotes, sharded batch jobs, stage fingerprints | [ADR-0010](../../../docs/architecture/adr/0010-dvc-with-s3-remote.md), [ADR-0014](../../../docs/architecture/adr/0014-sharded-batch-jobs.md), [ADR-0016](../../../docs/architecture/adr/0016-storage-responsibilities.md), [ADR-0018](../../../docs/architecture/adr/0018-curation-stages-without-datatrove.md) |
| Profiling & MLflow helpers | `trifact.lake.stats` (`ResourceMonitor`, `git_sha`, `log_mlflow`) |

## Cloud Execution & Artifact Persistence

Follow [references/cloud-compute.md](references/cloud-compute.md) for exact `[LOCAL]` and `[REMOTE]` commands:
- **Primary:** Official Google Colab CLI (`colab new/drivemount/exec/run/log/stop`, NVIDIA L4 when available/permitted, Colab CPU for non-GPU stages).
- **Secondary:** Kaggle CLI (`kaggle kernels push/status/output`, `python -m trifact.batch.kaggle`, 2× T4 fp16 within free weekly quota).
- **Pre-flight:** Verify CLI auth, runtime/GPU type, RAM/disk, quota, and `params.yaml: cloud.budget_usd`. Never start unapproved paid runs.
- **Storage:** Pull only required DVC targets from Drive (`verify=true`), push outputs immediately, snapshot ephemeral MLflow SQLite DBs to Drive, and stop cloud runtimes immediately on completion.

## Reproducibility Checklist

Every run must be reconstructible from `git SHA + DVC hashes + config` (`01-architecture.md §1`):
- Commit code before full runs (`git_sha` without `-dirty`).
- Declare all `deps`, `params`, and `outs` in `dvc.yaml`; commit `dvc.lock` with LF line endings.
- Pin Hugging Face models and tokenizers by commit SHA (`params.yaml`), never mutable branch names.
- Fix Python, NumPy, and PyTorch seeds; record CUDA/driver version and `uv.lock` / Docker image.

## MLflow & Model Registry

- One experiment per domain (`01-architecture.md §11`); nested runs for HPO trials and benchmark configs.
- Every run logs: flattened params + full config artifact, per-language/direction metrics (`<metric>/<lang>` or `<metric>/<q>-<d>` sanitized via `lake.stats._mlflow_name`), tags (`stage`, `status=final|superseded`), JSON reports/plots/per-example traces (ADR-0022), wall time, throughput, peak RSS/VRAM, GPU model, and USD/quota cost.
- Check `/health` before logging to an HTTP MLflow server so unreachable servers never stall jobs.
- Log models with signature + input example → register as `challenger`. Only the promotion gate moves `champion` (ADR-0009). Services load `models:/<name>@champion`.
- Rollback = alias reassignment. For embedders, switch model alias and Qdrant collection alias atomically (`01-architecture.md §11`).

## ML Test Pyramid

| Level | Checks | TriFact Examples |
|---|---|---|
| Unit | Pure logic on synthetic cases | Normalization, chunk offsets, BIO conversion, metrics vs. `seqeval`/`ranx` |
| Data & schema | Shard invariants | Pandera schemas, offset bounds, valid BIO, zero eval leakage |
| Integration | End-to-end stage on tiny fixture | `--limit`, `--root <tmp>`, manifest marked `partial` |
| Model regression | Metric on golden set within tolerance | NER F1 / nDCG vs. recorded baseline ± margin |
| Behavioral | Invariance & directional checks | Name transliteration, passage reorder, evidence removal → refusal |
| Performance | Throughput floor, RAM ceiling | Parser `pages/s`, CPU NER `chunks/s` |
| E2E | API & Web on fixtures | `eval_e2e`, Playwright (Stage 10) |

Mark network/large-model tests so default `pytest -q` and pre-commit stay fast. Extend `tests/test_architecture.py` whenever adding a structural invariant.

## Profiling & Job Resilience

- **Profiling:** `lake.stats.ResourceMonitor` (full process-tree RSS), `py-spy` (CPU hotspots), `memray` (Python allocations), Polars `explain()`, DuckDB `EXPLAIN ANALYZE`, `torch.cuda.max_memory_allocated` / `torch.profiler` / `nvidia-smi`. Benchmark `1/2/4/8` workers to find the throughput/RAM knee.
- **Resilience:** Idempotent shards with atomic writes and `_SUCCESS` markers; exponential backoff + timeouts on network calls; pre-run quota/budget checks; periodic Drive checkpoints (weights + optimizer + scheduler + step) for preemptible GPUs.

## Serving Optimization & Drift Monitoring

- **Serving:** Dynamic batching with max-wait cap and length sorting in Inference Service; ONNX Runtime int8 (CPU) / fp16 (GPU) warmed up before `ready` healthcheck; include `model_version` in Redis cache keys for query embeddings and reranker scores.
- **Drift:** Monitor question language/length, max retrieval/reranker scores, refusal/`contradicted`/`not_enough_info` rates, `ParseError` reasons, and OCR page share via PSI/JS-divergence (histograms), KS test (continuous), and centroid distance / two-sample classifier (embeddings). Derive alert thresholds from reference-window variance and link every alert to a runbook. Feed audited production samples into **new** versioned eval suites; never mutate frozen tests.

## Done Criteria

- Clean Git SHA, clean `dvc status --cloud`, pinned model revisions, and complete MLflow run persisted to Drive.
- Unit, data, regression, and performance tests pass.
- Registered model promoted only via gate with verified rollback; cloud runtime stopped.
- Production service has Prometheus metrics, Grafana dashboard, and reference-calibrated alerts.

## Sources

- Breck et al., *The ML Test Score* (IEEE Big Data 2017); Sculley et al., *Hidden Technical Debt* (NeurIPS 2015).
- Rabanser et al. (arXiv:1810.11953); Ribeiro et al., *CheckList* (arXiv:2005.04118).
- Official docs: `googlecolab/google-colab-cli`, `Kaggle/kaggle-cli`, `mlflow.org/docs`, `dvc.org/doc`, `pandera.readthedocs.io`.
