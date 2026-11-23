# 14. Sharded Idempotent Batch Jobs

Date: 2026-11-18

## Status

Accepted

Amends [10. DVC with S3-Compatible Remote](0010-dvc-with-s3-remote.md)

Amended by [20. Kaggle Batch GPU, NLLB MT, and llama.cpp Teacher](0020-kaggle-batch-gpu-mt-and-teacher.md)

Amended by [24. Google Drive for Data and Backups](0024-google-drive-storage.md)

Amended by [25. Cloud-Only Computing and Stage 1–3 Recovery](0025-cloud-only-computing.md)

## Context

Mass batch jobs (35M corpus embeddings, corpus NER annotation, synthetic teacher generation, `kk` XNLI translation) run for hours on preemptible cloud GPUs that cannot reach local SeaweedFS. Jobs must be crash-resumable and executable identically on pilot slices and full-scale cloud runs.

## Decision

- Run batch jobs as `python -m trifact.batch.<job> --input <layer/...> --output <layer/...> --shard i/N`, with one input Parquet shard as the unit of work. Write outputs to temp files, atomically rename, and emit a `_SUCCESS` marker; re-runs skip completed shards.
- Log `records/s`, peak RAM/VRAM, wall time, GPU model, and hourly cost to MLflow.
- Use one shared Docker/uv environment; pull only required input shards, execute, and push outputs immediately.
- Configure DVC remotes for local S3 (`local`, SeaweedFS) and cloud object storage (`cloud`; subsequently amended to Google Drive in ADR-0024/0025).
- Record cloud budget in `params.yaml` before launch (ROADMAP 3.6) and halt on budget breach.

## Consequences

Preemption loses at most one active shard. Pilot validation (`scale.profile: pilot`) uses identical code and parameters as full cloud runs. Because transferring ~72 GB of fp16 vectors is costly, corpus collections may be built directly on the Serving VM (ADR-0013).
