# 10. DVC with S3-Compatible Remote (SeaweedFS)

Date: 2026-11-18

## Status

Accepted

Amended by [14. Sharded Idempotent Batch Jobs](0014-sharded-batch-jobs.md)

Amended by [24. Google Drive for Data and Backups](0024-google-drive-storage.md)

## Context

Datasets and model weights do not belong in Git, yet every pipeline stage must remain reproducible.

## Decision

Track pipelines with DVC (`dvc.yaml`) and parameters in `params.yaml`, using SeaweedFS S3 (`s3://trifact-dvc`) as the local S3 remote (MinIO omitted because images are no longer published on Docker Hub). Store MLflow artifacts on its volume with `--serve-artifacts` (portable to S3 in cloud deployments). Pin every HF dataset revision by SHA in `params.yaml`.

## Consequences

`dvc repro` reproduces any stage; SeaweedFS can be swapped for any S3-compatible object store.
