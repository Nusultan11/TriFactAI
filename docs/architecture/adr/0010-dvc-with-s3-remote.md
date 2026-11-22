# 10. DVC для данных и моделей, S3-совместимое хранилище (SeaweedFS) как remote

Date: 2026-11-18

## Status

Accepted

Amended by [14. Шардированные идемпотентные batch-задания](0014-sharded-batch-jobs.md)

Amended by [24. Google Drive для данных и резервных копий](0024-google-drive-storage.md)

## Context

Датасеты и веса не помещаются в git, а эксперименты должны воспроизводиться.

## Decision

DVC-пайплайн (`dvc.yaml`) с параметрами в `params.yaml`; remote — SeaweedFS S3 (`s3://trifact-dvc`). MinIO не используется: его образы больше не публикуются на Docker Hub. MLflow хранит артефакты на своём volume с `--serve-artifacts`; при переезде в облако артефакты переносятся в тот же S3. Ревизия каждого HF-датасета фиксируется в `params.yaml`.

## Consequences

`dvc repro` воспроизводит любой этап; при переезде в облако SeaweedFS заменяется любым S3.
