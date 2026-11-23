# 24. Google Drive for Persistent Data and Backups

Date: 2026-11-22

## Status

Accepted

Amends [10. DVC with S3-Compatible Remote](0010-dvc-with-s3-remote.md)

Amends [14. Sharded Idempotent Batch Jobs](0014-sharded-batch-jobs.md)

Amended by [25. Cloud-Only Computing and Stage 1–3 Recovery](0025-cloud-only-computing.md)

## Context

Code and Git history live on GitHub, while datasets, models, and backups must be recoverable from private Google Drive. The native DVC Drive connector hit file-size limits and Google blocked DVC's built-in OAuth client.

## Decision

- Store persistent project assets in Google Drive under `TrifactAI/`: DVC cache (`dvc`), models (`models`), working artifacts & recovery maps (`workspace-data`), and Git/DVC-pointer backups (`backups`).
- On Windows, mount `dvc` in streaming mode via Google Drive for Desktop and configure DVC local remote `drive_desktop` (`verify=true`) in `.dvc/config.local`.
- Keep SeaweedFS for the local Docker serving stack. Cloud jobs access inputs via mounted Drive in Colab or job-specific input bundles.
- Clear local data copies after verifying cloud objects; preserve original paths and hashes in recovery maps.

## Consequences

Local `dvc pull` requires Google Drive for Desktop running and temporarily caches synced files on the system drive; clearing verified local copies reclaims laptop disk space.
