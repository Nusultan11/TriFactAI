# Cloud Compute Reference (Colab CLI, Kaggle CLI, Storage & Recovery)

Single operational reference for cloud-first execution ([AGENTS.md §3](../../../../AGENTS.md), [ADR-0020](../../../../docs/architecture/adr/0020-kaggle-batch-gpu-mt-and-teacher.md), [ADR-0024](../../../../docs/architecture/adr/0024-google-drive-storage.md), [ADR-0025](../../../../docs/architecture/adr/0025-cloud-only-computing.md)).

Execution flow: **Local Terminal → Cloud CLI → Remote Compute → Persistent Cloud Storage → Verified Results**

Command labels: **`[LOCAL]`** runs on the development PC; **`[REMOTE]`** runs inside the cloud runtime.

---

## 1. Pre-Flight Checklist (Mandatory)

1. Verify `params.yaml: cloud.budget_usd`. Never launch paid workloads without approved budget.
2. Commit code before full runs (`git status` clean, non-dirty SHA) and push to `origin/main`.
3. Verify CLI installation, authentication, target accelerator availability, RAM/disk limits, and quota.
4. Select only required DVC input targets — never pull the entire 220+ GiB DVC cache onto an ephemeral VM.
5. Run `scale.profile: pilot` or a bounded single shard before full-scale execution.

---

## 2. Google Colab CLI (Primary Interface)

Official repo: `https://github.com/googlecolab/google-colab-cli`.
- **OS support:** Linux and macOS natively. On Windows, run inside a Linux WSL distribution; if unavailable, report the gap and fall back to Kaggle CLI or `notebooks/cloud_stages.ipynb` per ADR-0025.
- **Accelerator preference:** NVIDIA `L4` (24 GB VRAM, bf16) when available and within quota/budget; `CPU` runtime for non-GPU DVC/EDA/curation stages.

### Install & Authenticate (`[LOCAL]`)

```bash
# [LOCAL] Install via uv (Linux / macOS / WSL)
uv tool install google-colab-cli

# [LOCAL] Authenticate with OAuth2 (stores refreshable token) and verify sessions
colab --auth=oauth2 sessions

# [LOCAL] Optional: GCP ADC if accessing GCP services
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

### Session Lifecycle & Execution (`[LOCAL]` controlling `[REMOTE]`)

```bash
# [LOCAL] Provision named runtime (L4 GPU or omit --gpu for CPU)
colab new -s trifact-job --gpu L4
colab status -s trifact-job

# [LOCAL] Mount Google Drive at /content/drive on the remote VM
colab drivemount -s trifact-job

# [LOCAL] Execute script or command on remote VM (state persists per session)
colab exec -s trifact-job -f bootstrap_and_run.py

# [LOCAL] Ephemeral one-shot job (provisions, runs, auto-terminates)
colab run --gpu L4 run_job.py

# [LOCAL] Inspect remote files, export logs, download small reports
colab ls -s trifact-job /content/TrifactAI/reports
colab log -s trifact-job
colab download -s trifact-job /content/TrifactAI/reports/run.json ./reports/run.json

# [LOCAL] ALWAYS release runtime immediately after outputs are persisted
colab stop -s trifact-job
```

### Remote Environment Bootstrap (`[REMOTE]` inside Colab)

```bash
# [REMOTE] Verify GPU/RAM/disk and Drive DVC store before pulling data
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
df -h /content && free -h
test -d /content/drive/MyDrive/TrifactAI/dvc/files/md5

# [REMOTE] Checkout pinned Git SHA and locked uv environment
git clone --no-checkout https://github.com/Nusultan11/TriFactAI.git /content/TrifactAI
git -C /content/TrifactAI checkout --detach <40-CHAR-GIT-SHA>
cd /content/TrifactAI
python -m pip install -q uv
uv sync --locked --extra ml --extra dev

# [REMOTE] Point DVC to mounted Drive store with hash verification
uv run --no-sync dvc remote add --local --force --default drive /content/drive/MyDrive/TrifactAI/dvc
uv run --no-sync dvc remote modify --local drive verify true
uv run --no-sync dvc config --local core.autostage false
```

### Bounded GPU Shard Runner (`[REMOTE]` in Colab)

```bash
# [REMOTE] Execute one bounded shard (teacher | translate | nli_score) with Drive checkpointing
uv run --no-sync python -m trifact.batch.colab <job> \
  --inputs /content/drive/MyDrive/TrifactAI/workspace-data/<shard_dir> \
  --drive /content/drive/MyDrive/TrifactAI \
  --timeout-minutes 45
```

- Runner validates Colab runtime, Drive mount, single L4 GPU, input/output IDs, and SHA-256 digests.
- Completed shards are skipped on retry; interrupted runs lose at most the active shard.

---

## 3. Kaggle CLI (Secondary Interface)

Official repo: `https://github.com/Kaggle/kaggle-cli`.
- **Hardware & limits:** 2× Tesla T4 (15 GB each, **fp16 only, no bf16**), 30 GPU-h/week free quota, 12 h max/session, max 2 concurrent GPU sessions.
- **Security:** upload only private input datasets and self-contained job scripts (`src/trifact/batch/jobs/<job>.py`). Never transmit `.env`, `HF_TOKEN`, or user secrets to Kaggle.

### Direct CLI Workflow (`[LOCAL]`)

```bash
# [LOCAL] Check credentials and remaining weekly quota
kaggle config view
kaggle quota

# [LOCAL] Initialize, push, monitor, and retrieve kernel outputs
kaggle kernels init -p <kernel_dir>
kaggle kernels push -p <kernel_dir> --accelerator NvidiaTeslaT4
kaggle kernels status <user>/<kernel-slug>
kaggle kernels output <user>/<kernel-slug> -p <out_dir>
```

### Project Batch Driver (`[LOCAL]`, ADR-0020)

```bash
# [LOCAL] Upload inputs as private dataset, push 2x T4 kernel, poll, download outputs + run.json
uv run --no-sync python -m trifact.batch.kaggle run <job> --inputs <in_dir> --out <out_dir> [--cpu] [--timeout-h 11]

# [LOCAL] Fetch outputs of an already completed run
uv run --no-sync python -m trifact.batch.kaggle fetch <job> --out <out_dir> [--version N]
```

- Use one OS process per GPU (`CUDA_VISIBLE_DEVICES=0`, `1`), never multi-threaded shared `torch` default dtype.

---

## 4. Persistent Storage, DVC & MLflow

### Selective DVC Sync (`[REMOTE]`)

```bash
# [REMOTE] Pull only required targets
uv run --no-sync dvc pull <target1> <target2>
uv run --no-sync dvc status --cloud --remote drive <target1> <target2>

# [REMOTE] Run single stage and push outputs immediately
uv run --no-sync dvc repro --single-item <stage>
uv run --no-sync dvc push <stage>
uv run --no-sync dvc status --cloud --remote drive <stage>
```

### MLflow in Ephemeral Runtimes (`[REMOTE]`)

- Run SQLite backend on local VM disk (`/content/trifact-<session>.db`) and write artifacts directly to Drive (`--default-artifact-root file:///content/drive/MyDrive/TrifactAI/workspace-data/mlflow/artifacts`).
- After every job, create an atomic SQLite backup (`sqlite3.Connection.backup()` + `PRAGMA integrity_check` + SHA-256 copy) into `/content/drive/MyDrive/TrifactAI/workspace-data/mlflow/` (implemented in `notebooks/cloud_stages.ipynb`).
- When a shared remote MLflow server is configured, set `MLFLOW_TRACKING_URI` and verify `/health` before logging.

### Checkpoints, Recovery & Cleanup

- Write model checkpoints (weights + optimizer + scheduler + step + RNG state) and Optuna RDB/Journal storage to `/content/drive/MyDrive/TrifactAI/workspace-data/<area>/`.
- Restore pinned non-DVC models via `uv run --no-sync python scripts/restore_model.py <parts.json> --dest <dir>` (verifies SHA-256).
- Backup Git history + DVC pointers to Drive: `uv run --no-sync python scripts/backup.py --tag <tag> --dest /content/drive/MyDrive/TrifactAI/backups --git-only --keep 0`.
- Before `colab stop`, verify all outputs exist on Drive (`dvc status --cloud` clean, MLflow snapshot written, checkpoints synced).
