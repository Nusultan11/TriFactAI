# TriFact AI — Agent Instructions

**Operating principle: Minimum context. Minimum tools. Maximum autonomy. Verified results.**

Applies to every AI agent and engineer modifying this repository.

## 1. Mission & Document Ownership

TriFact AI is a multilingual (KK / RU / EN) document and knowledge intelligence platform designed for tens of millions of records. Each fact has one owner; link to owners instead of duplicating.

| Document | Owns |
|---|---|
| `AGENTS.md` | Global agent rules, autonomy, cloud-first policy, tool hygiene |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Stages, substeps, completion criteria |
| [docs/PROGRESS.md](docs/PROGRESS.md) | Current state, blockers, stage log, downstream notes |
| [01-architecture.md](docs/architecture/docs/01-architecture.md) | System design, flows, experiments, quality gates, deployment |
| [docs/architecture/adr/](docs/architecture/adr/) | Accepted architectural decisions and supersessions |
| [docs/research/models.md](docs/research/models.md) | Verified model facts (size, VRAM, licenses, caveats) |
| [docs/research/datasets.md](docs/research/datasets.md) | Dataset sources, target volumes, license status |
| `params.yaml` | Numeric parameters, thresholds, scale profiles, `cloud.budget_usd` |
| [.agents/skills/](.agents/skills/README.md) | Canonical Agent Skills (task-specific ML & cloud procedures) |
| `README.md` | Overview, quick start, entry links |

## 2. Autonomy & Execution

- Execute assigned tasks end-to-end: investigate → implement → test → verify.
- Make engineering decisions independently using accepted ADRs, `trifact.core.interfaces`, and ROADMAP criteria.
- Do not ask for preliminary plans, intermediate reviews, approvals, or status check-ins.
- Request user input **only** for: genuine external blockers, mandatory human validation (e.g., eval review sheets), security-sensitive operations, or unapproved paid spend.
- Prefer verified code and experimental measurements over outdated documentation; update stale facts with date and source.
- Never claim completion without direct command, test, or metric evidence.
- Before starting stage work: read relevant sections of `docs/PROGRESS.md`, `docs/ROADMAP.md`, and applicable ADRs.
- If a change conflicts with an accepted ADR, write a superseding ADR first.

## 3. Cloud-First Computing & Storage

Execution model: **Local Terminal → Cloud CLI → Remote Compute → Persistent Cloud Storage → Verified Results**

### Compute Policy ([ADR-0020](docs/architecture/adr/0020-kaggle-batch-gpu-mt-and-teacher.md), [ADR-0025](docs/architecture/adr/0025-cloud-only-computing.md))

- Local PC is a control and development station only: coding, Git, CLI orchestration, and fast unit checks.
- Run substantial EDA, data curation, model training, HPO, batch inference, and ML experiments on cloud infrastructure.
- **Primary remote interface:** official Google Colab CLI (`google-colab-cli`). Prefer NVIDIA L4 when available and permitted by quota and budget; use Colab/cloud CPU for non-GPU workloads.
- **Secondary remote option:** Kaggle CLI (`kaggle`, 2× Tesla T4 15 GB fp16, free weekly quota, max 2 concurrent GPU sessions, 12 h/session). Do not run bf16 LoRA on T4.
- Launch, monitor, and retrieve remote jobs from the local terminal via official CLIs. Prefer official CLIs over browser automation or MCP.
- **Pre-flight check (mandatory before remote execution):** verify CLI installation/auth, runtime availability, accelerator type, RAM/disk limits, quota, and `params.yaml: cloud.budget_usd`.
- Never launch unapproved paid workloads. Respect `params.yaml: cloud.budget_usd`.
- Release/stop remote runtimes immediately after outputs are verified and persisted.

### Storage, Experiments & Recovery ([ADR-0024](docs/architecture/adr/0024-google-drive-storage.md), [ADR-0025](docs/architecture/adr/0025-cloud-only-computing.md))

- **GitHub (`origin`):** source code, DVC pointers (`dvc.lock`, `*.dvc`), configs, docs.
- **Google Drive + DVC:** persistent datasets, models, checkpoints, and artifacts (`drive_desktop` locally; mounted `/content/drive/MyDrive/TrifactAI/dvc` in Colab).
- Pull only the DVC targets needed for the current job; avoid full-dataset downloads or redundant transfers.
- Verify data integrity via manifests (`source.yaml`, `dataset.yaml`), checksums (`dvc status --cloud`, `python -m trifact.lake verify`), and pinned revisions.
- **MLflow:** log every experiment with git SHA (clean, non-dirty for full runs), DVC input hashes, config, per-language/direction metrics, runtime, hardware, and cost. In ephemeral runtimes, snapshot SQLite to Drive after each job.
- Design remote jobs to be resumable and idempotent (shard markers `_SUCCESS`, periodic checkpoints persisted to Drive). Never leave irreplaceable outputs only on temporary VM disk.
- Detailed Colab/Kaggle CLI commands and recovery procedures live in [cloud-compute.md](.agents/skills/trifact-mlops/references/cloud-compute.md).

## 4. Tooling: MCP & Agent Skills

### MCP Policy (`.mcp.json`)

- MCP servers are **optional**, never mandatory.
- Never auto-invoke MCP servers at session startup or call tools just because they exist.
- Prefer native CLI, direct file access, Git, and repository docs when sufficient.
- Do not require `shrimp` for task decomposition.
- Do not auto-invoke `context7` for every API; consult official docs only when authoritative current details are needed.
- Do not use `serena` for trivial file edits.
- Avoid `exa`, `huggingface`, `github`, or other external MCP calls when equivalent local information exists.
- Choose the simplest reliable tool that completes the task.

### Agent Skills Policy (`.agents/skills/`)

- `.agents/skills/` is the **single canonical directory** for Agent Skills (compliant with [agentskills.io](https://agentskills.io/specification)). Do not create `CLAUDE.md` or `.claude/skills/` mirrors.
- **Strict on-demand loading:**
  1. Discover skills by name and description only (see [.agents/skills/README.md](.agents/skills/README.md)).
  2. Load only the skill(s) directly relevant to the current task.
  3. Never load all skills automatically; loading unrelated skills is prohibited.
  4. Read `references/` files only when the task requires them.
  5. Skip skills for straightforward tasks requiring no specialized methodology.

## 5. Token & Context Efficiency

- Read only files and line ranges relevant to the task.
- Use targeted search (`rg`, `git grep`) instead of full-repository scans.
- Do not reread unchanged files already in context.
- Summarize long logs instead of pasting raw output.
- Avoid verbose narration, redundant re-verification, and unnecessary delegation.
- Never sacrifice correctness, required tests, or evidence to save tokens.

## 6. Engineering Standards

### Code & Repository Hygiene

- Python 3.12, `uv` (`pyproject.toml` groups: `serving`, `ingestion`, `ml`, `dev`).
- Pre-commit checks: `ruff check`, `ruff format --check`, `mypy src/trifact/core`, `pytest -q`.
- Modules depend on `trifact.core.interfaces`, not concrete model implementations.
- Keep LF line endings in `params.yaml`, `dvc.yaml`, and `src/**/*.py` (DVC hash reproducibility).
- No versioned file copies (`*_v1`, `*_v2`, `*_final`, `*_new`, `*_old`, `*_copy`). Experiment variants use `params.yaml` or CLI flags.
- Write only code needed for the task; remove dead code (`ruff check --select F401,F841`) and temp files outside the repo. Notebooks live in `notebooks/` with outputs stripped.
- Never commit `.env`, secrets, datasets, model weights, `__pycache__`, or build artifacts.

### Data & ML Discipline

- Parquet data lake layers, shards, and manifests follow [ADR-0011](docs/architecture/adr/0011-parquet-data-lake.md); `raw` is immutable.
- Use streaming/columnar engines (Polars lazy/streaming, DuckDB, PyArrow). Never load large datasets into RAM or use `pandas` on large corpora.
- Mass jobs are sharded and idempotent ([ADR-0014](docs/architecture/adr/0014-sharded-batch-jobs.md)).
- Split by source document; freeze eval before training ([ADR-0007](docs/architecture/adr/0007-leakage-safe-splits.md)). Silver and synthetic data are train-only ([ADR-0015](docs/architecture/adr/0015-silver-and-synthetic-data.md)).
- Record source license decision in the manifest before use.
- Compare against a baseline on the same frozen test with statistical significance ([ADR-0022](docs/architecture/adr/0022-evaluation-protocol.md)); promote models only through the gate ([ADR-0009](docs/architecture/adr/0009-model-registry-and-promotion-gate.md)).
- Log negative results in `docs/PROGRESS.md`.

### Git & Backups

- Work on `main`, remote `origin` (GitHub).
- Atomic Conventional Commits (`≤ 72` chars, present tense): `type(scope): summary` (`feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `data`, `exp`, `chore`, `build`).
- Run pre-commit checks before committing; fix root causes, never weaken tests.
- Backups: `python scripts/backup.py --tag <tag>` (or `--dest <Drive>/backups --git-only --keep 0` per ADR-0025) after stages and before destructive or schema changes. Back up data/models via `dvc push`.

## 7. Exceptions

| Condition | Action |
|---|---|
| External blocker (missing access, quota, human labels, unapproved budget) | Complete all unblocked work → record blocker in `docs/PROGRESS.md` → commit → report status and exact user action needed |
| Experiment shows no significant gain | Keep champion → record negative result in `docs/PROGRESS.md` |
| Accepted ADR no longer holds | Author new ADR (`Supersedes`), mark old (`Superseded by`) |
| Destructive delete or schema change | Run `backup.py --tag before-<change>` first |
| Checks fail before commit | Do not commit; fix root cause |
| Required CLI or MCP unavailable | Report exact gap; use sanctioned alternative if available, never assume operational |

## 8. Completion Report

Keep final reports short and evidence-based:
- **Changed files** (paths)
- **Key changes / results** (metrics per language, volumes, throughput, compute used, cost)
- **Verification** (exact commands run and pass/fail results)
- **Commit** (SHA)
- **Blockers / follow-ups** (only if genuine)
