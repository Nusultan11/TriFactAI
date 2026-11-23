# TriFact AI — Agent Skills

Canonical engineering methodologies compliant with the [Agent Skills specification](https://agentskills.io/specification). Each skill defines **how to execute and verify** a domain task; project decisions and numbers stay in their owner documents ([AGENTS.md §1](../../AGENTS.md)).

## Storage & Discovery

- **Single canonical directory:** `.agents/skills/<skill-name>/SKILL.md`
- **No mirrors:** `.claude/skills/` and `CLAUDE.md` are removed. Do not recreate duplicate skill directories.
- **Discovery:** Agent Skills–compliant clients (e.g., Codex, Antigravity) discover `.agents/skills/` directly via YAML frontmatter (`name`, `description`).
- **Compatibility note:** Clients that only scan `.claude/skills/` will not auto-discover `.agents/skills/` unless explicitly pointed to `.agents/skills/`. This limitation is intentional to prevent drift and duplication.
- **Validation:** `pytest tests/test_skills.py` and the `agent-skills` pre-commit hook verify frontmatter schema, `< 500` line limit, link integrity, and absence of Claude skill mirrors.

## On-Demand Loading Rule

1. Match the task against skill names and descriptions below.
2. Read **only** the directly relevant skill(s). Never bulk-load all skills.
3. Read `references/` files only when the subtask requires them.
4. Skip skills for simple tasks that need no specialized methodology.

## Skill Catalog

| Skill | Owns | Out of Scope |
|---|---|---|
| [trifact-data-pipeline](trifact-data-pipeline/SKILL.md) | Lake layers, cloud CPU/GPU curation, splits, silver/synthetic/MT, licenses, streaming & memory | Distribution analysis (`eda`) |
| [trifact-eda](trifact-eda/SKILL.md) | Cloud-scalable EDA of corpora & task sets, label noise, shift, finding→decision | Filter implementation (`data-pipeline`) |
| [trifact-document-processing](trifact-document-processing/SKILL.md) | PDF/DOCX/HTML/TXT parsing, OCR, layout, tables, routing, cloud batch eval | Retrieval chunking choice (`retrieval`) |
| [trifact-ner-linking](trifact-ner-linking/SKILL.md) | NER, DATE, Entity Linking, calibration, ONNX speedup, cloud `annotate_corpus` | Generic training/HPO (`training-hpo`) |
| [trifact-training-hpo](trifact-training-hpo/SKILL.md) | Cloud GPU training, Optuna HPO, seeds, checkpoints, ablations, cost/quota | Task-specific recipes (domain skills) |
| [trifact-retrieval](trifact-retrieval/SKILL.md) | Chunking, embeddings, hybrid, Qdrant at scale, reranker, 9-direction eval | LLM context assembly (`rag-pipeline`) |
| [trifact-rag-pipeline](trifact-rag-pipeline/SKILL.md) | End-to-end `ask` flow, context budget, prompt, refusal, injection defense, stage attribution | LLM fine-tuning (`llm-finetuning`) |
| [trifact-llm-finetuning](trifact-llm-finetuning/SKILL.md) | Qwen3.5 compatibility, baselines, SFT mix, cloud bf16 LoRA, GGUF/imatrix, serving | Claim verification (`verification`) |
| [trifact-verification](trifact-verification/SKILL.md) | Claims, windowed NLI, calibration, risk–coverage thresholds, refusal policy | End-to-end RAG metrics (`evaluation`) |
| [trifact-evaluation](trifact-evaluation/SKILL.md) | Metric selection, cluster bootstrap, LLM-as-judge limits, slicing, error analysis, gate | Component implementation |
| [trifact-mlops](trifact-mlops/SKILL.md) | DVC/Drive, MLflow/Registry, cloud compute CLI, ML tests, profiling, drift | Global git/autonomy rules (`AGENTS.md`) |

**Shared cloud CLI reference:** [trifact-mlops/references/cloud-compute.md](trifact-mlops/references/cloud-compute.md) (Google Colab CLI, Kaggle CLI, Drive/DVC/MLflow persistence, teardown).

## Task Routing

If two skills apply, load the domain skill first and the methodology skill (`evaluation` or `training-hpo`) second.

| Task | Load Skill(s) |
|---|---|
| Add raw source, fix stage OOM, build silver/synthetic/MT | `trifact-data-pipeline` |
| Audit dataset before training, detect label noise or shift | `trifact-eda` |
| Fix lost Kazakh OCR letters, table parsing, new file format | `trifact-document-processing` → `trifact-evaluation` |
| Train NER, improve kk F1, link Kazakh inflected mentions | `trifact-ner-linking` → `trifact-training-hpo` |
| Tune LR/epochs, run Optuna HPO, debug divergence | `trifact-training-hpo` |
| Choose embedder, configure RRF/Qdrant, train reranker | `trifact-retrieval` → `trifact-evaluation` |
| Diagnose wrong/uncited RAG answer, test prompt injection | `trifact-rag-pipeline` → `trifact-evaluation` |
| Curate SFT mix, run bf16 LoRA on Qwen3.5, quantize GGUF | `trifact-llm-finetuning` |
| Tune NLI thresholds, calibrate ECE, filter synthetic QA | `trifact-verification` |
| Prove model gain with CI, run error analysis, check gate | `trifact-evaluation` |
| Launch/recover Colab or Kaggle jobs, MLflow/DVC/drift | `trifact-mlops` |

## Authoring Rules

- Directory name `trifact-<domain>` must match frontmatter `name` (`^[a-z0-9]+(-[a-z0-9]+)*$`, ≤ 64 chars).
- Frontmatter uses only spec fields (`name`, `description`, optional `license`, `compatibility`, `metadata`, `allowed-tools`).
- `description` (≤ 1024 chars) states what the skill does and starts trigger guidance with `Use `.
- Keep `SKILL.md` body under 500 lines; place deep details in one-level `references/*.md`.
- Write concise, actionable English. Do not duplicate `AGENTS.md`, `params.yaml`, or `cloud-compute.md`.
