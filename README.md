# TriFact AI

Multilingual Document & Knowledge Intelligence Platform for **Kazakh, Russian and English**:
entity extraction, cross-lingual hybrid search, grounded answers with citations and claim verification.

## Architecture

| | |
|---|---|
| Architecture document | [docs/architecture/docs/01-architecture.md](docs/architecture/docs/01-architecture.md) |
| C4 model (Structurizr DSL) | [docs/architecture/workspace.dsl](docs/architecture/workspace.dsl) |
| Diagrams (Mermaid, exported from the model) | [docs/architecture/diagrams/](docs/architecture/diagrams/) |
| Architecture decisions | [docs/architecture/adr/](docs/architecture/adr/) |
| Roadmap and progress | [docs/ROADMAP.md](docs/ROADMAP.md), [docs/PROGRESS.md](docs/PROGRESS.md) |

View the interactive C4 model with docs and ADRs at http://localhost:8081:

```bash
docker run -it --rm -p 8081:8080 -v "$PWD/docs/architecture:/usr/local/structurizr" structurizr/structurizr local
```

After changing `workspace.dsl`, inspect the model and regenerate the diagrams: `python scripts/export_diagrams.py`.

## Layout

```
src/trifact/   core · lake · ingestion · extraction · retrieval · generation · verification
               api · worker · inference · data · batch · training · evaluation
configs/       NER label mapping, prompts
infra/         docker-compose, Dockerfile, Prometheus, Grafana
web/           Next.js client
data/          Parquet data lake raw → clean → curated → task → eval (DVC, not in git)
dvc.yaml       offline pipeline (acquire → curate → task datasets → index → train → eval → gate)
params.yaml    all parameters; scale.profile = pilot | full
```

## Quick start

```bash
cp .env.example .env
uv venv && uv pip install -e ".[serving,ingestion,ml,dev]"
docker compose -f infra/docker-compose.yml up -d     # Postgres, Qdrant, Redis, S3, MLflow, Prometheus, Grafana
dvc repro acquire                                    # needs HF_TOKEN for gated KazNERD / KazQAD / KazParC
```

Status: stage 0 (foundation) done — architecture, ADRs, configs and module skeleton. Modules are stubs; implementation follows [docs/ROADMAP.md](docs/ROADMAP.md), current state in [docs/PROGRESS.md](docs/PROGRESS.md).
