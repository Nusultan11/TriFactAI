# TriFact AI

Multilingual Document & Knowledge Intelligence Platform for **Kazakh, Russian and English**:
entity extraction, cross-lingual hybrid search, grounded answers with citations and claim verification,
built for corpora of tens of millions of records.

## Documents

| | |
|---|---|
| Working rules (read first) | [AGENTS.md](AGENTS.md) |
| Current state | [docs/PROGRESS.md](docs/PROGRESS.md) |
| Stages and done criteria | [docs/ROADMAP.md](docs/ROADMAP.md) |
| Architecture, repository layout | [docs/architecture/docs/01-architecture.md](docs/architecture/docs/01-architecture.md) |
| Decisions | [docs/architecture/adr/](docs/architecture/adr/) |
| Model and data catalogs | [docs/research/](docs/research/) |

## Quick start

```bash
cp .env.example .env
uv venv && uv pip install -e ".[serving,ingestion,ml,dev]"
docker compose -f infra/docker-compose.yml up -d     # Postgres, Qdrant, Redis, S3, MLflow, Prometheus, Grafana
dvc repro acquire                                    # needs HF_TOKEN for gated KazNERD / KazQAD / KazParC
```
