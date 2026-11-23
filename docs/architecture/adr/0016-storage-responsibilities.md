# 16. Storage Responsibilities

Date: 2026-11-18

## Status

Accepted

## Context

Storing tens of millions of corpus chunks (~50 GB text) and hundreds of millions of entity mentions in PostgreSQL would overload the online transactional database and duplicate the data lake, while the application still requires transactional state, query logs, and fast Entity Linking lookups.

## Decision

| Store | Owns | Must Not Store |
|---|---|---|
| **Data Lake (Parquet)** | Source of truth for all corpora, corpus chunks, annotations, vectors, task sets, and eval | Application state |
| **PostgreSQL** | `workspaces`, user `documents`/`chunks`/`mentions`, `queries`/`answers`/`claims`/`feedback`, `kb_entities` & `kb_aliases` (`COPY`-loaded from lake) | Corpus chunks and corpus mentions |
| **Qdrant** | Search indexes; payload = chunk text + filter fields (ADR-0013) | Sole copy of any data |
| **Object Storage (S3)** | User uploads, DVC `local` remote, MLflow artifacts, GGUF weights | — |
| **Redis** | Job queue, query/reranker caches | Durable state |

Manage PostgreSQL schema exclusively via Alembic migrations; bulk-load KB tables via `COPY`.

## Consequences

Every serving store is rebuildable (Qdrant from lake + PostgreSQL, KB from lake) and PostgreSQL stays compact. Corpus entities are exposed via Qdrant `entity_ids` and KB tables; corpus-wide mention analytics run offline (DuckDB over `task/annotations`).
