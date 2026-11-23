# 13. Two-Tier Qdrant Index: Corpus and Workspaces

Date: 2026-11-18

## Status

Accepted

Amends [2. Qdrant for Dense and BM25 Search](0002-qdrant-for-dense-and-sparse.md)

## Context

Two data tiers behave differently:
- **Corpus** (Wikipedia, FineWeb-2 `kk`, `multidomain_kk`): ~35M chunks, built offline, static between releases, tied to one embedder version. At 1024-d `fp32`, vectors require 143 GB vs. 15 GB local RAM.
- **User documents:** thousands to hundreds of thousands of chunks, mutated online, isolated per `workspace_id`.

A single collection would force identical storage/indexing settings for both and couple corpus re-indexing to user data.

## Decision

- `corpus_chunks__<embedder>@<version>`: one collection per embedder version; original vectors on disk in `float16`; quantized copy in RAM (`binary`, `scalar int8`, or `turbo` — selected in Stage 6 `RET-S`) with `rescore` + `oversampling`; HNSW built after bulk upload; payload on disk. Accessed via alias `corpus_chunks`, switched blue/green after promotion gate. Benchmark MRL `256`/`512` for Qwen3-Embedding.
- `workspace_chunks__<embedder>@<version>`: user chunks with in-RAM vectors, `workspace_id` payload index (`is_tenant=true`), and filters on `document_id`, `lang`, `entity_ids`.
- Vectors per collection: `dense` + `sparse` (BM25 with IDF modifier or BGE-M3 sparse, plus Qdrant 1.20 native BM25 text index as comparison).
- Fusion: server-side (`Query API` prefetch + RRF/DBSF) during serving, cross-collection RRF in application code; per-leg queries logged separately in eval mode for ablation.
- Prompt text comes from Qdrant payload; source of truth is the data lake (`task/chunks`) for corpus and PostgreSQL for user documents.

## Consequences

Zero-downtime corpus re-indexing and instant alias rollback. Serving RAM is bounded by quantized vectors. Stage 6 `RET-S` (`1M → 10M → 35M`) decides whether the full 35M index requires a cloud Serving VM (32–64 GB RAM) while local dev serves a subset.
