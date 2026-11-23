# 2. Qdrant for Dense and BM25 Search

Date: 2026-11-18

## Status

Accepted

Amended by [13. Two-Tier Qdrant Index](0013-two-tier-qdrant-index.md)

## Context

Hybrid retrieval requires both lexical (BM25) and dense vector indexing. Running a separate Elasticsearch/OpenSearch cluster adds ≥1–2 GB RAM overhead and cross-store synchronization complexity.

## Decision

Use a single Qdrant collection with named vectors: `dense` (BGE-M3, 1024-d, cosine) and `bm25` (sparse with IDF modifier), plus payload indexes on `document_id`, `lang`, and `entity_ids`. Compute RRF in application code to control `k` and log per-leg ranks.

## Consequences

Simpler infrastructure with a single consistency boundary. Kazakh morphology is weaker under Qdrant BM25 than Elasticsearch analyzers; mitigated via text normalization and benchmarking BGE-M3 sparse vectors as an alternative.
