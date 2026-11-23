# 3. BGE-M3 as Default Embedding Model

Date: 2026-11-18

## Status

Accepted

## Context

TriFact requires a multilingual embedder supporting Kazakh, long context, and cross-lingual retrieval that fits within a 4 GB local serving budget alongside the LLM.

## Decision

Adopt `BAAI/bge-m3` (dense + sparse in one model, 8,192 tokens) as default champion, with `intfloat/multilingual-e5-large` as initial challenger. Final selection is governed by `Recall@10` / `nDCG@10` across all 9 language directions (`RET-1`, `RET-2`).

## Consequences

Model choice remains metric-driven; the `Embedder` interface decouples application code from the concrete model.
