# 4. Hybrid Retrieval Independent of NER

Date: 2026-11-18

## Status

Accepted

## Context

Extracted entities aid filtering and navigation, but NER misses (especially in Kazakh) would cause hard retrieval failures if used as a mandatory filter.

## Decision

Default retrieval executes `BM25 + dense → RRF → cross-encoder reranker`. Entities act solely as a soft boost or an explicit user-selected filter — never a mandatory default filter.

## Consequences

Retrieval recall is insulated from NER errors; Extraction's contribution is measured separately in Stage 9 ablation.
