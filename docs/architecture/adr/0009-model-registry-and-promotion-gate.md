# 9. Model Registry and Metric-Driven Promotion Gate

Date: 2026-11-18

## Status

Accepted

Amended by [22. Evaluation Protocol](0022-evaluation-protocol.md)

## Context

The platform serves multiple models (NER, embedder, reranker, NLI, LLM adapter) and must guarantee that production only loads verified versions.

## Decision

Register all models in MLflow Model Registry. Services load alias `champion` on startup. New versions receive alias `challenger` and advance to `champion` only when the Evaluation Harness confirms `params.yaml: promotion` thresholds (`01-architecture.md §10`) on frozen test sets with bootstrap CI or ≥3 seeds. If fine-tuning fails to beat baseline, it is not promoted.

## Consequences

Every API response records model versions; rollback is an alias reassignment.
