# 7. Source-Document Splits and Frozen Evaluation Sets

Date: 2026-11-18

## Status

Accepted

Amended by [19. Corpus Deduplication and Decontamination Policy](0019-dedup-and-decontamination-policy.md)

## Context

QA datasets attach multiple questions to the same context; random row-level splitting leaks identical passages across train and test, inflating metrics.

## Decision

Split `train`/`val`/`test` strictly by source document (Wikipedia article or context). Assemble and freeze test and cross-lingual eval suites before any training, versioned in DVC with SHA-256 manifests. Enforce exact-hash and near-duplicate overlap checks during `validate` and `build_eval`.

## Consequences

Reported metrics are honest (typically lower) and remain comparable across all subsequent experiments.
