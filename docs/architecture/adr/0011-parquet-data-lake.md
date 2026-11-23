# 11. Layered Parquet Data Lake with Manifests

Date: 2026-11-18

## Status

Accepted

Amended by [18. Custom Sharded Runner for Curation Stages](0018-curation-stages-without-datatrove.md)

## Context

Corpora span tens of millions of documents (~40 GB raw Parquet) and task sets span millions of rows, exceeding 15 GB RAM. Monolithic CSV/JSONL files cannot be processed in-memory, partially rebuilt, or hash-verified per source, and each source requires independent license, revision, and schema tracking.

## Decision

- Store data under `data/` as `zstd`-compressed Parquet shards (~1M rows or ≤512 MB): `data/<layer>/<source>/<lang>/part-<NNNNN>.parquet`.
- Layers: `raw` (immutable post-acquisition) → `clean` (normalize, langid, quality filters) → `curated` (dedup, decontamination) → `task` (`ner`/`retrieval`/`sft`/`nli`/`chunks`/`annotations`/`vectors`) → `eval` (frozen, ADR-0007).
- Every raw source carries `data/raw/<source>/source.yaml` (HF ID, revision SHA, license decision, download date, per-lang rows/bytes, shard SHA-256s). Every derived dataset carries `dataset.yaml` (input manifests, git SHA, params, row counts) and `card.md`.
- Process via streaming columnar engines: Polars lazy/streaming, DuckDB (joins/aggregates), PyArrow (atomic shard writes); never use `pandas` on large sets. *(Original decision included `datatrove`, subsequently amended by ADR-0018.)*
- DVC tracks shard directories rather than individual files.

## Consequences

Any layer can be partially rebuilt and hash-verified by source/language, bounded by disk and CPU rather than RAM. Schema changes require a backed-up migration ([AGENTS.md §6](../../../AGENTS.md)).
