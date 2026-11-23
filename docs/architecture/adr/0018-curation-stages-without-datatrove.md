# 18. Custom Sharded Runner for Curation Stages (Without datatrove)

Date: 2026-11-18

## Status

Accepted

Amends [11. Layered Parquet Data Lake with Manifests](0011-parquet-data-lake.md) (replaces `datatrove` for filtering and MinHash dedup)

## Context

Stage 2 revealed four mismatches with `datatrove`:
1. `clean`, `dedup`, and `decontaminate` require the same shard-unit, completion-marker, fingerprint, and `dataset.yaml` protocol as [ADR-0014](0014-sharded-batch-jobs.md) and [ADR-0017](0017-raw-layer-acquisition-protocol.md); `datatrove` executors enforce a separate state/layout protocol.
2. Cross-source dedup needs global connected components across all sources with source-priority root selection and audit retention of removed duplicates.
3. Line-level decontamination (ADR-0019) has no off-the-shelf `datatrove` block.
4. Domain/language-specific quality filters map cleanly to single-pass vectorized Polars expressions, whereas uncapped Polars thread pools per worker caused Windows commit-limit OOMs (3× RSS) until capped at `cpu_count // workers`.

## Decision

- Run `trifact.data.{clean,dedup,decontaminate}` on shared runner `trifact.data.curation`:
  - `StageDir`: SHA-256 fingerprint of stage name, params, LF-normalized code, and input manifests (cleans dir on change; skips finished units on match).
  - `run_units`: worker pool with unit markers and `POLARS_MAX_THREADS = cpu_count // workers`.
  - `UnitWriter`: atomic zstd Parquet writes; `stage_report`: drop counts, throughput, peak RSS.
- Compute quality features in Polars, paragraph langid via `fasttext-predict`, MinHash/n-grams/13-grams in NumPy (`trifact.data.hashing`: 32-bit affine permutations over mixed 64-bit hashes, ~17 MB/s per core), and connected components via `scipy.sparse.csgraph`.
- Remove unused `datatrove` dependency.

## Consequences

One unified resumability, fingerprinting, and manifest protocol across all data stages; peak memory governed by `curation.workers`; MinHash unbiasedness and quality filters verified by unit tests (`tests/data/test_hashing.py`, `test_quality.py`). Filter heuristics (Gopher/FineWeb) live as `params.yaml: curation.quality`.
