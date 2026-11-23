# 17. Raw Layer Acquisition Protocol: Tables, Row Checks, and Licenses

Date: 2026-11-18

## Status

Accepted

Amends [11. Layered Parquet Data Lake with Manifests](0011-parquet-data-lake.md)

## Context

Stage 1 sources span heterogeneous formats (Parquet, multi-line CSV, headerless TSV, JSONL/gz, Arrow IPC) and multi-table layouts (mMARCO `collection`/`queries`/`triples`, MIRACL `topics`/`qrels`) with splits encoded in filenames. HF dataset viewer omits or partially counts script/large CSV datasets. Wikidata dumps download at ~1–1.5 MB/s per connection (HTTP 429 at 4+ connections), requiring resumable downloads.

## Decision

- **Tables:** Name shards `data/raw/<source>/<lang>/<table>-NNNNN.parquet` (`table=part` for single-table sources; `lang=multi` for parallel/Wikidata). Enforce identical schema across all shards of each `(lang, table)`.
- **Provenance columns:** Append only `_config` and `_split` (`train`/`validation`/`test`; normalize `valid`/`val`/`dev` → `validation`), preserving all source content verbatim.
- **Source spec (`params.yaml: sources.<name>`):** Require pinned commit-SHA `revision` (reject branch names), `files` glob mappings, and `license` (`use: allowed | research_only | blocked`, `release`, `evidence`, `note`). Refuse acquisition if `blocked` or undecided.
- **HF protocol:** One repo file = one unit. Download to `data/.cache/downloads` (outside DVC outs), verify LFS SHA-256, stream-convert to `_work/`, emit `_work/<idx>.json` marker. On completion, atomically finalize into shard layout (resumable via `_work/finalize.json`), write `source.yaml`, and clean `_work/`. Re-runs at matching revision only verify the manifest.
- **Row verification (`row_checks`):** Apply highest-trust available check: `hf_dataset_info` (pinned README), exact `hf_datasets_server` (never `partial`), `parquet_footer`, or independent `line_count` / `csv_record_count`. Any mismatch fails the stage.
- **Wikidata:** Pin dated dump (`revision` = date). Download in 64 MB HTTP ranges (≤3 connections + mirrors) with resume state and published SHA-1 verification; parse deterministically in worker processes with `_progress/` markers and independent `entity_count` verification. *(Block-range bz2 parsing replaced `indexed_bzip2` in Stage 1 completion; see ADR-0020 & PROGRESS.)*
- **DVC & smoke runs:** Mark `acquire` and `wikidata` outs `persist: true` with `hardlink` cache. Smoke runs (`--limit-rows`, `--limit-entities`, `--root <tmp>`) flag `partial: true` and stay out of DVC.

## Consequences

Interrupted downloads lose at most active files or one 64 MB range. Every manifest row count has independent verification, and unlicensed sources cannot be acquired accidentally, at the cost of a second pass for independent row counts and temporary `_work/` disk space.
