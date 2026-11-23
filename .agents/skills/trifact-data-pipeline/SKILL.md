---
name: trifact-data-pipeline
description: Cloud-first data engineering for TriFact corpora and task datasets at tens of millions of rows — Parquet lake and manifests, normalization, language ID, quality filters, MinHash dedup, 13-gram decontamination, leakage-safe splits, gold/silver/synthetic/MT sets, license lineage, DVC/Drive streaming, sharded cloud CPU and GPU batch jobs. Use when adding a source, writing or debugging trifact.data/lake/batch stages, building training or eval sets, or fixing OOM, resumability, or leakage.
---

# TriFact AI — Data Pipeline

## Scope

- Load when adding a raw source, writing/debugging `trifact.data`, `trifact.lake`, or `trifact.batch`, building task/eval sets, or fixing OOM, resumability, or leakage.
- Distribution analysis → `trifact-eda`. User file parsing → `trifact-document-processing`.
- Cloud CLI execution & DVC/Drive sync → [cloud-compute.md](../trifact-mlops/references/cloud-compute.md).

## Owner Documents

| Topic | Owner |
|---|---|
| Layers, shards, manifests | [ADR-0011](../../../docs/architecture/adr/0011-parquet-data-lake.md), [ADR-0017](../../../docs/architecture/adr/0017-raw-layer-acquisition-protocol.md), [01-architecture.md §5](../../../docs/architecture/docs/01-architecture.md) |
| Stage runner, fingerprints, markers | [ADR-0018](../../../docs/architecture/adr/0018-curation-stages-without-datatrove.md), `trifact.data.curation` |
| Sharded batch jobs & cloud compute | [ADR-0014](../../../docs/architecture/adr/0014-sharded-batch-jobs.md), [ADR-0020](../../../docs/architecture/adr/0020-kaggle-batch-gpu-mt-and-teacher.md), [ADR-0025](../../../docs/architecture/adr/0025-cloud-only-computing.md) |
| Dedup & decontamination | [ADR-0019](../../../docs/architecture/adr/0019-dedup-and-decontamination-policy.md) |
| Splits & frozen eval | [ADR-0007](../../../docs/architecture/adr/0007-leakage-safe-splits.md), `trifact.data.split`, `trifact.data.build_eval` |
| Silver, synthetic, licenses | [ADR-0015](../../../docs/architecture/adr/0015-silver-and-synthetic-data.md), [datasets.md §7](../../../docs/research/datasets.md) |
| Parameters & traps | `params.yaml` (`lake`, `curation`, `split`, `teacher`, `mt`, `sft`, `scale`), [PROGRESS](../../../docs/PROGRESS.md) |

## Cloud Execution & Stage Checklist

1. **Contract:** Define input manifests, output schema (`01-architecture.md §5.2`), unit of work (input shard), namespaced keys (`<source>/<lang>/<id>`), and drop reasons. Reuse `curation.run_units`, `StageDir`, `UnitWriter`, or `trifact.batch.*`.
2. **Memory & disk budget:** Estimate `bytes_per_row × batch × workers` before launch. Size batches by bytes (`curation.rows_per_batch`), not row count (wikitext rows are tens of KB). Set `POLARS_MAX_THREADS = cpu_count // workers`.
3. **Cloud CPU stages (`[REMOTE]`):** Pull only required DVC inputs, run `scale.profile: pilot` or `--limit` first, check manifest, drop reasons, and peak RSS (`lake.stats.ResourceMonitor`), then run full stage from a clean Git SHA and push outputs immediately (`dvc push <stage>` + `dvc status --cloud`).
4. **Cloud GPU batch jobs (MT, teacher, NLI scoring):** Prefer Colab L4 via `trifact.batch.colab` (bounded shard, Drive checkpointing) or Kaggle 2× T4 via `trifact.batch.kaggle` (one process per GPU, `CUDA_VISIBLE_DEVICES`, fp16 only). Verify quota and `cloud.budget_usd` first.
5. **Outputs:** `dataset.yaml` (inputs, fingerprint, row counts, drop reasons, licenses, `release` flag), `card.md`, `reports/<stage>.json`, and MLflow run (snapshot SQLite to Drive on ephemeral VMs).

## Tool Selection

| Task | Tool | Guardrail |
|---|---|---|
| Row features & filters per batch | Polars expressions over PyArrow `iter_batches` | Filtered views pin source buffers; materialize or write immediately |
| Large joins, group-by, intersections | DuckDB (`memory_limit`, `temp_directory`) | Out-of-core spill; verify peak RSS before switching to Polars streaming |
| Key-based subsetting of large shards | PyArrow `dataset` + `is_in` (`task_common.rows_with_keys`) | Row-group pushdown; avoids Polars lazy semi-join RAM blowup |
| Atomic shard writing | PyArrow / `UnitWriter` (`zstd`, `.tmp` → fsync → rename) | Resumable, crash-safe |
| MinHash, n-grams, hashing | NumPy (`trifact.data.hashing`) | 64-bit mixed hashes with Jaccard unbiasedness tests |
| Dedup clusters | `scipy.sparse.csgraph.connected_components` | Global cross-source pass |
| Schema validation | Pandera (Polars) on batches/samples | Catches cross-shard schema drift |

Never use `pandas` on large corpora, never load full datasets into RAM, and do not reintroduce `datatrove` (ADR-0018).

## Curation Rules

- **Normalize first:** Run `trifact.ingestion.normalize` identically online and offline before hashing, langid, dedup, or decontamination. Handle Kazakh letters (`ә ғ қ ң ө ұ ү һ і`) and Latin/Cyrillic homoglyphs.
- **Language ID:** Score document and paragraphs separately. Use Kazakh-letter share (`curation.downstream`) as a secondary signal. Re-evaluate `curation.langid.candidates` (`eval_langid`) if `ky`/`tt`/`uz` contamination appears.
- **Quality filters:** Resolve thresholds `default → language → domain` from human review, not guesswork. Record every drop reason and audit `raw → clean` shifts so `kk` or whole domains are not disproportionately dropped.
- **Dedup:** MinHash-LSH link probability for Jaccard `s` with `b` bands × `r` rows is $1 - (1 - s^r)^b$; verify with exact Jaccard on sampled pairs (`pair_check`). Keep templated Wikipedia stubs distinct within Wikipedia (ADR-0019). Reserve SemDeDup for synthetic queries/SFT, not 30M+ corpora.
- **Decontamination:** Strip lines matching eval 13-grams; drop document if match spans lines or >50% chars removed; ignore template grams (>100 docs); re-scan must yield 0 leaks.

## Task Dataset Rules

- **Leakage-safe splits:** Split by source document (`doc_key`) and verify `ctx_key` (context hash) separately. Keep translations and parallel pairs (KazParC) on the same split side.
- **Gold:** Validate BIO sequences, `answer_start` alignment, and `configs/labels.yaml` mapping.
- **Silver (ADR-0015):** Estimate noise against gold on shared surfaces. Primary defect is missed entities (false `O`); apply `ner.silver.capital_filter` and silver→gold curriculum.
- **Synthetic:** Order filters cheap→expensive: `parse → length → language → copy → round-trip → duplicate → NLI`. Log drop rate per filter, check diversity (distinct-n, question prefixes, topic coverage), and plan volume from measured throughput and quota. Never put unreviewed synthetic rows in eval.
- **MT (NLLB, ADR-0020):** Filter by back-translation chrF, langid, length ratio, and non-copy. Inherit `release: false`.
- **Mixes:** Enforce explicit language shares (`scale.min_kk_share`, `sft.mix`) and temperature sampling ($p \propto n^\alpha$, $\alpha = 0.3$); prevent short parallel sentences from dominating cross-lingual retrieval passages.

## Common Failures

| Symptom | Cause | Fix |
|---|---|---|
| OOM on some sources | Row-counted batches on long texts; uncapped Polars threads | Byte-sized batches, cap `POLARS_MAX_THREADS`, reduce workers |
| `dvc.lock` hash mismatch in clean clone | CRLF line endings | Run `tests/test_architecture.py::test_dvc_dependencies_use_lf_line_endings` |
| Duplicate IDs across languages | Non-unique source IDs (e.g., MIRACL `docid`) | Namespace IDs (`<source>/<lang>/<id>`) |
| Suspiciously high eval metric | Shared context or translation in train | Check `ctx_key`, normalized text hashes, `build_eval leakage` |
| Re-run recomputes everything | Unstable fingerprint input (timestamp/path) | Hash only stage name, params, LF code, and input manifests |
| Stage ran in `pilot` unintentionally | Default `scale.profile` | Verify `profile: full` in manifest and MLflow |

## Done Criteria

- `python -m trifact.lake verify` passes; row counts match source check methods.
- Sum of drop reasons equals input − output; reasons documented in `card.md`.
- Eval ↔ train leakage = 0 by document, context, and normalized text.
- Peak RSS ≤ `lake.peak_ram_gb`; throughput logged to manifest and MLflow.
- Re-running the stage is a no-op (idempotent).
- All input licenses and `release` flags recorded; no unlicensed source used.

## Sources

- Penedo et al., *FineWeb* (arXiv:2406.17557), *FineWeb2* (arXiv:2506.20920); Rae et al., *Gopher* (arXiv:2112.11446).
- Lee et al., *Deduplicating Training Data* (arXiv:2107.06499); Abbas et al., *SemDeDup* (arXiv:2303.09540); Brown et al., *GPT-3* App. C (arXiv:2005.14165).
- Conneau et al., *XLM-R* (arXiv:1911.02116); Gebru et al., *Datasheets* (arXiv:1803.09010); Pushkarna et al., *Data Cards* (arXiv:2204.01075).
