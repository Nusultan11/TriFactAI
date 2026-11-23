# 19. Corpus Deduplication and Decontamination Policy

Date: 2026-11-18

## Status

Accepted

Amends [7. Source-Document Splits and Frozen Evaluation Sets](0007-leakage-safe-splits.md) (defines train corpus decontamination against eval sources)

## Context

Stage 2 pilot runs (`wikipedia`, `fineweb2_kk`, `multidomain_kk`) showed:
- 16×8 MinHash-LSH (midpoint $J \approx 0.70$) removed 1.45% of docs with median exact Jaccard `0.65` (500 pairs), driven by ~1.1% templated Wikipedia stubs (`"X — село в …"`, `"X is a moth of the family …"`) describing distinct entities.
- Document-level 13-gram decontamination dropped 3.2% of corpus (4.7% of Wikipedia) because KazQAD/SberQuAD/SQuAD/MIRACL contexts come from Wikipedia, and template 13-grams (`"… is a species of …"`) match hundreds of docs.
- MinHash cannot detect paragraph-level eval overlap inside long documents (low whole-doc Jaccard).

## Decision

**Deduplication (`trifact.data.dedup`, `params.yaml: curation.minhash`):**
- Exact duplicates: matching hash of lowercased, whitespace-collapsed text.
- Near-duplicates: 14 bands × 9 rows (126 permutations, midpoint $J \approx 0.75$) over word 5-grams; global connected components across all sources.
- Keep highest-priority source (`curation.sources` order), then earliest shard.
- Exception `keep_near_within: [wikipedia]`: retain near-duplicates whose cluster root is also in Wikipedia (distinct entity stubs); still remove exact duplicates and cross-source Wikipedia copies.
- Write removed docs to `removed` (`kept_doc_id`, `kept_source`); log exact Jaccard on 500 random pairs (`pair_check`).

**Decontamination (`trifact.data.decontaminate`, `curation.decontamination`):**
- Index normalized word 13-grams across all splits of eval sources (MIRACL passages with `relevance > 0`).
- Ignore template grams appearing in `> max_gram_docs` (100) corpus docs; log ignored templates in manifest.
- Strip lines containing any eval 13-gram; drop entire document if a gram spans line breaks or `> max_removed_share` (50%) of chars are removed.
- Write removed/edited docs to `contaminated`. Re-scan output; any surviving eval 13-gram fails the stage. Do not use MinHash against eval.

## Consequences

Pilot results: dedup removed 0.31% (median Jaccard `0.83`, 98.4% ≥ `0.5`, caught 148 Wikipedia copies in `fineweb2_kk`); decontamination dropped 0.71% and edited 2.5% with 0 post-scan leaks. Retained Wikipedia stubs carry `minhash_cluster` and `dup_count` for optional down-weighting. Parameter changes require a new `pair_check` and drop-rate measurement.
