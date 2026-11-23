---
name: trifact-eda
description: Cloud-scalable EDA for TriFact corpora and task datasets in KK/RU/EN — schema integrity, length tails in target tokens, language/script mix, encoding damage, tokenizer fertility, imbalance, label noise (confident learning, cartography, agreement), duplicates, train/eval shift, annotation artifacts, OCR quality; every finding maps to a decision. Use before training or freezing a dataset, when profiling a source, auditing NER/retrieval/QA/SFT/NLI/synthetic data, or diagnosing metric drops via data.
---

# TriFact AI — EDA

Every EDA metric must inform a concrete decision (filter, threshold, `max_length`, mix weight, or model choice). Skip metrics that cannot change a decision.

## Scope

- Load before setting curation thresholds, before training, before freezing eval, or when diagnosing a language/metric drop.
- Filter implementation → `trifact-data-pipeline`. Model error analysis → `trifact-evaluation`.
- Cloud CPU execution & DVC/MLflow persistence → [cloud-compute.md](../trifact-mlops/references/cloud-compute.md).

## Project Entry Points

- Commands (`[REMOTE]` on cloud CPU for full layers; `[LOCAL]` only for pilot/unit checks):
  - `uv run --no-sync python -m trifact.data.eda raw|clean|curated|entities`
  - `uv run --no-sync python -m trifact.data.task_eda`
  - `uv run --no-sync python -m trifact.data.eda_report` → `reports/eda/report.md`, `findings.json`, MLflow `eda`
- Add new metrics as mergeable streaming aggregates (counters, fixed-bin histograms) inside these modules, not one-off scripts.
- Coverage requirements: [ROADMAP 2.6, 3.7](../../../docs/ROADMAP.md). Existing decisions: [PROGRESS](../../../docs/PROGRESS.md).

## Cloud-Scalable Method

1. **Pull selectively:** Pull only the target layer (`dvc pull data/curated/corpus` etc.) on cloud CPU; check disk and RAM (`lake.peak_ram_gb`) before running. Cap workers (e.g., 2 workers on ~12–16 GB RAM) and keep byte-bounded batches (~32 MB).
2. **State decisions first:** List which thresholds or design choices depend on the run.
3. **Full volume vs. deterministic sample:** Compute exact streaming aggregates (counts, log-bin histograms → quantiles) over 100% of rows. Run expensive checks (vocabulary, Zipf, n-grams, tokenizer fertility) on a seeded hash sample of keys — never the head of a shard.
4. **Mandatory slices:** `source × language × domain × split` (+ `origin`: gold/silver/synthetic/mt for task sets). Never average across languages without per-language breakdowns.
5. **Attach examples:** Include 5–20 concrete examples for every tail or anomaly; unverified numbers are not actionable.
6. **Finding → Decision:** Record `{id, evidence (value/share/CI), affected_share, decision, owner_doc}`. Prioritize by `affected_share × metric_impact`.
7. **Audit `raw → curated`:** Measure distribution shifts introduced by curation (dropped domains, languages, lengths).
8. **Persist:** Push JSON profiles/reports and snapshot MLflow SQLite to Drive before stopping the cloud runtime.

## Mandatory Baseline Profile

| Block | Measure | Typical Decision |
|---|---|---|
| Integrity | Types, nulls/empties, constant cols, duplicate IDs, cross-shard schema drift | Namespace keys, fix loader |
| Lengths | p1…p99.9 in chars, words, and **target-model tokens**; share > `max_length` | `max_length`, chunking, length filter |
| Language & script | `kk`/`ru`/`en` doc & paragraph shares, langid confidence, script mix, Kazakh-letter share, translit | Langid threshold, `kk` filter, mojibake drop |
| Encoding | `U+FFFD`, control chars, mojibake (`Ð`, `Ñ`, `Ã`, `â€`) | Re-encode or drop |
| Duplicates | Exact & near clusters, cross-source overlap matrix | Dedup, source priority/reweighting |
| Sensitive content | Email, phone, checksum-valid IIN, Luhn cards, toxicity | `curation.downstream.pii_mask` |
| Tokenization | Fertility (tokens/word) per model & language, `<unk>` / byte fallback | Context budget, model choice |

Always measure lengths in target-model tokens: whitespace word counts severely undercount agglutinative `kk` and inflected `ru`.

## Label Noise, Artifacts & Shift

- **Confident learning** (`cleanlab.find_label_issues`): compute k-fold out-of-fold probabilities on train; manually audit top 100–200 flagged items and extrapolate noise rate.
- **Dataset cartography:** track per-epoch gold-label confidence and variability (low confidence + low variability = mislabeled; high variability = hard).
- **Annotator agreement:** Cohen's $\kappa$ (2 raters) or Krippendorff's $\alpha$ (multi-rater); entity-level for NER.
- **Annotation artifacts:** run hypothesis-only baseline (NLI), question-only / lexical overlap (QA), and BM25-easiness (retrieval). Scores far above chance indicate shortcut learning.
- **Distribution shift:** Jensen–Shannon divergence (`task_eda.js`), PSI, and classifier two-sample test (TF-IDF or embeddings; AUC ≈ 0.5 means low shift). Check train ↔ eval entity/question/doc overlap (ADR-0007) and unseen-entity rate.

Domain-specific checklists (corpora, OCR, NER, EL/KB, retrieval, QA/SFT, NLI, synthetic) → [references/task-checklists.md](references/task-checklists.md).

## Common Errors

| Error | Risk | Fix |
|---|---|---|
| Mean instead of quantiles | Heavy tails hide real distribution | Report quantiles, ECDF, log scale |
| Sampling shard head | Source order biases topic/date | Seeded hash of `doc_id` |
| Global average without language split | `kk` defects hidden by `en` volume | Per-language tables/small multiples |
| Finding without decision | Wasted EDA | Record decision in `PROGRESS.md` and `params.yaml` |
| Eyeballing threshold from histogram | Unverified precision/recall | Calibrate on human review sample |

## Done Criteria

- Baseline profile + task checklist computed for every `source × lang × split`.
- Every finding has evidence, examples, affected share, and a recorded decision (`params.yaml` / `PROGRESS.md`).
- JSON profiles in `reports/eda/`, report and figures logged in MLflow (`eda`) and persisted to Drive.

## Sources

- Northcutt et al., *Confident Learning* (arXiv:1911.00068); Swayamdipta et al., *Dataset Cartography* (arXiv:2009.10795).
- Gururangan et al. (arXiv:1803.02324); Poliak et al. (arXiv:1805.01042).
- Lopez-Paz & Oquab, *Classifier Two-Sample Tests* (arXiv:1610.06545); Rabanser et al., *Failing Loudly* (arXiv:1810.11953).
