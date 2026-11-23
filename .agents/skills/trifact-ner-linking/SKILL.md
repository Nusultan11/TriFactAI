---
name: trifact-ner-linking
description: Cloud-first NER and entity linking for KK/RU/EN in TriFact — PER/ORG/LOC schema with rule-based DATE, subword alignment, XLM-R gold/silver curriculum, GLiNER baseline, partial-annotation noise, Kazakh case suffixes, unseen-entity F1, strict/partial span metrics, calibration, ONNX int8 and vocab trimming, Wikidata KB linking, sharded cloud annotate_corpus. Use for stage 5 work in trifact.extraction, train_ner, eval_ner, build_kb, or annotate_corpus.
---

# TriFact AI — NER & Entity Linking

## Scope

- Load for NER training/eval/optimization (NER-0…NER-5), DATE extraction, Wikidata KB + Entity Linking, and sharded `annotate_corpus`.
- General cloud training/HPO/seeds → `trifact-training-hpo`. Statistics & gate → `trifact-evaluation`.
- Cloud CLI execution & persistence → [cloud-compute.md](../trifact-mlops/references/cloud-compute.md).

## Owner Documents

| Topic | Owner |
|---|---|
| Label schema & rule-based DATE | [ADR-0006](../../../docs/architecture/adr/0006-unified-ner-label-schema.md), `configs/labels.yaml` |
| NER as soft signal only | [ADR-0004](../../../docs/architecture/adr/0004-hybrid-retrieval-independent-of-ner.md) |
| Silver data & curriculum | [ADR-0015](../../../docs/architecture/adr/0015-silver-and-synthetic-data.md) |
| Experiments, KB, gate | [01-architecture.md §7.2, §10](../../../docs/architecture/docs/01-architecture.md), `params.yaml` (`ner`, `linking`, `promotion`) |
| Models & silver noise notes | [models.md §1](../../../docs/research/models.md), [PROGRESS](../../../docs/PROGRESS.md) |

## Cloud Execution

- **Training (`train_ner`):** Run NER-1…NER-4 on **Colab L4 (bf16)** preferred, or **Kaggle 2× T4 (fp16 AMP, 1 process/GPU)** as secondary option. Pull only `data/task/ner` and `data/eval/ner`. Persist best/last checkpoints and MLflow SQLite snapshots to Drive.
- **Corpus annotation (`annotate_corpus`):** Run as a sharded, idempotent batch job (`--shard i/N`, `_SUCCESS` markers) on cloud GPU (fp16) or cloud CPU (ONNX int8) over `data/task/chunks`. Push `data/task/annotations` shards to Drive/DVC incrementally.

## Data & Alignment

- Assign the entity label to the **first subword** of each word and `-100` to continuation subwords (labeling all subwords overweights agglutinative `kk` tokens).
- Split sentences exceeding `ner.max_length` into overlapping windows instead of truncating.
- Kazakh case/possessive suffixes attach to the surface token (`Астанада` = `B-LOC`). Strip suffixes (`-да/-де/-та/-те`, `-ның/-нің/-дың/-дің/-тың/-тің`, `-ға/-ге/-қа/-ке`, `-ны/-ні/-ды/-ді/-ты/-ті`, possessives) during linking candidate generation and measure candidate recall before/after.

## Training Protocol

- **Baselines:** GLiNER multi v2.1 zero-shot (NER-0) and XLM-R base on gold (NER-1).
- **Curriculum (NER-2):** 1–2 epochs on silver → gold at lower LR (or down-weighted mix); select by dev `kk` strict F1.
- **Handling silver false-`O` noise** (cheapest → strongest):
  1. `ner.silver.capital_filter` (existing).
  2. Partial annotation: mask unlabeled capitalized tokens with `-100`.
  3. Self-training: label silver with NER-1 above confidence threshold.
- **Class weights / CRF:** Skip class weights unless per-type recall shows imbalance. Treat CRF head as an optional ≥3-seed experiment.
- Select checkpoints by dev strict entity F1 (never loss); run ≥3 seeds (`ner.seeds`).

## Evaluation & Error Taxonomy

- Primary metric: `seqeval` **strict IOB2** (`mode="strict"`, `scheme=IOB2`) per `lang × type`, paired cluster bootstrap by document.
- Separate boundary vs. type errors with `nervaluate` (`strict`/`exact`/`partial`/`ent_type`).
- Track **unseen-F1** (surfaces absent from train), especially for `kk`.
- Track error categories: ORG↔LOC metonymy, nested spans (`Алматы университеті`), initials/titles in PER, sentence-initial capitalized false PER, transliteration (`Абай`/`Abai`), mixed-language sentences.

## Calibration & Speedup (NER-5)

- Score entities by mean or min token probability; fit per-language temperature scaling on dev; verify with equal-mass ECE and reliability diagram.
- Pick `annotate_corpus` threshold on the precision–coverage curve (prefer higher precision for Qdrant payload filters).
- Optimize: ONNX export (`optimum`), dynamic int8 (CPU) / fp16 (GPU), XLM-R vocabulary trimming to `kk`/`ru`/`en` corpus tokens (~192M/279M params are embeddings), length-sorted batching. Verify F1 drop ≤ NER-5 tolerance and log `chunks/s`.

## Entity Linking

1. **KB (`build_kb`):** Wikidata entities with `kk`/`ru`/`en`/`mul` labels, aliases, redirects, and Wikipedia anchor frequencies → prior $p(e \mid m)$.
2. **Candidates:** Exact normalized alias (`kb_aliases` B-tree) + Kazakh suffix stripping + `pg_trgm` fallback. Measure **candidate recall@k** separately (system ceiling).
3. **Ranking:** `prior × NER/KB type compatibility × multilingual context–description similarity` (critical for `kk`, where Wikidata `kk` label coverage is <1% and descriptions are `ru`/`en`).
4. **NIL & eval:** Tune NIL threshold on dev; evaluate `accuracy@1`, candidate `recall@k`, NIL rate, and low-sitelink (tail entity) accuracy on human-labeled per-language sets. Escalate to BLINK / mGENRE / ReFinED only if candidate recall is high and ranking bottlenecks.

## Done Criteria

- Strict entity F1 exceeds baseline with CI across languages/types and passes `promotion.ner_*` gate (≥3 seeds, mean ± CI).
- Unseen-F1, error taxonomy, and CPU/GPU throughput logged in MLflow `ner`.
- Linking reports candidate `recall@k`, `accuracy@1`, and NIL rate per language.

## Sources

- Conneau et al., *XLM-R* (arXiv:1911.02116); Zaratiana et al., *GLiNER* (arXiv:2311.08526); Yeshpanov et al., *KazNERD* (arXiv:2111.13419).
- Augenstein et al. (arXiv:1701.02877); Mayhew et al. (CoNLL 2019); Abdaoui et al. (arXiv:2010.05609); Guo et al. (arXiv:1706.04599).
- Wu et al., *BLINK* (arXiv:1911.03814); De Cao et al., *mGENRE* (arXiv:2103.12528); Ayoola et al., *ReFinED* (arXiv:2207.04108); `seqeval`; `nervaluate`.
