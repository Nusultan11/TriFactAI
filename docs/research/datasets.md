# Dataset Catalog & Scale Plan

Markers: **✓** verified via HF datasets-server / API (2026-11-18); **●** measured during Stage 1–3 acquisition and reconciled in `data/raw/<source>/source.yaml`; **≈** card/paper estimate, unverified.

## 1. Text Corpora (RAG Index, Pretraining, Silver Annotation)

| Source | HF ID / Origin | Volume | Size | License |
|---|---|---|---|---|
| Wikipedia EN | `wikimedia/wikipedia` `20231101.en` | ✓ 6,407,814 articles | ✓ 11.6 GB parquet | CC BY-SA |
| Wikipedia RU | `wikimedia/wikipedia` `20231101.ru` | ✓ 1,945,063 articles | ✓ 4.9 GB | CC BY-SA |
| Wikipedia KK | `wikimedia/wikipedia` `20231101.kk` | ✓ 238,615 articles | ✓ 0.18 GB | CC BY-SA |
| FineWeb-2 KK | `HuggingFaceFW/fineweb-2` `kaz_Cyrl` | ✓ 3,375,998 docs | ✓ 6.9 GB | ODC-By |
| Multidomain Kazakh | `kz-transformers/multidomain-kazakh-dataset` | ● 24,883,808 CSV records (HF card/server: 15,510,000 partial) | ● 6.97 GB parquet | Apache-2.0 (`release: false`) |
| KazParC (`kk`/`ru`/`en`/`tr` parallel) | `issai/kazparc` | ● 15,103,193 rows (all configs, incl. synthetic) | ● 2.26 GB parquet | CC BY 4.0, gated |
| Wikidata | `dumps.wikimedia.org` JSON `20261116` | ● 122,308,973 entities (2026-11-19); `kk` labels 0.38%, `en` 76.4% | ● 103.9 GB bz2 → 5.29 GB parquet | CC0 |
| Wikipedia wikitext (full `kk`, `ru` 1–4, `en` 1–4) | `dumps.wikimedia.org` `20261101` `pages-articles` | ● articles `kk` 249,063, `ru` 790,327, `en` 216,913 + redirects `kk` 47,326, `ru` 795,340, `en` 119,478 (2026-11-23) | ● 5.38 GB Parquet, 24 shards | CC BY-SA 4.0 + GFDL |

## 2. NER

| Source | HF ID | Volume | Languages | License |
|---|---|---|---|---|
| KazNERD | `issai/kaznerd` | ● 112,702 sentences, 25 classes (✓ IOB2 verified) | `kk` | CC BY 4.0, gated |
| WikiANN | `unimelb-nlp/wikiann` | ✓ `kk` 3,000 · `ru` 40,000 · `en` 40,000 | `kk`/`ru`/`en` | Unknown → `research_only` |
| WikiNEuRal | `Babelscape/wikineural` | ✓ 1,027,506 (9 langs); acquired ● `en` 115,907, `ru` 115,440 | `ru`/`en` | CC BY-NC-SA 4.0 (`research_only`) |
| NEREL | `iluvvatar/NEREL` | ● 1,011 docs, ≈ 29 entity types | `ru` | Unknown → `research_only` |
| **Silver NER (Wikipedia)** | Constructed in-house | **Target: 5–10 M sentences** | `kk`/`ru`/`en` | CC BY-SA derivative |

Silver NER maps internal Wikipedia links → Wikidata QID → `P31`/`P279` class (`PER`/`ORG`/`LOC`), generating millions of labeled sentences across all three languages.

## 3. QA

| Source | HF ID | Volume | Language | License |
|---|---|---|---|---|
| SQuAD v2 | `rajpurkar/squad_v2` | ✓ 142,192 (incl. unanswerable) | `en` | CC BY-SA 4.0 |
| SberQuAD | `kuznetsoffandrey/sberquad` | ✓ 74,300 | `ru` | Unknown → `research_only` |
| KazQAD | `issai/kazqad` (`kazqad`, `nq-translate-kk`, `nq-original-en`) | ● `kk` 68,246 · `en` 68,770 | `kk`/`en` | CC BY-SA 4.0, gated |
| **Synthetic Grounded QA** | Teacher `Qwen3.5-35B-A3B` | **Target: 1–2 M filtered pairs** | `kk`/`ru`/`en` + cross-lingual | Derivative |

## 4. Retrieval

| Source | HF ID | Volume | Language |
|---|---|---|---|
| MS MARCO | `microsoft/ms_marco` | ● 1,112,939 query+passage rows (v1.1, v2.1) | `en` |
| mMARCO | `unicamp-dl/mmarco` (`russian`) | ● `ru` 9,758,627 (collection & queries), `multi` 39,780,811 (ID triples) | `ru` |
| MIRACL | `miracl/miracl` + `miracl/miracl-corpus` | ● queries/qrels `en` 43,952 · `ru` 54,585; corpus `en` 32,893,221 · `ru` 9,543,918 passages | `ru`/`en` |
| ru-HNP | `deepvk/ru-HNP` | ✓ 102,590 (hard negatives) | `ru` |
| bge-m3-data | `Shitao/bge-m3-data` | ≈ BGE-M3 training data | Multi |
| **Synthetic Queries** | Teacher-generated | **Target: 3–5 M `(query, chunk)`** | `kk`/`ru`/`en`, 9 directions |

## 5. NLI

| Source | HF ID | Volume | Language |
|---|---|---|---|
| XNLI | `facebook/xnli` | ✓ 400,202 per language (`ru`, `en`) | `ru`/`en` |
| XNLI-kk | Constructed (NLLB MT + QC filters) | Target: 400k | `kk` |

## 6. Scale Plan

| Layer | Target & Measured (`●`) Volume | Storage (Est. & Measured) |
|---|---|---|
| Raw corpora | Target ~27 M docs; ● 36.9 M (`multidomain_kk` = 24.9 M) | Est. ~40 GB; ● full raw layer 48.5 GB (19 sources incl. Wikidata + wikitext) |
| Curated (clean + dedup + decontam) | Target 60–80% of raw; ● 30.85 M docs = 84% (`kk` 19.86 M, `ru` 4.62 M, `en` 6.37 M; 2026-11-20) | Est. ~30 GB; ● corpus 23.7 GB |
| **RAG index** | **Target ~35 M chunks** (~20 M `en`, ~7 M `ru`, ~8 M `kk`); ● 59.63 M chunked (`kk` 29.59 M, `ru` 11.78 M, `en` 18.25 M; subset chosen in Stage 6) | fp16 vectors on disk ~72 GB; binary quant in RAM ~4.5 GB |
| NER | Target 5–10 M silver + ~1.3 M gold; ● silver 5.17 M (`kk` 0.23 M, `ru` 2.81 M, `en` 2.13 M — meets 5–10 M target), gold 0.44 M (2026-11-23) | ~5 GB |
| Retrieval pairs | Target 10–20 M; ● 15.05 M across 9 directions (meets target; `kk-kk` 86.6k; synthetic queries 102.0k) | ~10 GB |
| SFT grounded QA | Target 1–2 M; ● 338.7k (teacher synthetic QA 11.0k — `$0` budget / Kaggle quota deviation documented in `PROGRESS.md`) | ~5 GB |
| NLI | Target ~1.2 M; ● train 1.0 M (`en` 393k, `ru` 393k, `kk` 210k MT) | ~0.5 GB |
| Eval (frozen) | Target 30–50k (incl. 3.5k human-verified, 9 directions); ● `qa` 32.4k & `ner` frozen; `retrieval` 38.6k queries & `nli` 12.7k awaiting human review | — |
| Eval `docproc` (frozen, Stage 4) | ● 150 docs (18 × 8 kinds, 6 per `kk`/`ru`/`en`) + 6 mixed `kk`/`ru`/`en`; 239 PDF pages; 3 × 300-page PDFs; 17 hostile files (`clean/wikipedia`, CC-BY-SA-3.0) | 161 MB |

Index scale ladder (`RET-S`) and throughput estimates: [01-architecture.md §7.3, §9](../architecture/docs/01-architecture.md).

## 7. License Decisions (Stage 1, Checked 2026-11-18)

Stored in `params.yaml: sources.*.license` and copied to each `source.yaml` (ADR-0017). `allowed`: usable and releasable under license terms. `research_only`: train/eval only, derivatives inherit `release: false` (ADR-0015). `blocked`: never acquired.

| Source | License | Decision | `release` | Evidence |
|---|---|---|---|---|
| Wikipedia, MIRACL corpus | CC BY-SA 3.0 (+ GFDL) | `allowed` | yes | HF card, Wikimedia terms |
| Wikipedia wikitext (`20261101`) | CC BY-SA 4.0 + GFDL | `allowed` | yes | Wikimedia Terms of Use, `dumps.wikimedia.org/legal.html` |
| FineWeb-2 `kk` | ODC-By 1.0 + CommonCrawl ToU | `allowed` | yes | HF card |
| Multidomain Kazakh | Apache-2.0 (HF card) | `allowed` | **no** | Aggregates CC100, OSCAR, news, books; card does not clear third-party text |
| KazParC | CC BY 4.0 | `allowed` | yes | Missing on HF card; confirmed in arXiv:2403.19399 and ISSAI release page |
| KazNERD | CC BY 4.0 | `allowed` | yes | HF card |
| KazQAD | CC BY-SA 4.0 | `allowed` | yes | HF card |
| SQuAD v2 | CC BY-SA 4.0 | `allowed` | yes | HF card |
| MIRACL (topics, qrels) | Apache-2.0 | `allowed` | yes | HF card |
| ru-HNP | Apache-2.0 | `allowed` | yes | HF card (pairs generated by `gpt-3.5-turbo`) |
| Wikidata | CC0 1.0 | `allowed` | yes | `Wikidata:Licensing` |
| WikiANN | Unknown (text CC BY-SA) | `research_only` | no | HF card `unknown`; `github.com/afshinrahimi/mmner` code is Apache-2.0, Dropbox data unlicensed |
| NEREL | Unknown | `research_only` | no | No data license on HF or GitHub; RANLP 2021 paper is CC BY 4.0 |
| SberQuAD | Unknown | `research_only` | no | HF card `unknown`, TFDS "No known license" |
| WikiNEuRal | CC BY-NC-SA 4.0 | `research_only` | no | HF card |
| XNLI | CC BY-NC 4.0 | `research_only` | no | HF card |
| MS MARCO, mMARCO | MS MARCO terms (non-commercial research) | `research_only` | no | `microsoft.github.io/msmarco/Notice`; mMARCO inherits terms |

`research_only` for WikiANN, NEREL, or SberQuAD can be upgraded to `allowed` by editing `params.yaml` and rewriting the manifest once rights holders confirm licensing (no re-download needed).

## 8. External Evaluation-Only Benchmarks

| Suite | HF ID / Source | Purpose | Terms |
|---|---|---|---|
| LLM-AggreFact | `lytang/LLM-AggreFact` | Document-level claim support (11 English suites) | HF card: evaluation benchmark only — never use in training or pretraining (checked 2026-11-21) |

External benchmarks are excluded from train and corpus decontamination; they serve as auxiliary checks outside promotion gates.
