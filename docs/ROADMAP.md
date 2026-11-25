# TriFact AI — Roadmap

One stage = one milestone ([AGENTS.md](../AGENTS.md)). A stage closes only when **all** completion criteria are satisfied. Experiment IDs (`DOC-`, `NER-`, `RET-`, `GEN-`, `VER-`) are defined in [01-architecture.md §7](architecture/docs/01-architecture.md), model specs in [models.md](research/models.md), and target volumes in [datasets.md](research/datasets.md).

| # | Stage | Primary Deliverable |
|---|---|---|
| 0 | Project Foundation | Agent instructions, docs, git, backups |
| 1 | Data Platform & Acquisition | All sources in Parquet data lake with manifests, DVC + Drive recovery |
| 2 | Data Quality & Curation | Cleaned, deduplicated, decontaminated corpora, quality reports, full EDA |
| 3 | Task Datasets: Gold, Silver, Synthetic | Multi-million-row task datasets and frozen eval suites |
| 4 | Document Processing Engine | Multi-format parser & OCR router with quality/speed benchmark |
| 5 | Extraction | Champion NER and Entity Linking across KK/RU/EN |
| 6 | Retrieval at Scale | ~35M-chunk index, champion embedder and reranker |
| 7 | Generation | Champion grounded QA LLM across KK/RU/EN |
| 8 | Evidence Verification | Claim verifier with calibrated per-language thresholds |
| 9 | System Integration & Serving | Online pipeline, load tests, prompt-injection defense, ablation |
| 10 | Product, Ops & Final Evaluation | Web UI, monitoring, deployment, `real_docs` transfer, final report |

---

## Stage 0 — Project Foundation

1. Agent instructions (`AGENTS.md`; superseded earlier `CLAUDE.md` wrapper).
2. Verified model and dataset catalogs.
3. Architecture, ADRs, and roadmap sized for tens of millions of records.
4. Progress log (`docs/PROGRESS.md`).
5. Backup script, `main` branch on remote `origin`, initial commit, and snapshot.

**Done when:** Documents are consistent and non-overlapping; initial commit, push, and backup exist.

## Stage 1 — Data Platform & Acquisition

1. Environment: `uv.lock`, `ruff`, `pytest`, pre-commit; `docker compose` (Postgres, Qdrant, Redis, S3, MLflow) starts and passes healthchecks.
2. `trifact.lake`: layer layout and manifests per ADR-0011.
3. Streaming, parallel, resumable HF → Parquet loader with metadata row-count verification.
4. Acquire all sources declared in `params.yaml: sources`.
5. Stream-parse Wikidata JSON dump into `entities` without loading the dump into RAM.
6. Record license decision in each source manifest before use.
7. DVC: track shard directories, sync with Google Drive, and verify clean cloud recovery per [ADR-0025](architecture/adr/0025-cloud-only-computing.md).
8. Report volumes by source/language, download/parse throughput, and peak RSS.

**Done when:** Every source resides in `raw` with a valid manifest, row counts match metadata, `dvc status` is clean, loader tests pass, and Drive recovery is verified in an independent cloud runtime.

## Stage 2 — Data Quality & Curation

1. Vectorized text normalization (`01-architecture.md §5.3`) with edge-case tests.
2. Language ID: benchmark `lid.176` vs. `GlotLID` on reviewed paragraphs, select by macro-F1, tag documents and paragraphs.
3. Quality filters with thresholds calibrated on a human-reviewed sample.
4. Deduplication: exact hash + MinHash-LSH, within and across sources.
5. Decontamination of train corpora against all eval sources.
6. **Full streaming EDA** of every `raw` source (including `entities`) and `curated` layer across `source × language × domain × split` (run `raw` EDA before tuning step 2.3; `curated` EDA after 2.5):
   - **Schema & integrity:** dtypes, nulls/empties, constants, broken encodings/control chars, duplicate IDs, cross-shard schema drift.
   - **Volumes:** documents, paragraphs, chars, words, tokens; source and language shares.
   - **Lengths:** char/word/token/paragraph quantiles, tails, extreme examples.
   - **Language & script:** `kk`/`ru`/`en` doc & paragraph shares, langid confidence, intra-doc mixing, Cyrillic vs. Latin, Kazakh-letter share (`ә ғ қ ң ө ұ ү һ і`), transliteration.
   - **Lexicon:** frequency spectra, Zipf slope, rare-word share, boilerplate (repeated lines, nav, cookie banners).
   - **Quality signals:** filter feature distributions before/after filtering, correlations, alignment with human review.
   - **Duplicates:** exact/near cluster sizes, cross-source overlap matrix, eval overlap (input to 2.5).
   - **Domains & time:** domain/URL/TLD and publish-date distributions where available.
   - **Tokenizers:** XLM-R / Qwen3 / Qwen3.5 fertility by language/domain, `<unk>` and byte fallback, token length vs. model context limits.
   - **Sensitive content:** PII (email, phone, IIN) and toxicity rates by source.
   - **`entities` (Wikidata):** type distribution, `kk`/`ru`/`en` label/description coverage, sitelinks, aliases, graph connectivity.
   - **`raw → curated`:** distribution shifts and curation bias (dropped domains, languages, lengths).
   - **Decisions:** every finding maps to a concrete action in `PROGRESS.md` and numeric values in `params.yaml`.
7. Data card (`card.md`) for every curated dataset.

**Done when:** Curated layer is built, drop rates per reason are reported, filters are validated (full human review or autonomous dual-signal machine cross-validation with report-level human governance per [ADR-0026](architecture/adr/0026-calibrated-human-audit-validation-protocol.md), reporting per-language Wilson 95% CIs, disagreements, and `human: false` provenance), full `raw` and `curated` EDA covers all 2.6 items and is logged in MLflow (tables + figures), and every finding has a recorded decision.

## Stage 3 — Task Datasets: Gold, Silver, Synthetic

1. **NER:** normalize gold sets to unified schema (ADR-0006); build silver from Wikipedia links × Wikidata (ADR-0015); estimate silver noise against gold.
2. **Retrieval:** build pairs from public sets + KazParC; generate synthetic teacher queries with round-trip filtering; mine hard negatives.
3. **Grounded QA (SFT):** teacher synthetic QA (answerable, cross-lingual, unanswerable) with ADR-0015 filters.
4. **NLI:** XNLI `ru`/`en` + filtered `kk` MT.
5. **Eval:** build and freeze per ADR-0007 and [ADR-0026](architecture/adr/0026-calibrated-human-audit-validation-protocol.md) across all 9 language directions, with SHA-256 hashes in manifest (3,500 dual-signal cross-validated MT items with report-level human governance and explicit `human_verified` vs. `machine_cross_validated` row provenance); silver NER noise audit (600 cross-validated sentences with per-language bootstrap CIs).
6. Fix cloud generation budget and cost in `params.yaml` before launch.
7. **Full task-set EDA** across `language × source × split × origin` (`gold`, `silver`, `synthetic`, `mt`, `eval`):
   - **NER:** label distribution, span lengths/density, nested/overlapping spans, schema violations, silver vs. gold type confusion.
   - **QA:** question/context/answer lengths, answer position, unanswerable share, question types, lexical overlap.
   - **Retrieval:** positives per query, query/doc lengths, BM25 easiness, hard-negative difficulty, 9-direction balance.
   - **NLI:** class balance, lengths, hypothesis-only baseline artifacts.
   - **Synthetic:** topic/template distribution, diversity (distinct n-grams), per-filter drop rates.
   - **Train vs. eval:** distribution shift and entity/question/doc leakage check (ADR-0007).

**Done when:** All task sets are tracked in DVC with data cards, noise is quantified, eval is frozen and hashed, target volumes ([datasets.md §6](research/datasets.md)) are met or deviations documented in `PROGRESS.md`, task EDA is logged in MLflow, and defects are resolved before freezing eval.

## Stage 4 — Document Processing Engine

1. Reference benchmark corpus: PDF (text, scans, multi-column, tables), DOCX, HTML, TXT in `kk`/`ru`/`en`.
2. Experiments `DOC-0`…`DOC-3`.
3. Prepare chunking variants (`chunking.variants`); final choice deferred to Stage 6 retrieval metrics.
4. Robustness: corrupted, oversized, and mixed-language files never crash the worker.

**Done when:** Benchmark is logged, processing path per document kind is selected, and edge-case/hostile tests pass.

## Stage 5 — Extraction

1. Extract shared evaluation module: cluster paired bootstrap and per-example outputs (ADR-0022) from `eval_docproc`; verify `eval_ner` against `seqeval` reference cases.
2. Experiments `NER-0`…`NER-5` with HPO and ≥3 seeds: strict entity F1, unseen-F1, boundary vs. type errors.
3. Rule-based DATE extractor and evaluation.
4. Entity Linking: Wikidata KB, candidate generation (with Kazakh suffix stripping), ranking; report candidate `recall@k` and `accuracy@1` on human-labeled per-language sets.
5. Calibrate entity confidence and pick `annotate_corpus` threshold on precision–coverage curve.
6. Sharded cloud corpus annotation (`annotate_corpus`) with champion model.

**Done when:** Strict entity F1 beats baseline with CI across languages/types, champion passes gate, throughput is measured, and linking reports candidate `recall@k`, `accuracy@1`, and NIL rate per language.

## Stage 6 — Retrieval at Scale

1. Add cluster key to retrieval eval linking MT directions to source query (ADR-0022).
2. Compute corpus embeddings and build Qdrant index (`embed_corpus` → `index_corpus`).
3. Experiments `RET-0`…`RET-7`; log first-stage `candidate recall@100` separately.
4. `RET-C`: select chunking variant by retrieval metrics.
5. `RET-S`: scale ladder `1M → 10M → 35M`; compare ANN recall against exact search at every step.
6. Stage-attributed miss analysis across all 9 directions.

**Done when:** Champion does not regress on any of 9 directions, scale curves and memory budget are recorded, ANN/quantization loss is isolated from model loss, and Serving VM decision is finalized (ADR-0013).

## Stage 7 — Generation

0. **Precondition:** Approved cloud budget for Ampere+ GPU (`params.yaml: cloud.budget_usd`) — bf16 LoRA cannot run on Kaggle T4. Without paid budget, run steps 1, 2 (`GEN-0` only), 4, and 5 on baselines.
1. Qwen3.5 compatibility check in `transformers`, `PEFT`, `llama.cpp`, `vLLM` (ADR-0012): inspect actual linear module names, non-thinking chat template, and GGUF vs. `transformers` parity.
2. Experiments `GEN-0`…`GEN-3` (bf16 LoRA); compute faithfulness per ADR-0022 with human verifier audit on `verification.faithfulness_audit_claims` claims/language.
3. Adapter regression suite: false refusals on answerable queries, answer language, citation format, repetition loops, cross-lingual directions.
4. `GEN-Q`: GGUF quantization with multilingual `imatrix`, KL divergence vs. bf16, per-language metrics.
5. Throughput: `tokens/s`, TTFT, and memory on target serving hardware.

**Done when:** Fine-tuned model beats best baseline on faithfulness and answer F1 per language without regressions in step 3; otherwise record negative result in `PROGRESS.md` and serve best baseline.

## Stage 8 — Evidence Verification

1. Human-labeled claim test set from real system answers in `kk`/`ru`/`en`, including cross-lingual pairs.
2. Claim splitting: rules vs. LLM splitter benchmarked against human splits.
3. Experiments `VER-0`…`VER-2`: sentence windows, rule-based number/date verification.
4. Per-language calibration, risk–coverage thresholds, refusal policy (ADR-0008); revise starter `promotion.verification_*` thresholds on step 1 human labels.
5. Verification latency per answer.

**Done when:** Verifier cuts unsupported claims without significant recall loss, and `contradicted` precision + ECE pass gate per language.

## Stage 9 — System Integration & Serving

1. API, Ingestion Worker, Inference Service, and LLM Server per `01-architecture.md §6`.
2. PostgreSQL schema with Alembic migrations and `COPY` bulk load.
3. Dynamic batching, Redis caching, SSE streaming.
4. Load tests: RPS, p95 latency, degradation under concurrent document ingestion.
5. Prompt-injection defense and `kk`/`ru`/`en` attack suite in `eval_e2e` (ADR-0023).
6. End-to-end eval with per-stage traces, failing-stage attribution table, and ablation (`01-architecture.md §10`).

**Done when:** NFR targets (`01-architecture.md §1`) and end-to-end gate thresholds (including attack success rate) pass; ablation and failing-stage shares per language are recorded.

## Stage 10 — Product, Ops & Final Evaluation

1. Web UI (Next.js): upload, status, chat, citations, entities; Playwright e2e tests.
2. Monitoring (`01-architecture.md §12`): Prometheus metrics, Grafana dashboards, reference-calibrated drift alerts.
3. Deployment: Docker images, compose profiles, clean-start runbook.
4. `real_docs` suite: verify document-processing transfer (router thresholds, borderless tables) and e2e quality on real files.
5. Final report: all experiments, champion models, per-language metrics, pretraining contamination check (ADR-0022), and limitations.

**Done when:** Stack boots cleanly from scratch, demo works on real files, `real_docs` evaluation is logged, and final report is published.
