# TriFact AI — Progress Log

Read relevant sections before stage work ([AGENTS.md](../AGENTS.md)). Add new entries at the top.

## Current State

| Field | Status |
|---|---|
| **Active stage** | Stage 1–3 cloud reconciliation (2026-11-23). CPU execution started over the private Colab SSH relay. Full-scale EDA requires higher-throughput Drive access. Stage 2–3 acceptance still depends on human review and teacher-generation quota |
| **Last completed stage** | Stage 1 — Data Platform & Acquisition (2026-11-20; `ruwiki` part 4 added in `febc68c` awaits cloud `wikitext` run); Stage 4 — Document Processing Engine (2026-11-20, re-verified 2026-11-23) |
| **Remote repository** | `origin` → `https://github.com/Nusultan11/TriFactAI` (`main`) |
| **DVC & Google Drive** | `drive_desktop` default locally ([ADR-0024](architecture/adr/0024-google-drive-storage.md)). All **3,274 / 3,274** DVC-cached objects across `dvc.lock`, `data/external.dvc`, and `data/review.dvc` verified on Drive after stripping spurious `.pdf` extensions from 105 `docproc` cache objects (2026-11-23) |
| **Raw layer** | 19 sources: 17 HF + Wikidata `20261116` (122,308,973 entities) + Wikipedia wikitext `20261101`; all 584 shards / 270,766,215 rows passed deep verification in Colab CPU (`stage-01-03-3a88f9d273da/raw-integrity.json`, 0 problems) |

## Open Blockers (User Action, Quota, or Budget Required)

1. **Stage 3 human review (eval & silver NER):**
   - `data/eval/review/sheet-01.csv` — 3,500 items (MT retrieval queries, `kk` NLI; `0 / 3,500` human-verified). ROADMAP 3.5 requires full human labeling; `retrieval` and `nli` remain `candidate` until `python -m trifact.data.build_eval freeze`.
   - `data/review/ner_silver/to_label.csv` — 600 silver NER sentences (200/language, ADR-0015 noise audit).
   - AI/LLM judgments do not substitute for human labels.
2. **Stage 2 human review:**
   - `data/review/quality_human/to_label.csv` — 1,500 stratified docs (500/language: 350 kept, 150 dropped). Save completed sheet as `labels.csv`, then run `python -m trifact.evaluation.review_quality score --review-dir data/review/quality_human --reviewer "<name>"`.
   - Prior `data/review/quality` and 550 re-checked langid paragraphs were LLM-labeled (Claude, tagged in `reviewer`) and do not replace human review.
3. **Interactive Colab Google Drive OAuth (cloud CPU stages):**
   - `colab drivemount` requires interactive browser OAuth consent for the selected runtime. The verified SSH relay provides private access to Drive DVC objects at approximately 1 MiB/s, which would take more than a day to transfer the inputs for full-scale corpus EDA.
4. **Kaggle GPU quota for synthetic teacher:**
   - On 2026-11-23, `kaggle quota` reported `7.76 / 30.00` GPU-hours remaining (reset `2026-11-28 00:00 UTC`). NLI QA v4 outputs were recovered and verified as recorded below.
   - Full synthetic targets in `datasets.md §6` require additional weekly quota or approved `params.yaml: cloud.budget_usd > 0`.
5. **Colab GPU tier & budget:**
   - Hosted CPU runtime (12.67 GB RAM, 107.72 GB disk) verified with Drive. Account has `0.00` compute units (NVIDIA L4 unavailable without paid units); `params.yaml: cloud.budget_usd: 0`. Recovery protocol: [ADR-0025](architecture/adr/0025-cloud-only-computing.md).
6. **Optional MCP servers:**
   - `github` MCP: invalid `Authorization` header (`GITHUB_PERSONAL_ACCESS_TOKEN`).
   - `mlflow` MCP: `127.0.0.1:5000` refused connection (2026-11-22). Historical MLflow runs logged via local HTTP; shared cloud MLflow server remains to be qualified.

## Downstream Notes (Carry Forward)

- `fineweb2_kk` peaks at 5.2 GB RSS with 4 workers due to large input row groups (within `lake.peak_ram_gb: 10`); use `--workers 2` when sharing RAM.
- `multidomain_kk` CSV has 24,883,808 records (HF card/datasets-server show 15.51 M partial count); verified via full `csv_record_count`. Overlaps with FineWeb-2 (CC100, OSCAR, news) → cross-source dedup mandatory.
- `research_only` sources (WikiANN, NEREL, SberQuAD, WikiNEuRal, XNLI, MS MARCO, mMARCO) are train/eval only and propagate `release: false` ([datasets.md §7](research/datasets.md)); `multidomain_kk` is `allowed` with `release: false`.
- Separate KazParC synthetic configs (`sync`, `sync_raw` → `sync_pairs`, `sync_entries`) from human pairs so synthetic rows never enter eval (ADR-0015).
- **Polars memory trap:** filtered Polars views pin underlying string buffers. Subset large corpora via PyArrow key filter (`task_common.rows_with_keys`), not Polars lazy filter/semi-join.
- **Teacher synthetic throughput (ADR-0015, ADR-0020):** `llama.cpp` on 2× T4 achieved ~3.6 tok/s per slot (~56 tok/s total) at ~1,500-token prompts vs. 740 tok/s in short benchmark. Next run: shorten QA evidence, keep numeric passage IDs, measure prefill vs. decode separately.
- **Synthetic QA NLI filter:** scores each claim over 3-sentence passage windows (max over windows, min over claims). `teacher.filters.min_nli_entail: 0.7` is uncalibrated by humans (pass rate: 0.70 at 0.5, 0.58 at 0.7).
- `mDeBERTa-xnli` was trained on XNLI (validation agreement 1.0); treat its `kk` XNLI consistency score with that caveat until Stage 8 independent evaluation.
- XNLI hypothesis-only baseline: `0.52–0.56` vs. `0.33` chance across all languages; compare Stage 8 NLI models against this baseline.
- **EDA downstream rules (`curation.downstream`, Stages 6–7):** exclude `kk` docs with Kazakh-letter share `< 0.05` (1.06% of curated `kk`: broken encoding or `ru`); mask checksum-valid IIN and emails (do not mask phone regex — fires on 45% of `en` wikitext dates/coordinates).
- **Wikidata `kk` gap:** only `0.38%` of entities have `kk` labels (`en`: `76.4%`); `kk` entity linking must leverage `ru`/`en` labels, aliases, and `kkwiki` sitelinks (`task/kb`).
- **Retrieval `kk-kk` shortage:** ~63k pairs; ~97% of cross-lingual pairs are short KazParC sentences rather than passages. Supplement with teacher synthetic queries (Blocker 3) and reweight sources in Stage 6.
- Keep `HF_TOKEN` only in `.env` (never `.env.example`); required for `dvc repro` on gated sets (KazNERD, KazQAD, KazParC), not `dvc pull`.
- **Qwen3.5 (ADR-0012):** bf16 LoRA only (never QLoRA), `vLLM >= 0.17`; verify PEFT + `llama.cpp` compatibility at Stage 7 start.
- Full 35M Qdrant index likely exceeds 15 GB RAM alongside the serving stack; `RET-S` decides Serving VM (ADR-0013). Build vector collections where served to avoid transferring ~72 GB fp16 vectors (ADR-0014).
- **Document processing (ADR-0021):** worker needs Tesseract 5 binary (`TRIFACT_TESSERACT` on Linux) and ~2.1 GB RAM/parser process with Docling; scans/tables take 2–9 s/page on CPU → separate load-test scenario in Stage 9. Verify synthetic benchmark transfer on `real_docs` (≥50 files, Stage 10).
- `chunking.variants` are chosen in Stage 6; `chunk_corpus` currently chunks flat, while `DocumentChunker` respects sections.
- `dvc.yaml` lists `build_eval` outputs individually (`data/eval/{ner,qa,retrieval,nli,review}`) because `data/eval/docproc` belongs to `build_docproc`.

---

## Stage 1–3 · Cloud Reconciliation (2026-11-23)

- **NLI QA v4:** `kaggle kernels status nurs2005/trifact-nli-score-qa` reported `COMPLETE`. The explicit version-4 output contained two successful T4 workers and **218,451** scored windows (109,226 in 377.8 s and 109,225 in 411.1 s, $0). An independent recovery matched every score and log SHA-256 against Drive. The staged NLI input had **0 missing IDs, 0 extra IDs, 0 duplicate score IDs, and 0 invalid probabilities**. Input Parquet SHA-256 is `f2aa3aa725bb938ba974738ae89e21195dc8b06660c4d5fae485cd3e14e28c56`.
- **Pinned teacher inputs:** verified second-run receipts (12 query artifacts and 4 QA artifacts), with no ID overlaps across the original and second runs. Raw generations were 12,624 + 20,642 for queries (second run includes judge prompts) and 4,554 + 5,629 for QA. Added `data/stage3_teacher.dvc` (30 files, 94,445,420 bytes, `3b87db61e0dbc7a1b6086b0dc7598db8.dir`), pushed 31 DVC objects to private Drive and confirmed `dvc status --cloud --json` returned `{}`. Canonical synthesis now requires both teacher runs and the exact scored claim windows and model config. These counts describe the input runs.
- **CPU access:** existing Colab `trifact-s13` reported 12,975 MiB RAM and 101 GiB free disk. The approved private Drive DVC store was reachable from the VM over the official Colab SSH relay with a loopback-only DVC endpoint. Colab's interactive Drive mount was still waiting for OAuth when the private relay was qualified. No paid resources were requested.
- **Synthetic execution attempt:** `trifact-synth-reconcile` restored the verified language-ID model and the versioned teacher bundle. Over the HTTP remote, the combined DVC pull omitted the older external bundle and the granular chunk manifest, so `synthetic_queries` exited before building. Its failed summary was retained on Drive in `workspace-data/stage-01-03-reconcile/2026-11-23-cloud`. Pulling the granular manifest on its own recovered that file. The runner now restores these three targets separately.
- **Synthetic datasets rebuilt on Colab CPU (`6bbf6db`, 153.6 s, $0):** kept **101,970 queries** (kk 34,123, ru 33,954, en 33,893) and **11,015 QA** (kk 7,385, ru 1,908, en 1,722). Exact NLI input content and model config matched the scored run, and QA dropped 5,889 of 16,904 pre-NLI candidates at the configured threshold. Both datasets passed deep shard and duplicate-ID verification. DVC pushed and verified 9 objects per dataset. Both datasets have `release: false` through the corpus license lineage. The [run report](../reports/stage-01-03/synthetic.json) records all drops. Drive holds the checked MLflow SQLite snapshot with three `FINISHED` runs and the two dataset manifests. Synthetic volumes remain below ROADMAP targets.
- **NER isolation audit:** the historical gold set has 36 Russian training rows and 6 Russian validation rows whose normalized sentences also occur in frozen NER eval. Kazakh and English had zero overlaps. The gold builder now quarantines matching training and validation sentences against its test split and reports the drops by language and source.
- **Human review:** reused the existing Drive packets in `workspace-data/human-review`. They contain machine suggestions for 1,500/1,500 curation, 457/600 silver NER and 3,403/3,500 eval items, respectively. Human verification remains required as described above.

## Stage 1–4 · Storage Repair, EDA Fix & Verification (2026-11-23)

- **Stage 1 & 4 Drive DVC cache repair:** Discovered that Google Drive had appended a spurious `.pdf` extension to **105** extensionless DVC cache files under `TrifactAI/dvc/files/md5/`, causing `dvc pull data/eval/docproc` to fail on 105 files (`[WinError 2]`). Stripped the `.pdf` suffix from all 105 objects and audited **3,274 / 3,274 (100%)** DVC-cached objects referenced by `dvc.lock`, `data/external.dvc`, and `data/review.dvc` on Drive (the 4 uncached outputs are `{cache: false}` JSON reports archived in `workspace-data/files/reports/`).
- **Stage 1 cloud integrity confirmed:** Audited Colab CPU verification log `workspace-data/colab-validation/stage-01-03-3a88f9d273da/raw-integrity.json` — all **19 raw sources, 584 shards, 270,766,215 rows** passed deep verification with **0 problems** (2,457.2 s). `ruwiki` part 4 (added to `params.yaml` in `febc68c`) remains queued for cloud CPU `wikitext` execution once Colab Drive OAuth is authorized.
- **Stage 2 EDA memory ceiling fixed (`7d9a4f6`):** Updated [eda.py](../src/trifact/data/eda.py) to merge shard profiles incrementally (`accumulate()`) instead of buffering all shard dicts in driver RAM, reduced `BATCH_BYTES` (`32 MiB → 16 MiB`), added `gc.collect()` per shard, and capped default workers to `min(curation.workers, 2)` to stay within `lake.peak_ram_gb: 10` and 12.67 GB Colab CPU RAM. Added regression test in [test_eda.py](../tests/data/test_eda.py).
- **Stage 3 eval & synthetic QA NLI:** Verified `data/eval/{ner,qa,retrieval,nli}` via `trifact.lake verify --layer eval` (4/4 OK) and `build_eval status` (`ner` and `qa` frozen & intact; `retrieval` 8,390 and `nli` 2,650 `needs_review` intact as `candidate`; `sheet-01.csv` `0 / 3,500` human-verified). Queued Kaggle 2× T4 job `nurs2005/trifact-nli-score-qa` v4 (`$0` USD, within `7.76 / 30.00 h` free quota) to score claim pairs of the combined 16,904 synthetic QA candidates (`stage-01-03-2026d1dadd71/candidate-qa/nli-input`).
- **Stage 1 `ruwiki` part 4 acquired on Colab CPU (`trifact-wt4`, `$0` USD):** Downloaded `ruwiki-20261101-pages-articles4.xml-p2198270p3698269.bz2` (774.2 MB, Wikimedia SHA-1 verified) and parsed **322,989 articles + 265,322 redirects** into 3 Parquet shards (`ru/part-004-00.parquet` 540.6 MB, `ru/part-004-01.parquet` 367.8 MB, `ru/redirects-004-00.parquet` 7.6 MB) in **699.9 s** at **612.8 MB peak RSS**. Verified SHA-256 on Drive (`workspace-data/wikitext-part4/`) and terminated `trifact-wt4`.
- **Turnkey cloud CPU runner (`40e5b70`):** Added [cloud_run_stages_1_3.py](../scripts/cloud_run_stages_1_3.py) to execute `wikitext_merge` → `build_ner` (>5M silver NER) → `synthetic` (combined queries2 + qa2/NLI) → `train_and_eval` (`build_pairs`, `build_sft`, `eval_leakage`, `task_eda`) → `eda_rss_check` (`entities` + `curated` <10 GiB RSS) with automatic Drive DVC push and atomic MLflow SQLite snapshots once Colab Drive OAuth is authorized.
- **Human-review workflow verified (`workspace-data/human-review/`):** All 3 packets have standalone interactive `review.html` UIs + `machine_labels.csv` with Qwen3.5-35B-A3B judge suggestions, independent signals, and disagreement triage (`stage-2-quality`: 1,500 rows, 1,428 agree, 420 priority-1; `stage-3-silver-ner`: 600 rows, 335 agree, 549 priority-1; `stage-3-eval`: 3,500 rows, 2,220 agree, 1,716 priority-1).
- **Stage 4 verified end-to-end:** Clean `dvc pull data/eval/docproc` restored all **324 files**; **323 / 323** SHA-256 hashes in `data/eval/docproc/manifest.yaml` passed (`0` mismatches); **57 / 57** `tests/ingestion` + `tests/evaluation/test_eval_docproc.py` passed (1 skipped); **17 / 17** live hostile-file robustness checks passed through `SafeParser` (`restarts=0`).

---

## Stage 1–3 · Cloud Requalification (2026-11-22)

- **Colab CPU verification:** Restored `data/review.dvc` (9 files), `data/external.dvc` (242 files), `lid.176` with SHA-256 check, and Git bundle (`32bc2a6`) from Drive in a clean VM. Clean `dvc status` (local + remote). Receipt: `workspace-data/colab-recovery/02a36f308d6d4126b9c1ca9e6f567ce3.json` (174.5 s).
- **Stage 2–3 pilots (`workspace-data/colab-validation/stage-01-03-2026d1dadd71`):** raw EDA on NEREL (3 profiles, 0.7 s, peak RSS 312.4 MiB); QA import of both teacher runs yielded 16,904 pre-NLI candidates; MLflow snapshot contains 2 runs. Fixed retrieval import Polars expression context bug and added regression test for both query types.
- Added [notebooks/cloud_stages.ipynb](../notebooks/cloud_stages.ipynb) for pinned Git checkout, verified Drive DVC mount, CPU/GPU stages, and MLflow snapshots. Colab L4 and full independent cloud restore remain open; no stored datasets or DVC pointers were altered.
- **Stage 1 audit:** Verified 19 `source.yaml` manifests + MD5s, licenses, row checks, and 584 Parquet shard sizes on mounted Drive; HF confirmed 17 pinned revisions; Wikimedia returned HTTP 200 on dump/checksum/mirror/9 wikitext HEADs. Full shard re-hashing in an independent cloud VM remains open. `lid.176` reconstructed from 2 parts with exact SHA-256 match.
- **Stage 2 audit:** Located raw/clean/curated/entities EDA outputs on Drive (41/14/12/1 profiles); recorded raw/curated peak RSS 10,513 / 11,200 MiB (>10 GiB). Updated `dvc.yaml` so each EDA layer receives only its input. Cloud rerun and human review remain open.
- **Stage 3 audit:** Duplicate verdicts block eval freeze; leakage check fails on non-zero overlap outside shared MIRACL passage pool; historical 198 retrieval query overlaps require re-check after train rebuild.
- **Kaggle 2× T4 outputs archived on Drive (`workspace-data/kaggle`):** v1 `trifact-teacher-queries2` (15,042 retrieval + 5,600 judge outputs / 65,600 prompts, 0.524 req/s, 270 length-truncated) and `trifact-teacher-qa2` (5,629 QA outputs / 24,000 prompts, 0.142 req/s, 210 length-truncated). Parquet row counts and IDs verified; raw outputs awaiting import/filtering. Separate NLI pilot (512 pairs) submitted as v1 (`nurs2005/trifact-cloud-nli-pilot`).
- **Assisted review packets:** Private Drive folder holds raw sheets, `review.html`, and `machine_labels.csv` (judge v1 suggestions: 1,500/1,500 docs, 3,403/3,500 eval items, 457/600 silver NER). No accepted human verdicts yet.
- Uploading prompt metadata as a new private Kaggle dataset was blocked by automated permissions check; import bundles remain available on Drive for Colab.

---

## Architecture & Stage Alignment Review (2026-11-21)

Audited docs against `.agents/skills`; Stage 1–4 code, data, and `dvc status` unchanged.

| Gap | Resolution |
|---|---|
| `faithfulness` in LLM gate undefined | [ADR-0022](architecture/adr/0022-evaluation-protocol.md): verifier-supported claim share + human audit (`verification.faithfulness_audit_claims`) |
| Example-level bootstrap on dependent items | ADR-0022: cluster paired bootstrap with joint seed resampling |
| No LLM-judge or contamination rules | ADR-0022: human-calibrated judge only; non-Wikipedia & `real_docs` slices |
| Prompt injection unspecified/unmeasured | [ADR-0023](architecture/adr/0023-untrusted-evidence-in-prompts.md): spotlighting, independent output checks, attack suite, `promotion.e2e_injection_no_worse` |
| Missing Verification gate thresholds | Added `verification_contradicted_precision_min`, `verification_ece_max` (starter values) |
| Gate numbers duplicated in `01-architecture.md §10` | Replaced with `params.yaml: promotion` key references |
| Stages 5–10 lacked per-stage diagnostics | Added candidate recall, ANN vs. exact (`RET-S`), `RET-C`, adapter regressions, system-answer verifier test, e2e traces, drift, `real_docs` |
| bf16 LoRA incompatible with Kaggle T4 at $0 budget | Added ROADMAP 7.0 precondition and `01-architecture.md §17` risk |
| VLM-OCR & claim-verifier facts lived in skills | Moved to `models.md §5, §6.4` and `datasets.md §8` |

C4 model (`workspace.dsl`) unchanged.

---

## Agent Skills Library (2026-11-21; modernized 2026-11-23)

- Built 11 domain skills under [.agents/skills/](../.agents/skills/README.md) (`trifact-{data-pipeline,eda,document-processing,ner-linking,training-hpo,retrieval,rag-pipeline,llm-finetuning,verification,evaluation,mlops}`) following `agentskills.io`.
- Historical note: initially mirrored to `.claude/skills/` via `scripts/sync_skills.py` and verified in Codex (`codex-cli 0.144.0`, 18/18 routing accuracy) and Claude Code desktop. On 2026-11-23 (`4bc2093`), `CLAUDE.md`, `.claude/skills/`, and `scripts/sync_skills.py` were removed so `.agents/skills/` is the single canonical directory.
- Fixed stale docstrings in `train_llm`, `llm_client` (ADR-0012), and `retrieval.store` (ADR-0013).

---

## Stage 3 — Task Datasets: Stopped (2026-11-21)

All work independent of human labels and Kaggle quota is built, verified, and logged in DVC/MLflow. Stopped on Blockers 1 & 3 (`retrieval`/`nli` eval unfrozen, 600 silver NER sentences unreviewed, teacher synthetic below target).

**Completed**
- **3.1 NER** ([build_ner.py](../src/trifact/data/build_ner.py), [build_kb.py](../src/trifact/data/build_kb.py)): mapped KazNERD, WikiANN, WikiNEuRal, NEREL to PER/ORG/LOC (ADR-0006); fixed 16 KazNERD BIO errors; built silver via wikitext links → redirects → KB titles → Wikidata type (3-pass, 200k-sentence chunks). Switched KB typing from label-priority (`PER > ORG > LOC`, which misclassified countries/cities as ORG via "sovereign state") to closest `P279` class → ORG agreement with gold jumped: `kk 0.32 → 0.93`, `ru 0.49 → 0.79`, `en 0.60 → 0.82`.
- **3.2 Retrieval** ([build_pairs.py](../src/trifact/data/build_pairs.py)): MS MARCO, mMARCO-ru, MIRACL, ru-HNP, QA pairs (incl. MT queries, NQ `kk↔en`), KazParC (human + `synthetic_mt`-tagged sync), teacher queries; source + BM25 hard negatives; isolated process per builder.
- **3.3 SFT** ([build_sft.py](../src/trifact/data/build_sft.py)): `0.60/0.25/0.15` mix, `kk >= 0.30`, frozen QA eval questions excluded.
- **Synthetic teacher** ([synthetic.py](../src/trifact/data/synthetic.py), ADR-0015, ADR-0020): Qwen3.5-35B-A3B (`UD-IQ4_XS`) via `llama.cpp` on Kaggle 2× T4. Query filters: parse, length, language, copy, BM25 round-trip, duplicate. QA filters: parse, language, citation, refusal, 3-sentence-window NLI.
- **3.4 NLI** ([build_nli.py](../src/trifact/data/build_nli.py)): XNLI `ru`/`en` + NLLB-200-distilled-1.3B `kk` MT (`chrF >= 30`, langid, `mDeBERTa-xnli` label consistency).
- **3.5 Eval** ([build_eval.py](../src/trifact/data/build_eval.py), ADR-0007): `ner` and `qa` frozen (`frozen.json`, SHA-256); `retrieval` and `nli` candidate.
- **3.6 Budget:** `$0` cloud spend (`cloud.budget_usd: 0`), ~29 / 30 free Kaggle GPU-hours used.
- **3.7 Task EDA** ([task_eda.py](../src/trifact/data/task_eda.py)) → `reports/eda/task.json`, MLflow `eda/eda-task`.
- **Corpus chunks** ([chunk_corpus.py](../src/trifact/data/chunk_corpus.py)): ≤350 XLM-R tokens, overlap 50, exact offsets.

**Results**

| Dataset | kk | ru | en | Total | Target (`datasets.md §6`) |
|---|---|---|---|---|---|
| NER gold sentences (`train/val/test`) | 90,135 / 11,948 / 12,307 | 121,262 / 22,216 / 23,081 | 112,030 / 21,376 / 21,597 | 435,952 | ~1.3 M (all public gold < target) |
| NER silver (train) | 239,225 | 2,006,149 | 2,274,843 | 4,520,217 | 5–10 M (below; expandable via ru/en dump parts) |
| Retrieval pairs | see directions | — | — | 14,984,724 | 10–20 M ✓ |
| SFT | 0.30 share | — | — | 307,966 | 1–2 M (below; Blocker 3) |
| NLI train | 210,255 | 392,645 | 392,645 | 995,545 | ~1.2 M |
| Corpus chunks | 29.59 M | 11.78 M | 18.25 M | 59.63 M | ~35 M (subset selected in Stage 6 `RET-S`) |
| Synthetic queries / QA | 11,161 / 947 | 11,150 / 583 | 11,160 / 294 | 33,471 / 1,824 | Below (Blocker 3) |
| KB entities | `kk` labels 0.38% | — | — | 105.6 M | — |

- Retrieval directions: `en-en` 705,382; `ru-ru` 959,945; `kk-kk` 63,547; cross-lingual 2.16–2.27 M each (6 directions).
- Eval sets: `ner` (gold test, frozen), `qa` (32,427, frozen), `retrieval` (38,640 queries, 94,105 qrels, 35,237 docs, candidate), `nli` (12,670, candidate), human review sheet (3,500 items).
- Leakage check: `retrieval` queries 198/38,640; `retrieval` passages 2,573/35,237 (shared MIRACL corpus negatives); NLI 0.
- Silver NER noise vs. gold: type agreement on shared surfaces PER `0.92–0.99`, LOC `0.92–0.96`, ORG `0.79–0.93`; gazetteer lower bound F1 `0.12` (KazNERD) – `0.31` (WikiANN `kk`).
- Teacher synthetic (6 h on 2× T4): 12,624/30,000 queries (~56 tok/s total); 4,554/10,000 QA → 1,824 after NLI (pass rate 0.58 at 0.7 threshold).
- NLI `kk`: 418,122 / 528,819 translated sentences passed QC; label consistency on test: `en 0.873`, `kk 0.768`.

**Experiments & Fixes**
- NLI QA filter pass rate: full answer vs. 2 passages `0.44`; per-claim vs. full passage `0.40`; **per-claim vs. 3-sentence windows `0.58` (adopted)** — full-passage NLI scored verbatim-supported claims at `0.24` and `0.001`.
- Numeric citations `[n]` adopted over long `chunk_id` (753/4,554 = 16.5% truncated at `max_tokens`).
- **KazQAD leakage:** shared contexts across IDs put 2,257 `en` and 1,206 `kk` test contexts in train; added `ctx_key` split guard → dropped 13,787 train/val rows, frozen QA eval clean.
- **SberQuAD:** excluded unlabeled official test (12 IDs overlapped train).
- **MIRACL:** namespaced `doc_id = miracl/<lang>/<docid>` (1.06 M Wikipedia page IDs overlapped between `ru` and `en`).
- **OOM fixes:** chunked streaming silver NER, per-builder process + PyArrow `rows_with_keys` + UInt32 mMARCO IDs in `build_pairs`, hashed `train_keys` in `build_eval`, single-pass review sampling, one scan/language in `scan_raw`.
- Switched default `scale.profile` from `pilot` to `full` and rebuilt sets. Excluded 4.2% mojibake passages in mMARCO-ru. Added `-kvu` in `llama.cpp` (305 → 740 tok/s batched) and used prebuilt CUDA binary with `--list-devices` check.

---

## Stage 2 — Data Quality & Curation: Stopped (2026-11-21)

Full `curated` layer and full EDA complete; stopped on Blocker 2 (human review of ≥500 docs/language pending; prior review was LLM-labeled).

**Completed & Results**
- **2.1 Normalization** ([normalize.py](../src/trifact/ingestion/normalize.py)): vectorized with edge-case tests.
- **2.2 Language ID** (`eval_langid`, `reports/langid.json`): 1,800 paragraphs (`kk`/`ru`/`en`/`ky`/`tt`, 550 LLM-audited). `lid.176` beat `GlotLID` by `+0.028` macro-F1 (95% CI `[0.016, 0.039]`), 5.7× faster; at `min_score: 0.65`, macro-F1 `0.945`, `kk` F1 `0.981`.
- **2.3 Quality filters** ([quality.py](../src/trifact/data/quality.py), `reports/clean.json`): kept Wikipedia `99.0%`, FineWeb-2 `kk` `98.2%`, `multidomain_kk` `84.2%` (2.89 M were `ru`, 3.6 M too short).
- **2.4 Dedup** ([dedup.py](../src/trifact/data/dedup.py)): `32,770,131 → 30,879,083` docs (`-5.77%`; 87,416 exact, 1,803,632 near MinHash-LSH; sampled LSH median Jaccard `0.83`, 96% ≥ `0.5`).
- **2.5 Decontamination** ([decontaminate.py](../src/trifact/data/decontaminate.py)): dropped 31,361 docs, edited `0.54%`; re-scan = 0 leaks. Final `curated`: **30,847,722 docs** (`kk` 19.86 M, `ru` 4.62 M, `en` 6.37 M).
- **2.6 Full streaming EDA** ([eda.py](../src/trifact/data/eda.py), [eda_report.py](../src/trifact/data/eda_report.py)): `raw` (41 profiles, 57 min), `clean` (14 profiles, 37 min, 8.7 GB peak RSS), `curated` (12 profiles, 70 min), `entities` (1 profile, 0.6 min) → `reports/eda/report.md`, 88 findings, 43 figures, MLflow `eda`.
- **Tokenizer fertility (tokens/word):** Qwen3.5 `kk 3.64 / ru 2.44 / en 1.49` vs. Qwen3 `kk 4.85 / ru 3.06 / en 1.51` (`kk` 25% cheaper on Qwen3.5, still 2.4× `en`).
- **2.7 Data cards:** `card.md` alongside every `dataset.yaml`; human review sheet `data/review/quality_human/to_label.csv` (1,500 stratified docs).

**Key EDA Findings → Decisions (`reports/eda/findings.json`)**
- Duplicate IDs in KazParC, MIRACL qrels, mMARCO, MS MARCO, NEREL → composite keys / dedup by `qid`.
- SberQuAD 12 test/train ID overlaps → unlabeled test excluded (Stage 3).
- mMARCO-ru 4.2% mojibake → excluded from retrieval pairs.
- PII: mask checksum-valid IIN (0.02% `kk`) and emails (`curation.downstream.pii_mask: [iin, email]`); skip phone regex (45% false positives on `en` wikitext).
- `kk` docs with <5% Kazakh letters (1.06% curated `kk`) → excluded via `curation.downstream.kk_min_kk_letter_share: 0.05` and queued for human review.
- FineWeb-2 `kk` `ru`-labeled docs with 5.1% Kazakh letters (~4.7k docs, 0.02%) and 6–7% Latin in `kk`/`ru` Wikipedia titles → retained.
- Missing `title`/`url`/`date` in `multidomain`/`FineWeb`/`Wikipedia` → documented in data cards.
- Wikidata `kk` labels 0.38% (72% no sitelinks) → rely on `ru`/`en` labels, aliases, `kkwiki` titles.
- XNLI hypothesis-only accuracy `0.52–0.56` → retained; compared as baseline in Stage 8.
- 5 SFT/eval question overlaps (MT SQuAD) → excluded from SFT.
- Langid `min_score: 0.65` (macro-F1 `0.945`) kept over `0.30` (`0.970`) to limit `ky`/`tt` false accepts (`2.7%`/`5.2%`); revisit after human langid labels.
- **Fixes:** bounded EDA batch size to ~32 MB (fixed 6.5 GB/worker wikitext OOM), capped vocab scan to first 200k chars, sanitized `>` in MLflow metric keys.

---

## Stage 1 — Data Platform & Acquisition: Completed (2026-11-20)

Closed by completing Wikidata `20261116` and `wikipedia_wikitext` `20261101` (rest of Stage 1 logged below on 2026-11-18).

- **Wikidata `20261116`** ([wikidata.py](../src/trifact/data/wikidata.py)): 103.9 GB bz2 dump downloaded with SHA-1 match; parsed in 194 × 512 MB bz2 block ranges across 8 workers (1 h 56 min, 14.25 MB/s compressed) using magic `0x314159265359` + 64-bit tail feed, stopping at EOS + 64 bits. Verified by [test_wikidata.py](../tests/data/test_wikidata.py). Result: **122,308,973 entities**, 5.29 GB Parquet, 211 shards, `entity_count` ✓, 17,588 rows/s, 3,190 MB peak RSS.
- **`wikipedia_wikitext` `20261101`** ([wikitext.py](../src/trifact/data/wikitext.py)): full `kk`, `ru` parts 1–3, `en` parts 1–4 + redirects: `en` 336,391, `kk` 296,389, `ru` 997,356, 4.46 GB Parquet, 21 shards, `xml_page_count` ✓, 593 rows/s, 1,306 MB peak RSS (2 workers).
- **Total raw layer:** 19 sources, 270.7 M rows (146.8 M HF + 122.3 M Wikidata + 1.63 M wikitext), 48.5 GB Parquet.
- **Fixes:** replaced `indexed_bzip2` (OOM on 100 GB stream) with custom bz2 block-range parser; fixed 69-entity boundary drop (+64-bit feed) and EOS `Invalid data stream`. Manifests recorded `-dirty` SHA; parser code committed unchanged in `5037d41`, `9c6c0bf`.

---

## Stage 4 — Document Processing Engine: Completed (2026-11-20)

- **4.1 Frozen benchmark** ([build_docproc.py](../src/trifact/data/build_docproc.py), `data/eval/docproc`, SHA-256 `manifest.yaml`): 150 docs (18 per kind × 8 kinds: `pdf_text`, `pdf_columns`, `pdf_tables`, `pdf_scan`, `pdf_word`, `docx`, `html`, `txt` — 6 per `kk`/`ru`/`en` + 6 mixed `kk`/`ru`/`en`, 239 PDF pages) built from `clean/wikipedia` + 3 × 300-page PDFs + 17 hostile files.
- **Parsers & Sandbox** ([parser.py](../src/trifact/ingestion/parser.py), [pdf.py](../src/trifact/ingestion/pdf.py), [docling_backend.py](../src/trifact/ingestion/docling_backend.py), [ocr.py](../src/trifact/ingestion/ocr.py), [engines.py](../src/trifact/ingestion/engines.py), [sandbox.py](../src/trifact/ingestion/sandbox.py)): unified `ParsedDoc`, content-sniffed format detection, PyMuPDF fast path + Docling (Heron/TableFormer/Tesseract CLI `kaz+rus+eng`, tessdata_best 4.1.0 SHA-256 pinned), `DOC-3` per-page router, isolated `SafeParser` (timeout + RSS cap + auto-restart).
- **4.3 Chunking** ([chunker.py](../src/trifact/ingestion/chunker.py)): `DocumentChunker` with `s350`, `s200`, `s512`, `flat350` variants.
- **4.2 Benchmark** ([eval_docproc.py](../src/trifact/evaluation/eval_docproc.py), `reports/docproc.json`, MLflow `docproc`, 12 `status=final` runs, isolated process/config, CPU 8 threads):

| Kind | DOC-0 PyMuPDF (CER/WER%, order, hdg F1, cell F1, p/s) | DOC-1 Docling | DOC-3 Router (Champion) |
|---|---|---|---|
| `pdf_text` | `0.35 / 1.8`, `0.999`, `1.00`, —, **80** | `0.38 / 2.6`, `0.999`, `1.00`, —, `0.64` | `0.35 / 1.8`, `0.999`, `1.00`, —, **79** |
| `pdf_columns` | `0.36 / 1.6`, `1.000`, `0.98`, —, `72` | `0.41 / 2.5`, `1.000`, `1.00`, —, `0.63` | `0.36 / 1.6`, `1.000`, `0.98`, —, `73` |
| `pdf_tables` | `6.01 / 9.8`, `1.000`, `1.00`, **0.00**, `99` | `0.75 / 3.1`, `1.000`, `1.00`, `0.99`, `0.39` | `0.79 / 2.8`, `1.000`, `1.00`, `0.99`, `0.43` |
| `pdf_scan` | `100` (no text) | `100` (no text) | `0.59 / 2.7`, `0.999`, `0.99`, `0.84`, `0.12` |
| `pdf_word` | `3.40 / 5.0`, `1.000`, `1.00`, `0.00`, `103` | `0.19 / 1.0`, `1.000`, `1.00`, `0.97`, `0.41` | `0.16 / 0.9`, `1.000`, `1.00`, `0.97`, `0.47` |
| `mixed` (`kk+ru+en`, 1 scan) | `43.7` | `43.9` | `0.39 / 2.1`, `1.000`, `1.00`, —, `0.28` |
| **All PDFs** | `23.4` | `21.8` | **0.45 [0.35, 0.56]** |

- **DOC-3 by language (all PDFs):** CER `kk 0.70%`, `ru 0.34%`, `en 0.31%`; page acc `0.994 / 0.995 / 0.986`; heading F1 `0.99 / 0.99 / 0.99`; cell F1 `0.94 / 0.97 / 0.98`.
- **DOC-2 OCR (`pdf_scan` CER `kk/ru/en`%, `s/p`):** Docling+Tesseract `0.74/0.54/0.50` (`8.6`), PyMuPDF+Tesseract `3.5/3.0/3.6` (`5.4`), RapidOCR `20.9/19.3/10.3` (`10.1`), EasyOCR `36.1/5.4/5.4` (`27.0`).
- **Office/Text:** native DOCX/HTML/TXT give CER `0`, order `1.0`, heading & cell F1 `1.0` at `97 / 143 / 342 docs/s` (beat Docling DOCX heading F1 `0.86`, HTML CER `0.55%`, `59 / 25 docs/s`).
- **NFRs & Robustness:** 3 × 300-page PDFs → DOC-3 `165 p/s`, `1.05 GB` RSS (80× above ≥2 p/s target); peak RSS: DOC-0 `129 MB`, DOC-1 `1.7 GB`, DOC-3 `2.1 GB`, EasyOCR `4.1 GB` (all <10 GB); 17/17 hostile files + sandbox crash/hang/OOM tests passed.
- **Chunking (150 DOC-3 docs):** `s350` 1,157 chunks (avg 235 tok, 11.4% page-crossing, 0 offset errors, 99.6% lang acc); `s200` 1,799/152/7.2%; `s512` 942/287/13.8%; `flat350` 914/310/22.0%.
- **Key decisions ([ADR-0021](architecture/adr/0021-document-processing-router.md)):** `router.columns: false` (PyMuPDF 100× faster, equal quality), `router.scans: docling`, `table_mode: fast` (cell F1 `0.993` vs. `1.000`, 1.4× faster). Negative results: EasyOCR/RapidOCR drop Kazakh letters; Docling DOCX/HTML slower and worse; DOC-1 120× slower on text PDFs with no significant CER gain.
- **Fixes:** `tessdata_dir` creates `configs/tsv` (fixed `KeyError: 'text'`), `DoclingEngine.warmup`, table-header false-heading filter, sequential block-search for reading order, per-config process isolation (superseded earlier MLflow runs), explicit HTML encoding + `huge_tree` + `RecursionError` fallback.
- **Commits & Backup:** from `0ef610a`; backup tag `stage-04` in `C:\Users\nurs\TrifactAI_backups\`.

---

## Stage 1 — Data Platform & Acquisition: Stopped (2026-11-18)

Paused on Wikidata dump download (subsequently completed 2026-11-20 above).

- **Completed:** `uv.lock` (Python 3.12), pre-commit, healthy [docker-compose](../infra/docker-compose.yml) + `s3-init`, `trifact.lake` (atomic `.tmp → fsync → rename` Parquet zstd shards, `source.yaml`, `verify`/`report`), streaming resumable HF loader ([acquire.py](../src/trifact/data/acquire.py), [ADR-0017](architecture/adr/0017-raw-layer-acquisition-protocol.md)), 17/17 HF sources acquired with verified row counts and licenses ([datasets.md §7](research/datasets.md)), MLflow `data` experiment.
- **HF acquisition summary (`python -m trifact.lake report`, ~37 GB input → 38.76 GB Parquet, 146.8 M rows, 352 shards, ~9 MB/s net-bound, peak RSS ≤5.2 GB, `lake verify` 42 s):**
  - `fineweb2_kk`: `kk` 3,375,998 (6.84 GB, 14 shards, 5,215 MB RSS, ODC-By-1.0)
  - `kaznerd`: `kk` 112,702 (0.01 GB, 3 shards, CC-BY-4.0)
  - `kazparc`: `multi` 15,103,193 (2.26 GB, 39 shards, CC-BY-4.0)
  - `kazqad`: `en` 68,770, `kk` 68,246 (0.05 GB, 5 shards, CC-BY-SA-4.0)
  - `miracl`: `en` 43,952, `ru` 54,585 (0.00 GB, 12 shards, Apache-2.0)
  - `miracl_corpus`: `en` 32,893,221, `ru` 9,543,918 (7.32 GB, 86 shards, Apache-2.0)
  - `mmarco_ru`: `multi` 39,780,811, `ru` 9,758,627 (2.14 GB, 52 shards, `research_only`)
  - `ms_marco`: `en` 1,112,939 (1.64 GB, 12 shards, `research_only`)
  - `multidomain_kk`: `kk` 24,883,808 (6.97 GB, 32 shards, 3,626 MB RSS, Apache-2.0 / `release: false`)
  - `nerel`: `ru` 1,011 (0.00 GB, 5 shards, `research_only`)
  - `ru_hnp`: `ru` 102,590 (0.06 GB, 3 shards, Apache-2.0)
  - `sberquad`: `ru` 74,300 (0.01 GB, 3 shards, `research_only`)
  - `squad_v2`: `en` 142,192 (0.01 GB, 2 shards, CC-BY-SA-4.0)
  - `wikiann`: `en` 40,000, `kk` 3,000, `ru` 40,000 (0.00 GB, 9 shards, `research_only`)
  - `wikineural`: `en` 115,907, `ru` 115,440 (0.02 GB, 6 shards, `research_only`)
  - `wikipedia`: `en` 6,407,814, `kk` 238,615, `ru` 1,945,063 (11.38 GB, 63 shards, CC-BY-SA-3.0 + GFDL)
  - `xnli`: `en` 400,202, `ru` 400,202 (0.08 GB, 6 shards, `research_only`)
- **DVC & fault tolerance verified:** `dvc push -r local` (386 objects in 4 m 51 s), clean-clone `dvc pull` (3 sources in 24 s, `verify` OK), SIGKILL resume on `xnli`, CSV field-limit resume on `multidomain_kk`, Wikidata resume (1,534/1,549 chunks), 32 automated tests.
- **Fixes:** MLflow `/health` pre-check, batch-wide JSONL schema union, lifted CSV field limit, normalized CRLF → LF with `test_dvc_dependencies_use_lf_line_endings` in [tests/test_architecture.py](../tests/test_architecture.py).

---

## Stage 0 — Project Foundation: Completed (2026-11-18)

- Scaled architecture from 267k MVP to tens-of-millions design: [01-architecture.md](architecture/docs/01-architecture.md), [workspace.dsl](architecture/workspace.dsl) (15 views, 0 Structurizr inspector errors), exported [diagrams/](architecture/diagrams/) via [scripts/export_diagrams.py](../scripts/export_diagrams.py), ADRs `0011`–`0016` (+ ADR-0009 significance amendment), `params.yaml`, `dvc.yaml` (29 stages), code skeletons, and [tests/test_architecture.py](../tests/test_architecture.py).
- Deduplicated single-owner docs (`AGENTS.md`, `ROADMAP.md`, `models.md`, `datasets.md`, `README.md`), fixed `dvc.yaml` quoting and Structurizr links, connected GitHub `origin`, pushed `main`, and created initial backup.
