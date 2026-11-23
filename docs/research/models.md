# Model Catalog

Verified facts (parameters, architecture, context length, license) collected from Hugging Face Hub API (`config.json`, safetensors metadata, model cards) on 2026-11-18 unless otherwise dated. Memory estimates are analytical (`fp16`/`bf16` ≈ 2 bytes/param; `GGUF Q4_K_M` ≈ 0.6 bytes/param, excluding KV cache and activations).

Compute and deployment profiles: [01-architecture.md §1, §9, §13](../architecture/docs/01-architecture.md) and [ADR-0025](../architecture/adr/0025-cloud-only-computing.md). Local GPU (4 GB VRAM, sm86, bf16) is a serving reference only; training and batch inference run on cloud infrastructure.

## 1. Extraction (NER)

| Model | Params | Architecture | Context | License | fp16 Size |
|---|---|---|---|---|---|
| `FacebookAI/xlm-roberta-base` | 279 M | XLM-R, 12L, H=768, 12H | 512 | MIT | ~0.56 GB |
| `FacebookAI/xlm-roberta-large` | 561 M | XLM-R, 24L, H=1024, 16H | 512 | MIT | ~1.1 GB |
| `urchade/gliner_multi-v2.1` | 289 M | GLiNER (span-based, mDeBERTa-v3) | — | Apache-2.0 | ~0.6 GB |

- **XLM-R:** pretrained on 100 CC-100 languages including **Kazakh** (`kk`). Vocab = 250,002 (~192 M of 279 M base params are embeddings). **Optimization:** trim vocabulary to `kk`/`ru`/`en` tokens (Stage 5). Base is the primary corpus-annotation candidate; Large is the cloud quality ceiling.
- **GLiNER:** zero-shot span NER baseline; Kazakh is not explicitly listed — benchmarked in §6.5.
- **Inference speedup:** ONNX Runtime dynamic int8 (CPU), fp16 (GPU), length-sorted batching.

## 2. Retrieval: Embedders

| Model | Params | Architecture | Context | Dim | License | fp16 Size |
|---|---|---|---|---|---|---|
| `BAAI/bge-m3` | ~568 M | XLM-R large | 8,192 | 1024 dense + sparse + multi-vec | MIT | ~1.1 GB |
| `intfloat/multilingual-e5-large` | 560 M | XLM-R large | 512 | 1024 | MIT | ~1.1 GB |
| `intfloat/multilingual-e5-large-instruct` | 560 M | XLM-R large | 512 | 1024 | MIT | ~1.1 GB |
| `Qwen/Qwen3-Embedding-0.6B` | 596 M | Qwen3, 28L, GQA 16/8 | 32,768 | 32–1024 (MRL) | Apache-2.0 | ~1.2 GB |
| `Qwen/Qwen3-Embedding-4B` | 4,022 M | Qwen3, 36L | 40,960 | up to 2560 (MRL) | Apache-2.0 | ~8 GB (cloud) |

- **BGE-M3:** emits dense, sparse (lexical weights), and ColBERT vectors in one forward pass; 100+ languages.
- **mE5:** multilingual baseline with `kk`; requires `query: ` / `passage: ` prefixes; 512-token context.
- **Qwen3-Embedding:** instruction-aware with Matryoshka (MRL) truncation to 256–512 dims (2–4× smaller 35M index); verify `kk` on TriFact eval.
- 35M index memory budget and quantization: [01-architecture.md §7.3](../architecture/docs/01-architecture.md).

## 3. Retrieval: Rerankers

| Model | Params | Architecture | Context | License |
|---|---|---|---|---|
| `BAAI/bge-reranker-v2-m3` | 568 M | XLM-R large cross-encoder | 8,192 | Apache-2.0 |
| `Qwen/Qwen3-Reranker-0.6B` | 596 M | Qwen3 LLM-reranker (yes/no logits) | 40,960 | Apache-2.0 |
| `Qwen/Qwen3-Reranker-4B` | 4,022 M | Qwen3 | 40,960 | Apache-2.0 (cloud) |

Reranking dominates online query latency ($N$ pairs × full pass). Tune `top-N` (`50 → 20–30`) on recall/latency curve, use fp16/ONNX, cap chunk length, and cache scores in Redis.

## 4. Generation (LLM)

| Model | Params | Architecture | Context | License | Q4_K_M | Role |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3.5-0.8B` | 873 M* | Gated DeltaNet + Gated Attn (3:1), 24L, KV 2 | 262,144 | Apache-2.0 | ~0.5 GB | Fast baseline, speculative draft |
| `Qwen/Qwen3.5-2B` | 2,274 M* | Same, H=2048, 24L, KV 2 | 262,144 | Apache-2.0 | ~1.4 GB | **Primary local serving candidate** |
| `Qwen/Qwen3.5-4B` | 4,660 M* | Same, H=2560, 32L, KV 4 | 262,144 | Apache-2.0 | ~2.8 GB | Higher quality; at 4 GB local limit |
| `Qwen/Qwen3.5-9B` | 9,653 M* | H=4096, 32L | 262,144 | Apache-2.0 | ~5.8 GB | Cloud; small-model quality ceiling |
| `Qwen/Qwen3.5-35B-A3B` | 35,952 M (~3B active) | MoE + Gated DeltaNet | 262,144 | Apache-2.0 | — | **Synthetic teacher** (`llama.cpp` GGUF IQ4_XS on Kaggle 2× T4, ADR-0020; §6.5) |
| `Qwen/Qwen3-1.7B` | 2,032 M | Qwen3 dense, 28L, GQA 16/8 | 40,960 | Apache-2.0 | ~1.2 GB | Previous-gen control baseline |
| `issai/LLama-3.1-KazLLM-1.0-8B` | 8,030 M | Llama 3.1 | 128k | **CC BY-NC 4.0**, gated | ~4.8 GB | Kazakh eval-only baseline (NC) |
| `inception42/Llama-3.1-Sherkala-8B-Chat` | 8,288 M | Llama 3.1 | 128k | **CC BY-NC-SA 4.0**, gated | ~5 GB | Kazakh eval-only baseline (NC) |

\* Qwen3.5 parameter counts include a vision encoder; only the language tower is used.

- **Qwen3.5 notes:** 3 linear-attention blocks (Gated DeltaNet) per full-attention block → KV cache grows in only 1/4 of layers (2–4 KV heads). Claims 201 languages, 248,320 vocab (vs. Qwen3 151,936). Use non-thinking mode for grounded QA latency. Requires `transformers`/`llama.cpp` with `qwen3_5` support (`unsloth/Qwen3.5-{2B,4B}-GGUF`) and `vLLM >= 0.17`.
- **Fine-tuning (Unsloth docs, verified 2026-11-18):** 4-bit QLoRA is **not recommended** for Qwen3.5 due to elevated quantization error → **bf16 LoRA** ([ADR-0012](../architecture/adr/0012-qwen35-lora-generation.md)). Estimated bf16 LoRA VRAM: `0.8B ≈ 3 GB`, `2B ≈ 5 GB`, `4B ≈ 10 GB`, `9B ≈ 22 GB`.
- **NC licenses:** KazLLM and Sherkala are non-commercial; use for evaluation baselines only, never as released weights.

## 5. Verification (NLI)

| Model | Params | Architecture | Languages | License |
|---|---|---|---|---|
| `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | 279 M | DeBERTa-v2, 12L | 27 (`ru`, `en`; **no `kk`**) | MIT |
| `MoritzLaurer/bge-m3-zeroshot-v2.0` | 568 M | XLM-R large (BGE-M3) | Multilingual, 8,192 ctx | MIT |

- **Gap:** no pretrained NLI model includes Kazakh; addressed via `kk` XNLI MT (Stage 3, ADR-0015) and `VER-2` fine-tuning (Stage 8).
- **Document claim verifiers (checked 2026-11-21):** MiniCheck (Tang et al., EMNLP 2024, arXiv:2404.10774; `MiniCheck-Flan-T5-L` 0.8B, `Bespoke-MiniCheck-7B`) is English-only and unevaluated multilingually. Reuse only its perturbation methodology for `kk`/`ru` synthetic training pairs; external benchmark LLM-AggreFact is in `datasets.md §8`.

## 6. Auxiliary Models & Empirical Benchmarks

| Task | Selected / Candidates |
|---|---|
| Language ID | **Selected: fastText `lid.176`** (176 langs, CC-BY-SA-3.0); challenger `cis-lmu/glotlid` (2000+ langs, Apache-2.0) — §6.1 |
| OCR | **Selected: Tesseract 5 `kaz+rus+eng`** (`tessdata_best 4.1.0`); EasyOCR, RapidOCR — §6.3 |
| Parsing | **Selected: PyMuPDF → Docling router (`DOC-3`)** ([ADR-0021](../architecture/adr/0021-document-processing-router.md)) — §6.3 |
| Dates | Regex + `dateparser` (`kk`/`ru`/`en`) |

### 6.1 Language ID Benchmark (Stage 2, 2026-11-18)

1,800 reviewed paragraphs (`kk` 506, `ru` 500, `en` 298, `ky` 220, `tt` 233, other 43; `ky`/`tt` from `wikimedia/wikipedia@b04c8d1`). Source: `reports/langid.json` (`eval_langid`, single CPU process).

| Model | macro-F1 @ `0.65` | F1 `kk / ru / en` | Best threshold → macro-F1 | False accept `ky / tt` as `kk` | Para/s | RSS (MB) |
|---|---|---|---|---|---|---|
| `lid.176` | `0.945` | `0.981 / 0.952 / 0.943` | `0.3 → 0.970` | `2.7% / 5.2%` | 19,509 | 133 |
| `GlotLID v3` (`model.bin`) | `0.918` | `0.988 / 0.956 / 0.845` | `0.1 → 0.968` | `0.0% / 0.4%` | 3,065 | 753 |
| `GlotLID small` (`model_v4_small.bin`) | `0.886` | `0.971 / 0.940 / 0.794` | `0.0 → 0.951` | `0.5% / 0.4%` | 14,360 | 132 |

- `lid.176 − GlotLID` at `0.65`: 95% bootstrap CI `[0.016, 0.039]` (vs. small: `[0.046, 0.074]`), 6.4× faster. `GlotLID` remains configurable (`curation.langid.model`) for sources with `ky`/`tt` contamination.
- **Case sensitivity (checked 2026-11-21 on curated `multidomain_kk`):** `lid.176` misclassifies ALL-CAPS Kazakh headlines (`ҚАЗАҚСТАН АРМЕНИЯНЫ СОҢҒЫ РЕТ…`) as `ru` (`p = 0.3–0.5`); lowercasing fixes 47% of low-confidence `ru` docs to `kk` (`p >= 0.65`; GlotLID agreed `kk` on 80%). Since 2026-11-21, `LangId.predict` lowercases strings with no lowercase letters (benchmark unchanged: macro-F1 `0.946`, `kk` F1 `0.981`).

### 6.2 Tokenizer Fertility (Stage 2, 2026-11-18)

Tokens per whitespace word on up to 3,000 curated docs per `(source, lang)` (first 4,000 chars), pinned `tokenizer.json` revisions (`params.yaml: curation.stats.tokenizers`). Source: `reports/corpus_stats.json`.

| Tokenizer | Vocab | `kk` | `ru` | `en` |
|---|---|---|---|---|
| XLM-R (`FacebookAI/xlm-roberta-base`) | 250,002 | `2.13` | `1.98` | `1.54` |
| Qwen3 (`Qwen/Qwen3-1.7B`) | 151,669 | `4.87` | `2.78` | `1.54` |
| Qwen3.5 (`Qwen/Qwen3.5-2B`) | 248,070 | `3.65` | `2.21` | `1.54` |

- `kk` uses 2.3× more tokens on Qwen3 and 1.7× more on Qwen3.5 vs. XLM-R (directly affects Stage 7 context budget and latency). XLM-R `<unk>` ≈ `1e-5`; Qwen byte-level BPE has zero `<unk>`.

### 6.3 Parsing & OCR Benchmark (Stage 4, 2026-11-20)

Versions: Docling `2.150.1` (MIT; Heron layout, TableFormer), PyMuPDF `2.1` (AGPL-3.0 / Artifex commercial), Tesseract `5.4.0` + `tessdata_best 4.1.0` (`kaz`/`rus`/`eng`, Apache-2.0), EasyOCR `1.7.2` (Apache-2.0), RapidOCR `3.10` PP-OCRv5 `cyrillic` (Apache-2.0). Reference host: Ryzen 7 5800HS, 8 threads, CPU torch; dataset `data/eval/docproc`; source `reports/docproc.json`.

- **Kazakh glyph coverage:** EasyOCR `cyrillic_g2` (207 chars) includes `ғ қ ө ү і` but lacks `ә ң ұ һ` (no Kazakh model). RapidOCR `cyrillic` substitutes `қ→к`, `ң→н`, `ү→у`.
- **Scan CER (`pdf_scan`, %, 95% bootstrap CI):**

| Engine | `kk` | `ru` | `en` | s/page | Peak RSS (MB) |
|---|---|---|---|---|---|
| Docling + Tesseract CLI `kaz+rus+eng` | `0.74 [0.43, 1.02]` | `0.54 [0.39, 0.72]` | `0.50 [0.29, 0.70]` | `8.6` | 1,535 |
| PyMuPDF + Tesseract (full page, 300 dpi) | `3.52 [1.06, 7.17]` | `2.97 [0.64, 5.81]` | `3.56 [0.87, 6.30]` | `5.4` | 587 |
| Docling + RapidOCR `cyrillic` | `20.85` | `19.28` | `10.30` | `10.1` | 1,890 |
| Docling + EasyOCR `ru,en` | `36.06` | `5.43` | `5.41` | `27.0` | 4,053 |

- PyMuPDF + Tesseract lags mainly due to columnar table misreads and missed headings (heading F1 `0.70` vs. `0.99`). Docling converts em-dashes to hyphens (+0.03 pp CER on text PDFs vs. PyMuPDF).

### 6.4 VLM-OCR Candidates (Published Numbers, Checked 2026-11-21)

Sources: OmniDocBench v1.6 leaderboard (`opendatalab/OmniDocBench`, README 2026-11-12) and `allenai/olmocr` README. **Published external figures — not TriFact measurements.**

| Model | Size | Weight License | OmniDocBench v1.6 Overall | olmOCR-Bench |
|---|---|---|---|---|
| `PaddleOCR-VL-1.6` | 0.9B | Apache-2.0 | `96.34` | `80.0` (v1, authors) |
| `MinerU2.5-Pro` | 1.2B | Apache-2.0 | `95.75` | — |
| `GLM-OCR` | 0.9B | MIT | `95.22` | — |
| `dots.ocr` | 3B | MIT | `90.77` | — |
| `olmOCR 2` | 7B | Apache-2.0 | — | `82.4` |
| Tesseract 5 (current, §6.3) | — | Apache-2.0 | not ranked | not ranked |

- OmniDocBench (1,651 pages) is mostly Chinese/English (~1/3 formulas); none test Kazakh. On Wild-OmniDocBench photos, v1 PaddleOCR-VL and MinerU2.5 drop ~20 pts, dots.ocr ~10 pts. All require cloud GPU; evaluation protocol in `trifact-document-processing/references/engines.md`.

### 6.5 Teacher, Pre-Labeling & Data Classifiers (Stages 2–3, 2026-11-21)

- **Teacher `Qwen3.5-35B-A3B` (`UD-IQ4_XS`, 16.28 GiB) on `llama.cpp b12032`, Kaggle 2× T4 (`--split-mode layer -fa on -kvu`), `llama-batched-bench`, 16 slots:**

| `-b / -ub` | Prefill tok/s (prompt 550 / 1,500) | Decode tok/s |
|---|---|---|
| `2048 / 512` | `1,210 / 1,134` | `~186` |
| `2048 / 1024` | `1,327 / 1,254` | `~190` |
| `4096 / 2048` | `1,345 / 1,496` | `~188` |

  - 32 slots × 550: decode `220` vs. `201` tok/s at 16 slots (marginal gain). Single-sequence `llama-bench`: `pp512 = 911`, `tg64 = 57` tok/s. Server run (2026-11-20, `-ub 512`) reached ~50% of benchmark (`322` prefill + `49–56` decode tok/s); prompts are 5.6–6.6× longer than responses, making prefill dominant.
- **`textdetox/bert-multilingual-toxicity-classifier` (`8fe5526e`, mBERT 178M, OpenRAIL++):** trained on `en`/`ru`/`uk`/`tt`/`de`/`es`/… without `kk` (`kk` scores unverified); CPU ~24 docs/s at 256 tokens.
- **GLiNER `urchade/gliner_multi-v2.1` on `kk`/`ru`/`en` silver NER (600 sentences, threshold 0.5):** exact span agreement with silver: `kk 67%`, `ru 75%`, `en 74%`. Frequent false positives on common nouns (`«отец|PER»`, `«The museum|ORG»`, `«Похоронен|PER»`) → useful as disagreement signal only, not noise estimator (CPU 0.7 s/sentence).
- **`bge-m3` MT query QC:** across 3,000 eval MT queries, `cos(src, mt) < 0.5` flags clear NLLB errors (`«wafer at communion» → «тостаған»`, `«ductile» → «упруго»`, `«Құдатғу біліг» → «Code of Conduct»`). 10th-percentile cosine ranges from `0.62` (`en-kk`) to `0.80` (`kk-ru`).
