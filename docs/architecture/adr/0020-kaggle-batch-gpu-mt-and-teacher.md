# 20. Kaggle Batch GPU, NLLB Machine Translation, and llama.cpp Teacher

Date: 2026-11-18

## Status

Accepted

Amends [14. Sharded Idempotent Batch Jobs](0014-sharded-batch-jobs.md) (GPU execution host) and [15. Silver and Synthetic Data](0015-silver-and-synthetic-data.md) (`kk` XNLI translation engine)

Amended by [25. Cloud-Only Computing and Stage 1–3 Recovery](0025-cloud-only-computing.md) (execution platform selection)

## Context

With `params.yaml: cloud.budget_usd: 0`, the local 4 GB RTX 3050 cannot run `Qwen3.5-35B-A3B` or mass MT. Kaggle provides free 2× Tesla T4 (15 GB each, **sm75, no bf16**), 30 GPU-h/week quota, 12 h max/session, and at most 2 concurrent GPU sessions.

Pilot checks on Kaggle showed:
- `vLLM` cannot run `Qwen3.5-35B-A3B` on T4 (Marlin GPTQ/AWQ MoE kernels require sm80; FP8 requires sm89).
- `NLLB-200-distilled-1.3B` (fp16, beam 2) translates at 27 sent/s on 1× T4; `distilled-600M` back-translates at 76 sent/s.
- Multi-threaded multi-GPU model loading fails with `expected scalar type Half but found Float` because `torch.set_default_dtype` is global and not thread-safe.

## Decision

- **Stage 3 GPU jobs on Kaggle:** `python -m trifact.batch.kaggle run <job>` uploads inputs as a private dataset, runs a self-contained private kernel from `src/trifact/batch/jobs/` using public HF weights only (never sending tokens or user data), downloads outputs, and records `run.json` (kernel version, wall time, quota before/after, `$0` USD).
- **One OS process per GPU** (`CUDA_VISIBLE_DEVICES`), never threads.
- **NLLB-200 for `kk` MT** (`facebook/nllb-200-distilled-1.3B` + `distilled-600M` back-translation, pinned in `params.yaml: mt`): deterministic, 20–30× cheaper than LLM generation; CC-BY-NC-4.0 propagates `release: false` in `dataset.yaml`.
- **Teacher (`Qwen3.5-35B-A3B` GGUF `UD-IQ4_XS`, 17.5 GB):** `llama.cpp` CUDA server on 2× T4 (`--split-mode layer`, 16 slots, non-thinking); use prebuilt CUDA binary if `--list-devices` detects GPUs, else compile for sm75. Plan synthetic volume from pilot throughput and quota.
- **Keep ADR-0015 filters mandatory:** back-translation chrF, langid, length ratio, non-copy, `nli_score` (`mDeBERTa-xnli`), and synthetic query/QA filters (`trifact.data.synthetic`).
- **Wikidata parsing:** use bz2 block-range workers (`trifact.data.wikidata.parse_range`) instead of `indexed_bzip2` (~45 MB index per unpacked GB on ~1 TB JSON).

## Consequences

Zero paid cloud spend, but synthetic volume is bounded by 30 free GPU-h/week on 2× T4 (`datasets.md §6` synthetic targets are unreachable inside one week's quota; log deviations in `PROGRESS.md`). MT-derived eval sets require human review (Stage 3.5) before freezing.
