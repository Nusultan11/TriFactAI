# 12. Qwen3.5 Generation Family, bf16 LoRA Instead of QLoRA

Date: 2026-11-18

## Status

Accepted

Supersedes [5. Qwen3-1.7B + QLoRA](0005-qwen3-qlora-local-gguf-serving.md)

## Context

Qwen3.5 Small (`0.8B`/`2B`/`4B`/`9B`) uses a 3:1 Gated DeltaNet + Gated Attention hybrid (2–4 KV heads, 262k context, 201 languages, 248k vocab), growing KV cache in only 1/4 of layers and reducing Kazakh tokenizer fertility. Unsloth documentation advises against 4-bit QLoRA for Qwen3.5 due to elevated quantization errors and specifies bf16 LoRA VRAM requirements (`0.8B ≈ 3 GB`, `2B ≈ 5 GB`, `4B ≈ 10 GB`, `9B ≈ 22 GB`). Requires `vLLM >= 0.17` and current `transformers`/`llama.cpp`.

## Decision

- Primary local serving target: `Qwen/Qwen3.5-2B` (GGUF `Q4_K_M` ≈ 1.4 GB on `llama.cpp`, non-thinking mode); additional candidates `0.8B` (fast/draft) and `4B`/`9B` (cloud quality); control baseline `Qwen/Qwen3-1.7B`.
- Fine-tuning: **bf16 LoRA** (PEFT or Unsloth) on cloud Ampere+ GPU (24 GB+).
- Synthetic teacher: `Qwen/Qwen3.5-35B-A3B` (ADR-0015; execution subsequently amended by ADR-0020 and ADR-0025).
- Serving: merged GGUF on `llama.cpp` locally; `vLLM` with LoRA adapter on GPU servers, using the exact training chat template.
- Final size selection governed by Stage 7 promotion gate (ADR-0009); serve best baseline if fine-tuning does not win.

## Consequences

Cloud bf16 LoRA costs more than QLoRA but is more stable. Verify `qwen3_5` compatibility in PEFT, `llama.cpp`, and `vLLM` at Stage 7 start; measure `Q4`/`Q5`/`Q8` quantization impact on frozen eval. Non-commercial KazLLM / Sherkala remain evaluation-only baselines.
