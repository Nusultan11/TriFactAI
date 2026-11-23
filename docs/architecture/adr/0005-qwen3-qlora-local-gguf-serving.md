# 5. Qwen3-1.7B + QLoRA, Cloud Training, Local GGUF Serving

Date: 2026-11-18

## Status

Superseded by [12. Qwen3.5 Generation Family, bf16 LoRA](0012-qwen35-lora-generation.md)

## Context

Required an LLM supporting KK/RU/EN that could be fine-tuned in the cloud and served within a 4 GB RTX 3050 local budget.

## Decision

Use `Qwen/Qwen3-1.7B` with cloud QLoRA (NF4, `r=16`, 24 GB+ GPU). Serve locally via merged GGUF `Q4_K_M` on `llama.cpp` server (OpenAI-compatible API) and on GPU servers via `vLLM` with LoRA.

## Consequences

Quantization impact (`fp16` vs. `Q4`) must be measured on frozen eval; OpenAI-compatible client keeps the model easily swappable.
