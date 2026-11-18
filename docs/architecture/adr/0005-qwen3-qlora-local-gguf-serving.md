# 5. Qwen3-1.7B + QLoRA, обучение в облаке, GGUF локально

Date: 2026-11-18

## Status

Superseded by [12. Семейство Qwen3.5, bf16 LoRA](0012-qwen35-lora-generation.md)

## Context

Нужна LLM с поддержкой RU/KZ/EN, которую реально дообучить и обслуживать на RTX 3050 4 GB.

## Decision

Базовая модель `Qwen/Qwen3-1.7B`. Обучение QLoRA (NF4, r=16) на облачной GPU 24 GB+. Локальный сервинг: merge адаптера → GGUF Q4_K_M → llama.cpp server (OpenAI-совместимый API). На GPU-сервере — vLLM с LoRA. Локально — только smoke-тесты обучения.

## Consequences

Квантизация Q4 может снизить качество: сравниваем метрики fp16 и Q4 на eval-наборе. Клиент генерации работает с любым OpenAI-совместимым сервером, поэтому модель легко заменить.
