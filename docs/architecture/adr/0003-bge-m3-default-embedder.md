# 3. BGE-M3 как embedding-модель по умолчанию

Date: 2026-11-18

## Status

Accepted

## Context

Нужна мультиязычная модель с поддержкой казахского, длинного контекста и межъязыкового поиска, помещающаяся в 4 GB вместе с LLM.

## Decision

Champion по умолчанию — `BAAI/bge-m3` (dense + sparse в одной модели, 8192 токена). Challenger — `intfloat/multilingual-e5-large`. Выбор фиксируется по Recall@10/nDCG@10 на 9 языковых направлениях (эксперименты RET-1 и RET-2).

## Consequences

Решение пересматривается по метрикам; интерфейс `Embedder` скрывает конкретную модель.
