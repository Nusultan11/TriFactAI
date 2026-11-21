# 9. Model Registry и гейт продвижения по метрикам

Date: 2026-11-18

## Status

Accepted

Amended by [22. Протокол оценки](0022-evaluation-protocol.md)

## Context

Моделей несколько (NER, embedder, reranker, LLM-адаптер), и нужно гарантировать, что сервис использует проверенные версии.

## Decision

Все модели регистрируются в MLflow Model Registry. Сервисы при старте загружают версию с alias `champion`. Новая версия получает `challenger` и становится `champion` только если Evaluation Harness подтвердил пороги из `params.yaml: promotion` (архитектура §10) на замороженном тесте, с bootstrap CI или ≥ 3 сидами. Если fine-tuning не обыграл baseline, он не продвигается.

## Consequences

Каждый ответ API содержит версии моделей; откат — перенос alias.
