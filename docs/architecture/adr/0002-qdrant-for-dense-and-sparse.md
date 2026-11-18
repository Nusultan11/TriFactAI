# 2. Qdrant для dense- и BM25-поиска

Date: 2026-11-18

## Status

Accepted

Amended by [13. Двухуровневый индекс Qdrant](0013-two-tier-qdrant-index.md)

## Context

Гибридный поиск требует лексического (BM25) и векторного индекса. Отдельный Elasticsearch/OpenSearch — ещё один тяжёлый сервис (≥1–2 GB RAM) и синхронизация двух индексов.

## Decision

Одна коллекция Qdrant с named vectors: `dense` (BGE-M3, 1024, cosine) и `bm25` (sparse с IDF-модификатором). Фильтры по `document_id`, `lang`, `entity_ids` через payload-индексы. RRF считаем в приложении, чтобы контролировать `k` и логировать ранги каждого метода.

## Consequences

Меньше инфраструктуры и одна точка консистентности. Казахская морфология в BM25 слабее, чем у Elasticsearch с анализаторами; компенсируем нормализацией текста и сравнением sparse-векторов BGE-M3 как альтернативы (эксперимент).
