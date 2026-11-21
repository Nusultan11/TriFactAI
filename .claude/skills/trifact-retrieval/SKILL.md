---
name: trifact-retrieval
description: Cross-lingual retrieval over tens of millions of chunks in TriFact — chunking choice, embedders (BGE-M3, mE5, Qwen3-Embedding, MRL), BM25/sparse, hybrid fusion (RRF, DBSF), Qdrant quantization, rescoring and HNSW, rerankers and calibration, embedder and reranker fine-tuning with hard negatives and false-negative filtering, scale ladder, evaluation over 9 language directions. Use for stage 6 work in trifact.retrieval, embed_corpus, index_corpus, train_retriever, train_reranker, eval_retrieval (поиск, эмбеддинги, Qdrant, reranker).
---

# Retrieval TriFact AI

## Когда применять

- Эксперименты RET-0…RET-7 и RET-S, выбор чанкинга, индексация корпуса, дообучение эмбеддера и reranker.
- Нужный фрагмент не находится или находится на неправильном языке; поиск медленный или не помещается в RAM.

Сборка контекста для LLM и сквозная диагностика ответа — `trifact-rag-pipeline`. Статистика сравнения — `trifact-evaluation`.

## Опорные документы

| Тема | Владелец |
|---|---|
| Эксперименты, бюджет памяти 35M, метрики | архитектура [§7.3](../../../docs/architecture/docs/01-architecture.md), §8.2 |
| Qdrant: dense + sparse, два уровня, alias | [ADR-0002](../../../docs/architecture/adr/0002-qdrant-for-dense-and-sparse.md), [ADR-0013](../../../docs/architecture/adr/0013-two-tier-qdrant-index.md) |
| Эмбеддер по умолчанию | [ADR-0003](../../../docs/architecture/adr/0003-bge-m3-default-embedder.md) |
| NER — только мягкий сигнал | [ADR-0004](../../../docs/architecture/adr/0004-hybrid-retrieval-independent-of-ner.md) |
| Batch-эмбеддинги и сборка коллекции | [ADR-0014](../../../docs/architecture/adr/0014-sharded-batch-jobs.md) |
| Модели | [models.md §2–3](../../../docs/research/models.md) |
| Параметры | `params.yaml: retrieval`, `chunking`, `scale.index_steps` |
| Данные: мало kk-kk пар, чанков больше цели | [PROGRESS «Что учесть дальше»](../../../docs/PROGRESS.md) |

## Сначала оценка

- Замороженный `data/eval/retrieval`, qrels и прогоны в ranx. Метрики Recall@5/10, nDCG@10, MRR — по каждому из 9 направлений (язык запроса × язык источника) и среднее.
- Для первой стадии отдельно **candidate recall@50/100**: это потолок reranker.
- Пул оценок закрыт, а неоценённый документ не обязательно нерелевантен. Для новой системы считается judged@10. При низкой доле оценённых документов выборка неоценённых топ-результатов проверяется вручную.
- Ресурсы: latency p50/p95 при целевой конкурентности, QPS, RAM, диск, время индексации.
- Gold-пассаж привязывается к чанку по пересечению смещений. При сравнении вариантов чанкинга это делается одинаково для всех.

## Чанкинг

Варианты `chunking.variants` выбираются по метрикам retrieval, а не по интуиции. Дешёвые усиления, которые проверяются абляцией:

- заголовок документа и раздела в тексте, который эмбеддится (contextual chunk header);
- late chunking для длинноконтекстного эмбеддера: эмбеддинг всего документа, затем пулинг по границам чанков;
- смежные чанки одного документа склеиваются при выдаче, если обе части релевантны.

## Эмбеддер

Критерии выбора по порядку:

1. Качество на своём eval по 9 направлениям. MTEB/MMTEB почти не проверяют kk.
2. Межъязыковое выравнивание: разрыв худшего направления со средним (гейт cross-lingual).
3. Память индекса: размерность, MRL, бинаризуемость.
4. Скорость кодирования 35M+ чанков.
5. Лицензия.

Практика:

- Префиксы и инструкции строго по карточке: mE5 — `query: ` / `passage: `; Qwen3-Embedding — инструкция только у запроса.
- Векторы L2-нормализуются.
- Доля чанков длиннее `max_length` эмбеддера контролируется (EDA).
- Смена эмбеддера означает новую коллекцию и переключение alias после гейта.

## Sparse и гибрид

- Sparse-нога — BM25 с IDF (Qdrant) или sparse BGE-M3. Казахская морфология без стемминга ослабляет BM25, поэтому обе ноги сравниваются по kk-направлениям.
- Слияние RRF (`retrieval.rrf_k`), DBSF или взвешенное. Решение принимается по каждому направлению: на межъязыковых направлениях лексическое пересечение мало, и sparse может мешать. Варианты — веса слияния по паре языков или перевод запроса для sparse-ноги (эксперимент).
- В режиме eval ноги запрашиваются отдельно и логируются их ранги (ADR-0013), чтобы видеть вклад каждой.

## Reranker

- Top-N выбирается по кривой «качество против латентности» (models.md §3), с ограничением длины пары и fp16 или ONNX.
- Оценка reranker — логит, а не вероятность. Если нужен порог «нет релевантных доказательств» (ранний отказ), калибровать Platt-масштабированием на dev по языкам.
- Кандидаты для reranker берутся из реальной первой стадии, той же, что в сервинге.

## Дообучение (RET-6, RET-7)

- **Данные.** Пары этапа 3 и hard negatives. Сэмплирование по направлениям уравновешивается, чтобы короткие параллельные предложения KazParC не доминировали над пассажами.
- **Майнинг негативов** текущей моделью из диапазона рангов (например 10–100). Ложные негативы отсекаются оценкой cross-encoder: `max_score` и относительный запас к позитиву (positive-aware mining, NV-Retriever). `sentence_transformers.util.mine_hard_negatives` умеет это из коробки.
- **Loss:** `CachedMultipleNegativesRankingLoss` (большой батч на малой памяти) или `CachedGISTEmbedLoss` с guide-моделью против ложных in-batch негативов; батч-сэмплер `NO_DUPLICATES`. Для сокращения размерности — `MatryoshkaLoss`.
- **Забывание.** Низкий LR, смесь с общими парами, контроль на всех 9 направлениях и на внешнем наборе (MIRACL). Гейт требует отсутствия регресса ни на одном направлении.
- **Reranker:** BCE или listwise loss на (запрос, позитив, k негативов из первой стадии). Опция — дистилляция оценок LLM-reranker.

## Масштаб (RET-S)

Ступени `scale.index_steps`, квантизация, HNSW в RAM или на диске, bulk-загрузка — [references/qdrant-at-scale.md](references/qdrant-at-scale.md). Главное правило: потери ANN и квантизации меряются отдельно от потерь модели. На подвыборке точный поиск (`exact=true`) служит эталоном для recall приближённого.

## Диагностика промахов

Для каждого непойманного gold-пассажа определить стадию:

1. Пассажа нет в индексе: отфильтрован курацией или kk-фильтром, потерян при разборе.
2. Ответ разрезан между чанками.
3. Пассаж не попал в top-100 ни одной ноги: представление модели.
4. Пассаж был в ногах, но потерян при слиянии.
5. Потеря ANN или квантизации: есть при точном поиске, нет при приближённом.
6. Пассаж был в кандидатах, но понижен reranker.
7. Ошибка разметки: найден релевантный, но неоценённый пассаж.

Доли считаются по направлениям. На худшем направлении вручную разбираются ≥ 50 случаев. Частый kk-специфичный эффект — предпочтение документов на языке запроса при межъязыковом поиске.

## Критерии готовности

- Champion не хуже предыдущего ни на одном из 9 направлений, среднее лучше с CI > 0 (`promotion.retrieval_*`); худшее направление в пределах `promotion.worst_direction_max_gap`.
- Кривые масштаба записаны: recall против точного поиска, латентность, RAM, диск, время индексации. Решение о Serving VM принято (ADR-0013).
- Анализ промахов по стадиям и направлениям записан в MLflow `retrieval`.

## Источники

- Chen et al., *BGE M3-Embedding*, arXiv:2402.03216; Wang et al., *Multilingual E5*, arXiv:2402.05672; Zhang et al., *Qwen3 Embedding*, arXiv:2506.05176.
- Enevoldsen et al., *MMTEB*, arXiv:2502.13595; Zhang et al., *MIRACL*, arXiv:2210.09984.
- Cormack et al., *Reciprocal Rank Fusion*, SIGIR 2009; Kusupati et al., *Matryoshka Representation Learning*, arXiv:2205.13147.
- Moreira et al., *NV-Retriever: positive-aware hard-negative mining*, arXiv:2407.15831; Qu et al., *RocketQA*, arXiv:2010.08191; Solatorio, *GISTEmbed*, arXiv:2402.16829; Gao et al., *GradCache*, arXiv:2101.06983.
- Günther et al., *Late Chunking*, arXiv:2409.04701; Anthropic, *Introducing Contextual Retrieval* (2024), anthropic.com/news/contextual-retrieval.
- sentence-transformers docs: loss overview, `mine_hard_negatives` (проверено 2026-11-21); ranx — github.com/AmenRa/ranx.
