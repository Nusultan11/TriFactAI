---
name: trifact-ner-linking
description: NER and entity linking for KK/RU/EN in TriFact — PER/ORG/LOC schema with rule-based DATE, subword alignment, XLM-R gold/silver curriculum, GLiNER baseline, partial-annotation noise, Kazakh case suffixes, unseen entities, strict and partial span metrics, calibration, ONNX int8 and vocabulary trimming, Wikidata candidate generation, ranking, NIL and accuracy@1. Use for stage 5 work in trifact.extraction, train_ner, eval_ner, build_kb, annotate_corpus (NER, извлечение сущностей, связывание сущностей).
---

# NER и Entity Linking TriFact AI

## Когда применять

- Обучение, оценка, ускорение NER (эксперименты NER-0…NER-5) и DATE-экстрактора.
- KB, генерация кандидатов и ранжирование для linking.
- Массовая разметка корпуса `annotate_corpus`.

Общая методика обучения, сиды и HPO — `trifact-training-hpo`. Статистика и гейт — `trifact-evaluation`.

## Опорные документы

| Тема | Владелец |
|---|---|
| Схема меток, DATE правилами | [ADR-0006](../../../docs/architecture/adr/0006-unified-ner-label-schema.md), `configs/labels.yaml` |
| NER не участвует в отборе документов для поиска | [ADR-0004](../../../docs/architecture/adr/0004-hybrid-retrieval-independent-of-ner.md) |
| Silver, curriculum | [ADR-0015](../../../docs/architecture/adr/0015-silver-and-synthetic-data.md) |
| Эксперименты, KB, метрики | архитектура [§7.2](../../../docs/architecture/docs/01-architecture.md); гейт §10 и `params.yaml: promotion` |
| Модели | [models.md §1](../../../docs/research/models.md) |
| Шум silver, покрытие kk-меток Wikidata | [PROGRESS, этап 3 и «Что учесть дальше»](../../../docs/PROGRESS.md) |
| Параметры | `params.yaml: ner`, `linking` |

## Данные и выравнивание

- Метка ставится на первый субтокен слова, остальным `-100`. Эвристика «метка на все субтокены» завышает вклад длинных слов: в kk слова длиннее в токенах (фертильность — models.md §6.2).
- Предложения длиннее `ner.max_length` режутся окнами с перекрытием, а не обрезаются. Долю обрезаемых даёт EDA.
- Казахские падежные окончания входят в токен («Астанада» = B-LOC). Для NER это не проблема, но для linking и газеттиров поверхность нормализуется: снимаются окончания (-да/-де/-та/-те, -ның/-нің/-дың/-дің/-тың/-тің, -ға/-ге/-қа/-ке, -ны/-ні/-ды/-ді/-ты/-ті, притяжательные). Recall кандидатов меряется до и после.

## Обучение (что реально работает)

- **Baseline:** GLiNER multi zero-shot (NER-0) и XLM-R base на gold (NER-1). Без них улучшение не доказать.
- **Curriculum silver → gold** (NER-2): сначала эпоха или две на silver, затем gold с меньшим LR. Альтернатива — смесь с пониженным весом silver. Выбирается по dev F1 kk.
- **Шум silver — в основном пропуски.** Сущность без ссылки размечена `O`. Варианты по возрастанию стоимости:
  1. фильтр предложений с неразмеченными заглавными словами (уже есть);
  2. не учить `O` на неразмеченных заглавных токенах (`-100`), то есть частичная разметка;
  3. self-training: модель NER-1 доразмечает silver с порогом уверенности.
- **Дисбаланс.** Для token classification на трансформере взвешивание классов обычно не нужно. Сначала проверить per-type recall, потом решать.
- **CRF-голова** даёт на трансформерах небольшой и непостоянный прирост. Добавлять только как эксперимент с ≥ 3 сидами.
- Гиперпараметры, сиды, чекпоинты — `trifact-training-hpo`. Модель выбирается по entity F1 на dev, не по loss.

## Оценка

- seqeval в **строгом** режиме (`mode="strict"`, `scheme=IOB2`) — основная метрика. В режиме по умолчанию seqeval засчитывает спаны, начатые с `I-`. Метрики по языку × типу, бутстреп по предложениям, парный для сравнения моделей.
- Разделение ошибок границ и типа: `nervaluate` (strict / exact / partial / ent_type). Ошибки границ лечатся данными и нормализацией, ошибки типа — контекстом и схемой.
- **Unseen-F1:** F1 на сущностях, чьей поверхности нет в train. Это главный показатель обобщения, особенно для kk.
- Таксономия ошибок, которую стоит считать:
  - метонимия ORG↔LOC («Қазақстан келісімге қол қойды»);
  - вложенность (LOC внутри ORG: «Алматы университеті»);
  - инициалы и титулы в PER;
  - заглавное первое слово предложения как ложный PER;
  - транслитерация (Абай / Abai);
  - смешанный язык в предложении.

## Калибровка и пороги

- Оценка сущности — среднее или минимум вероятностей её токенов. Температура подбирается на dev по каждому языку.
- Проверка: reliability diagram и ECE (equal-mass бины). Порог для `annotate_corpus` выбирается по кривой precision–coverage: в payload Qdrant лучше меньше, но точнее, потому что сущности служат фильтром.

## Ускорение (NER-5)

1. Экспорт в ONNX (optimum), динамическая int8-квантизация для CPU, fp16 на GPU.
2. Обрезка словаря XLM-R до токенов kk/ru/en-корпуса: у base ~192M из 279M параметров приходится на эмбеддинги (models.md §1).
3. Сортировка по длине и батчинг.
4. Каждое ускорение сравнивается с исходной моделью на том же тесте: просадка F1 ≤ лимита NER-5. Throughput меряется в чанках/с на целевом железе.

## Entity Linking

1. **KB.** Сущности Wikidata с метками kk/ru/en/mul, алиасы, редиректы, якорные тексты ссылок Wikipedia с частотами → приор `p(e | m)`.
2. **Кандидаты.** Точный поиск по нормализованному алиасу, затем триграммы (`pg_trgm`) и снятие окончаний. Метрика стадии — recall@k кандидатов: это потолок всей системы, считается отдельно.
3. **Ранжирование:** приор × совместимость типа NER с типом KB × сходство контекста упоминания с описанием сущности. Сходство считает мультиязычный эмбеддер: kk-контекст против ru/en-описания, потому что kk-описаний мало.
4. **NIL.** Порог на итоговую оценку подбирается на dev; доля NIL меряется отдельно.
5. **Оценка.** accuracy@1 на ручной выборке по языкам, отдельно recall@k кандидатов и accuracy на редких сущностях (мало sitelinks), где приор смещает к популярным.

Альтернативы для следующих экспериментов — bi-encoder + cross-encoder (BLINK), генеративный mGENRE, ReFinED. Брать их, если ранжирование «приор × тип × контекст» упирается в потолок при высоком recall кандидатов.

## Критерии готовности

- Entity F1 (strict) по каждому языку и типу выше baseline со значимостью, условие гейта (`promotion.ner_*`) выполнено.
- Unseen-F1 и таксономия ошибок записаны в MLflow `ner` вместе с throughput на CPU и GPU.
- Для linking записаны recall@k кандидатов, accuracy@1 и доля NIL по языкам.
- ≥ 3 сидов, отчёт — среднее ± CI.

## Источники

- Conneau et al., *XLM-R*, arXiv:1911.02116; Zaratiana et al., *GLiNER*, arXiv:2311.08526.
- Yeshpanov et al., *KazNERD*, arXiv:2111.13419.
- Augenstein et al., *Generalisation in Named Entity Recognition*, arXiv:1701.02877.
- Mayhew et al., *Named Entity Recognition with Partially Annotated Training Data*, CoNLL 2019.
- Abdaoui et al., *Load What You Need: Smaller Versions of Multilingual BERT*, arXiv:2010.05609.
- seqeval — github.com/chakki-works/seqeval; nervaluate — github.com/MantisAI/nervaluate.
- Wu et al., *BLINK*, arXiv:1911.03814; De Cao et al., *mGENRE*, arXiv:2103.12528; Ayoola et al., *ReFinED*, arXiv:2207.04108.
- Guo et al., *On Calibration of Modern Neural Networks*, arXiv:1706.04599.
