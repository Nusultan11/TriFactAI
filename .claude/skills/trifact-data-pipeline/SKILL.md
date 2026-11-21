---
name: trifact-data-pipeline
description: Data engineering for TriFact corpora and task datasets at tens of millions of rows — Parquet lake and manifests, normalization, language ID, quality filters, MinHash dedup, 13-gram decontamination, leakage-safe splits, gold/silver/synthetic/MT sets, license lineage, DVC, memory-bounded Polars/DuckDB/PyArrow streaming. Use when adding a source, writing or debugging a stage in trifact.data, trifact.lake or trifact.batch, building training or eval sets, or fixing OOM or leakage (подготовка данных, дедупликация, утечки, синтетика).
---

# Data pipeline TriFact AI

## Когда применять

- Новая стадия или новый источник в `trifact.data`, `trifact.lake`, `trifact.batch`.
- Сборка task-наборов: NER gold/silver, retrieval-пары, SFT, NLI, eval.
- Стадия падает по памяти, работает медленно, не возобновляется или даёт утечку train/eval.

Анализ распределений и находок — скилл `trifact-eda`. Парсинг пользовательских файлов — `trifact-document-processing`.

## Опорные документы (не пересказывать, читать)

| Тема | Владелец |
|---|---|
| Слои, шарды, манифесты | [ADR-0011](../../../docs/architecture/adr/0011-parquet-data-lake.md), [ADR-0017](../../../docs/architecture/adr/0017-raw-layer-acquisition-protocol.md), архитектура [§5](../../../docs/architecture/docs/01-architecture.md) |
| Раннер стадий, отпечатки, маркеры | [ADR-0018](../../../docs/architecture/adr/0018-curation-stages-without-datatrove.md), `trifact.data.curation` |
| Batch-задания, Kaggle | [ADR-0014](../../../docs/architecture/adr/0014-sharded-batch-jobs.md), [ADR-0020](../../../docs/architecture/adr/0020-kaggle-batch-gpu-mt-and-teacher.md) |
| Дедуп и деконтаминация | [ADR-0019](../../../docs/architecture/adr/0019-dedup-and-decontamination-policy.md) |
| Сплиты и замороженный eval | [ADR-0007](../../../docs/architecture/adr/0007-leakage-safe-splits.md), `trifact.data.split`, `trifact.data.build_eval` |
| Silver, синтетика, лицензии | [ADR-0015](../../../docs/architecture/adr/0015-silver-and-synthetic-data.md), [datasets.md §7](../../../docs/research/datasets.md) |
| Пороги и параметры | `params.yaml`: `lake`, `curation`, `split`, `teacher`, `mt`, `sft`, `scale` |
| Известные ловушки памяти и данных | [PROGRESS «Что учесть дальше»](../../../docs/PROGRESS.md) |

## Порядок работы над стадией

1. **Контракт.** Входные манифесты, схема выхода (архитектура §5.2), единица работы (входной шард), ключи (`doc_id` с пространством имён `<source>/<lang>/<id>`), причины отсева. Новый цикл обработки не пишется: используется `curation.run_units`, `StageDir` и `UnitWriter` или задание по шаблону `trifact.batch`.
2. **Бюджет памяти до запуска.** Оценить байты на строку × батч × воркеры. Батч режется по байтам (`curation.rows_per_batch`), а не по строкам: строки wikitext весят десятки KB. В каждом процессе `POLARS_MAX_THREADS = cpu_count // workers`.
3. **Пилот.** `scale.profile: pilot` или `--limit`; затем проверить манифест, причины отсева и пик RSS (`lake.stats.ResourceMonitor` меряет всё дерево процессов).
4. **Полный прогон** только из закоммиченного кода, иначе `code_version` будет `-dirty`. Профиль `full` проверяется в манифесте.
5. **Выход:** `dataset.yaml` (входы, отпечаток, строки, отсев по причинам, лицензии входов, `release`), `card.md`, `reports/<стадия>.json`, run в MLflow.

## Выбор инструмента

| Задача | Инструмент | Почему и чем рискуем |
|---|---|---|
| Признаки и фильтры по строкам, батч за батчем | Polars (выражения, `iter_batches` PyArrow) | векторизовано, один проход; отфильтрованные строки держат string views на весь буфер, поэтому результат копируется или пишется сразу |
| Большие join, group by, подсчёт пересечений | DuckDB (`memory_limit`, `temp_directory`) | out-of-core: соблюдает `memory_limit` и сбрасывает промежуточные данные на диск; перед переходом на Polars streaming замерить пик RSS на полном объёме |
| Выборка строк по ключам из больших шардов | PyArrow `dataset` + фильтр `is_in` (`task_common.rows_with_keys`) | pushdown по row group; semi join по ленивому скану Polars раздувал память |
| Запись шардов | PyArrow/`UnitWriter`: zstd, `.tmp` → fsync → rename | атомарность и возобновление |
| MinHash, n-граммы, хеши | numpy (`trifact.data.hashing`) | 64-битные смешанные хеши, тесты несмещённости Jaccard |
| Кластеры дублей | `scipy.sparse.csgraph.connected_components` | глобальный шаг по всем источникам |
| Схемы и инварианты | pandera (Polars) на каждом батче или выборке | ловит дрейф схемы между шардами |
| GPU-задания (MT, учитель, NLI-оценка) | Kaggle по ADR-0020, процесс на GPU | T4 без bf16; квота, а не деньги |

`pandas` и загрузка набора целиком в RAM запрещены ([AGENTS §6.2](../../../AGENTS.md)). datatrove не используется (ADR-0018).

## Курация

- **Нормализация до всего остального.** Хеши, langid, дедуп и деконтаминация считаются по тексту после `trifact.ingestion.normalize`, онлайн и офлайн одинаково. Казахские буквы `ә ғ қ ң ө ұ ү һ і` и гомоглифы латиницы и кириллицы — частый источник ложных различий.
- **Язык.** Документ и абзацы отдельно. Для kk полезен второй сигнал — доля специфичных казахских букв (`curation.downstream`). Если в источнике есть ky/tt/uz, переоценить модель (`curation.langid.candidates`, отчёт `eval_langid`).
- **Фильтры качества.** Пороги разрешаются default → язык → домен. Подбираются по ручной выборке, а не на глаз. Каждый отсев получает причину. После подбора сравнить распределения raw → clean по языку и домену: фильтр не должен непропорционально выбивать kk или целый домен.
- **Дедуп.** Вероятность связать пару с Jaccard `s` при `b` бандах по `r` строк равна `1 − (1 − s^r)^b`. Схему выбирают по этой кривой и проверяют настоящим Jaccard на случайных парах (`pair_check`). Шаблонные заготовки Wikipedia похожи по 5-граммам, но описывают разные сущности (ADR-0019). Семантический дедуп (SemDeDup) уместен для синтетических запросов и SFT, но не для корпуса 30M+.
- **Деконтаминация.** Строки с 13-граммой eval удаляются, шаблонные граммы игнорируются, повторный скан выхода обязан дать 0. MinHash против eval не годится: абзац внутри длинного документа даёт малый Jaccard.

## Task-наборы

- **Сплит по исходному документу.** Ключи `doc_key` и `ctx_key` (хеш контекста) проверяются отдельно: в KazQAD один контекст переиспользовался под разными id. Перевод тестового примера — тот же пример. Параллельные корпуса (KazParC) и MT-производные проверяются на межъязыковую утечку через общий `doc_key`.
- **Gold.** BIO валиден, `answer_start` указывает на ответ, метки отображены через `configs/labels.yaml`.
- **Silver** ([ADR-0015](../../../docs/architecture/adr/0015-silver-and-synthetic-data.md)). Шум оценивается против gold по типу на общих поверхностях. Главный дефект silver NER — пропущенные сущности, то есть ложные `O`. Отсюда фильтры (`ner.silver.capital_filter`) и обучение по curriculum.
- **Синтетика.** Фильтры идут от дешёвых к дорогим: parse → length → language → copy → round-trip → duplicate → NLI. У каждого фильтра записывается доля отсева. Проверяется разнообразие: distinct-n, первые n-граммы вопросов, покрытие тем корпуса. Объём планируется от замеренной пропускной способности учителя и квоты, а не от цели. В eval синтетика не попадает без ручной проверки.
- **MT** (NLLB, ADR-0020). Фильтры: chrF обратного перевода, langid, отношение длин, отсутствие копий. Производные наследуют `release: false`.
- **Смеси.** Доли языков задаются явно (`scale.min_kk_share`, `sft.mix`). Если источники несоизмеримы, применяется температурное сэмплирование `p ∝ n^α`; в XLM-R и mT5 `α = 0.3`. Короткие параллельные предложения не должны доминировать в межъязыковых направлениях над пассажами.

## Типичные ошибки

| Симптом | Причина | Проверка и исправление |
|---|---|---|
| OOM на части источников | батч по строкам при длинных текстах; Polars занял все ядра в каждом воркере | батч по байтам, лимит потоков, меньше воркеров |
| Хеши `dvc.lock` не воспроизводятся в чистом клоне | CRLF в рабочей копии | `tests/test_architecture.py::test_dvc_dependencies_use_lf_line_endings` |
| Один и тот же id в двух языках | id источника не уникален (MIRACL docid) | id с пространством имён |
| Метрика eval подозрительно высока | общий контекст или перевод в train | `ctx_key`, хеши нормализованных текстов, `metrics.leakage` у `build_eval` |
| Повторный запуск всё пересчитал | отпечаток зависит от нестабильного параметра (время, путь) | отпечаток только от имени стадии, параметров, LF-кода и входных манифестов |
| Стадия пошла в pilot | значение по умолчанию `scale.profile` | профиль в манифесте и в MLflow |
| Mojibake в части пассажей | двойное декодирование cp1251/latin-1 | детектор mojibake EDA, исключение или перекодирование |

## Критерии готовности

- `python -m trifact.lake verify` (или проверка `dataset.yaml`) проходит; строки сходятся с методом проверки.
- Сумма отсева по причинам равна разнице входа и выхода, причины есть в data card.
- Утечка eval ↔ train равна 0 по документу, контексту и нормализованному тексту.
- Пик RSS дерева процессов ≤ `lake.peak_ram_gb`; throughput записан в манифест и MLflow.
- Повторный запуск только проверяет готовое и ничего не пересчитывает.
- Лицензии входов и `release` перечислены; источник без решения по лицензии не используется.

## Источники

- Penedo et al., *The FineWeb Datasets*, arXiv:2406.17557; *FineWeb2*, arXiv:2506.20920 — фильтры и дедуп на веб-масштабе по языкам.
- Rae et al., *Gopher*, arXiv:2112.11446 — эвристики качества (повторы строк и n-грамм, стоп-слова).
- Lee et al., *Deduplicating Training Data Makes Language Models Better*, arXiv:2107.06499; Broder, *On the resemblance and containment of documents*, 1997.
- Abbas et al., *SemDeDup*, arXiv:2303.09540.
- Brown et al., *Language Models are Few-Shot Learners*, arXiv:2005.14165, прил. C — 13-граммная деконтаминация.
- Conneau et al., *XLM-R*, arXiv:1911.02116 — сэмплирование языков с α = 0.3.
- Gebru et al., *Datasheets for Datasets*, arXiv:1803.09010; Pushkarna et al., *Data Cards*, arXiv:2204.01075.
