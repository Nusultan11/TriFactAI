---
name: trifact-mlops
description: MLOps for TriFact — reproducible DVC stages and pinned revisions, MLflow runs and metric naming, Model Registry aliases champion/challenger and rollback with Qdrant aliases, ML test pyramid (data, regression, behavioral, performance), profiling RSS/VRAM and throughput, resumable sharded jobs, serving batching, ONNX and caches, quality and drift monitoring with Prometheus and Grafana. Use when making a stage reproducible, adding data or model tests, registering or promoting a model, profiling, hardening jobs, or designing alerts (MLOps, MLflow, мониторинг, дрейф).
---

# MLOps TriFact AI

Процедура этапа, проверки перед коммитом, git и бэкапы — [AGENTS.md §5–§8](../../../AGENTS.md); здесь их нет. Скилл о том, как сделать ML-часть воспроизводимой, проверяемой и наблюдаемой.

## Опорные документы

| Тема | Владелец |
|---|---|
| Офлайн-вычисления, MLOps, наблюдаемость, развёртывание | архитектура [§9, §11–§13](../../../docs/architecture/docs/01-architecture.md) |
| Registry и гейт | [ADR-0009](../../../docs/architecture/adr/0009-model-registry-and-promotion-gate.md) |
| DVC, remotes, batch-задания | [ADR-0010](../../../docs/architecture/adr/0010-dvc-with-s3-remote.md), [ADR-0014](../../../docs/architecture/adr/0014-sharded-batch-jobs.md) |
| Ответственность хранилищ | [ADR-0016](../../../docs/architecture/adr/0016-storage-responsibilities.md) |
| Раннер стадий, отпечатки | [ADR-0018](../../../docs/architecture/adr/0018-curation-stages-without-datatrove.md) |
| Замеры и MLflow-хелперы | `trifact.lake.stats` (`ResourceMonitor`, `git_sha`, `log_mlflow`) |

## Воспроизводимость run

Run восстанавливается по git SHA + DVC-хешам + конфигу (НФТ архитектуры §1). Для этого:

- код закоммичен до полного прогона (`git_sha` без `-dirty`);
- стадия в `dvc.yaml` объявляет все `deps` (код модуля, входы), `params` (ключи `params.yaml`) и `outs`, а `dvc.lock` закоммичен;
- модели и токенизаторы закреплены ревизией HF (SHA), а не веткой;
- сиды Python, numpy и torch фиксированы. Для GPU — `torch.use_deterministic_algorithms` там, где это не ломает скорость; остаточная недетерминированность учитывается сидами;
- окружение задано `uv.lock` и образом Docker; версия CUDA и драйвера пишется в run.

## MLflow

- Эксперимент на направление (архитектура §11), run на прогон, вложенные runs для trials HPO и конфигураций бенчмарка.
- В run пишутся:
  - параметры плоским списком с полным конфигом как артефакт;
  - метрики по языкам и направлениям с именами `<metric>/<lang>` или `<metric>/<q>-<d>`, очищенными от недопустимых символов (`lake.stats._mlflow_name`);
  - теги `stage` и `status=final|superseded`;
  - артефакты: отчёты JSON, графики, выходы по примерам и трассы e2e (ADR-0022), матрицы ошибок;
  - ресурсы: время, throughput, пик RSS/VRAM, тип GPU, стоимость.
- Модель логируется с сигнатурой и примером входа, затем регистрируется как новая версия с alias `challenger`. `champion` переносит только гейт (ADR-0009). Сервисы грузят `models:/<name>@champion` при старте.
- Откат — перенос alias. Для эмбеддера alias модели и alias коллекции Qdrant переключаются вместе (архитектура §11).
- Перед логированием — быстрая проверка `/health` сервера, чтобы недоступный MLflow не подвешивал задание (решение этапа 1).

## Пирамида тестов ML

| Уровень | Что проверяет | Пример в TriFact |
|---|---|---|
| Unit | чистые функции на ручных случаях | нормализация, смещения чанков, BIO-конвертация, метрики против эталонных библиотек (seqeval, ranx) на игрушечных данных |
| Данные и схемы | схема и инварианты шардов | pandera-схемы; смещения в пределах текста; BIO валиден; утечка eval = 0 |
| Интеграция | стадия на крошечной фикстуре целиком | `--limit`, `--root <tmp>`, манифест помечен `partial` |
| Регрессия модели | метрика на фиксированном golden-наборе в допуске | NER F1 и nDCG на малом наборе против записанного значения ± допуск |
| Поведенческие | инвариантность и направленные тесты | транслит имён, перестановка пассажей, удаление доказательства → отказ |
| Производительность | нижняя граница throughput, верхняя — RAM | стр/с парсера, чанков/с NER на CPU |
| E2E | API и web на фикстурах | `eval_e2e`, Playwright (этап 10) |

- Тесты с сетью и большими моделями помечаются отдельным маркером и не идут в `pytest -q` по умолчанию: pre-commit должен оставаться быстрым.
- Тесты зеркалят `src/` ([AGENTS §6.5](../../../AGENTS.md)). Архитектурные тесты (`tests/test_architecture.py`) расширяются, когда появляется новое структурное правило.

## Профилирование

| Что | Инструмент |
|---|---|
| RAM всего дерева процессов | `lake.stats.ResourceMonitor` |
| Горячие места CPU в Python | py-spy (sampling, без правки кода) |
| Аллокации Python | memray |
| Планы запросов | Polars `explain()`, DuckDB `EXPLAIN ANALYZE` |
| GPU | `torch.cuda.max_memory_allocated`, `torch.profiler`, nvidia-smi |
| Масштабирование по воркерам | throughput и пик RSS при 1/2/4/8 воркерах — найти перегиб, а не брать максимум |

## Устойчивость заданий

- Единица работы — шард; атомарная запись; маркеры готовности; повтор без пересчёта готового (ADR-0014, ADR-0018).
- Сетевые операции: повтор с экспоненциальной задержкой и таймауты.
- Квота и бюджет проверяются до запуска, задание останавливается при превышении (`cloud.budget_usd`, квота Kaggle).
- Обучение сохраняет чекпоинт с состоянием оптимизатора и шагом, чтобы прерванная GPU продолжила работу.

## Производительность сервинга

- Динамический батчинг в Inference Service с ограничением ожидания; сортировка по длине.
- ONNX Runtime int8 для encoder-моделей на CPU, fp16 на GPU. Прогрев моделей при старте до healthcheck `ready`.
- Кеши Redis: эмбеддинг запроса, оценки reranker по ключу (запрос, chunk_id, версия модели). Версия в ключе обязательна, иначе после смены champion кеш вернёт старые оценки.

## Мониторинг и дрейф

- Технические и качественные сигналы перечислены в архитектуре §12.
- Признаки дрейфа:
  - распределение языков вопросов и длины запросов;
  - распределения максимальной оценки поиска и reranker;
  - доли отказов, `contradicted`, `not_enough_info`;
  - причины `ParseError` и доля страниц с OCR.
- Методы: PSI или JS-дивергенция гистограмм против опорного окна; KS-тест для непрерывных; для эмбеддингов — расстояние центроидов или классификаторный тест двух выборок на выборках.
- Пороги алертов берутся из разброса опорных окон, а не из головы. Алерт ведёт в runbook: что проверить первым.
- Обратная связь: фидбек и выборочная ручная проверка пополняют **новые** версионированные eval-наборы. Замороженный тест не меняется.

## Критерии готовности

- Любой run восстанавливается: SHA чистый, `dvc status` чистый, конфиг и ревизии моделей в run.
- Для нового компонента есть unit- и data-тесты, регрессионный тест метрики и тест производительности.
- Модель зарегистрирована, alias переносит только гейт, откат проверен.
- Для сервиса есть метрики, дашборд и алерт с порогом из опорного окна.

## Источники

- Breck et al., *The ML Test Score: A Rubric for ML Production Readiness*, IEEE Big Data 2017.
- Sculley et al., *Hidden Technical Debt in Machine Learning Systems*, NeurIPS 2015.
- Rabanser et al., *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift*, arXiv:1810.11953.
- Ribeiro et al., *CheckList*, arXiv:2005.04118.
- MLflow Model Registry: aliases (mlflow.org/docs); pandera — pandera.readthedocs.io; DVC — dvc.org/doc.
