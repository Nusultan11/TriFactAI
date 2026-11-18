# TriFact AI — журнал прогресса

Читать перед любой работой ([AGENTS.md §0](../AGENTS.md)). Новые записи добавляются сверху.

## Текущее состояние

| | |
|---|---|
| **Текущий этап** | 1 — Data Platform & Acquisition (не начат) |
| **Последний завершённый этап** | 0 — Фундамент проекта (2026-11-18) |
| **Декомпозиция текущего этапа** | shrimp: задачи S1.1–S1.7 (`.shrimp/`, не в git) |

## Открытые вопросы (нужно решение пользователя)

1. **Облачный провайдер GPU и S3-бакет** для DVC remote `cloud` и бюджет (`params.yaml: cloud.budget_usd`, сейчас 0). Это блокирует облачные задания с этапа 3, но не этап 1.
2. **Доступ к gated-наборам** KazNERD, KazQAD, KazParC: условия на huggingface.co принимаются вручную, затем `HF_TOKEN` прописывается в `.env`.
3. **MCP-серверы**: `huggingface` требует авторизации (`/mcp`); `github` не подключается — неверный заголовок Authorization (переменная `GITHUB_PERSONAL_ACCESS_TOKEN` не задана или задана с ошибкой); `mlflow` и `serena` не стартовали за 30 с (MLflow ещё не запущен, serena ставится через `uvx` из git — первый запуск долгий).

## Что учесть дальше

- XNLI имеет лицензию CC BY-NC 4.0, поэтому производный kk-NLI наследует `release: false` (ADR-0015). Для выпускаемого верификатора нужен источник с разрешительной лицензией, или верификатор публикуется только как исследовательский.
- Qwen3.5: только bf16 LoRA, QLoRA не применять; нужен vLLM ≥ 0.17 (ADR-0012). Совместимость PEFT и llama.cpp проверить первым шагом этапа 7.
- Полный индекс 35M, скорее всего, не поместится в 15 GB RAM вместе со стеком. Ступени RET-S (1M/10M/35M) решат, нужен ли Serving VM (ADR-0013).
- Перекачка векторов корпуса (~72 GB fp16) дорогая: коллекцию лучше собирать там, где она будет обслуживаться (ADR-0014).
- Перед первым запуском DVC: `dvc init` и remotes (задача S1.6).

---

## Этап 0 — Фундамент проекта: завершён (2026-11-18)

**Что сделано**

- Архитектура пересобрана под масштаб десятков миллионов записей и согласована с ROADMAP и AGENTS. Раньше `01-architecture.md`, `params.yaml`, `dvc.yaml` и ADR-0005 описывали MVP на 267k записей и Qwen3-1.7B + QLoRA, что противоречило ROADMAP (35M чанков, Qwen3.5, Parquet-lake).
  - [01-architecture.md](architecture/docs/01-architecture.md): масштаб и профили `pilot` / `full`, NFR, три контура (Serving / Data & ML Platform / Observability), data lake и схемы шардов, онлайн-потоки и API v1, эксперименты DOC/NER/RET/GEN/VER по этапам, бюджет памяти индекса 35M, хранилища, офлайн-вычисления, гейт со значимостью, MLOps, наблюдаемость, развёртывание Local/Cloud, трассировка этапов → компонентов → модулей → DVC → ADR, риски.
  - [workspace.dsl](architecture/workspace.dsl): новые контейнеры Data Lake и Batch Compute, внешние Wikimedia Dumps и Cloud Object Storage, 15 представлений, включая ModelLifecycle и DeploymentLocal/Cloud. Проверено MCP `structurizr` и `structurizr/structurizr inspect`: 0 ошибок, 0 предупреждений.
  - Диаграммы ([diagrams/](architecture/diagrams/)) экспортированы из модели скриптом [scripts/export_diagrams.py](../scripts/export_diagrams.py).
  - Новые ADR 0011–0016: Parquet lake, Qwen3.5 + bf16 LoRA (заменяет 0005), двухуровневый индекс Qdrant (дополняет 0002), шардированные batch-задания и два DVC remote (дополняет 0010), silver/синтетика и родословная лицензий, ответственность хранилищ. В ADR-0009 гейт дополнен требованием значимости.
- Конфигурация: [params.yaml](../params.yaml) (все 17 источников, профили масштаба, курация, индекс, учитель, бюджет облака) и [dvc.yaml](../dvc.yaml) (29 стадий этапов 1–9; `acquire` разворачивается в 16 источников).
- Каркас кода приведён к стадиям: добавлены `trifact.lake`, `trifact.batch`, стадии курации и KB, `train_nli`, `eval_docproc`, `eval_verification`, `gate`. Схемы дополнены `RetrievalScope`, `RetrievalStats`, `workspace_id`, `SourceType.CORPUS`. Python 3.12, зависимости Polars / DuckDB / PyArrow / datatrove / Optuna / ONNX Runtime.
- Архитектурные тесты [tests/test_architecture.py](../tests/test_architecture.py): `core` ни от чего не зависит; все команды DVC указывают на существующие модули; источники `acquire` объявлены в params; на каждый ADR есть ссылка; JSON-контракт ответа из документа проходит валидацию схемой `Answer`.
- Каталог моделей дополнен фактами о дообучении Qwen3.5 (exa). Документация Qdrant 1.20 (prefetch + RRF/DBSF, квантизация, multitenancy) проверена через context7.
- Этап 1 декомпозирован в shrimp на 7 задач с зависимостями.
- Git-репозиторий на `main`, первый коммит, бэкап.

**Проблемы и решения**

- Тесты нашли невалидный YAML в `dvc.yaml`: `${item}` внутри flow-последовательности. DVC тоже не смог бы его прочитать; исправлено кавычками.
- На ADR-0003 не было ссылок из документации; ссылка добавлена в §7.3.
- Инспектор Structurizr нашёл несвязанные элементы (Dynamic Batcher, Training Jobs в облаке); добавлены связи и узел S3-бакета.

**Удалено**: `data/download.py`, `data/map_labels.py`, `data/build_rag_corpus.py`, `data/index_corpus.py` (заменены стадиями `acquire`, `build_ner`, `chunk_corpus`, `batch.index_corpus`), устаревшая диаграмма `Deployment.mmd`, MVP-бюджет как единственный режим (теперь это профиль `pilot`).
