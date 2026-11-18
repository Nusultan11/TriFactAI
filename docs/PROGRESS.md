# TriFact AI — журнал прогресса

Читается перед любой работой ([AGENTS.md §3](../AGENTS.md)). Записи этапов добавляются сверху.

## Текущее состояние

| | |
|---|---|
| **Текущий этап** | 1 — Data Platform & Acquisition (остановлен: ждёт дамп Wikidata, см. блокер 1) |
| **Последний завершённый этап** | 0 — Фундамент проекта (2026-11-18) |
| **Декомпозиция текущего этапа** | shrimp: S1.1–S1.4, S1.6 выполнены; S1.5 (Wikidata) ждёт дамп, S1.7 (отчёт) закрывается после S1.5 (`.shrimp/`, не в git) |
| **Удалённый репозиторий** | `origin` → https://github.com/Nusultan11/TriFactAI (`main`) |
| **Raw-слой** | 17 из 18 источников (все HF): 146.8 M строк, 38.76 GB Parquet, 352 шарда; `dvc status` чистый, `python -m trifact.lake verify` — 17/17 OK |

## Блокеры (нужно действие пользователя или время)

1. **Дамп Wikidata `20261116` (103.9 GB bz2) качается ~32 ч.** Это внешняя причина. dumps.wikimedia.org отдаёт 0.3–1 MB/s на соединение и отвечает HTTP 429 на четвёртое. Зеркало your.org работает в том же диапазоне. Итого ~0.8 MB/s при канале ~9 MB/s. На 2026-11-18 14:43 скачано 112/1549 чанков по 64 MB (≈7 GB), оставшиеся ~92 GB при 0.82 MB/s займут ~31 ч. Загрузка докачивается с места остановки (`.state.json`). После неё выполнить `uv run dvc repro wikidata` (sha1 сверяется автоматически), затем `uv run dvc push`. Сторонние HF-копии дампа отклонены: у них нет проверяемого происхождения и ревизии. Парсер проверен на 50k сущностей реального дампа (замер ниже).
2. **DVC remote `cloud`** объявлен (`s3://trifact-dvc-cloud`), но не проверен: нет провайдера, endpoint, ключей и бюджета (`params.yaml: cloud.budget_usd: 0`). Нужно выбрать провайдера, создать бакет, прописать `endpointurl` и ключи в `.dvc/config.local`, затем выполнить `dvc push -r cloud`. Этап 2 не блокирует.
3. **MCP `github`** не подключается: неверный заголовок Authorization (`GITHUB_PERSONAL_ACCESS_TOKEN`). **MCP `exa`** не подключается по таймауту (2026-11-18). Push через git работает. Лицензии проверены через MCP `huggingface` и веб-поиск.

## Что учесть дальше

- `fineweb2_kk` даёт пик RSS 5.2 GB при 4 воркерах, потому что входные row group большие. Лимит `lake.peak_ram_gb` = 10 GB соблюдён. Если параллельно идут тяжёлые задания, запускать с `--workers 2`.
- В CSV `multidomain_kk` 24 883 808 записей, а карточка и datasets-server дают 15.51 M (частичный подсчёт). Сверка идёт полным чтением CSV (`csv_record_count`). Состав корпуса (CC100, OSCAR, новости) пересекается с FineWeb-2, поэтому межисточниковая дедупликация на этапе 2 обязательна.
- Источники с `research_only` (WikiANN, NEREL, SberQuAD, WikiNEuRal, XNLI, MS MARCO, mMARCO) идут только в train/eval, их производные наследуют `release: false` ([datasets.md §7](research/datasets.md)). `multidomain_kk` — allowed, но тоже `release: false`.
- В KazParC есть синтетические конфиги `sync` и `sync_raw` (таблицы `sync_pairs`, `sync_entries`). На этапе 3 отделить их от ручных пар, чтобы синтетика не попала в eval (ADR-0015).
- Таблица `entities` (Wikidata) нужна для silver-NER этапа 3, а этапу 2 не нужна. Этап 2 можно начинать параллельно с загрузкой дампа.
- `HF_TOKEN` хранится только в `.env`, никогда в `.env.example`. Gated-наборы (KazNERD, KazQAD, KazParC) требуют токен для `dvc repro`, но не для `dvc pull`.
- Qwen3.5: только bf16 LoRA, QLoRA не применять; нужен vLLM ≥ 0.17 (ADR-0012). Совместимость PEFT и llama.cpp проверить первым шагом этапа 7.
- Полный индекс 35M, скорее всего, не поместится в 15 GB RAM вместе со стеком. Ступени RET-S решат, нужен ли Serving VM (ADR-0013).
- Перекачка векторов корпуса (~72 GB fp16) дорогая: коллекцию лучше собирать там, где она будет обслуживаться (ADR-0014).

---

## Этап 1 — Data Platform & Acquisition: остановлен (2026-11-18)

Остановлен по блокеру 1 (дамп Wikidata). Всё, что от дампа не зависит, сделано и проверено.

**Что сделано**

- **Окружение.**
  - `uv.lock`: Python 3.12, группы `ml` и `dev`.
  - pre-commit: ruff check и format, mypy core, pytest.
  - В [docker-compose](../infra/docker-compose.yml) у всех сервисов есть healthcheck. Одноразовый сервис `s3-init` создаёт бакеты.
  - `docker compose up -d --wait`: все сервисы healthy (проверено).
- **`trifact.lake`** ([layout](../src/trifact/lake/layout.py), [manifest](../src/trifact/lake/manifest.py), [stats](../src/trifact/lake/stats.py), [CLI](../src/trifact/lake/__main__.py)).
  - Шарды Parquet (zstd) пишутся атомарно: `.tmp` → fsync → rename.
  - Манифест `source.yaml` хранит происхождение, ревизию, решение по лицензии, sha256 и число строк каждого шарда, сверки строк, throughput и пик RSS.
  - `verify` проверяет состав шардов, размеры, строки, sha256, сверки и лицензию. `report` выдаёт таблицу объёмов.
- **Загрузчик HF** ([acquire.py](../src/trifact/data/acquire.py)). Протокол описан в [ADR-0017](architecture/adr/0017-raw-layer-acquisition-protocol.md).
  - Источник: закреплённая SHA-ревизия.
  - Чтение потоковое, батчами: parquet, arrow, csv, tsv, jsonl(.gz).
  - Файлы обрабатываются параллельно.
  - Докачка: маркеры готовых файлов в `_work/`, недокачанные HF-файлы продолжаются с места обрыва, sha256 LFS сверяется.
  - Число строк сверяется с README `dataset_info`, datasets-server, футерами parquet или полным подсчётом строк и записей.
  - Повторный запуск при той же ревизии только проверяет манифест.
  - Источник с лицензией `blocked` не скачивается.
- **Все HF-источники** из `params.yaml: sources` загружены (17, включая добавленный `miracl_corpus`). Все сверки строк сошлись.
- **Wikidata** ([wikidata.py](../src/trifact/data/wikidata.py)).
  - Загрузка: датированный дамп скачивается диапазонами по 64 MB в несколько соединений и через зеркала, с докачкой и сверкой sha1 по опубликованному `sha1sums`.
  - Парсинг потоковый: параллельная распаковка bzip2 (indexed_bzip2) → чанки строк → пул процессов (orjson) → детерминированные шарды таблицы `entities` по диапазонам сущностей с маркерами прогресса → сверка `entity_count`.
- **Лицензии**: решение по каждому источнику записано в его манифест ([datasets.md §7](research/datasets.md)). Измеренные объёмы перенесены в datasets.md (пометка ●).
- **DVC.**
  - Remotes: `local` (SeaweedFS S3, по умолчанию) и `cloud`.
  - Каталоги `data/raw/<источник>` — persist-выходы стадий `acquire@*` и `wikidata`.
  - `dvc.lock` закоммичен, `dvc status` чистый.
  - Результаты push и восстановления — ниже.
- **MLflow**: эксперимент `data`. Каждая реальная загрузка — один run. Холостые повторы run не создают.

**Результаты** (`python -m trifact.lake report`, вход с HF ~37 GB)

| Источник | Строк по языкам | Parquet, GB | Шардов | Проверка строк | Throughput, строк/с | MB/s | Пик RSS, MB | Лицензия |
|---|---|---|---|---|---|---|---|---|
| fineweb2_kk | kk 3 375 998 | 6.84 | 14 | ✓ hf_datasets_server, parquet_footer | 4340.3 | 8.42 | 5215.3 | ODC-By-1.0 (allowed) |
| kaznerd | kk 112 702 | 0.01 | 3 | ✓ hf_dataset_info | 24552.8 | 6.95 | 168.8 | CC-BY-4.0 (allowed) |
| kazparc | multi 15 103 193 | 2.26 | 39 | ✓ csv_record_count, hf_dataset_info | 36233.7 | 14.0 | 1863.3 | CC-BY-4.0 (allowed) |
| kazqad | en 68 770, kk 68 246 | 0.05 | 5 | ✓ hf_datasets_server, line_count | 10955.3 | 3.95 | 501.0 | CC-BY-SA-4.0 (allowed) |
| miracl | en 43 952, ru 54 585 | 0.00 | 12 | ✓ line_count | 25205.3 | 0.64 | 139.6 | Apache-2.0 (allowed) |
| miracl_corpus | en 32 893 221, ru 9 543 918 | 7.32 | 86 | ✓ line_count | 50881.2 | 7.59 | 642.8 | Apache-2.0 (text CC-BY-SA-3.0 Wikipedia) (allowed) |
| mmarco_ru | multi 39 780 811, ru 9 758 627 | 2.14 | 52 | ✓ line_count | 95003.0 | 12.34 | 1475.4 | MS MARCO terms (non-commercial research) (research_only) |
| ms_marco | en 1 112 939 | 1.64 | 12 | ✓ hf_dataset_info, parquet_footer | 4706.6 | 9.37 | 1214.0 | MS MARCO terms (non-commercial research) (research_only) |
| multidomain_kk | kk 24 883 808 | 6.97 | 32 | ✓ csv_record_count | 2075.3 | 10.17 | 3625.7 | Apache-2.0 (allowed) |
| nerel | ru 1 011 | 0.00 | 5 | ✓ line_count | 316.8 | 2.83 | 192.6 | unknown (research_only) |
| ru_hnp | ru 102 590 | 0.06 | 3 | ✓ hf_datasets_server, parquet_footer | 6735.4 | 5.59 | 456.4 | Apache-2.0 (allowed) |
| sberquad | ru 74 300 | 0.01 | 3 | ✓ hf_dataset_info, parquet_footer | 9938.9 | 2.52 | 347.0 | unknown (research_only) |
| squad_v2 | en 142 192 | 0.01 | 2 | ✓ hf_dataset_info, parquet_footer | 28657.3 | 3.41 | 465.8 | CC-BY-SA-4.0 (allowed) |
| wikiann | en 40 000, kk 3 000, ru 40 000 | 0.00 | 9 | ✓ hf_dataset_info, parquet_footer | 8872.3 | 0.66 | 210.8 | unknown (text CC-BY-SA-3.0) (research_only) |
| wikineural | en 115 907, ru 115 440 | 0.02 | 6 | ✓ hf_datasets_server, parquet_footer | 27328.5 | 2.29 | 317.7 | CC-BY-NC-SA-4.0 (research_only) |
| wikipedia | en 6 407 814, kk 238 615, ru 1 945 063 | 11.38 | 63 | ✓ hf_dataset_info, parquet_footer | 5668.0 | 10.5 | 2012.5 | CC-BY-SA-3.0 AND GFDL (allowed) |
| xnli | en 400 202, ru 400 202 | 0.08 | 6 | ✓ hf_dataset_info, parquet_footer | 41556.4 | 6.01 | 412.3 | CC-BY-NC-4.0 (research_only) |

- Итого 146.8 M строк, 38.76 GB Parquet, 352 шарда.
- Throughput ограничен сетью: ~9 MB/s суммарно.
- Пик RSS ≤ 5.2 GB у всех загрузок при лимите 10 GB.
- `lake verify` с полной проверкой sha256 всех шардов занимает 42 с.

Wikidata, реальный дамп, первые 50 000 сущностей:

- 17.4 с на 8 воркерах, пик RSS 1.43 GB;
- распаковка bzip2 — ~239 MB/s на 8 потоках, разбор — ~130 MB/s на процесс, степень сжатия ~9.9.

Полный дамп — это ~1 TB JSON. По этим цифрам его парсинг займёт порядка 1.5–2 ч на 12 воркерах. Это оценка, не замер.

**DVC (проверено)**

- `dvc push -r local`: 386 объектов (352 шарда, манифесты, `.dir`) в SeaweedFS за 4 мин 51 с.
- Восстановление с remote в чистый клон с пустым кешем: `dvc pull` трёх источников (kaznerd, xnli, ms_marco) — 24 с, `lake verify` (sha256) OK, `dvc status` чистый.
- Локальное восстановление удалённого `data/raw/kaznerd` через `dvc checkout` — 3 с, `verify` OK.
- `dvc status acquire` — «up to date». Стадия `wikidata` ждёт дамп (блокер 1), стадии этапов 2–9 ещё не запускались.

**Отказоустойчивость (проверено на реальных данных)**

- `xnli`: процесс убит SIGKILL после 2 файлов. Повторный запуск докачал оставшиеся 4, `verify` OK.
- `multidomain_kk`: упал на CSV-поле длиннее 131 072 символов. После исправления продолжил с 3 из 5 файлов, 2 готовых переиспользованы.
- Wikidata: загрузка остановлена и перезапущена. Продолжила с 1534 оставшихся чанков из 1549, без повторной загрузки.
- Автотесты (32).
  - Загрузчик: докачка после сбоя, идемпотентность, смена лицензии без перекачки, расхождение с метаданными, подмена шарда (sha256), `blocked`, ревизия не-SHA, частичный прогон, огромное CSV-поле.
  - Wikidata: разбор, шардирование, возобновление после краха, докачка и sha1, неверная контрольная сумма, зеркала.

**Проблемы и решения**

- На усечённом файле indexed_bzip2 бросает `RuntimeError`, а не `EOFError`. Теперь при `allow_truncated` обрабатываются оба исключения.
- Когда сервер MLflow недоступен, логирование висело на ретраях по нескольку минут. Теперь перед логированием идёт быстрый `/health`-запрос.
- Схема jsonl выводилась по первой строке. Теперь она объединяется по всему батчу.
- `multidomain_kk` упал на лимите поля CSV. Лимит снят, и стейджинг-файл теперь сохраняется при ошибке конвертации.
- Рабочая копия части файлов была с CRLF (`core.autocrlf=true`), а git хранит LF. `dvc.lock` фиксировал хеши, которые чистый клон не воспроизводит: это показала проверка восстановления. Файлы приведены к LF, lock пересобран, добавлен тест [test_dvc_dependencies_use_lf_line_endings](../tests/test_architecture.py).
- Часть стадий DVC запускалась со старой версией `acquire.py`. Повторный `dvc repro acquire` только проверил манифесты: 17/17 за 2 мин 41 с, без перекачки.

**Удалено**: ничего из репозитория. Стейджинг и логи лежат вне git (`data/.cache/`).

## Этап 0 — Фундамент проекта: завершён (2026-11-18)

**Доработка инструкций (2026-11-18)**

- [AGENTS.md](../AGENTS.md) переписан как набор чек-листов: карта документов (у каждого факта один владелец), «перед началом работы», цикл этапа, завершение этапа, правила, нештатные ситуации, бэкапы, инструменты, шаблон отчёта. В завершение этапа добавлен `git push`.
- [ROADMAP.md](ROADMAP.md) ссылается на идентификаторы экспериментов архитектуры и не пересказывает модели. Исправлено противоречие с ADR-0012: на этапе 7 было «LoRA/QLoRA», стало bf16 LoRA. Критерий этапа 9 ссылался на §7 вместо §1 (НФТ).
- Убраны дубли: таблица масштаба (владелец — datasets.md §6), железо (архитектура §1), бюджет памяти 35M (архитектура §7.3), оценки throughput (архитектура §9), сводка VRAM в models.md (архитектура §13), раскладка репозитория в README (архитектура §15).
- Исправлены ссылки: ADR-0006 → NER-4 (было NER-3), web/README → этап 10 (было 8), номера разделов AGENTS в архитектуре и ADR-0011.
- Подключён удалённый репозиторий GitHub, `main` запушен.

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
