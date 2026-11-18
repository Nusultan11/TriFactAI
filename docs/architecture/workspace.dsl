workspace "TriFact AI" "Multilingual Document & Knowledge Intelligence Platform (KK / RU / EN). Масштаб: десятки миллионов чанков." {

    !identifiers hierarchical

    model {
        properties {
            "structurizr.groupSeparator" "/"
        }

        # ---------------------------------------------------------------- people
        user = person "Пользователь" "Студент или исследователь: загружает документы KK/RU/EN, задаёт вопросы на любом из трёх языков, проверяет источники."
        mlEngineer = person "ML-инженер" "Собирает и курирует данные, обучает и оценивает модели, продвигает champion-версии."

        # ------------------------------------------------------- external systems
        hfHub = softwareSystem "Hugging Face Hub" "Датасеты (Wikipedia, FineWeb-2, KazNERD, KazQAD, WikiANN, MS MARCO, MIRACL, XNLI, ...) и базовые модели (XLM-R, BGE-M3, Qwen3.5, Qwen3-Embedding/Reranker)." "External"
        wikimedia = softwareSystem "Wikimedia Dumps" "JSON-дамп Wikidata (100M+ сущностей, CC0) для базы сущностей и silver-NER." "External"
        cloudGpu = softwareSystem "Cloud GPU" "Арендуемые GPU 24-80 GB: обучение, массовые эмбеддинги, учитель-LLM для синтетики." "External"
        cloudStorage = softwareSystem "Cloud Object Storage" "S3-совместимый бакет для обмена данными с облачными GPU-заданиями (DVC remote 'cloud')." "External"

        # ------------------------------------------------------------ the system
        trifact = softwareSystem "TriFact AI" "Извлечение сущностей, межъязыковой гибридный поиск, ответы со ссылками и проверка утверждений по документам на трёх языках." {

            !docs docs
            !adrs adr

            # ======================== SERVING PLANE
            group "Serving" {

                web = container "Web App" "Загрузка документов, статусы, чат со стримингом, карточки сущностей, подсветка подтверждающих фрагментов." "Next.js, TypeScript" "Web"

                api = container "API" "REST + SSE: рабочие пространства, документы, вопросы, сущности, фидбек. Оркестрирует онлайн-пайплайн ответа." "Python, FastAPI" {
                    documentsRouter = component "Documents Router" "Загрузка файлов, статус обработки, список и удаление документов рабочего пространства." "FastAPI router"
                    askRouter = component "Ask Router" "Вопрос, язык ответа, scope (documents | corpus | all); стрим ответа с цитатами." "FastAPI router, SSE"
                    entitiesRouter = component "Entities Router" "Поиск сущностей, карточка QID, документы и фрагменты с упоминаниями." "FastAPI router"
                    queryService = component "Query Service" "Оркестратор: анализ запроса -> retrieval -> rerank -> generation -> verification -> журнал." "Python, asyncio"
                    queryAnalyzer = component "Query Analyzer" "Язык вопроса, сущности запроса (мягкий сигнал, ADR-0004)." "Python"
                    hybridRetriever = component "Hybrid Retriever" "Dense + sparse поиск в Qdrant (prefetch + RRF на сервере), слияние результатов двух коллекций." "Python, qdrant-client"
                    rerankerClient = component "Reranker Client" "Переранжирование top-N кандидатов cross-encoder'ом, кеш оценок." "HTTP client"
                    promptBuilder = component "Grounded Prompt Builder" "Контекст с [chunk_id], язык ответа, инструкция отвечать только по доказательствам." "Python, Jinja"
                    generatorClient = component "Generator Client" "Стриминговый вызов LLM через OpenAI-совместимый API." "HTTP client"
                    evidenceVerifier = component "Evidence Verifier" "Разбиение на утверждения, проверка ссылок, NLI-вердикты, политика отказа (ADR-0008)." "Python"
                    cache = component "Query Cache" "Кеш эмбеддингов запросов и оценок reranker." "Redis"
                    repository = component "Repositories" "Документы, чанки, сущности, журнал запросов и ответов." "SQLAlchemy"
                }

                worker = container "Ingestion Worker" "Асинхронная обработка загруженных документов: парсинг, OCR, нормализация, язык, чанки, NER, linking, индексация." "Python, arq" {
                    parser = component "Document Parser" "Двухуровневый парсинг: быстрый путь PyMuPDF, Docling для сложной структуры; page/section/offset." "PyMuPDF, Docling"
                    ocr = component "OCR Fallback" "Распознавание сканов без текстового слоя (kaz+rus+eng); движок выбирается бенчмарком этапа 4." "Docling OCR / Tesseract"
                    cleaner = component "Text Normalizer" "NFC, казахские буквы, гомоглифы латиница/кириллица, колонтитулы. Общий код с офлайн-курацией." "Python (trifact.ingestion)"
                    langDetector = component "Language Detector" "Язык документа и абзаца (kk/ru/en), различение kk/ky/tt." "fastText lid.176 / GlotLID"
                    chunker = component "Chunker" "Секции -> предложения -> чанки с перекрытием. Тот же код, что строит корпусный индекс." "Python (trifact.ingestion)"
                    nerExtractor = component "Entity Extractor" "NER по чанкам: PER/ORG/LOC (+ расширенные классы для kk)." "HTTP -> Inference"
                    dateExtractor = component "Date Extractor" "Даты kk/ru/en." "Regex + dateparser"
                    entityLinker = component "Entity Linker" "Кандидаты из базы сущностей по нормализованным алиасам + ранжирование по контексту -> Wikidata QID." "Python"
                    indexer = component "Indexer" "Dense+sparse векторы -> Qdrant (коллекция workspace), метаданные и упоминания -> PostgreSQL." "Python"
                }

                inference = container "Inference Service" "Онлайн-хостинг encoder-моделей: эмбеддинги, reranker, NER, NLI. Динамический батчинг." "Python, FastAPI, PyTorch / ONNX Runtime" {
                    embedEndpoint = component "Embedding Endpoint" "Dense + sparse векторы запросов и чанков пользователя." "BGE-M3 / Qwen3-Embedding / mE5"
                    rerankEndpoint = component "Rerank Endpoint" "Релевантность (query, chunk)." "bge-reranker-v2-m3 / Qwen3-Reranker"
                    nerEndpoint = component "NER Endpoint" "Token classification." "XLM-R (fine-tuned, ONNX int8)"
                    nliEndpoint = component "NLI Endpoint" "entailment / contradiction / neutral." "mDeBERTa-xnli / bge-m3-zeroshot (fine-tuned kk)"
                    batcher = component "Dynamic Batcher" "Собирает запросы в батчи по длине, ограничивает очередь." "asyncio"
                    modelLoader = component "Model Loader" "Загружает версии моделей по alias 'champion' из MLflow Model Registry." "mlflow"
                }

                llm = container "LLM Server" "Grounded generation на KK/RU/EN: базовая модель + LoRA-адаптер, стриминг." "llama.cpp server (локально, GGUF) / vLLM (GPU-сервер), OpenAI-compatible API" "LLM"

                postgres = container "PostgreSQL" "Состояние приложения: workspaces, documents, chunks и mentions пользовательских документов, журнал запросов/ответов/вердиктов, фидбек; база сущностей kb_entities/kb_aliases (миллионы строк)." "PostgreSQL 16" "Database"
                qdrant = container "Qdrant" "Коллекции: corpus_chunks (десятки миллионов, квантизация, on-disk, read-only, alias на версию эмбеддера) и workspace_chunks (пользовательские документы, tenant = workspace_id)." "Qdrant 1.20" "Database"
                redis = container "Redis" "Очередь задач обработки документов и кеш запросов." "Redis 7" "Queue"
                objectStore = container "Object Storage" "Исходные файлы пользователей, DVC remote 'local', артефакты MLflow, GGUF/веса для сервинга." "SeaweedFS (S3 API)" "Storage"
            }

            # ======================== DATA & ML PLANE
            group "Data & ML Platform" {

                dataLake = container "Data Lake" "Parquet (zstd), шарды ~1M строк / <=512 MB, партиции source/lang. Слои raw -> clean -> curated -> task -> eval; манифест у каждого источника (ADR-0011)." "Parquet on SSD, DVC" "Storage"

                dataPipeline = container "Data Pipeline" "CPU-стадии DVC: загрузка, курация, сборка task-наборов, заморозка eval. Потоковая колоночная обработка без загрузки набора в RAM." "Python, Polars (lazy/streaming), DuckDB, PyArrow, datatrove, DVC" {
                    hfLoader = component "Source Loader" "Потоковая загрузка HF -> Parquet-шарды, докачка, проверка числа строк, манифест (revision SHA, лицензия, хеши)." "huggingface_hub, PyArrow"
                    wikidataParser = component "Wikidata Parser" "Потоковый разбор JSON-дампа -> entities(qid, type, labels/aliases kk/ru/en, sitelinks)." "orjson, PyArrow"
                    normalizer = component "Normalizer & LangID" "Векторизованная нормализация и определение языка документов и абзацев." "Polars, fastText / GlotLID"
                    qualityFilter = component "Quality Filters" "Длина, доля букв, повторы, boilerplate; пороги по ручной выборке." "datatrove"
                    deduplicator = component "Deduplicator" "Точный хеш + MinHash-LSH внутри и между источниками." "datatrove"
                    decontaminator = component "Decontaminator" "Удаляет из train-корпусов тексты, пересекающиеся с eval-источниками." "n-gram / MinHash"
                    splitter = component "Leakage-safe Splitter" "train/val/test по исходному документу (ADR-0007)." "Polars"
                    nerBuilder = component "NER Builder" "Gold-наборы -> общая схема меток; silver NER из ссылок Wikipedia + типов Wikidata (5-10M предложений)." "Polars, DuckDB"
                    pairBuilder = component "Retrieval Pair Builder" "MS MARCO / mMARCO / MIRACL / KazParC + синтетические запросы; hard negatives." "Polars"
                    sftBuilder = component "SFT Builder" "Фильтры синтетического grounded QA (NLI-согласованность, ответ в контексте, дубли), состав смеси." "Polars"
                    nliBuilder = component "NLI Builder" "XNLI ru/en + kk-перевод с фильтрацией." "Polars"
                    evalBuilder = component "Eval Builder" "Замороженный eval 30-50k по 9 направлениям; хеш фиксируется до обучения." "Polars"
                    corpusChunker = component "Corpus Chunker" "curated -> чанки корпуса тем же Chunker, что и онлайн." "trifact.ingestion"
                    kbBuilder = component "KB Builder" "Сущности и алиасы kk/ru/en + приоры из ссылок Wikipedia -> PostgreSQL (COPY)." "DuckDB, psycopg COPY"
                }

                batchCompute = container "Batch Compute" "Шардированные идемпотентные GPU/CPU-задания с чекпойнтом на шард: эмбеддинги корпуса, массовая NER-разметка, учитель-LLM, загрузка индекса (ADR-0014)." "Python, PyTorch, vLLM, ONNX Runtime" "Batch" {
                    shardRunner = component "Shard Runner" "Раздаёт шарды, пропускает готовые (_SUCCESS), пишет метрики throughput и пиковую память." "Python"
                    embedJob = component "Embed Corpus Job" "Dense + sparse векторы для десятков миллионов чанков (fp16, сортировка по длине)." "sentence-transformers / FlagEmbedding"
                    annotateJob = component "Annotate Corpus Job" "Массовая NER + linking champion-моделью -> аннотации в lake." "ONNX Runtime int8"
                    teacherJob = component "Teacher Generation Job" "Синтетические запросы, grounded QA, kk-перевод NLI учителем." "vLLM, Qwen3.5-35B-A3B"
                    indexLoader = component "Index Loader" "Bulk upload в новую коллекцию Qdrant, HNSW после загрузки, переключение alias после гейта." "qdrant-client"
                }

                training = container "Training Jobs" "Обучение NER, embedder, reranker, NLI и LoRA-адаптеров LLM; HPO и >=3 сидов." "PyTorch, Transformers, PEFT, sentence-transformers, Optuna" "Batch"

                evaluation = container "Evaluation Harness" "Офлайн-метрики по языкам и 9 направлениям, значимость, ablation, гейт продвижения (ADR-0009)." "Python, seqeval, ranx, Locust" "Batch" {
                    docprocEval = component "DocProc Eval" "CER/WER, порядок чтения, таблицы, страниц/с." "jiwer"
                    nerEval = component "NER Eval" "Entity-level P/R/F1 по языку и типу; linking accuracy@1." "seqeval"
                    retrievalEval = component "Retrieval Eval" "Recall@5/10, nDCG@10, MRR по 9 направлениям; кривые масштаба 1M/10M/35M." "ranx"
                    genEval = component "Generation Eval" "Answer EM/F1, faithfulness, citation accuracy, refusal P/R." "Python + NLI"
                    verifEval = component "Verification Eval" "Точность вердиктов, ECE, калибровка порогов." "Python"
                    e2eEval = component "End-to-End Eval" "Полный пайплайн через API, ablation, нагрузочные тесты." "Python, Locust"
                    significance = component "Significance" "Bootstrap CI, сравнение сидов." "numpy"
                    promotionGate = component "Promotion Gate" "Проверяет пороги на замороженном тесте, ставит alias 'champion'." "mlflow"
                }

                mlflow = container "MLflow" "Трекинг экспериментов (git SHA, DVC-хеши, конфиг, метрики по языкам, железо) и Model Registry (champion / challenger)." "MLflow 3" "Tool"
            }

            group "Observability" {
                prometheus = container "Prometheus" "Метрики API, воркера, моделей, Qdrant: латентность по стадиям, ошибки, очередь, качество." "Prometheus" "Tool"
                grafana = container "Grafana" "Дашборды Service и Quality, алерты." "Grafana" "Tool"
            }
        }

        # ============================================================ relationships
        # --- people
        user -> trifact.web "Загружает документы, задаёт вопросы" "HTTPS"
        mlEngineer -> trifact.dataPipeline "Запускает стадии" "dvc repro"
        mlEngineer -> trifact.batchCompute "Запускает шардированные задания" "CLI"
        mlEngineer -> trifact.mlflow "Сравнивает эксперименты, продвигает модели" "HTTPS"
        mlEngineer -> trifact.grafana "Смотрит дашборды" "HTTPS"

        # --- web / api
        trifact.web -> trifact.api.documentsRouter "Загрузка и статусы" "JSON/HTTPS"
        trifact.web -> trifact.api.askRouter "Вопросы, стрим ответа" "SSE/HTTPS"
        trifact.web -> trifact.api.entitiesRouter "Сущности" "JSON/HTTPS"

        trifact.api.documentsRouter -> trifact.objectStore "Сохраняет исходный файл" "S3 API"
        trifact.api.documentsRouter -> trifact.api.repository "Создаёт запись документа" "in-process"
        trifact.api.documentsRouter -> trifact.redis "Ставит задачу process_document" "Redis"
        trifact.api.entitiesRouter -> trifact.api.repository "Читает сущности и упоминания" "in-process"
        trifact.api.askRouter -> trifact.api.queryService "Передаёт вопрос" "in-process"
        trifact.api.queryService -> trifact.api.queryAnalyzer "Анализ запроса" "in-process"
        trifact.api.queryService -> trifact.api.hybridRetriever "Кандидаты (top-50)" "in-process"
        trifact.api.queryService -> trifact.api.rerankerClient "top-50 -> top-8" "in-process"
        trifact.api.queryService -> trifact.api.promptBuilder "Контекст" "in-process"
        trifact.api.queryService -> trifact.api.generatorClient "Генерация" "in-process"
        trifact.api.queryService -> trifact.api.evidenceVerifier "Проверка утверждений" "in-process"
        trifact.api.queryService -> trifact.api.repository "Журнал запроса и ответа" "in-process"
        trifact.api.queryAnalyzer -> trifact.inference.nerEndpoint "Сущности запроса" "HTTP"
        trifact.api.hybridRetriever -> trifact.api.cache "Эмбеддинг запроса из кеша" "in-process"
        trifact.api.hybridRetriever -> trifact.inference.embedEndpoint "Эмбеддинг запроса" "HTTP"
        trifact.api.hybridRetriever -> trifact.qdrant "Prefetch dense + sparse, RRF; две коллекции" "gRPC"
        trifact.api.rerankerClient -> trifact.api.cache "Кеш оценок" "in-process"
        trifact.api.rerankerClient -> trifact.inference.rerankEndpoint "Оценки пар" "HTTP"
        trifact.api.generatorClient -> trifact.llm "chat/completions (stream)" "HTTP"
        trifact.api.evidenceVerifier -> trifact.inference.nliEndpoint "claim vs evidence" "HTTP"
        trifact.api.cache -> trifact.redis "GET/SET" "Redis"
        trifact.api.repository -> trifact.postgres "Читает/пишет" "SQL"

        # --- worker
        trifact.worker.parser -> trifact.redis "Забирает задачи" "Redis"
        trifact.worker.parser -> trifact.objectStore "Читает исходный файл" "S3 API"
        trifact.worker.parser -> trifact.worker.ocr "Скан без текстового слоя" "in-process"
        trifact.worker.parser -> trifact.worker.cleaner "Сырой текст + структура" "in-process"
        trifact.worker.cleaner -> trifact.worker.langDetector "Нормализованный текст" "in-process"
        trifact.worker.langDetector -> trifact.worker.chunker "Текст с языковой разметкой" "in-process"
        trifact.worker.chunker -> trifact.worker.nerExtractor "Чанки" "in-process"
        trifact.worker.nerExtractor -> trifact.inference.nerEndpoint "Token classification" "HTTP"
        trifact.worker.nerExtractor -> trifact.worker.dateExtractor "Чанки для дат" "in-process"
        trifact.worker.nerExtractor -> trifact.worker.entityLinker "Упоминания" "in-process"
        trifact.worker.entityLinker -> trifact.postgres "Кандидаты из kb_aliases" "SQL"
        trifact.worker.entityLinker -> trifact.worker.indexer "Чанки + QID" "in-process"
        trifact.worker.indexer -> trifact.inference.embedEndpoint "Векторы чанков" "HTTP"
        trifact.worker.indexer -> trifact.qdrant "Upsert в workspace_chunks" "gRPC"
        trifact.worker.indexer -> trifact.postgres "Чанки, упоминания, статус" "SQL"

        # --- inference: every endpoint goes through the dynamic batcher, models come from the registry
        trifact.inference.embedEndpoint -> trifact.inference.batcher "Запросы в батч" "in-process"
        trifact.inference.rerankEndpoint -> trifact.inference.batcher "Пары в батч" "in-process"
        trifact.inference.nerEndpoint -> trifact.inference.batcher "Чанки в батч" "in-process"
        trifact.inference.nliEndpoint -> trifact.inference.batcher "Пары в батч" "in-process"
        trifact.inference.batcher -> trifact.inference.modelLoader "Загруженные модели" "in-process"
        trifact.inference.modelLoader -> trifact.mlflow "Версия 'champion' и её веса" "HTTP"
        trifact.llm -> trifact.objectStore "GGUF / LoRA-адаптер champion" "S3 API"

        # --- offline: data pipeline
        trifact.dataPipeline.hfLoader -> hfHub "Скачивает датасеты (revision SHA)" "HTTPS"
        trifact.dataPipeline.hfLoader -> trifact.dataLake "raw-шарды + манифест" "Parquet"
        trifact.dataPipeline.wikidataParser -> wikimedia "Скачивает JSON-дамп" "HTTPS"
        trifact.dataPipeline.wikidataParser -> trifact.dataLake "raw/wikidata" "Parquet"
        trifact.dataPipeline.normalizer -> trifact.dataPipeline.qualityFilter "Нормализованные документы с языком" "in-process"
        trifact.dataPipeline.qualityFilter -> trifact.dataPipeline.deduplicator "Прошедшие фильтры" "in-process"
        trifact.dataPipeline.deduplicator -> trifact.dataPipeline.decontaminator "Уникальные документы" "in-process"
        trifact.dataPipeline.decontaminator -> trifact.dataLake "curated-слой + отчёт качества" "Parquet"
        trifact.dataPipeline.normalizer -> trifact.dataLake "Читает raw" "Parquet"
        trifact.dataPipeline.splitter -> trifact.dataLake "Сплиты gold-наборов" "Parquet"
        trifact.dataPipeline.nerBuilder -> trifact.dataLake "task/ner" "Parquet"
        trifact.dataPipeline.pairBuilder -> trifact.dataLake "task/retrieval" "Parquet"
        trifact.dataPipeline.sftBuilder -> trifact.dataLake "task/sft" "Parquet"
        trifact.dataPipeline.nliBuilder -> trifact.dataLake "task/nli" "Parquet"
        trifact.dataPipeline.evalBuilder -> trifact.dataLake "eval (заморожен, хеш)" "Parquet"
        trifact.dataPipeline.corpusChunker -> trifact.dataLake "curated -> chunks" "Parquet"
        trifact.dataPipeline.kbBuilder -> trifact.dataLake "Читает wikidata + ссылки Wikipedia" "DuckDB"
        trifact.dataPipeline.kbBuilder -> trifact.postgres "kb_entities, kb_aliases" "COPY"

        trifact.dataLake -> trifact.objectStore "Версии шардов" "dvc push (remote local)"
        trifact.dataLake -> cloudStorage "Обмен с GPU-заданиями" "dvc push/pull (remote cloud)"

        # --- offline: batch compute
        trifact.batchCompute.shardRunner -> trifact.batchCompute.embedJob "Шард чанков" "in-process"
        trifact.batchCompute.shardRunner -> trifact.batchCompute.annotateJob "Шард чанков" "in-process"
        trifact.batchCompute.shardRunner -> trifact.batchCompute.teacherJob "Шард заданий генерации" "in-process"
        trifact.batchCompute.shardRunner -> trifact.dataLake "Читает шарды, пишет результат + _SUCCESS" "Parquet"
        trifact.batchCompute.shardRunner -> cloudStorage "Вход/выход облачных запусков" "S3 API"
        trifact.batchCompute.shardRunner -> trifact.mlflow "Throughput, память, стоимость" "HTTP"
        trifact.batchCompute.shardRunner -> cloudGpu "Выполняется на" "SSH / job API"
        trifact.batchCompute.teacherJob -> hfHub "Веса учителя" "HTTPS"
        trifact.batchCompute.embedJob -> trifact.mlflow "Веса champion/challenger эмбеддера" "HTTP"
        trifact.batchCompute.annotateJob -> trifact.mlflow "Веса champion NER" "HTTP"
        trifact.batchCompute.indexLoader -> trifact.dataLake "Читает векторы и payload" "Parquet"
        trifact.batchCompute.indexLoader -> trifact.qdrant "Bulk upload в corpus_chunks__<embedder>, alias" "gRPC"

        # --- offline: training & evaluation
        trifact.training -> trifact.dataLake "Читает task-наборы" "Parquet"
        trifact.training -> cloudStorage "Данные и чекпойнты облачных запусков" "S3 API"
        trifact.training -> hfHub "Базовые модели" "HTTPS"
        trifact.training -> trifact.mlflow "Параметры, метрики, модели (challenger)" "HTTP"
        trifact.training -> cloudGpu "Выполняется на" "SSH / job API"

        trifact.evaluation.docprocEval -> trifact.dataLake "Бенчмарк документов" "Parquet"
        trifact.evaluation.nerEval -> trifact.dataLake "Замороженный eval" "Parquet"
        trifact.evaluation.retrievalEval -> trifact.qdrant "Поиск по каждому методу отдельно" "gRPC"
        trifact.evaluation.retrievalEval -> trifact.dataLake "Замороженный eval" "Parquet"
        trifact.evaluation.genEval -> trifact.llm "Генерация на eval" "HTTP"
        trifact.evaluation.verifEval -> trifact.inference "NLI на eval" "HTTP"
        trifact.evaluation.e2eEval -> trifact.api "Прогон теста и нагрузка" "HTTP"
        trifact.evaluation.nerEval -> trifact.evaluation.significance "Метрики по примерам" "in-process"
        trifact.evaluation.retrievalEval -> trifact.evaluation.significance "Метрики по запросам" "in-process"
        trifact.evaluation.genEval -> trifact.evaluation.significance "Метрики по примерам" "in-process"
        trifact.evaluation.e2eEval -> trifact.evaluation.significance "Метрики по примерам" "in-process"
        trifact.evaluation.verifEval -> trifact.evaluation.significance "Метрики по примерам" "in-process"
        trifact.evaluation.docprocEval -> trifact.evaluation.significance "Метрики по документам" "in-process"
        trifact.evaluation.significance -> trifact.evaluation.promotionGate "CI и дельты" "in-process"
        trifact.evaluation.promotionGate -> trifact.mlflow "Метрики, решение, alias champion" "HTTP"

        # --- observability
        trifact.prometheus -> trifact.api "Скрейпит /metrics" "HTTP"
        trifact.prometheus -> trifact.worker "Скрейпит /metrics" "HTTP"
        trifact.prometheus -> trifact.inference "Скрейпит /metrics" "HTTP"
        trifact.prometheus -> trifact.llm "Скрейпит /metrics" "HTTP"
        trifact.prometheus -> trifact.qdrant "Скрейпит /metrics" "HTTP"
        trifact.grafana -> trifact.prometheus "PromQL" "HTTP"
        trifact.grafana -> trifact.postgres "Метрики качества из журнала ответов" "SQL"

        # ============================================================ deployment
        local = deploymentEnvironment "Local" {
            deploymentNode "Developer laptop" "Windows 11, RTX 3050 Laptop 4 GB, Ryzen 7 5800HS, 15 GB RAM, SSD 663 GB" "Docker Desktop + WSL2" {
                deploymentNode "docker compose" "Онлайн-стек и MLOps-инфраструктура" "Docker Compose" {
                    containerInstance trifact.web
                    containerInstance trifact.api
                    containerInstance trifact.worker
                    containerInstance trifact.inference
                    deploymentNode "llama.cpp server" "Qwen3.5-2B Q4_K_M GGUF (~1.4 GB VRAM)" "llama.cpp" {
                        containerInstance trifact.llm
                    }
                    containerInstance trifact.postgres
                    containerInstance trifact.qdrant
                    containerInstance trifact.redis
                    containerInstance trifact.objectStore
                    containerInstance trifact.mlflow
                    containerInstance trifact.prometheus
                    containerInstance trifact.grafana
                }
                deploymentNode "uv venv" "CPU-стадии данных, небольшие эксперименты, eval" "Python 3.12" {
                    containerInstance trifact.dataLake
                    containerInstance trifact.dataPipeline
                    containerInstance trifact.evaluation
                }
            }
        }

        cloud = deploymentEnvironment "Cloud" {
            provider = deploymentNode "GPU cloud provider" "Почасовая аренда; провайдер выбирается в этапе 1" "IaaS" {
                bucket = infrastructureNode "Object Storage bucket" "DVC remote 'cloud': входные шарды, результаты заданий, чекпойнты." "S3-compatible"
                gpu = deploymentNode "GPU instance" "A10G / L4 / A100 / H100, 24-80 GB" "Linux, CUDA, Docker" {
                    batchJobs = containerInstance trifact.batchCompute
                    trainJobs = containerInstance trifact.training
                }
            }
            provider.gpu.batchJobs -> provider.bucket "Читает шарды, пишет векторы / аннотации / синтетику" "S3 API"
            provider.gpu.trainJobs -> provider.bucket "Читает task-наборы, пишет чекпойнты и адаптеры" "S3 API"
            deploymentNode "Serving VM (опционально)" "32-64 GB RAM, GPU 16-24 GB: полный индекс 35M и vLLM" "Linux, Docker Compose" {
                containerInstance trifact.api
                containerInstance trifact.inference
                containerInstance trifact.llm
                containerInstance trifact.qdrant
                containerInstance trifact.postgres
                containerInstance trifact.redis
            }
        }
    }

    views {
        systemContext trifact "SystemContext" "Кто использует TriFact AI и с чем он связан." {
            include *
            autoLayout lr
        }

        container trifact "Containers" "Serving, Data & ML Platform, Observability." {
            include *
            autoLayout tb
        }

        container trifact "ServingContainers" "Онлайн-контур: загрузка документов и ответы на вопросы." {
            include user trifact.web trifact.api trifact.worker trifact.inference trifact.llm trifact.postgres trifact.qdrant trifact.redis trifact.objectStore trifact.mlflow
            autoLayout lr
        }

        container trifact "MlPlatformContainers" "Офлайн-контур: данные, массовые задания, обучение, оценка, продвижение." {
            include mlEngineer hfHub wikimedia cloudGpu cloudStorage trifact.dataLake trifact.dataPipeline trifact.batchCompute trifact.training trifact.evaluation trifact.mlflow trifact.qdrant trifact.postgres trifact.objectStore
            autoLayout lr
        }

        component trifact.api "ApiComponents" "Онлайн-пайплайн ответа на вопрос." {
            include *
            autoLayout lr
        }

        component trifact.worker "WorkerComponents" "Пайплайн обработки загруженного документа." {
            include *
            autoLayout lr
        }

        component trifact.inference "InferenceComponents" "Модели, обслуживаемые Inference Service." {
            include *
            autoLayout lr
        }

        component trifact.dataPipeline "DataPipelineComponents" "Загрузка, курация и сборка task-наборов." {
            include *
            autoLayout lr
        }

        component trifact.batchCompute "BatchComputeComponents" "Шардированные массовые задания." {
            include *
            autoLayout lr
        }

        component trifact.evaluation "EvaluationComponents" "Офлайн-оценка, значимость и гейт продвижения." {
            include *
            autoLayout lr
        }

        dynamic trifact.api "AskFlow" "Вопрос на казахском по документам на трёх языках." {
            trifact.web -> trifact.api.askRouter "Вопрос (kk), answer_lang = kk, scope = all"
            trifact.api.askRouter -> trifact.api.queryService "Передаёт вопрос"
            trifact.api.queryService -> trifact.api.hybridRetriever "Кандидаты"
            trifact.api.hybridRetriever -> trifact.inference.embedEndpoint "Dense + sparse вектор запроса"
            trifact.api.hybridRetriever -> trifact.qdrant "workspace_chunks || corpus_chunks: prefetch + RRF -> top-50"
            trifact.api.queryService -> trifact.api.rerankerClient "Переранжирование"
            trifact.api.rerankerClient -> trifact.inference.rerankEndpoint "top-50 -> top-8"
            trifact.api.queryService -> trifact.api.promptBuilder "Контекст с [chunk_id]"
            trifact.api.queryService -> trifact.api.generatorClient "Генерация"
            trifact.api.generatorClient -> trifact.llm "Ответ на kk со ссылками [chunk_id] или отказ"
            trifact.api.queryService -> trifact.api.evidenceVerifier "Проверка"
            trifact.api.evidenceVerifier -> trifact.inference.nliEndpoint "Каждое утверждение vs цитируемый фрагмент"
            trifact.api.queryService -> trifact.api.repository "Запрос, ответ, вердикты, версии моделей, латентность"
            autoLayout lr
        }

        dynamic trifact.worker "IngestFlow" "Обработка загруженного документа." {
            trifact.worker.parser -> trifact.redis "Забирает задачу"
            trifact.worker.parser -> trifact.objectStore "Читает файл"
            trifact.worker.parser -> trifact.worker.cleaner "Текст + структура"
            trifact.worker.cleaner -> trifact.worker.langDetector "Нормализованный текст"
            trifact.worker.langDetector -> trifact.worker.chunker "Текст с языками"
            trifact.worker.chunker -> trifact.worker.nerExtractor "Чанки"
            trifact.worker.nerExtractor -> trifact.inference.nerEndpoint "NER"
            trifact.worker.nerExtractor -> trifact.worker.entityLinker "Упоминания"
            trifact.worker.entityLinker -> trifact.postgres "Кандидаты QID"
            trifact.worker.entityLinker -> trifact.worker.indexer "Чанки + QID"
            trifact.worker.indexer -> trifact.inference.embedEndpoint "Векторы"
            trifact.worker.indexer -> trifact.qdrant "Upsert (tenant = workspace_id)"
            trifact.worker.indexer -> trifact.postgres "Метаданные, статус = processed"
            autoLayout lr
        }

        dynamic trifact "ModelLifecycle" "От сырых данных до champion-модели в сервинге." {
            trifact.dataPipeline -> trifact.dataLake "raw -> clean -> curated -> task -> eval"
            trifact.dataLake -> cloudStorage "dvc push (remote cloud)"
            trifact.batchCompute -> trifact.dataLake "Векторы, аннотации, синтетика"
            trifact.training -> trifact.dataLake "Читает task-наборы"
            trifact.training -> trifact.mlflow "Регистрирует challenger"
            trifact.evaluation -> trifact.mlflow "Гейт: challenger -> champion"
            trifact.batchCompute -> trifact.qdrant "Новая коллекция корпуса, переключение alias"
            trifact.inference -> trifact.mlflow "Загружает champion"
            autoLayout lr
        }

        deployment trifact local "DeploymentLocal" "Локальная разработка и демо на RTX 3050 4 GB." {
            include *
            autoLayout tb
        }

        deployment trifact cloud "DeploymentCloud" "Облачные GPU-задания и опциональный сервер полного индекса." {
            include *
            autoLayout tb
        }

        styles {
            element "Element" {
                color #ffffff
            }
            element "Person" {
                shape Person
                background #1f4e79
            }
            element "Software System" {
                background #2e75b6
            }
            element "External" {
                background #7f7f7f
            }
            element "Container" {
                background #4472c4
            }
            element "Component" {
                background #8faadc
                color #000000
            }
            element "Web" {
                shape WebBrowser
            }
            element "Database" {
                shape Cylinder
            }
            element "Storage" {
                shape Folder
            }
            element "Queue" {
                shape Pipe
            }
            element "LLM" {
                shape Hexagon
                background #c55a11
            }
            element "Batch" {
                shape Box
                background #548235
            }
            element "Tool" {
                background #7030a0
            }
            relationship "Relationship" {
                color #707070
            }
        }
    }

    configuration {
        scope softwaresystem
    }
}
