# Agent Skills TriFact AI

Инженерные методики для агента (Codex, Claude Code и любой клиент стандарта [Agent Skills](https://agentskills.io/specification)). Скилл отвечает на вопрос «как сделать хорошо и как проверить». Что решено в проекте и какие у него цифры, скилл не пересказывает, а ссылается на документ-владелец ([AGENTS.md §2](../../AGENTS.md)).

## Где лежат

| Путь | Кто читает | Правка |
|---|---|---|
| `.agents/skills/<name>/` | Codex, другие клиенты стандарта | **источник истины**, править только здесь |
| `.claude/skills/<name>/` | Claude Code | зеркало, генерируется `python scripts/sync_skills.py` |

После правки скилла: `python scripts/sync_skills.py`, затем `pytest tests/test_skills.py`. Тест и хук pre-commit падают, если зеркало разошлось с источником, нарушен формат или сломана ссылка. Скрипт управляет только каталогами `trifact-*`, поэтому собственные скиллы Claude Code в `.claude/skills/` не трогаются.

## Состав и границы

| Скилл | Отвечает за | Не отвечает за |
|---|---|---|
| [trifact-data-pipeline](trifact-data-pipeline/SKILL.md) | lake, стадии данных, курацию, сплиты, silver/синтетику/MT, лицензии, потоковую обработку и память | анализ распределений (eda) |
| [trifact-eda](trifact-eda/SKILL.md) | EDA корпусов и task-наборов, шум меток, сдвиги, перевод находок в решения | реализацию фильтров (data-pipeline) |
| [trifact-document-processing](trifact-document-processing/SKILL.md) | парсинг, OCR, layout, таблицы, маршрутизацию движков, бенчмарк извлечения | выбор чанкинга по поиску (retrieval) |
| [trifact-ner-linking](trifact-ner-linking/SKILL.md) | NER, DATE, Entity Linking, калибровку и ускорение экстракторов | общую методику обучения (training-hpo) |
| [trifact-training-hpo](trifact-training-hpo/SKILL.md) | методологию обучения, HPO в Optuna, сиды, чекпоинты, абляции, железо | задачно-специфичные рецепты (в профильных скиллах) |
| [trifact-retrieval](trifact-retrieval/SKILL.md) | чанкинг, эмбеддинги, гибрид, Qdrant, reranker, их дообучение и оценку по 9 направлениям | сборку контекста для LLM (rag-pipeline) |
| [trifact-rag-pipeline](trifact-rag-pipeline/SKILL.md) | сквозной ask-поток, контекст, промпт, отказ, prompt injection, атрибуцию ошибок по стадиям | дообучение LLM (llm-finetuning) |
| [trifact-llm-finetuning](trifact-llm-finetuning/SKILL.md) | Qwen3.5: baseline, SFT-данные, bf16 LoRA, квантизацию, сервинг | проверку утверждений (verification) |
| [trifact-verification](trifact-verification/SKILL.md) | claims, NLI, калибровку, пороги и политику отказа | метрики RAG целиком (evaluation) |
| [trifact-evaluation](trifact-evaluation/SKILL.md) | выбор метрик, статистику, LLM-as-judge, slicing, анализ ошибок, выбор champion | реализацию компонентов |
| [trifact-mlops](trifact-mlops/SKILL.md) | воспроизводимость, MLflow и Registry, тесты ML, профилирование, мониторинг и дрейф | процедуру этапа и коммитов (AGENTS.md) |

## Типовые задачи → скилл

Таблица выбора проверялась по описаниям скиллов (2026-11-21). Если задача касается двух скиллов, первым загружается профильный, вторым — методический (evaluation или training-hpo).

| Задача | Скиллы |
|---|---|
| Добавить источник в raw-слой, починить OOM стадии, собрать silver или синтетику | data-pipeline |
| Проверить датасет перед обучением, найти шум меток или сдвиг train/eval | eda |
| На сканах теряются казахские буквы, таблица читается по столбцам, новый формат файла | document-processing → evaluation |
| Обучить NER, поднять F1 kk, связать «Астанада» с QID | ner-linking → training-hpo |
| Подобрать lr и эпохи, спланировать Optuna-study, модель расходится | training-hpo |
| Выбрать эмбеддер, настроить RRF и квантизацию, дообучить reranker | retrieval → evaluation |
| Ответ неверный или без ссылки: на какой стадии ошибка | rag-pipeline → evaluation |
| SFT-смесь, LoRA Qwen3.5, GGUF-квантизация, скорость генерации | llm-finetuning |
| Пороги NLI, ECE, политика отказа, фильтр синтетики по NLI | verification |
| Доказать, что модель лучше, посчитать CI, разобрать ошибки | evaluation |
| Регрессионные тесты моделей, регистрация в MLflow, мониторинг дрейфа | mlops |

## Правила для новых скиллов

- Имя `trifact-<область>`, в frontmatter только поля спецификации (`name`, `description`, при необходимости `license`, `compatibility`, `metadata`). `name` совпадает с каталогом.
- `description` говорит, что делает скилл и когда его применять, с ключевыми словами на английском и русском, не длиннее 1024 символов.
- Тело `SKILL.md` короче 500 строк, детали — в `references/` на один уровень вложенности.
- Числа проекта не копируются: ссылка на `params.yaml`, PROGRESS или ADR. Внешний факт сопровождается источником.
- Скилл не противоречит действующим ADR. Альтернатива действующему решению описывается как эксперимент, который проходит гейт ([ADR-0009](../../docs/architecture/adr/0009-model-registry-and-promotion-gate.md)).
