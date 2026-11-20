# Каталог данных и план масштаба

Строки помечены: **✓** — число получено из HF datasets-server / API 2026-11-18; **●** — посчитано при загрузке этапа 1 (2026-11-18) и сверено с метаданными, итог в `data/raw/<источник>/source.yaml`; **≈** — по карточке или публикации, не проверено.

## 1. Текстовые корпуса (RAG-индекс, предобучение, silver-разметка)

| Источник | HF ID | Объём | Размер | Лицензия |
|---|---|---|---|---|
| Wikipedia EN | `wikimedia/wikipedia` `20231101.en` | ✓ 6 407 814 статей | ✓ 11.6 GB parquet | CC BY-SA |
| Wikipedia RU | `wikimedia/wikipedia` `20231101.ru` | ✓ 1 945 063 статей | ✓ 4.9 GB | CC BY-SA |
| Wikipedia KK | `wikimedia/wikipedia` `20231101.kk` | ✓ 238 615 статей | ✓ 0.18 GB | CC BY-SA |
| FineWeb-2 KK | `HuggingFaceFW/fineweb-2` `kaz_Cyrl` | ✓ 3 375 998 документов | ✓ 6.9 GB | ODC-By |
| Multidomain Kazakh | `kz-transformers/multidomain-kazakh-dataset` | ● 24 883 808 записей CSV (карточка и datasets-server дают 15 510 000 — частичный подсчёт) | ● 6.97 GB parquet | Apache-2.0 |
| KazParC (параллельный kk/ru/en/tr) | `issai/kazparc` | ● 15 103 193 строк (все конфиги, вкл. синтетику) | ● 2.26 GB parquet | CC BY 4.0, gated |
| Wikidata | dumps.wikimedia.org (JSON) | ≈ 100M+ сущностей | ≈ 100+ GB сжатый | CC0 |

## 2. NER

| Источник | HF ID | Объём | Языки | Лицензия |
|---|---|---|---|---|
| KazNERD | `issai/kaznerd` | ● 112 702 предложения, 25 классов (✓ классы проверены по IOB2-файлу) | kk | CC BY 4.0, gated |
| WikiANN | `unimelb-nlp/wikiann` | ✓ kk 3 000 · ru 40 000 · en 40 000 | kk/ru/en | не указана → проверить |
| WikiNEuRal | `Babelscape/wikineural` | ✓ 1 027 506 предложений (9 языков); загружены ● en 115 907, ru 115 440 | ru/en | проверить (CC BY-NC-SA у оригинала) |
| NEREL | `iluvvatar/NEREL` | ● 1 011 документов, ≈ 29 типов сущностей | ru | проверить |
| **Silver NER из Wikipedia** | строим сами | **цель: 5–10 M предложений** | kk/ru/en | производная CC BY-SA |

Silver NER: ссылки внутри статей Wikipedia → QID Wikidata → тип по `P31` (human → PER, organization → ORG, geographic → LOC). Так строились WikiANN/WikiNEuRal; мы получаем миллионы размеченных предложений на каждом языке, включая казахский.

## 3. QA

| Источник | HF ID | Объём | Язык | Лицензия |
|---|---|---|---|---|
| SQuAD v2 | `rajpurkar/squad_v2` | ✓ 142 192 (вкл. неотвечаемые) | en | CC BY-SA 4.0 |
| SberQuAD | `kuznetsoffandrey/sberquad` | ✓ 74 300 | ru | не указана → проверить |
| KazQAD | `issai/kazqad` (`kazqad`, `nq-translate-kk`, `nq-original-en`) | ● kk 68 246 · en 68 770 | kk/en | CC BY-SA 4.0, gated |
| **Синтетический grounded QA** | строим сами (учитель Qwen3.5-35B-A3B) | **цель: 1–2 M пар после фильтрации** | kk/ru/en + межъязыковые | производная |

## 4. Retrieval

| Источник | HF ID | Объём | Язык |
|---|---|---|---|
| MS MARCO | `microsoft/ms_marco` | ● 1 112 939 строк запрос + пассажи (v1.1, v2.1) | en |
| mMARCO | `unicamp-dl/mmarco` (`russian`) | ● ru 9 758 627 (коллекция и запросы), multi 39 780 811 (тройки id) | ru |
| MIRACL | `miracl/miracl` (`ru`, `en`) + `miracl/miracl-corpus` | ● запросы с оценками en 43 952 · ru 54 585; корпус en 32 893 221 · ru 9 543 918 пассажей | ru/en |
| ru-HNP | `deepvk/ru-HNP` | ✓ 102 590 (hard negatives) | ru |
| bge-m3-data | `Shitao/bge-m3-data` | ≈ обучающие данные BGE-M3 | мульти |
| **Синтетические запросы** | строим сами | **цель: 3–5 M (запрос, чанк)** | kk/ru/en, 9 направлений |

## 5. NLI

| Источник | HF ID | Объём | Язык |
|---|---|---|---|
| XNLI | `facebook/xnli` | ✓ 400 202 на язык (ru, en) | ru/en |
| XNLI-kk | строим сами (перевод учителем + фильтр) | цель: 400k | kk |

## 6. План масштаба

| Слой | Целевой объём | Хранение (оценка) |
|---|---|---|
| raw-корпуса | ~27 M документов (Wikipedia 8.6 M + FineWeb-2 kk 3.4 M + multidomain kk 15.5 M) | ~40 GB parquet |
| curated (после чистки/дедупа) | ожидаемо 60–80 % от raw | ~30 GB |
| **RAG-индекс** | **~35 M чанков** (оценка: ~20 M en, ~7 M ru, ~8 M kk) | векторы fp16 на диске ~72 GB; binary quantization в RAM ~4.5 GB |
| NER | 5–10 M silver + ~1.3 M gold/реальных предложений | ~5 GB |
| Retrieval pairs | 10–20 M (MS MARCO, mMARCO, синтетика, KazParC) | ~10 GB |
| SFT grounded QA | 1–2 M | ~5 GB |
| NLI | ~1.2 M | ~0.5 GB |
| Eval (заморожен) | 30–50k, из них ≥ 3k проверены вручную, все 9 языковых направлений | — |
| Eval `docproc` (заморожен, этап 4) | ● 150 документов: по 18 на вид (pdf_text, pdf_columns, pdf_tables, pdf_scan, pdf_word, docx, html, txt; по 6 на kk/ru/en) + 6 смешанных kk/ru/en; 239 стр. PDF; 3 длинных PDF по 300 стр.; 17 враждебных файлов. Текст — абзацы `clean/wikipedia` (CC-BY-SA-3.0) | 161 MB |

Ступени построения индекса — эксперимент RET-S, оценки пропускной способности — [архитектура §7.3, §9](../architecture/docs/01-architecture.md).

## 7. Лицензии — решения этапа 1

Решения записаны в `params.yaml: sources.*.license` и копируются в манифест каждого источника (ADR-0017). `allowed` — условия разрешают использование, в том числе выпуск производных в рамках лицензии. `research_only` — только обучение и оценка, производные наследуют `release: false` (ADR-0015). `blocked` — не скачивается. Проверено 2026-11-18.

| Источник | Лицензия | Решение | release | Основание |
|---|---|---|---|---|
| Wikipedia, MIRACL corpus | CC BY-SA 3.0 (+ GFDL) | allowed | да | карточка HF, условия Wikimedia |
| FineWeb-2 kk | ODC-By 1.0 + CommonCrawl ToU | allowed | да | карточка HF |
| Multidomain Kazakh | Apache-2.0 (карточка) | allowed | **нет** | собран из CC100, OSCAR, новостей и книг; лицензия карточки не снимает права третьих лиц на тексты |
| KazParC | CC BY 4.0 | allowed | да | в карточке HF лицензии нет; авторы: arXiv:2403.19399 и страница релиза ISSAI |
| KazNERD | CC BY 4.0 | allowed | да | карточка HF |
| KazQAD | CC BY-SA 4.0 | allowed | да | карточка HF |
| SQuAD v2 | CC BY-SA 4.0 | allowed | да | карточка HF |
| MIRACL (topics, qrels) | Apache-2.0 | allowed | да | карточка HF |
| ru-HNP | Apache-2.0 | allowed | да | карточка HF; пары сгенерированы gpt-3.5-turbo |
| Wikidata | CC0 1.0 | allowed | да | Wikidata:Licensing |
| WikiANN | не указана (текст CC BY-SA) | research_only | нет | карточка HF «unknown»; github.com/afshinrahimi/mmner — Apache-2.0 для кода, данные на Dropbox без лицензии |
| NEREL | не указана | research_only | нет | нет лицензии ни в карточке, ни в github.com/nerel-ds/NEREL; статья RANLP 2021 — CC BY 4.0 |
| SberQuAD | не указана | research_only | нет | карточка HF «unknown», TFDS «No known license» |
| WikiNEuRal | CC BY-NC-SA 4.0 | research_only | нет | карточка HF |
| XNLI | CC BY-NC 4.0 | research_only | нет | карточка HF |
| MS MARCO, mMARCO | условия MS MARCO (некоммерческие исследования) | research_only | нет | microsoft.github.io/msmarco/Notice; mMARCO — перевод MS MARCO |

Решение `research_only` для WikiANN, NEREL и SberQuAD можно повысить до `allowed`, если правообладатель подтвердит лицензию. Для этого достаточно правки `params.yaml` и перезаписи манифеста, повторная загрузка не нужна.
