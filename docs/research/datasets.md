# Каталог данных и план масштаба

Строки помечены: **✓** — число получено из HF datasets-server / API 2026-11-18; **≈** — по карточке или публикации, проверить при загрузке (этап 1).

## 1. Текстовые корпуса (RAG-индекс, предобучение, silver-разметка)

| Источник | HF ID | Объём | Размер | Лицензия |
|---|---|---|---|---|
| Wikipedia EN | `wikimedia/wikipedia` `20231101.en` | ✓ 6 407 814 статей | ✓ 11.6 GB parquet | CC BY-SA |
| Wikipedia RU | `wikimedia/wikipedia` `20231101.ru` | ✓ 1 945 063 статей | ✓ 4.9 GB | CC BY-SA |
| Wikipedia KK | `wikimedia/wikipedia` `20231101.kk` | ✓ 238 615 статей | ✓ 0.18 GB | CC BY-SA |
| FineWeb-2 KK | `HuggingFaceFW/fineweb-2` `kaz_Cyrl` | ✓ 3 375 998 документов | ✓ 6.9 GB | ODC-By |
| Multidomain Kazakh | `kz-transformers/multidomain-kazakh-dataset` | ✓ 15 510 000 строк | ≈ | Apache-2.0 |
| KazParC (параллельный kk/ru/en/tr) | `issai/kazparc` | ≈ 100k–1M пар | — | gated, проверить |
| Wikidata | dumps.wikimedia.org (JSON) | ≈ 100M+ сущностей | ≈ 100+ GB сжатый | CC0 |

## 2. NER

| Источник | HF ID | Объём | Языки | Лицензия |
|---|---|---|---|---|
| KazNERD | `issai/kaznerd` | ≈ 112 702 предложения, 25 классов (✓ классы проверены по IOB2-файлу) | kk | CC BY 4.0, gated |
| WikiANN | `unimelb-nlp/wikiann` | ✓ kk 3 000 · ru 40 000 · en 40 000 | kk/ru/en | не указана → проверить |
| WikiNEuRal | `Babelscape/wikineural` | ✓ 1 027 506 предложений (9 языков, вкл. ru/en) | ru/en | проверить (CC BY-NC-SA у оригинала) |
| NEREL | `iluvvatar/NEREL` | ≈ 29 типов сущностей | ru | проверить |
| **Silver NER из Wikipedia** | строим сами | **цель: 5–10 M предложений** | kk/ru/en | производная CC BY-SA |

Silver NER: ссылки внутри статей Wikipedia → QID Wikidata → тип по `P31` (human → PER, organization → ORG, geographic → LOC). Так строились WikiANN/WikiNEuRal; мы получаем миллионы размеченных предложений на каждом языке, включая казахский.

## 3. QA

| Источник | HF ID | Объём | Язык | Лицензия |
|---|---|---|---|---|
| SQuAD v2 | `rajpurkar/squad_v2` | ✓ 142 192 (вкл. неотвечаемые) | en | CC BY-SA 4.0 |
| SberQuAD | `kuznetsoffandrey/sberquad` | ✓ 74 300 | ru | не указана → проверить |
| KazQAD | `issai/kazqad` (`kazqad`, `nq-translate-kk`, `nq-original-en`) | ≈ | kk/en | CC BY-SA 4.0, gated |
| **Синтетический grounded QA** | строим сами (учитель Qwen3.5-35B-A3B) | **цель: 1–2 M пар после фильтрации** | kk/ru/en + межъязыковые | производная |

## 4. Retrieval

| Источник | HF ID | Объём | Язык |
|---|---|---|---|
| MS MARCO | `microsoft/ms_marco` | ≈ 8.8 M пассажей, ~500k обучающих запросов | en |
| mMARCO | `unicamp-dl/mmarco` (`russian`) | ≈ 8.8 M пассажей (перевод) | ru |
| MIRACL | `miracl/miracl` (`ru`, `en`) | ≈ корпуса в миллионы пассажей, ручные оценки | ru/en |
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

Индекс строится ступенями **1 M → 10 M → 35 M** чанков; на каждой ступени фиксируются recall, latency p50/p95, RAM и время индексации. Ступени — эксперимент о масштабировании, а не только цель.

**Оценки пропускной способности (проверить замерами):** эмбеддинги XLM-R-large-класса на RTX 3050 — порядка сотен чанков/с → 35 M чанков ≈ несколько суток; на облачной A100/H100 — часы. Массовое кодирование корпуса планируется в облаке, локально — инкрементальная индексация пользовательских документов.

## 7. Лицензии — обязательная проверка в этапе 1

WikiANN, SberQuAD, NEREL, KazParC, bge-m3-data — лицензия не указана в метаданных HF; WikiNEuRal исходно CC BY-NC-SA. Решение по каждому записывается в манифест источника.
