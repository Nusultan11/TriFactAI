---
name: trifact-evaluation
description: Evaluation, statistics and error analysis for TriFact components and the end-to-end system — metric choice and pitfalls (CER, entity F1, nDCG, answer F1, faithfulness, citations, refusal, ECE, latency, cost), frozen-test hygiene and contamination, paired and cluster bootstrap, seeds, multiple comparisons, power, LLM-as-a-judge limits, slicing, root-cause analysis, champion selection via the gate. Use when comparing models, claiming an improvement, building a benchmark, writing eval_* or gate code, or explaining a low metric (оценка, метрики, значимость, анализ ошибок).
---

# Evaluation TriFact AI

## Когда применять

- Любое утверждение «стало лучше», выбор champion, гейт.
- Новый бенчмарк или метрика, код `trifact.evaluation.*`.
- Метрика низкая или упала, и нужна первопричина.

## Опорные документы

- Наборы, значимость, гейт, ablation — архитектура [§10](../../../docs/architecture/docs/01-architecture.md); [ADR-0007](../../../docs/architecture/adr/0007-leakage-safe-splits.md), [ADR-0009](../../../docs/architecture/adr/0009-model-registry-and-promotion-gate.md).
- Протокол оценки проекта (кластерный бутстреп, определение faithfulness, LLM-судьи, загрязнение, выходы по примерам) — [ADR-0022](../../../docs/architecture/adr/0022-evaluation-protocol.md). Скилл объясняет методы, ADR фиксирует решение.
- Числа: `params.yaml: evaluation` (бутстреп, уровень CI, ручная проверка), `promotion` (пороги гейта).
- Рабочий пример бутстрепа и агрегации по виду × языку — `trifact.evaluation.eval_docproc` (`bootstrap_ci`, `aggregate`).
- Правило: проверка ИИ не заменяет человеческую (`build_eval`, PROGRESS блокеры).

## Выбор метрики

| Компонент | Основная | Ловушки |
|---|---|---|
| Документы | CER/WER, порядок чтения, F1 заголовков, F1 ячеек / TEDS | канонизация текста одинакова для всех систем; CER не видит перестановку |
| NER | entity P/R/F1, seqeval strict IOB2, по языку × типу; unseen-F1 | режим seqeval по умолчанию мягче strict; micro прячет редкие типы — показывать и macro |
| Entity Linking | accuracy@1; recall@k кандидатов | без recall кандидатов не видно потолок |
| Retrieval | Recall@5/10, nDCG@10, MRR по 9 направлениям; candidate recall@100 | MRR учитывает только первый релевантный; закрытый пул завышает новые системы |
| Генерация | answer EM/F1, faithfulness, точность и полнота ссылок, refusal P/R, язык ответа | нормализация EM/F1 по языку (NFC, регистр, пунктуация); F1 по токенам несправедлив к kk-морфологии — дублировать символьной метрикой (chrF) |
| Verification | P/R вердиктов, ECE, Brier, risk-coverage | ECE зависит от бинов |
| Система | p50/p95 латентности по стадиям, RPS, пиковая RAM/VRAM, стоимость | p95 под нагрузкой, а не на одиночных запросах |

Метрика считается по каждому языку и направлению, а среднее — дополнительно, не вместо.

## Гигиена теста

- Тест заморожен (sha256 в манифесте), на нём не тюнят и по нему не выбирают чекпоинт. Отбор идёт на validation.
- Утечки train ↔ eval проверяются по документу, контексту и нормализованному тексту, включая переводы (ADR-0007).
- **Загрязнение предобучения.** Eval из Wikipedia мог быть в предобучении LLM и эмбеддеров. Если система выигрывает только на Wikipedia-подмножестве и не выигрывает на `real_docs` или на подмножестве не из Wikipedia, это сигнал загрязнения.
- Сравнение — на одних и тех же примерах, с одним декодированием и одной версией кода метрики.

## Статистика

Улучшение без значимости улучшением не считается ([AGENTS §6.3](../../../AGENTS.md)). Методы, формулы и выбор — [references/statistics.md](references/statistics.md). Ключевое:

- **парный** бутстреп по примерам: одни и те же индексы для обеих систем, CI разности;
- **кластерный** бутстреп, если примеры зависимы (несколько вопросов на документ, переводы одного запроса);
- ≥ 3 сидов для обучаемых моделей; дисперсия сида и данных учитывается вместе;
- поправка на множественные сравнения при решении по многим направлениям, если правило гейта не задаёт иное;
- мощность: на малых kk-наборах CI широкие, нужный объём планируется заранее.

## LLM-as-a-judge

- Известные смещения: позиция, длина ответа, предпочтение собственных ответов. На казахском судья слабее.
- Протокол:
  - рубрика и эталонный ответ;
  - парное сравнение с обменом позиций;
  - температура 0;
  - фиксированная версия судьи.
- Перед использованием судья калибруется на человеческой разметке того же типа и языка: согласие (κ) и смещение по классам записываются. Без этого его оценка — диагностика, а не основание для гейта.
- Где есть измеримая метрика (EM/F1, NLI на человеческой разметке, точность ссылок), гейт опирается на неё.

## Ручная оценка

Стратифицированная выборка по языкам и направлениям, инструкция с примерами, двойная разметка подмножества (κ или α), разбор расхождений. Размер выбирается по требуемой ширине CI (references/statistics.md).

## Slicing

Обязательные разрезы: язык, направление, источник, origin (human/mt/synthetic), длина, тип документа, тип вопроса, отвечаемость, частотность сущностей. Для поиска новых проблемных срезов метрика считается по комбинациям признаков, а срезы ранжируются по «размер × отставание». Худший срез отчитывается отдельно (гейт cross-lingual).

## Анализ ошибок

Пошаговая методика от выборки провалов до проверяемой гипотезы и повторного замера — [references/error-analysis.md](references/error-analysis.md). Сквозная атрибуция ошибки RAG по стадиям — `trifact-rag-pipeline`.

## Выбор champion

1. Кандидаты сравниваются с champion на замороженном тесте по правилу гейта компонента (`promotion`).
2. При компромиссе качество против латентности или памяти строится фронт Парето, решение принимается по ограничениям НФТ.
3. Победа без значимости не продвигает модель: champion остаётся, результат записывается как отрицательный (AGENTS §7).

## Критерии готовности

- Метрики по языкам и направлениям с CI, тест не использовался для отбора.
- Сравнение парное, на одних примерах, с поправкой на зависимость примеров.
- Анализ ошибок с долями категорий и гипотезой следующего шага записан в MLflow.

## Источники

- Koehn, *Statistical Significance Tests for Machine Translation Evaluation*, EMNLP 2004; Berg-Kirkpatrick et al., *An Empirical Investigation of Statistical Significance in NLP*, EMNLP 2012.
- Dror et al., *The Hitchhiker's Guide to Testing Statistical Significance in NLP*, ACL 2018.
- Card et al., *With Little Power Comes Great Responsibility*, arXiv:2010.06595.
- Bouthillier et al., *Accounting for Variance in ML Benchmarks*, arXiv:2103.03098.
- Zheng et al., *Judging LLM-as-a-Judge*, arXiv:2306.05685.
- Ribeiro et al., *CheckList*, arXiv:2005.04118.
