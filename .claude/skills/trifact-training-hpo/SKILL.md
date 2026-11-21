---
name: trifact-training-hpo
description: Training methodology and hyperparameter optimization for TriFact models (NER, embedders, rerankers, NLI, LoRA) — baselines, validation discipline, losses, AdamW and LR schedules, regularization, precision per GPU (RTX 3050, Kaggle T4 without bf16, cloud), checkpoint selection, seeds, learning curves, Optuna samplers and pruners, ablations, cost logging. Use when implementing or running trifact.training jobs, choosing hyperparameters, planning HPO, or when a model diverges, overfits or misses its baseline (обучение, гиперпараметры, Optuna, абляции).
---

# Обучение и HPO TriFact AI

Задачно-специфичные рецепты лежат в профильных скиллах: `trifact-ner-linking`, `trifact-retrieval`, `trifact-verification`, `trifact-llm-finetuning`. Здесь общая методика.

## Опорные документы

- Цикл эксперимента и запись — [AGENTS §4, §6.3](../../../AGENTS.md); состав run — архитектура [§11](../../../docs/architecture/docs/01-architecture.md).
- Значимость и гейт — архитектура §10, [ADR-0009](../../../docs/architecture/adr/0009-model-registry-and-promotion-gate.md), `params.yaml: evaluation`, `promotion`.
- Где считать: архитектура §1 (железо), §9; [ADR-0014](../../../docs/architecture/adr/0014-sharded-batch-jobs.md), [ADR-0020](../../../docs/architecture/adr/0020-kaggle-batch-gpu-mt-and-teacher.md); бюджет `cloud.budget_usd`.
- Стартовые гиперпараметры проекта — `params.yaml: ner`, `llm`, `retrieval`.

## Дизайн эксперимента до кода

1. Гипотеза и ожидаемый эффект на целевой метрике конкретного языка или направления.
2. Baseline сильный и дешёвый: zero-shot модель, предыдущий champion, простое правило. Без baseline результат не интерпретируется.
3. Одна переменная на сравнение. Абляция меняет ровно один компонент.
4. Отбор только на validation. Замороженный test открывается один раз, для гейта. Early stopping и HPO на test запрещены.
5. Бюджет: GPU-часы, деньги (`cloud.budget_usd`), квота Kaggle. Пилот (`scale.profile: pilot`) до полного прогона.

## Железо и точность

| Где | Точность | Ограничения и приёмы |
|---|---|---|
| RTX 3050 Laptop 4 GB (sm 8.6) | bf16 доступен | smoke-тесты, малые encoder-модели; gradient checkpointing, малый батч + accumulation, 8-bit AdamW (bitsandbytes) |
| Kaggle 2× T4 (sm 7.5) | **только fp16**, bf16 нет | fp16 AMP с loss scaling для encoder-моделей; процесс на GPU (ADR-0020). bf16 LoRA Qwen3.5 по ADR-0012 здесь невозможна |
| Облако Ampere+ (A10/L4/A100/H100) | bf16 | LoRA LLM, large-модели, массовые эмбеддинги; только при утверждённом бюджете |

Перед полным запуском обязательны три проверки:

- модель переобучается на 64–100 примерах (иначе данные или метки сломаны);
- батч после коллатора декодирован глазами;
- скрипт оценки на известной модели даёт ожидаемое число.

## Стартовые рецепты (диапазоны для поиска)

| Модель | Оптимизация | Стартовая зона |
|---|---|---|
| Encoder fine-tune (NER, NLI, cross-encoder) | AdamW, linear decay, warmup 6–10 % шагов, weight decay 0.01 | lr 1e-5…5e-5 (base), 5e-6…2e-5 (large), 3–5 эпох; layer-wise LR decay — опция |
| Bi-encoder (эмбеддер) | InfoNCE / MultipleNegativesRankingLoss, in-batch + hard negatives | большой эффективный батч через CachedMNRL/GradCache, lr 1e-5…3e-5, 1–2 эпохи |
| LoRA LLM | см. `trifact-llm-finetuning` | LR LoRA ≈ 10× LR полного fine-tune |

Нестабильность fine-tuning BERT-подобных моделей лечится меньшим LR с warmup, большим числом шагов и коррекцией смещения Adam, а не перебором сидов до удачного.

## Сиды и дисперсия

- Минимум 3 сида (`ner.seeds`). Отчёт — среднее, std и CI. Сравнение с baseline — по парному бутстрепу на примерах в связке с сидами (`trifact-evaluation`).
- Дисперсия между сидами сопоставима с типичными улучшениями. Один удачный сид — не результат.
- Порядок данных, инициализация головы и dropout — отдельные источники дисперсии. Сид фиксируется для всех трёх и для сэмплера HPO.

## Диагностика

| Симптом | Проверка | Действие |
|---|---|---|
| Loss = NaN в fp16 | рост grad norm, переполнение | loss scaling, меньше LR, gradient clipping 1.0, bf16 там, где есть |
| Метрика на уровне случайной | выравнивание меток, перемешанные лейблы, маски | тест переобучения на малом наборе |
| Dev растёт, test нет | утечка, переобучение на dev при отборе | проверка утечек, меньше итераций HPO, новый dev-срез |
| Большой разрыв train/dev | переобучение | weight decay, dropout, меньше эпох, больше данных (кривая обучения) |
| kk отстаёт от ru/en | мало kk, фертильность, обрезка | доля kk в смеси, `max_length`, kk-специфичные данные |
| OOM | батч, длина, активации | checkpointing, accumulation, сортировка по длине, 8-bit оптимизатор |

**Кривая обучения** (10 / 30 / 100 % данных) показывает, поможет ли увеличение данных. Если метрика ещё растёт линейно по логарифму объёма, данные ценнее HPO.

## Выбор чекпоинта

- По dev-метрике, совпадающей с целевой (entity F1, nDCG@10, faithfulness), а не по loss.
- Хранить лучший и последний чекпоинты. Усреднение чекпоинтов или EMA — опция с замером.
- В MLflow записываются шаг, метрика и путь выбранного чекпоинта.

## HPO в Optuna

- **Что тюнить:** сначала LR (лог-шкала), затем warmup, эпохи, батч, регуляризацию. Архитектурные и данные-параметры — отдельными абляциями, не в одной study.
- **Сэмплер:**

  | Ситуация | Сэмплер |
  |---|---|
  | По умолчанию, 100–1000 trials | `TPESampler(multivariate=True, seed=...)` |
  | Малый бюджет (десятки trials), непрерывные параметры | `GPSampler` |
  | Неясно, что выбрать | `AutoSampler` (OptunaHub) |
  | Качество против латентности или памяти | multi-objective: `NSGAIISampler` или TPE, выбор по Парето с ограничением |

- **Pruning** по промежуточной dev-метрике: `MedianPruner` или `HyperbandPruner`. Для Hyperband нужно больше trials, чем `n_startup_trials` TPE × число брекетов. `WilcoxonPruner` подходит, когда trial оценивается по набору примеров или задач.
- **Возобновляемость:** storage RDB (SQLite/PostgreSQL) или `JournalStorage`, фиксированный `study_name`. Каждый trial — вложенный run MLflow с параметрами и метрикой.
- **Протокол:** HPO на dev-срезе или пилоте → топ-3 конфигурации переобучаются на полном объёме с 3 сидами → выбор по среднему с CI. LR обычно переносится с малого объёма на большой, число эпох — нет.

## Абляции и стоимость

- Абляция удаляет один компонент из champion-конфигурации (leave-one-out). Отчёт — дельта с CI, включая отрицательные.
- В каждом run: samples/s или tokens/s, пиковая VRAM (`torch.cuda.max_memory_allocated`), пиковая RAM, GPU-часы, тип GPU, стоимость.

## Критерии готовности

- Есть baseline, ≥ 3 сидов, CI; конфигурация выбрана на dev, test использован один раз.
- HPO-study воспроизводима (storage, seed, `study_name`), trials лежат в MLflow.
- Ресурсы и стоимость записаны. Отрицательный результат записан в PROGRESS.

## Источники

- Akiba et al., *Optuna*, arXiv:1907.10902; документация Optuna: samplers, `GPSampler`, `HyperbandPruner` (optuna.readthedocs.io, проверено 2026-11-21).
- Li et al., *Hyperband*, arXiv:1603.06560; Li et al., *ASHA*, arXiv:1810.05934.
- Google Research, *Deep Learning Tuning Playbook* — github.com/google-research/tuning_playbook.
- Mosbach et al., *On the Stability of Fine-tuning BERT*, arXiv:2006.04884; Dodge et al., *Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping*, arXiv:2002.06305.
- Bouthillier et al., *Accounting for Variance in Machine Learning Benchmarks*, arXiv:2103.03098.
- Sun et al., *How to Fine-Tune BERT for Text Classification?*, arXiv:1905.05583.
- Gao et al., *GradCache*, arXiv:2101.06983.
