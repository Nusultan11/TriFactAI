---
name: trifact-llm-finetuning
description: LLM engineering for grounded KK/RU/EN QA with Qwen3.5 in TriFact — compatibility checks (transformers, PEFT, llama.cpp, vLLM), zero-shot baselines, chat template and non-thinking mode, SFT data mix and selection, bf16 LoRA per ADR-0012, adapter evaluation and regressions, GGUF quantization with imatrix, AWQ, serving speed (TTFT, tokens/s). Use for stage 7 work in train_llm, eval_generation, generation.llm_client, or when the model ignores evidence, invents citations or over-refuses (дообучение LLM, LoRA, SFT, квантизация).
---

# LLM: SFT, LoRA, квантизация TriFact AI

## Когда применять

- Эксперименты GEN-0…GEN-3 и GEN-Q, обучение `train_llm`, оценка `eval_generation`, клиент `generation.llm_client`.
- Подготовка SFT-смеси, сервинг адаптера, замер скорости.

Сборка контекста и промпт — `trifact-rag-pipeline`. Общая методика обучения — `trifact-training-hpo`. Синтетика учителем — `trifact-data-pipeline`.

## Опорные документы

| Тема | Владелец |
|---|---|
| Семейство, bf16 LoRA, сервинг | [ADR-0012](../../../docs/architecture/adr/0012-qwen35-lora-generation.md) (заменил [ADR-0005](../../../docs/architecture/adr/0005-qwen3-qlora-local-gguf-serving.md)) |
| Эксперименты, данные, гейт | архитектура [§7.4](../../../docs/architecture/docs/01-architecture.md), §10 |
| Факты о моделях, VRAM, фертильность | [models.md §4, §6.2](../../../docs/research/models.md) |
| Синтетика, учитель | [ADR-0015](../../../docs/architecture/adr/0015-silver-and-synthetic-data.md), [ADR-0020](../../../docs/architecture/adr/0020-kaggle-batch-gpu-mt-and-teacher.md) |
| Параметры | `params.yaml: llm`, `sft`, `teacher` |

**Ограничение ADR-0012:** QLoRA для Qwen3.5 не применяется, только bf16 LoRA на GPU Ampere+. Kaggle T4 bf16 не поддерживает. Обучение LLM требует облачного бюджета (`cloud.budget_usd`), локально — только smoke-тест на 0.8B.

## Шаг 0 — совместимость (ROADMAP 7.1)

1. `transformers` загружает `qwen3_5`; `apply_chat_template(..., enable_thinking=False)` даёт ожидаемый текст; токенизация round-trip без потерь.
2. Имена модулей берутся из `model.named_modules()`, а не из памяти. В гибридной архитектуре слои Gated DeltaNet имеют свои проекции, и `q_proj`/`k_proj` есть не во всех слоях. Список `llm.lora.target_modules` сверяется с фактическим, число обучаемых параметров печатается.
3. Слитая модель конвертируется в GGUF текущим llama.cpp, генерация совпадает с transformers на нескольких промптах (greedy).
4. vLLM ≥ 0.17 загружает базу с LoRA-адаптером.

Расхождение версий или имён фиксируется в models.md с датой и источником.

## Baseline (GEN-0)

- Все кандидаты и контрольный baseline — с одним и тем же промптом, контекстом и декодированием.
- Метрики по языкам и направлениям на `qa` eval: answer F1, faithfulness (определение и ручная проверка вердиктов — [ADR-0022](../../../docs/architecture/adr/0022-evaluation-protocol.md)), точность ссылок, refusal P/R, язык ответа.
- KazLLM и Sherkala (NC) используются только для оценки.
- Лучший GEN-0 по каждому языку — планка, которую дообучение обязано пройти (гейт).

## SFT-данные

- Формат примера — ровно промпт сервинга плюс ответ ассистента. Loss считается только на токенах ответа.
- Смесь `sft.mix`, доля kk не ниже `scale.min_kk_share`.
- Неотвечаемые примеры с тематически близкими дистракторами, а не со случайными контекстами: иначе модель учится отказывать по несвязности темы.
- Отбор по качеству важнее объёма (LIMA):
  - NLI-фильтр утверждений;
  - валидность ссылок;
  - MinHash-дедуп вопросов;
  - кластеризация вопросов эмбеддингами и квоты на кластер против перекоса тем;
  - длина в пределах `llm.max_seq_len` (по EDA).
- Вопросы замороженного eval исключены (`build_sft`). Отдельный dev SFT нужен для loss и генеративной оценки по ходу обучения.

## Настройка LoRA

| Решение | Обоснование |
|---|---|
| LoRA на всех линейных слоях, включая MLP | attention-only LoRA заметно хуже даже при равном числе параметров (Thinking Machines, 2025; Biderman et al., 2024) |
| LR ≈ 10× LR полного fine-tune; почти не зависит от ранга при `α/r`-масштабировании | там же; стартовое значение — `llm.lr`, уточняется коротким LR-sweep на dev |
| Ранг: малые ранги быстро упираются в ёмкость на больших SFT | для 0.3–2M примеров ранг проверяется sweep, rsLoRA (`α/√r`) — опция |
| 1–3 эпохи, ранняя остановка по генеративной dev-метрике | loss падает и при росте галлюцинаций |
| Packing с корректными масками (без внимания через границу примеров) | padding-free или `position_ids` в TRL; иначе примеры смешиваются |
| Gradient checkpointing, bf16 | VRAM по models.md §4 |

PEFT + TRL или Unsloth выбираются по итогам шага 0: поддержка `qwen3_5`, память и скорость на целевой GPU.

## Оценка адаптера

- Генеративная, на том же тесте и декодировании, что GEN-0. Loss для сравнения не используется.
- Регрессии, которые проверяются всегда:
  - доля отказов на отвечаемых (over-refusal);
  - язык ответа;
  - валидность формата ссылок;
  - длина и повторы (особенно в kk);
  - межъязыковые направления отдельно.
- Забывание: небольшой набор общих мультиязычных вопросов без контекста до и после. LoRA забывает меньше полного FT, но не ноль.
- Гейт ADR-0009: faithfulness ≥ `promotion.faithfulness_min`, answer F1 и faithfulness лучше лучшего baseline по каждому языку с CI. Если нет — в сервинг идёт baseline, результат пишется в PROGRESS.

## Квантизация (GEN-Q)

1. Слить адаптер в bf16, конвертировать в GGUF.
2. Матрица важности `llama-imatrix` на in-domain тексте **со всеми тремя языками** (промпты и ответы SFT dev). Без kk в калибровке низкие кванты сильнее портят kk.
3. Кандидаты `Q4_K_M` / `Q5_K_M` / `Q8_0` (llama.cpp) и AWQ (vLLM).
4. Каждый квант мерится KL-дивергенцией к bf16 (`llama-perplexity --kl-divergence`) и задачными метриками по языкам. Выбор — по кривой «качество против памяти и скорости» (`llm.serve_quant`).

## Сервинг и скорость

- Шаблон чата при сервинге совпадает с обучением.
- Для гибридных моделей llama.cpp нужен унифицированный KV-кэш при параллельных слотах (`-kvu`; замер этапа 3 в `params.yaml: teacher.serve`).
- Метрики — TTFT, tokens/s на один поток и суммарно при N слотах, пик VRAM и RAM, отдельно префилл и декод. Цели — НФТ архитектуры §1, раскладка памяти — §13.
- Спекулятивное декодирование с Qwen3.5-0.8B как драфтом — опция; выигрыш меряется на kk, где приёмка драфта может быть ниже.

## Типичные проблемы

| Симптом | Вероятная причина | Что делать |
|---|---|---|
| Ответ из «памяти», а не из контекста | мало примеров, где контекст противоречит общеизвестному | контрфактические и межъязыковые примеры в SFT, проверка faithfulness |
| Ссылки на несуществующие id | длинные id, слабый формат | нумерация пассажей, проверка формата в фильтре SFT |
| Ответ не на том языке | мало межъязыковых примеров, язык не задан явно | доля `cross_lingual`, явный язык в промпте, langid-метрика |
| Отказывает слишком часто | доля неотвечаемых велика или дистракторы слишком похожи | ablation смеси (GEN-2), порог отказа по dev |
| Зацикливание и повторы на kk | токенизация, низкая температура | repetition penalty, проверка шаблона и EOS |
| После квантизации kk просел сильнее ru/en | калибровка без kk | imatrix с kk, более высокий квант |

## Критерии готовности

- Шаг 0 пройден и записан. GEN-0 по всем кандидатам записан в MLflow `generation`.
- Адаптер прошёл гейт или отрицательный результат записан в PROGRESS.
- Для выбранного кванта есть KL к bf16, метрики по языкам, TTFT и tokens/s на целевом железе.

## Источники

- Hu et al., *LoRA*, arXiv:2106.09685; Kalajdzievski, *rsLoRA*, arXiv:2312.03732.
- Biderman et al., *LoRA Learns Less and Forgets Less*, arXiv:2405.09673; Thinking Machines Lab, *LoRA Without Regret* (2025-09-29), thinkingmachines.ai/blog/lora.
- Dettmers et al., *QLoRA*, arXiv:2305.14314 (контекст решения ADR-0012); Unsloth docs по Qwen3.5 — models.md §4.
- Zhou et al., *LIMA*, arXiv:2305.11206.
- Lin et al., *AWQ*, arXiv:2306.00978; llama.cpp `tools/imatrix`, `tools/perplexity` — github.com/ggml-org/llama.cpp.
