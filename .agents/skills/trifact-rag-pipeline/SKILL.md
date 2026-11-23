---
name: trifact-rag-pipeline
description: End-to-end grounded RAG in TriFact — retrieval, reranking, context assembly, [chunk_id]-cited prompt, cloud/remote inference and evaluation, context token budget, evidence dedup, cross-lingual answers, conflicting sources, refusal, prompt-injection defense, latency budget, stage-attributed failure diagnosis and ablation. Use when wiring or debugging the ask flow (api.query_service, generation.prompt), running eval_e2e, or diagnosing wrong, uncited, wrong-language, or refused answers.
---

# TriFact AI — RAG Pipeline

## Scope

- Load when building or debugging `POST /v1/ask` (`01-architecture.md §6.2`), context assembly, prompt/refusal behavior, prompt-injection defenses, or Stage 9 `eval_e2e` and ablations.
- LLM fine-tuning → `trifact-llm-finetuning`. Retrieval → `trifact-retrieval`. NLI verification → `trifact-verification`.
- Cloud CLI execution & persistence → [cloud-compute.md](../trifact-mlops/references/cloud-compute.md).

## Owner Documents

| Topic | Owner |
|---|---|
| Ask flow, response contract, API | [01-architecture.md §6.2](../../../docs/architecture/docs/01-architecture.md) |
| Ablation, e2e gate, observability | [01-architecture.md §10, §12](../../../docs/architecture/docs/01-architecture.md) |
| Refusal & evidence verification | [ADR-0008](../../../docs/architecture/adr/0008-evidence-verification-and-refusal.md) |
| Prompt-injection defense | [ADR-0023](../../../docs/architecture/adr/0023-untrusted-evidence-in-prompts.md), [01-architecture.md §14](../../../docs/architecture/docs/01-architecture.md) |
| E2E traces & faithfulness | [ADR-0022](../../../docs/architecture/adr/0022-evaluation-protocol.md), `configs/prompts/grounded_answer.md` |
| Parameters | `params.yaml`: `retrieval.candidates_k`, `retrieval.context_k`, `verification`, `promotion` |

## Core Principles

1. Measure every stage both standalone and end-to-end; e2e score alone hides which stage failed.
2. Start with the simplest chain: `hybrid retrieval → reranker → top-k → LLM → NLI`. Add query rewriting, HyDE, or multi-hop only if ablation proves a CI gain within latency NFRs.
3. Keep the exact same prompt and chat template across SFT, eval, and serving (ADR-0012).
4. Run bulk `eval_e2e` and ablation sweeps on cloud GPU (Colab L4 or approved cloud GPU) with per-example traces logged to MLflow and persisted to Drive.

## Context Assembly

- **Dedup:** Collapse near-duplicate chunks (mirrors, templates), keeping the highest-reranked copy.
- **Merge:** Concatenate adjacent chunks from the same document to preserve continuity and save header tokens.
- **Order:** Sort by descending reranker score (strongest evidence first); keep `k` compact to avoid lost-in-the-middle degradation.
- **Token budget by generator tokenizer:** `kk` has ~2.4× higher token fertility than `en` in Qwen3.5 (`models.md §6.2`). Enforce a Qwen3.5 token budget rather than a fixed chunk count.
- **Passage header:** Use short numeric IDs `[n]` in the prompt (mapped back to `chunk_id` post-generation to prevent truncation), plus document title, section, page, and language.

## Prompt, Refusal & Prompt Injection

- **Language & citations:** Explicitly specify `answer_lang`; translate cross-lingual evidence without adding facts; verify output language with `langid`. Require `[n]` after every fact; reject any citation outside supplied IDs and track invalid-citation rate.
- **Decoding:** Non-thinking mode, low temperature, explicit stop sequences, capped output tokens.
- **Conflicting sources:** Surface both claims with their respective citations; let NLI verification flag contradictions rather than silently picking one.
- **Refusal:** Use the exact canonical refusal string per language (`grounded_answer.md`). Optional early refusal (calibrated max reranker score < $\tau$) saves latency but monitor cross-lingual false refusals. Final refusal follows ADR-0008.
- **Indirect prompt injection (ADR-0023):** Treat document passages as untrusted data. Isolate passages with spotlighting delimiters, forbid tool use by the generator, and validate outputs independently (citation IDs, NLI, language). Never mutate/strip passage text (preserves citation offsets). Include `kk`/`ru`/`en` attack suite in `eval_e2e` (`promotion.e2e_injection_no_worse`).

## Stage-Attributed Failure Diagnosis

Log full traces per eval example (per-leg ranks, post-fusion rank, post-rerank rank, post-budget prompt presence, answer, claims, verdicts). Assign each failure to the first "No":

| # | Check | Failing Stage |
|---|---|---|
| 1 | Required fact exists in some chunk? | Document processing / curation / chunking |
| 2 | Gold in first-stage `top-candidates_k`? | Retrieval |
| 3 | Gold in `top-context_k` after reranker? | Reranker |
| 4 | Gold survives token-budget truncation in prompt? | Context assembly |
| 5 | Answer correct when gold is in prompt? | Generation (evidence utilization) |
| 6 | Citations point to supporting passage? | Generation (attribution) |
| 7 | Verifier verdict matches human label? | Verification |

Slice shares by language and direction; fix the largest contributor first. Track auxiliary RAGChecker diagnostics: claim recall, context precision, context utilization, noise sensitivity, hallucination rate.

## Latency & Ablation

- Levers: Redis query-embedding cache, reranker `top-N`, parallel `workspace`/`corpus` Qdrant queries, SSE token streaming, batched NLI across all claims. Measure p95 under target concurrency, not single requests.
- Stage 9 ablation (same frozen test): `RAG w/o Extraction → + Extraction signals → + fine-tuned Retrieval → + fine-tuned LLM → + Verification`.

## Done Criteria

- End-to-end gate passes (`promotion.e2e_*`), citation accuracy ≥ champion, prompt-injection ASR ≤ champion.
- Per-language stage-attribution table and ablation deltas (with CI) logged to MLflow `e2e`.

## Sources

- Ru et al., *RAGChecker* (arXiv:2408.08067); Es et al., *RAGAS* (arXiv:2309.15217); Gao et al., *ALCE* (arXiv:2305.14627).
- Liu et al., *Lost in the Middle* (arXiv:2307.03172); Barnett et al. (arXiv:2401.05856).
- Greshake et al. (arXiv:2302.12173); Hines et al., *Spotlighting* (arXiv:2403.14720); Yi et al., *BIPIA* (arXiv:2312.14197).
