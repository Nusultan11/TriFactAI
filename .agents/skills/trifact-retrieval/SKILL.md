---
name: trifact-retrieval
description: Cloud-first cross-lingual retrieval at 35M+ chunks in TriFact — chunking experiments, embedders (BGE-M3, mE5, Qwen3-Embedding, MRL), BM25/sparse, hybrid fusion (RRF, DBSF), Qdrant quantization/rescoring/HNSW, reranker training and calibration, positive-aware hard-negative mining, scale ladder, 9-direction evaluation. Use for stage 6 work in trifact.retrieval, embed_corpus, index_corpus, train_retriever, train_reranker, or eval_retrieval.
---

# TriFact AI — Retrieval at Scale

## Scope

- Load for RET-0…RET-7, RET-C, RET-S, corpus embedding/indexing, embedder/reranker fine-tuning, and retrieval debugging.
- LLM context assembly & e2e attribution → `trifact-rag-pipeline`. Statistics & gate → `trifact-evaluation`.
- Cloud CLI execution & persistence → [cloud-compute.md](../trifact-mlops/references/cloud-compute.md).

## Owner Documents

| Topic | Owner |
|---|---|
| Experiments, 35M memory budget, metrics | [01-architecture.md §7.3, §8.2](../../../docs/architecture/docs/01-architecture.md) |
| Qdrant two-tier index & aliases | [ADR-0002](../../../docs/architecture/adr/0002-qdrant-for-dense-and-sparse.md), [ADR-0013](../../../docs/architecture/adr/0013-two-tier-qdrant-index.md) |
| Default embedder & NER independence | [ADR-0003](../../../docs/architecture/adr/0003-bge-m3-default-embedder.md), [ADR-0004](../../../docs/architecture/adr/0004-hybrid-retrieval-independent-of-ner.md) |
| Sharded batch jobs & cloud compute | [ADR-0014](../../../docs/architecture/adr/0014-sharded-batch-jobs.md), [ADR-0025](../../../docs/architecture/adr/0025-cloud-only-computing.md) |
| Models, params, data notes | [models.md §2–3](../../../docs/research/models.md), `params.yaml` (`retrieval`, `chunking`, `scale.index_steps`), [PROGRESS](../../../docs/PROGRESS.md) |

## Cloud Execution & Resource Management

- **Embedding (`embed_corpus`):** Run sharded (`--shard i/N`, `_SUCCESS` markers) on cloud GPU (Colab L4 preferred; approved cloud GPU for 35M full scale). Store fp16 dense + sparse vectors in Parquet shards; avoid re-transferring ~72 GB vectors — build Qdrant collections where they will be served (ADR-0014).
- **Fine-tuning (`train_retriever`, `train_reranker`):** Run on Colab L4 (bf16) with `CachedMNRL`/GradCache for large effective batch sizes; persist checkpoints and MLflow snapshots to Drive.
- **Benchmarking (`eval_retrieval`, RET-S):** Measure ANN recall vs. exact search, latency p50/p95 under target concurrency, QPS, RAM, disk, and index build time on the target serving profile.

## Evaluation Protocol First

- Use frozen `data/eval/retrieval` with `ranx`. Report `Recall@5/10`, `nDCG@10`, `MRR` across all **9 language directions** (`q_lang × doc_lang`) plus macro average.
- Report first-stage **candidate recall@50/100** separately (reranker ceiling).
- Track `judged@10` on closed qrel pools; manually audit a sample of unjudged top results for novel systems.
- Use **cluster paired bootstrap** grouping translations of the same query into one cluster (ADR-0022).
- Map gold passages to chunks identically across chunking variants via offset overlap.

## Chunking (RET-C)

Select from `chunking.variants` strictly by retrieval metrics. Test cheap ablations:
- Prepending document + section title to embedded text (contextual header).
- Late chunking for long-context embedders (encode full document, pool by chunk spans).
- Merging adjacent retrieved chunks from the same document at serving time.

## Embedder & Hybrid Fusion

- **Selection priority:** (1) 9-direction eval quality (MMTEB under-tests `kk`), (2) worst-direction gap (cross-lingual gate), (3) index memory (dim, MRL, binarization tolerance), (4) encoding throughput for 35M+ chunks, (5) license.
- Follow model-card prefixes strictly (`query: `/`passage: ` for mE5; query instruction only for Qwen3-Embedding); L2-normalize vectors; monitor truncation share > `max_length`.
- Compare BM25 (Qdrant IDF) vs. BGE-M3 sparse on `kk` (un-stemmed Kazakh morphology hurts BM25).
- Compare RRF (`retrieval.rrf_k`), DBSF, and weighted fusion per direction; sparse can hurt cross-lingual pairs. Log per-leg ranks in eval mode (ADR-0013).
- Embedder swap = new collection `corpus_chunks__<embedder>@<ver>` + atomic alias switch after gate.

## Reranker & Fine-Tuning (RET-5, RET-6, RET-7)

- Pick reranker `top-N` from quality vs. latency curve (fp16 GPU / ONNX CPU); calibrate logits via per-language Platt scaling on dev if using an early-refusal threshold.
- Train reranker on real first-stage candidates matching serving.
- **Hard-negative mining:** Mine rank window (e.g., 10–100) with current model; filter false negatives via cross-encoder `max_score` + positive-aware margin (`sentence_transformers.util.mine_hard_negatives`, NV-Retriever). Balance sampling across 9 directions so short KazParC sentences do not dominate passages.
- **Losses:** `CachedMultipleNegativesRankingLoss` or `CachedGISTEmbedLoss` + `NO_DUPLICATES` batch sampler; wrap in `MatryoshkaLoss` for MRL. Prevent catastrophic forgetting with low LR, general-pair mix, and regression checks on all 9 directions + MIRACL.
- **Reranker loss:** BCE or listwise loss over `(query, pos, k first-stage negs)` or LLM-reranker distillation.

## Scale Ladder (RET-S) & Miss Attribution

Scale steps (`1M → 10M → 35M`), quantization, HNSW tuning, and bulk loading → [references/qdrant-at-scale.md](references/qdrant-at-scale.md). Always separate ANN/quantization loss from model loss using `exact=true` on a query subsample.

Attribute every missed gold passage to its first failing stage (per direction; manually inspect ≥50 misses on worst direction):
1. Missing from index (filtered/dropped/parse loss).
2. Split across chunk boundary.
3. Absent from both legs' top-100 (representation miss).
4. Present in legs, dropped by fusion.
5. Lost to ANN/quantization (found by `exact=true`).
6. Demoted by reranker.
7. Unjudged relevant passage.

## Done Criteria

- Champion beats baseline on mean nDCG@10 (CI > 0), regresses on 0 of 9 directions (`promotion.retrieval_*`), and meets `promotion.worst_direction_max_gap`.
- RET-S scale curves (ANN recall vs. exact, p95, RAM, disk, build time) recorded; Serving VM decision made (ADR-0013).
- Stage miss attribution logged to MLflow `retrieval`.

## Sources

- Chen et al., *BGE-M3* (arXiv:2402.03216); Wang et al., *mE5* (arXiv:2402.05672); Zhang et al., *Qwen3 Embedding* (arXiv:2506.05176).
- Enevoldsen et al., *MMTEB* (arXiv:2502.13595); Zhang et al., *MIRACL* (arXiv:2210.09984); Cormack et al., *RRF* (SIGIR 2009); Kusupati et al., *MRL* (arXiv:2205.13147).
- Moreira et al., *NV-Retriever* (arXiv:2407.15831); Solatorio, *GISTEmbed* (arXiv:2402.16829); Günther et al., *Late Chunking* (arXiv:2409.04701); `sentence-transformers`; `ranx`.
