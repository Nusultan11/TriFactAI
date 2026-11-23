# Qdrant at Scale (1M → 10M → 35M+)

Project decisions: [ADR-0013](../../../../docs/architecture/adr/0013-two-tier-qdrant-index.md), [01-architecture.md §7.3](../../../../docs/architecture/docs/01-architecture.md), `params.yaml: retrieval.corpus_index`. Server version: `infra/docker-compose.yml`.

## Bulk Collection Build Order

1. Create `corpus_chunks__<embedder>@<ver>` with named vectors `dense` (`float16`, `on_disk: true`) and `sparse` (IDF modifier for BM25); store payload on disk.
2. Create payload indexes (`lang`, `source`, `entity_ids`) **before** bulk upload to avoid expensive post-hoc rebuilds.
3. Bulk-upload `task/vectors` shards with HNSW disabled (`indexing_threshold: 0` or `m: 0`) via idempotent sharded jobs (`_SUCCESS` markers, ADR-0014).
4. Enable HNSW (`m`, `ef_construct` from `params.yaml`) and quantization; wait for collection status `green`.
5. Verify point count, sample queries, and approximate vs. exact (`exact=true`) recall.
6. Switch alias `corpus_chunks` only after passing the promotion gate; retain old collection until verified for instant rollback.

## Quantization Trade-offs

| Type | Memory vs. fp32 | When to Use | Risk |
|---|---|---|---|
| `scalar` (int8) | 4× smaller | Serving VM with RAM headroom | Minimal recall loss |
| `binary` (1-bit) | 32× smaller | Tight RAM, high-dim models | Large loss without rescoring; model-dependent |
| 1.5/2-bit / asymmetric | Between binary & int8 | Recall/RAM middle ground | Benchmark on TriFact eval |
| `turbo` (1–4 bit) | Configurable | RET-S candidate | Measure recall and latency |
| `product` | 16–64× smaller | Extreme RAM constraint | High recall loss, slow rescoring |

- Always pair aggressive quantization with `rescore: true` and `oversampling > 1` (`retrieval.corpus_index.oversampling`). Rescoring reads fp16 vectors from disk, so latency depends on NVMe/SSD speed.
- Measure recall vs. `oversampling` separately per embedder. MRL truncation (256/512) + binary quantization compound losses — test on a grid.

## HNSW Tuning

- Level-0 graph RAM ≈ $N \times m \times 2 \times 4$ bytes of links. Tune query-time `hnsw_ef` to meet target recall@100 vs. `exact=true`.
- `on_disk: true` saves RAM at the cost of random disk I/O; measure p95 on both cold and warm OS page cache.
- Low-selectivity payload filters can bypass HNSW; index filter fields and set `is_tenant: true` on `workspace_id`.

## Measurements per Scale Step

Record at 1M, 10M, and 35M: approximate vs. exact Recall@k (1–5k queries), eval nDCG@10, p50/p95 latency under target concurrency, Qdrant RAM, disk footprint, upload time, and HNSW build time. If 35M exceeds local/target NFRs, deploy full corpus to a Serving VM and keep a subset in local dev (ADR-0013).
