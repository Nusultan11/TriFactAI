"""Comprehensive Stage 6 tests: embedder, fusion, two-tier Qdrant store, eval/ranx parity,
promotion gate + atomic rollback, miss attribution, sharded embed/index, and training guards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trifact.batch.embed_corpus import (
    kk_letter_share,
    mask_pii,
    passes_downstream_filter,
    run_embed_corpus,
)
from trifact.batch.index_corpus import benchmark_ann_vs_exact, index_vector_shards
from trifact.core.schemas import Chunk, Lang, RetrievalScope, ScoredChunk, SourceType
from trifact.evaluation.eval_retrieval import (
    ALL_DIRECTIONS,
    attribute_retrieval_misses,
    compare_multi_seed_retrieval,
    compare_retrieval_runs,
    evaluate_rankings,
    verify_ranx_parity,
)
from trifact.evaluation.gate import (
    evaluate_retrieval_gate,
    promote_retrieval_with_qdrant,
    rollback_retrieval_with_qdrant,
)
from trifact.retrieval.embedder import (
    BM25SparseEncoder,
    HashEmbedder,
    apply_mrl,
    dense_exact_search,
    detect_model_family,
    estimate_vector_bytes,
    format_passage_text,
    format_query_text,
    sparse_dot_search,
    tokenize_sparse,
)
from trifact.retrieval.hybrid import (
    apply_entity_soft_boost,
    dbsf_fuse,
    merge_adjacent_chunks,
    rrf_fuse,
    weighted_fuse,
)
from trifact.retrieval.reranker import LexicalOverlapReranker, fit_platt_scaling, platt_sigmoid
from trifact.retrieval.store import (
    CORPUS_ALIAS,
    WORKSPACE_ALIAS,
    CorpusIndexConfig,
    QdrantHybridRetriever,
    QdrantStore,
    versioned_collection_name,
)
from trifact.training.train_retriever import filter_false_negatives_positive_aware


def _make_chunk(cid: str, doc_id: str, text: str, lang: Lang = Lang.EN, ordinal: int = 0) -> Chunk:
    return Chunk(
        document_id=doc_id,
        chunk_id=cid,
        ordinal=ordinal,
        language=lang,
        char_start=ordinal * 100,
        char_end=ordinal * 100 + len(text),
        text=text,
        source_type=SourceType.CORPUS,
        corpus_source="wikipedia",
        token_count=max(1, len(text.split())),
    )


def test_embedder_prefixes_mrl_and_kk_stemming() -> None:
    assert detect_model_family("BAAI/bge-m3") == "bge_m3"
    assert detect_model_family("intfloat/multilingual-e5-large") == "me5"
    assert detect_model_family("intfloat/multilingual-e5-large-instruct") == "me5_instruct"
    assert detect_model_family("Qwen/Qwen3-Embedding-0.6B") == "qwen3"

    assert format_query_text("Astana", "me5") == "query: Astana"
    assert format_passage_text("Capital city", "me5") == "passage: Capital city"
    assert format_query_text("Astana", "qwen3").startswith("Instruct:")
    assert format_passage_text("Body", "bge_m3", title="Title") == "Title\nBody"

    vecs = np.array([[3.0, 4.0, 0.0, 0.0]], dtype=np.float32)
    mrl = apply_mrl(vecs, 2)
    assert mrl.shape == (1, 2)
    assert np.isclose(np.linalg.norm(mrl[0]), 1.0)

    # Kazakh suffix stripping maps inflected forms to shared stem
    stems = {tokenize_sparse(w, stem_kk=True)[0] for w in ("Қазақстанда", "Қазақстанның", "Қазақстанға")}
    assert len(stems) == 1

    est = estimate_vector_bytes(35_000_000, 1024, "binary")
    assert 60.0 < est["fp16_disk_gb"] < 70.0
    assert 4.0 < est["quant_ram_gb"] < 4.5


def test_dense_and_sparse_exact_search_and_fusion() -> None:
    emb = HashEmbedder(dim=64)
    docs = ["Astana is the capital of Kazakhstan", "Almaty is the largest city", "Paris is in France"]
    doc_ids = ["d1", "d2", "d3"]
    d_vecs = np.asarray(emb.embed_dense(docs), dtype=np.float32)
    q_vecs = np.asarray(emb.embed_queries(["capital of Kazakhstan Astana"]), dtype=np.float32)
    res_d = dense_exact_search(q_vecs, d_vecs, doc_ids, top_k=3)
    assert res_d[0][0][0] == "d1"

    sp_enc = BM25SparseEncoder()
    res_s = sparse_dot_search(
        sp_enc.encode_queries(["Astana Kazakhstan"]), sp_enc.encode_documents(docs), doc_ids, top_k=3
    )
    assert res_s[0][0][0] == "d1"

    # Fusion methods preserve per-leg ranks
    c1 = ScoredChunk(chunk=_make_chunk("c1", "d1", docs[0]), score=0.9, ranks={"dense": 1})
    c2 = ScoredChunk(chunk=_make_chunk("c2", "d2", docs[1]), score=0.5, ranks={"dense": 2})
    s1 = ScoredChunk(chunk=_make_chunk("c2", "d2", docs[1]), score=10.0, ranks={"sparse": 1})
    s2 = ScoredChunk(chunk=_make_chunk("c1", "d1", docs[0]), score=8.0, ranks={"sparse": 2})

    for fn in (rrf_fuse, dbsf_fuse, weighted_fuse):
        fused = fn({"dense": [c1, c2], "sparse": [s1, s2]}, top_k=2)
        assert len(fused) == 2
        assert all("dense" in x.ranks and "sparse" in x.ranks for x in fused)

    # Entity soft boost NEVER drops chunks (ADR-0004)
    boosted = apply_entity_soft_boost([c1, c2], {"c2": ["Q1520"]}, ["Q1520"], boost=1.5)
    assert len(boosted) == 2
    assert boosted[0].chunk.chunk_id == "c2"

    # Adjacent chunk merge
    ch_a = ScoredChunk(chunk=_make_chunk("d1#0", "d1", "Part one.", ordinal=0), score=0.8)
    ch_b = ScoredChunk(chunk=_make_chunk("d1#1", "d1", "Part two.", ordinal=1), score=0.7)
    merged = merge_adjacent_chunks([ch_a, ch_b])
    assert len(merged) == 1
    assert "Part one." in merged[0].chunk.text and "Part two." in merged[0].chunk.text


def test_two_tier_qdrant_isolation_aliases_and_retrieval() -> None:
    store = QdrantStore(in_memory=True)
    emb = HashEmbedder(dim=32)
    reranker = LexicalOverlapReranker()

    c_v1 = versioned_collection_name(CORPUS_ALIAS, "BAAI/bge-m3", "v1")
    c_v2 = versioned_collection_name(CORPUS_ALIAS, "BAAI/bge-m3", "v2")
    ws_col = versioned_collection_name(WORKSPACE_ALIAS, "BAAI/bge-m3", "v1")

    store.create_corpus_collection(c_v1, CorpusIndexConfig(dim=32, quantization="binary"))
    store.create_corpus_collection(c_v2, CorpusIndexConfig(dim=32, quantization="scalar"))
    store.create_workspace_collection(ws_col, dim=32)
    store.switch_alias(CORPUS_ALIAS, c_v1)
    store.switch_alias(WORKSPACE_ALIAS, ws_col)

    corpus_chunks = [
        _make_chunk("corp#0", "corp_doc", "Astana is the capital of Kazakhstan", Lang.EN),
        _make_chunk("corp#1", "corp_doc2", "Астана Қазақстанның астанасы", Lang.KK),
    ]
    store.upsert_chunks(
        c_v1,
        corpus_chunks,
        emb.embed_dense([c.text for c in corpus_chunks]),
        emb.embed_sparse([c.text for c in corpus_chunks]),
        entity_ids_by_chunk={"corp#0": ["Q1520"]},
        embedder_version=emb.version,
    )

    ws_a_chunks = [_make_chunk("wsA#0", "docA", "Secret project alpha budget Kazakhstan", Lang.EN)]
    ws_b_chunks = [_make_chunk("wsB#0", "docB", "Confidential workspace beta Kazakhstan", Lang.EN)]
    store.upsert_chunks(
        ws_col,
        ws_a_chunks,
        emb.embed_dense([c.text for c in ws_a_chunks]),
        emb.embed_sparse([c.text for c in ws_a_chunks]),
        workspace_id="ws_alpha",
    )
    store.upsert_chunks(
        ws_col,
        ws_b_chunks,
        emb.embed_dense([c.text for c in ws_b_chunks]),
        emb.embed_sparse([c.text for c in ws_b_chunks]),
        workspace_id="ws_beta",
    )

    retriever = QdrantHybridRetriever(store, emb, reranker=reranker, prefetch_k=10, candidates_k=10)

    # Strict workspace isolation
    with pytest.raises(ValueError, match="workspace_id"):
        retriever.search("Kazakhstan", top_k=5, scope=RetrievalScope.DOCUMENTS)

    res_a = retriever.search("Kazakhstan", top_k=5, scope=RetrievalScope.DOCUMENTS, workspace_id="ws_alpha")
    assert [r.chunk.chunk_id for r in res_a] == ["wsA#0"]
    assert "rerank" in res_a[0].ranks

    # Corpus search works WITHOUT entity annotations (ADR-0004)
    res_corp = retriever.search("Астана", top_k=5, scope=RetrievalScope.CORPUS, lang="kk")
    assert [r.chunk.chunk_id for r in res_corp] == ["corp#1"]

    # Explicit user-selected entity_filter restricts when requested
    res_ent = retriever.search("Astana", top_k=5, scope=RetrievalScope.CORPUS, entity_filter=["Q1520"])
    assert [r.chunk.chunk_id for r in res_ent] == ["corp#0"]

    # Scope ALL fuses both collections
    res_all = retriever.search("Kazakhstan", top_k=5, scope=RetrievalScope.ALL, workspace_id="ws_alpha")
    ids_all = {r.chunk.chunk_id for r in res_all}
    assert "wsA#0" in ids_all and "corp#0" in ids_all

    # Atomic alias switch & rollback
    assert store.get_alias_target(CORPUS_ALIAS) == c_v1
    sw = store.switch_alias(CORPUS_ALIAS, c_v2)
    assert sw["previous"] == c_v1 and sw["current"] == c_v2
    assert store.get_alias_target(CORPUS_ALIAS) == c_v2
    store.switch_alias(CORPUS_ALIAS, c_v1)
    assert store.get_alias_target(CORPUS_ALIAS) == c_v1


def test_eval_ranx_parity_gate_rollback_and_miss_attribution(tmp_path: Path) -> None:
    # Build synthetic 9-direction eval with shared cluster_ids
    q_rows = []
    qr_rows = []
    base_rank: dict[str, list[tuple[str, float]]] = {}
    better_rank: dict[str, list[tuple[str, float]]] = {}
    run_for_ranx: dict[str, dict[str, float]] = {}

    for c_idx in range(12):
        cid = f"cluster_{c_idx}"
        for d in ALL_DIRECTIONS:
            ql, dl = d.split("-")
            qid = f"{cid}/{d}"
            gold = f"doc_{c_idx}_{dl}"
            neg = f"neg_{c_idx}_{dl}"
            q_rows.append(
                {
                    "query_id": qid,
                    "query": f"query {c_idx} {d}",
                    "q_lang": ql,
                    "d_lang": dl,
                    "direction": d,
                    "source": "miracl" if dl == "ru" else "kazqad",
                    "origin": "gold" if ql == dl else "mt_query",
                    "verification_tier": "human_verified" if ql == dl else "machine_cross_validated",
                    "cluster_id": cid,
                }
            )
            qr_rows.append({"query_id": qid, "doc_id": gold, "relevance": 1, "lang": dl})
            qr_rows.append({"query_id": qid, "doc_id": neg, "relevance": 0, "lang": dl})
            # Baseline puts gold at rank 3, challenger puts gold at rank 1
            base_rank[qid] = [(neg, 0.9), (f"other_{c_idx}", 0.8), (gold, 0.7)]
            better_rank[qid] = [(gold, 0.95), (neg, 0.5), (f"other_{c_idx}", 0.2)]
            run_for_ranx[qid] = {gold: 0.95, neg: 0.5, f"other_{c_idx}": 0.2}

    q_df = pl.DataFrame(q_rows)
    qr_df = pl.DataFrame(qr_rows)

    # Verify exact ranx parity
    pos_map = {r["query_id"]: {r["doc_id"]: 1} for r in qr_rows if r["relevance"] == 1}
    parity = verify_ranx_parity(run_for_ranx, pos_map)
    assert parity["max_abs_diff"] < 1e-5

    ev_base = evaluate_rankings(q_df, qr_df, base_rank, n_bootstrap=200, seed=42)
    ev_cand = evaluate_rankings(q_df, qr_df, better_rank, n_bootstrap=200, seed=42)
    assert ev_cand["macro"]["ndcg@10"] > ev_base["macro"]["ndcg@10"]
    assert ev_cand["n_directions"] == 9
    assert "human_verified" in ev_cand["by_verification_tier"]
    assert "machine_cross_validated" in ev_cand["by_verification_tier"]

    cmp = compare_retrieval_runs(ev_base["per_query"], ev_cand["per_query"], n_bootstrap=200, seed=42)
    assert cmp["macro"]["significant"] is True

    # Multi-seed comparison
    ms = compare_multi_seed_retrieval(
        [ev_base["per_query"]] * 3,
        [ev_cand["per_query"]] * 3,
        n_bootstrap=100,
        seed=42,
    )
    assert ms["macro"]["significant"] is True

    # Promotion gate PASS + atomic MLflow + Qdrant alias + rollback
    promo_cfg = {"retrieval_no_direction_worse": True, "worst_direction_max_gap": 0.10}
    verdict = evaluate_retrieval_gate(cmp, ev_cand, promo_cfg)
    assert verdict["passed"] is True

    # Atomic MLflow + Qdrant promotion and rollback
    store = QdrantStore(in_memory=True)
    col1 = versioned_collection_name(CORPUS_ALIAS, "bge-m3", "v1")
    col2 = versioned_collection_name(CORPUS_ALIAS, "bge-m3", "v2")
    store.create_corpus_collection(col1, CorpusIndexConfig(dim=16, quantization="none"))
    store.create_corpus_collection(col2, CorpusIndexConfig(dim=16, quantization="none"))
    store.switch_alias(CORPUS_ALIAS, col1)

    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    p1 = promote_retrieval_with_qdrant(
        "trifact-embedder",
        (tmp_path / "m1").as_uri(),
        {"passed": False},
        store=store,
        qdrant_alias=CORPUS_ALIAS,
        target_collection=col1,
        tracking_uri=tracking_uri,
        bootstrap_champion_if_none=True,
    )
    assert p1["promoted"] is True and p1["champion"] == "1"

    p2 = promote_retrieval_with_qdrant(
        "trifact-embedder",
        (tmp_path / "m2").as_uri(),
        verdict,
        store=store,
        qdrant_alias=CORPUS_ALIAS,
        target_collection=col2,
        tracking_uri=tracking_uri,
    )
    assert p2["promoted"] is True and p2["champion"] == "2"
    assert store.get_alias_target(CORPUS_ALIAS) == col2

    rb = rollback_retrieval_with_qdrant(
        "trifact-embedder",
        "1",
        store=store,
        qdrant_alias=CORPUS_ALIAS,
        previous_collection=col1,
        tracking_uri=tracking_uri,
    )
    assert rb["version"] == "1"
    assert store.get_alias_target(CORPUS_ALIAS) == col1

    # Stage-attributed miss analysis
    miss_q = pl.DataFrame(
        [
            {"query_id": "q1", "query": "a", "direction": "kk-kk"},
            {"query_id": "q2", "query": "b", "direction": "kk-kk"},
            {"query_id": "q3", "query": "c", "direction": "ru-kk"},
            {"query_id": "q4", "query": "d", "direction": "en-kk"},
            {"query_id": "q5", "query": "e", "direction": "en-en"},
        ]
    )
    miss_qr = pl.DataFrame(
        [{"query_id": f"q{i}", "doc_id": f"g{i}", "relevance": 1, "lang": "kk"} for i in range(1, 6)]
    )
    attr = attribute_retrieval_misses(
        miss_q,
        miss_qr,
        indexed_doc_ids={"g2", "g3", "g4", "g5"},  # g1 missing from index
        dense_top100={"q2": [], "q3": [], "q4": ["g4"], "q5": ["g5"]},
        sparse_top100={"q2": [], "q3": [], "q4": [], "q5": []},
        fused_top50={"q2": [], "q3": [], "q4": [], "q5": ["g5"]},
        final_top10={"q1": ["x"], "q2": ["x"], "q3": ["x"], "q4": ["x"], "q5": ["x"]},
        exact_dense_top100={"q2": ["g2"], "q3": [], "q4": ["g4"], "q5": ["g5"]},
    )
    assert attr["overall_counts"]["missing_from_index"] == 1
    assert attr["overall_counts"]["ann_quantization_loss"] == 1
    assert attr["overall_counts"]["representation_miss"] == 1
    assert attr["overall_counts"]["dropped_by_fusion"] == 1
    assert attr["overall_counts"]["demoted_by_reranker"] == 1


def test_sharded_embed_index_resume_pilot_isolation_and_downstream_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ensure cloud-only guard triggers when not on cloud runtime
    monkeypatch.delenv("TRIFACT_CLOUD_RUNTIME", raising=False)
    in_dir = tmp_path / "chunks"
    in_dir.mkdir()
    out_dir = tmp_path / "vectors"

    tbl = pa.table(
        {
            "chunk_id": ["d1#0", "d2#0", "d3#0"],
            "doc_id": ["d1", "d2", "d3"],
            "source": ["wikipedia", "fineweb2_kk", "fineweb2_kk"],
            "lang": ["kk", "kk", "en"],
            "title": ["Астана", "BadKK", "Contact"],
            "ordinal": [0, 0, 0],
            "char_start": [0, 0, 0],
            "char_end": [30, 20, 40],
            "n_tokens": [10, 5, 10],
            "text": [
                "Қазақстанның астанасы Астана қаласы",
                "plain russian text without kazakh letters",  # kk_letter_share = 0.0 -> filtered!
                "Email user@example.com for info",
            ],
        }
    )
    pq.write_table(tbl, in_dir / "part-00000.parquet")

    assert kk_letter_share("Қазақстан") > 0.05
    assert not passes_downstream_filter("plain russian text", "kk", 0.05)
    assert "[EMAIL]" in mask_pii("Contact test@example.com now")

    emb = HashEmbedder(dim=16)
    with pytest.raises(RuntimeError, match="ADR-0025"):
        run_embed_corpus(in_dir, out_dir, emb, allow_local_pilot=False)

    # Limited pilot creates .PILOT.json, NEVER .SUCCESS.json
    pilot = run_embed_corpus(in_dir, out_dir, emb, limit_chunks=2, allow_local_pilot=True)
    assert pilot["pilot"] is True
    assert (out_dir / "part-00000.PILOT.json").is_file()
    assert not (out_dir / "part-00000.SUCCESS.json").exists()

    # Full run with downstream filter drops the bad kk row and masks email
    full = run_embed_corpus(
        in_dir, out_dir, emb, apply_downstream_filter=True, contextual_header=True, allow_local_pilot=True
    )
    assert full["total_rows"] == 2
    assert (out_dir / "part-00000.SUCCESS.json").is_file()

    # Idempotent resume skips already-verified shard
    resumed = run_embed_corpus(
        in_dir, out_dir, emb, apply_downstream_filter=True, contextual_header=True, allow_local_pilot=True
    )
    assert resumed["shards_skipped"] == 1 and resumed["shards_processed"] == 0

    # Index into in-memory Qdrant and verify ANN vs exact benchmark
    store = QdrantStore(in_memory=True)
    col = versioned_collection_name(CORPUS_ALIAS, "hash", "v1")
    idx_rep = index_vector_shards(
        store,
        out_dir,
        col,
        CorpusIndexConfig(dim=16, quantization="none"),
        switch_alias=True,
    )
    assert idx_rep["verified_count"] == 2
    assert store.get_alias_target(CORPUS_ALIAS) == col

    q_vecs = np.asarray(emb.embed_queries(["Астана", "Contact"]), dtype=np.float32)
    ann_rep = benchmark_ann_vs_exact(store, col, q_vecs, top_k=2, candidate_k=2)
    assert ann_rep["ann_vs_exact_recall@10"] == 1.0


def test_positive_aware_hard_negative_filter_and_platt() -> None:
    kept = filter_false_negatives_positive_aware(
        pos_score=0.90,
        candidate_scores=[("fn1", 0.89), ("hn1", 0.72), ("hn2", 0.65), ("hn3", 0.50)],
        margin_ratio=0.95,
        max_negatives=2,
    )
    assert kept == ["hn1", "hn2"]

    cal = fit_platt_scaling([2.0, 1.5, -1.0, -2.0], [1, 1, 0, 0])
    assert platt_sigmoid(2.0, cal["a"], cal["b"]) > platt_sigmoid(-2.0, cal["a"], cal["b"])
