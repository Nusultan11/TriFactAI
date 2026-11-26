"""Cloud runner for Stage 6 — Retrieval at Scale (RET-0..RET-7, RET-C, RET-S, and GPU annotate_corpus).

Designed for free Colab T4 / L4 GPU or Kaggle 2x T4 ($0 USD, ADR-0025), persisting all
JSON reports, per-query traces, checkpoints, and MLflow SQLite snapshots to Drive.

Usage:
    python scripts/cloud_run_stage6.py phase1 --out-dir reports/stage6
    python scripts/cloud_run_stage6.py phase2 --out-dir reports/stage6
    python scripts/cloud_run_stage6.py phase3 --out-dir reports/stage6
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import psutil

from trifact.core.params import ROOT, load_params
from trifact.data.bm25 import BM25
from trifact.evaluation.eval_retrieval import (
    attribute_retrieval_misses,
    compare_multi_seed_retrieval,
    compare_retrieval_runs,
    evaluate_rankings,
    load_frozen_retrieval_eval,
    verify_ranx_parity,
)
from trifact.evaluation.gate import (
    evaluate_retrieval_gate,
)
from trifact.evaluation.stats import write_per_example_jsonl
from trifact.ingestion.chunker import chunk_text, hf_counter
from trifact.lake.stats import git_sha
from trifact.retrieval.embedder import (
    MultilingualEmbedder,
    apply_mrl,
    dense_exact_search,
    estimate_vector_bytes,
    sparse_dot_search,
    tokenize_sparse,
)
from trifact.retrieval.hybrid import fuse_id_rankings
from trifact.retrieval.reranker import CrossEncoderReranker
from trifact.retrieval.store import CorpusIndexConfig, QdrantStore
from trifact.training.train_reranker import train_cross_encoder
from trifact.training.train_retriever import (
    build_balanced_pairs,
    mine_positive_aware_negatives,
    train_biencoder,
)

log = logging.getLogger("trifact.cloud_stage6")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def _gpu_info() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {"device": "cpu", "name": "cpu", "vram_gb": 0.0}
    props = torch.cuda.get_device_properties(0)
    return {
        "device": "cuda",
        "name": props.name,
        "vram_gb": round(props.total_memory / (1024**3), 2),
        "bf16": bool(torch.cuda.is_bf16_supported()),
    }


def _clear_gpu() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _snapshot_mlflow(local_db: Path, drive_dir: Path | None) -> None:
    if drive_dir is None or not local_db.is_file():
        return
    drive_dir.mkdir(parents=True, exist_ok=True)
    dst = drive_dir / local_db.name
    src_conn = sqlite3.connect(str(local_db))
    dst_conn = sqlite3.connect(str(dst))
    with dst_conn:
        src_conn.backup(dst_conn)
    src_conn.close()
    dst_conn.close()


def _log_mlflow_run(
    tracking_uri: str,
    run_name: str,
    params_dict: dict[str, Any],
    report: dict[str, Any],
    artifacts: list[Path] | None = None,
) -> None:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("retrieval")
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("stage", "6")
        mlflow.set_tag("git_sha", git_sha(ROOT))
        for k, v in params_dict.items():
            mlflow.log_param(k, str(v)[:250])
        macro = report.get("macro") or {}
        for mk, mv in macro.items():
            if isinstance(mv, (int, float)):
                mlflow.log_metric(f"macro_{mk.replace('@', '_at_')}", float(mv))
        if "worst_direction_gap" in report:
            mlflow.log_metric("worst_direction_gap", float(report["worst_direction_gap"]))
        for d, d_metrics in (report.get("by_direction") or {}).items():
            if "ndcg@10" in d_metrics:
                mlflow.log_metric(f"ndcg_at_10_{d.replace('-', '_')}", float(d_metrics["ndcg@10"]))
        for a in artifacts or []:
            if a.is_file():
                mlflow.log_artifact(str(a))


# ---------------------------------------------------------------- Phase 1: RET-0 .. RET-4


def run_phase1(out_dir: Path, eval_dir: Path, tracking_uri: str, drive_dir: Path | None) -> dict[str, Any]:
    """Execute RET-0 (BM25 / kk_stem), RET-1 (mE5), RET-2 (BGE-M3), RET-3 (Qwen3 MRL), RET-4 (Fusion)."""
    params = load_params()
    n_boot = int(params["evaluation"]["bootstrap_samples"])
    seed = int(params["seed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    queries, qrels, docs, frozen_check = load_frozen_retrieval_eval(eval_dir, verify_frozen=True, seed=seed)
    doc_ids = docs["doc_id"].to_list()
    doc_texts = [
        f"{(t or '').strip()}\n{(x or '').strip()}".strip()
        for t, x in zip(docs["title"].to_list(), docs["text"].to_list(), strict=True)
    ]
    q_ids = queries["query_id"].to_list()
    q_texts = queries["query"].to_list()
    q_dirs = dict(zip(q_ids, queries["direction"].to_list(), strict=True))
    doc_id_set = set(doc_ids)

    summary: dict[str, Any] = {
        "phase": "phase1_ret0_to_ret4",
        "git_sha": git_sha(ROOT),
        "gpu": _gpu_info(),
        "frozen_check": frozen_check,
        "n_queries": len(q_ids),
        "n_docs": len(doc_ids),
        "experiments": {},
    }

    # --- RET-0: BM25 & BM25 + Kazakh suffix stemming ---
    bm25_rankings: dict[str, list[tuple[str, float]]] = {}
    for variant, stem_kk in (("RET-0_bm25", False), ("RET-0_bm25_kk_stem", True)):
        t0 = time.perf_counter()
        p_docs = [" ".join(tokenize_sparse(t, stem_kk=True)) for t in doc_texts] if stem_kk else doc_texts
        p_qs = [" ".join(tokenize_sparse(t, stem_kk=True)) for t in q_texts] if stem_kk else q_texts
        bm25 = BM25(p_docs)
        idx_mat, sc_mat = bm25.search(p_qs, k=100)
        rk: dict[str, list[tuple[str, float]]] = {}
        for i, qid in enumerate(q_ids):
            rk[qid] = [
                (doc_ids[int(idx_mat[i, j])], float(sc_mat[i, j]))
                for j in range(idx_mat.shape[1])
                if idx_mat[i, j] >= 0
            ]
        dt = time.perf_counter() - t0
        rep = evaluate_rankings(queries, qrels, rk, first_stage_rankings=rk, n_bootstrap=n_boot, seed=seed)
        pq_rows = rep.pop("per_query")
        rep["seconds"] = round(dt, 2)
        if not stem_kk:
            bm25_rankings = rk
            # Verify ranx parity on 1,000-query slice
            pos_map = defaultdict(dict)
            for qid, did, rel in qrels.select("query_id", "doc_id", "relevance").iter_rows():
                if int(rel) > 0:
                    pos_map[str(qid)][str(did)] = int(rel)
            sample_qids = q_ids[:1000]
            run_s = {q: dict(rk[q][:100]) for q in sample_qids}
            rep["ranx_parity"] = verify_ranx_parity(
                run_s, {q: pos_map[q] for q in sample_qids if q in pos_map}
            )
        rep_path = out_dir / f"{variant}.json"
        pq_path = out_dir / f"{variant}_per_query.jsonl"
        _write_json(rep_path, rep)
        write_per_example_jsonl(pq_path, pq_rows)
        _log_mlflow_run(tracking_uri, variant, {"exp": variant, "stem_kk": stem_kk}, rep, [rep_path])
        summary["experiments"][variant] = {
            "macro": rep["macro"],
            "worst_direction": rep["worst_direction"],
            "worst_direction_gap": rep["worst_direction_gap"],
            "seconds": rep["seconds"],
        }
        log.info("%s: macro nDCG@10=%.4f (%.1fs)", variant, rep["macro"]["ndcg@10"], dt)

    # --- RET-1: mE5-large-instruct ---
    dense_runs: dict[str, dict[str, list[tuple[str, float]]]] = {}
    per_query_cache: dict[str, list[dict[str, Any]]] = {}

    for exp_id, model_id, max_len, bs in (
        ("RET-1_me5_large_instruct", "intfloat/multilingual-e5-large-instruct", 512, 64),
        ("RET-2_bge_m3", "BAAI/bge-m3", 512, 64),
        ("RET-3_qwen3_emb_0.6b", "Qwen/Qwen3-Embedding-0.6B", 512, 32),
    ):
        t0 = time.perf_counter()
        emb = MultilingualEmbedder(
            model_id,
            max_length=max_len,
            batch_size=bs,
            use_fp16=True,
            use_bge_sparse=(exp_id == "RET-2_bge_m3"),
        )
        if exp_id == "RET-2_bge_m3":
            d_vecs, d_sparse = emb.embed_both(doc_texts, is_query=False)
            q_vecs, q_sparse = emb.embed_both(q_texts, is_query=True)
        else:
            d_vecs = emb.embed_passages_np(doc_texts)
            q_vecs = emb.embed_queries_np(q_texts)
            d_sparse, q_sparse = None, None
        enc_s = time.perf_counter() - t0
        enc_docs_per_s = round((len(doc_texts) + len(q_texts)) / max(enc_s, 1e-6), 1)

        # Save doc/query embeddings for RET-2 BGE-M3 so Phase 2/3 can reuse them cleanly!
        if exp_id == "RET-2_bge_m3":
            np.save(out_dir / "bge_m3_doc_vecs_f16.npy", d_vecs.astype(np.float16))
            np.save(out_dir / "bge_m3_q_vecs_f16.npy", q_vecs.astype(np.float16))

        # Evaluate full dim (and MRL 512/256 for Qwen3-Embedding)
        dims_to_eval = [None, 512, 256] if exp_id == "RET-3_qwen3_emb_0.6b" else [None]
        for mrl_d in dims_to_eval:
            tag = f"{exp_id}_mrl{mrl_d}" if mrl_d else f"{exp_id}_dense"
            dv = apply_mrl(d_vecs, mrl_d) if mrl_d else d_vecs
            qv = apply_mrl(q_vecs, mrl_d) if mrl_d else q_vecs
            res_list = dense_exact_search(qv, dv, doc_ids, top_k=100)
            rk = dict(zip(q_ids, res_list, strict=True))
            rep = evaluate_rankings(
                queries, qrels, rk, first_stage_rankings=rk, n_bootstrap=n_boot, seed=seed
            )
            pq_rows = rep.pop("per_query")
            rep["encode_seconds"] = round(enc_s, 2)
            rep["texts_per_s"] = enc_docs_per_s
            rep["truncation"] = emb.truncation_stats.to_dict()
            rep_path = out_dir / f"{tag}.json"
            pq_path = out_dir / f"{tag}_per_query.jsonl"
            _write_json(rep_path, rep)
            write_per_example_jsonl(pq_path, pq_rows)
            _log_mlflow_run(
                tracking_uri, tag, {"exp": tag, "model": model_id, "mrl_dim": mrl_d or 1024}, rep, [rep_path]
            )
            summary["experiments"][tag] = {
                "macro": rep["macro"],
                "worst_direction": rep["worst_direction"],
                "worst_direction_gap": rep["worst_direction_gap"],
                "texts_per_s": enc_docs_per_s,
            }
            if mrl_d is None:
                dense_runs[exp_id] = rk
                per_query_cache[tag] = pq_rows
            log.info("%s: macro nDCG@10=%.4f (%.1f texts/s)", tag, rep["macro"]["ndcg@10"], enc_docs_per_s)

        # BGE-M3 sparse-only leg
        if exp_id == "RET-2_bge_m3" and d_sparse is not None and q_sparse is not None:
            t_sp = time.perf_counter()
            sp_res = sparse_dot_search(q_sparse, d_sparse, doc_ids, top_k=100)
            bge_sparse_rk = dict(zip(q_ids, sp_res, strict=True))
            rep_sp = evaluate_rankings(
                queries,
                qrels,
                bge_sparse_rk,
                first_stage_rankings=bge_sparse_rk,
                n_bootstrap=n_boot,
                seed=seed,
            )
            pq_sp = rep_sp.pop("per_query")
            rep_sp["seconds"] = round(time.perf_counter() - t_sp, 2)
            _write_json(out_dir / "RET-2_bge_m3_sparse.json", rep_sp)
            write_per_example_jsonl(out_dir / "RET-2_bge_m3_sparse_per_query.jsonl", pq_sp)
            _log_mlflow_run(tracking_uri, "RET-2_bge_m3_sparse", {"exp": "RET-2_bge_m3_sparse"}, rep_sp)
            summary["experiments"]["RET-2_bge_m3_sparse"] = {
                "macro": rep_sp["macro"],
                "worst_direction": rep_sp["worst_direction"],
                "worst_direction_gap": rep_sp["worst_direction_gap"],
            }
            dense_runs["bge_m3_sparse"] = bge_sparse_rk
            log.info("RET-2_bge_m3_sparse: macro nDCG@10=%.4f", rep_sp["macro"]["ndcg@10"])

        del emb, d_vecs, q_vecs
        _clear_gpu()

    # --- RET-4: Hybrid Fusion (RRF k=60, RRF k=20, DBSF, Weighted, Direction-Aware Weighted) ---
    bge_dense = dense_runs["RET-2_bge_m3"]
    bge_sparse = dense_runs.get("bge_m3_sparse", bm25_rankings)

    fusion_configs = [
        ("RET-4_bge_bm25_rrf60", bge_dense, bm25_rankings, "rrf", 60, {"dense": 1.0, "sparse": 1.0}, False),
        ("RET-4_bge_bm25_rrf20", bge_dense, bm25_rankings, "rrf", 20, {"dense": 1.0, "sparse": 1.0}, False),
        ("RET-4_bge_bm25_dbsf", bge_dense, bm25_rankings, "dbsf", 60, {"dense": 1.0, "sparse": 1.0}, False),
        (
            "RET-4_bge_bm25_weighted_dir",
            bge_dense,
            bm25_rankings,
            "weighted",
            60,
            {"dense": 0.75, "sparse": 0.25},
            True,
        ),
        (
            "RET-4_bge_dense_sparse_rrf60",
            bge_dense,
            bge_sparse,
            "rrf",
            60,
            {"dense": 1.0, "sparse": 1.0},
            False,
        ),
        (
            "RET-4_bge_dense_sparse_w_dir",
            bge_dense,
            bge_sparse,
            "weighted",
            60,
            {"dense": 0.75, "sparse": 0.25},
            True,
        ),
    ]

    best_ret4_name = ""
    best_ret4_ndcg = -1.0
    best_ret4_rk: dict[str, list[tuple[str, float]]] = {}

    for fname, d_rk, s_rk, method, rrf_k, weights, dir_aware in fusion_configs:
        fused_rk: dict[str, list[tuple[str, float]]] = {}
        for qid in q_ids:
            ql, dl = q_dirs[qid].split("-")
            w = dict(weights)
            if dir_aware and ql != dl:
                # Sparse lexical leg is weaker cross-lingually -> downweight sparse for cross-lingual pairs
                w = {"dense": 0.90, "sparse": 0.10}
            f = fuse_id_rankings(
                {"dense": d_rk.get(qid, []), "sparse": s_rk.get(qid, [])},
                method=method,
                rrf_k=rrf_k,
                weights=w,
                top_k=100,
            )
            fused_rk[qid] = [(doc_id, sc) for doc_id, sc, _ in f]
        rep = evaluate_rankings(
            queries, qrels, fused_rk, first_stage_rankings=fused_rk, n_bootstrap=n_boot, seed=seed
        )
        pq_rows = rep.pop("per_query")
        rep["comparison_vs_bge_dense"] = compare_retrieval_runs(
            per_query_cache["RET-2_bge_m3_dense"], pq_rows, n_bootstrap=n_boot, seed=seed
        )
        rep_path = out_dir / f"{fname}.json"
        _write_json(rep_path, rep)
        write_per_example_jsonl(out_dir / f"{fname}_per_query.jsonl", pq_rows)
        _log_mlflow_run(
            tracking_uri,
            fname,
            {"exp": fname, "method": method, "rrf_k": rrf_k, "dir_aware": dir_aware},
            rep,
            [rep_path],
        )
        summary["experiments"][fname] = {
            "macro": rep["macro"],
            "worst_direction": rep["worst_direction"],
            "worst_direction_gap": rep["worst_direction_gap"],
            "vs_bge_dense_diff": rep["comparison_vs_bge_dense"]["macro"]["diff"],
            "vs_bge_dense_sig": rep["comparison_vs_bge_dense"]["macro"]["significant"],
        }
        if rep["macro"]["ndcg@10"] > best_ret4_ndcg:
            best_ret4_ndcg = rep["macro"]["ndcg@10"]
            best_ret4_name = fname
            best_ret4_rk = fused_rk
        log.info(
            "%s: macro nDCG@10=%.4f (diff vs dense=%+.4f)",
            fname,
            rep["macro"]["ndcg@10"],
            rep["comparison_vs_bge_dense"]["macro"]["diff"],
        )

    # Save best RET-4 top-50 candidates for Phase 2 (RET-5 & RET-7)
    cand_dump = {qid: best_ret4_rk[qid][:50] for qid in q_ids}
    _write_json(out_dir / "best_ret4_candidates.json", {"name": best_ret4_name, "candidates": cand_dump})

    # Stage-attributed miss analysis on best RET-4
    miss_report = attribute_retrieval_misses(
        queries,
        qrels,
        doc_id_set,
        dense_top100={q: [d for d, _ in bge_dense.get(q, [])] for q in q_ids},
        sparse_top100={q: [d for d, _ in bge_sparse.get(q, [])] for q in q_ids},
        fused_top50={q: [d for d, _ in best_ret4_rk.get(q, [])[:50]] for q in q_ids},
        final_top10={q: [d for d, _ in best_ret4_rk.get(q, [])[:10]] for q in q_ids},
    )
    _write_json(out_dir / "miss_attribution_ret4.json", miss_report)
    summary["best_ret4"] = best_ret4_name
    summary["miss_attribution_ret4"] = {
        "miss_rate": miss_report["miss_rate"],
        "overall_shares": miss_report["overall_shares"],
    }
    _write_json(out_dir / "phase1_summary.json", summary)

    if drive_dir is not None:
        drive_dir.mkdir(parents=True, exist_ok=True)
        for f in out_dir.glob("*.json"):
            shutil.copy2(f, drive_dir / f.name)
    return summary


# ---------------------------------------------------------------- Phase 2: RET-5, RET-6, RET-7, RET-C


def run_phase2(
    out_dir: Path, eval_dir: Path, retrieval_dir: Path, tracking_uri: str, drive_dir: Path | None
) -> dict[str, Any]:
    """Execute RET-5 (rerankers + top-N curve), RET-6 (HPO + ablation + 3-seed embedder),
    RET-7 (3-seed reranker + Platt calibration), and RET-C (chunking variants x headers)."""
    params = load_params()
    n_boot = int(params["evaluation"]["bootstrap_samples"])
    seed = int(params["seed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    queries_full, qrels_full, docs, _ = load_frozen_retrieval_eval(eval_dir, verify_frozen=True, seed=seed)
    doc_ids = docs["doc_id"].to_list()
    doc_texts = [
        f"{(t or '').strip()}\n{(x or '').strip()}".strip()
        for t, x in zip(docs["title"].to_list(), docs["text"].to_list(), strict=True)
    ]
    doc_text_map = dict(zip(doc_ids, doc_texts, strict=True))

    # Stratified 200 queries/direction = 1,800 queries for fast reranker top-N & challenger comparison
    queries_sub, qrels_sub, _, _ = load_frozen_retrieval_eval(
        eval_dir, verify_frozen=False, max_per_direction=200, seed=seed
    )
    sub_qids = queries_sub["query_id"].to_list()
    sub_qtexts = dict(zip(sub_qids, queries_sub["query"].to_list(), strict=True))

    cand_file = out_dir / "best_ret4_candidates.json"
    if not cand_file.is_file():
        raise FileNotFoundError(f"Run phase1 first: {cand_file} missing")
    cand_data = json.loads(cand_file.read_text(encoding="utf-8"))
    ret4_cands: dict[str, list[list[Any]]] = cand_data["candidates"]

    # Baseline RET-4 on 1,800-query slice
    ret4_sub_rk = {q: [(str(d), float(s)) for d, s in ret4_cands.get(q, [])] for q in sub_qids}
    ret4_sub_ev = evaluate_rankings(
        queries_sub, qrels_sub, ret4_sub_rk, first_stage_rankings=ret4_sub_rk, n_bootstrap=n_boot, seed=seed
    )
    ret4_sub_pq = ret4_sub_ev.pop("per_query")

    summary: dict[str, Any] = {
        "phase": "phase2_ret5_ret6_ret7_retc",
        "git_sha": git_sha(ROOT),
        "gpu": _gpu_info(),
        "ret5": {},
        "ret6": {},
        "ret7": {},
        "ret_c": {},
    }

    # --- RET-5: Rerankers (bge-reranker-v2-m3 vs Qwen3-Reranker-0.6B) + top-N curve ---
    for rr_tag, rr_model, bs in (
        ("RET-5_bge_reranker_v2_m3", "BAAI/bge-reranker-v2-m3", 64),
        ("RET-5_qwen3_reranker_0.6b", "Qwen/Qwen3-Reranker-0.6B", 32),
    ):
        rr = CrossEncoderReranker(rr_model, batch_size=bs, max_length=384, use_fp16=True)
        # Score top-50 once, then evaluate top-N in {10, 20, 30, 50} + measure latency
        all_pairs: list[tuple[str, str]] = []
        offsets: list[tuple[str, int, int]] = []
        for qid in sub_qids:
            cs = ret4_sub_rk[qid][:50]
            s_idx = len(all_pairs)
            for d, _ in cs:
                all_pairs.append((sub_qtexts[qid], doc_text_map.get(d, "")))
            offsets.append((qid, s_idx, len(all_pairs)))

        t0 = time.perf_counter()
        scores = rr.score_pairs(all_pairs)
        total_s = time.perf_counter() - t0
        ms_per_pair = (total_s * 1000.0) / max(len(all_pairs), 1)

        curve: dict[str, Any] = {}
        for top_n in (10, 20, 30, 50):
            rk_n: dict[str, list[tuple[str, float]]] = {}
            for qid, s_idx, e_idx in offsets:
                cs = ret4_sub_rk[qid][:top_n]
                sc_slice = scores[s_idx : s_idx + len(cs)]
                rescored = sorted(
                    zip((d for d, _ in cs), sc_slice, strict=True), key=lambda kv: (-kv[1], kv[0])
                )
                rk_n[qid] = rescored
            ev_n = evaluate_rankings(
                queries_sub, qrels_sub, rk_n, first_stage_rankings=ret4_sub_rk, n_bootstrap=n_boot, seed=seed
            )
            pq_n = ev_n.pop("per_query")
            cmp_n = compare_retrieval_runs(ret4_sub_pq, pq_n, n_bootstrap=n_boot, seed=seed)
            gate_n = evaluate_retrieval_gate(
                cmp_n,
                ev_n,
                params["promotion"],
                candidate_name=f"{rr_tag}_top{top_n}",
                baseline_name="RET-4",
                component="reranker",
            )
            curve[f"top{top_n}"] = {
                "macro": ev_n["macro"],
                "worst_direction": ev_n["worst_direction"],
                "worst_direction_gap": ev_n["worst_direction_gap"],
                "est_latency_ms_per_query": round(ms_per_pair * top_n, 2),
                "vs_ret4_diff": cmp_n["macro"]["diff"],
                "vs_ret4_sig": cmp_n["macro"]["significant"],
                "gate_passed": gate_n["passed"],
            }
            if top_n == 30:
                _write_json(
                    out_dir / f"{rr_tag}_top30.json",
                    {**ev_n, "comparison_vs_baseline": cmp_n, "gate": gate_n},
                )
                write_per_example_jsonl(out_dir / f"{rr_tag}_top30_per_query.jsonl", pq_n)
                _log_mlflow_run(tracking_uri, f"{rr_tag}_top30", {"model": rr_model, "top_n": 30}, ev_n)
                if rr_tag == "RET-5_bge_reranker_v2_m3":
                    pass

        summary["ret5"][rr_tag] = {
            "model": rr_model,
            "total_pairs": len(all_pairs),
            "total_seconds": round(total_s, 2),
            "ms_per_pair": round(ms_per_pair, 3),
            "curve": curve,
        }
        del rr
        _clear_gpu()
        log.info(
            "%s complete: top30 nDCG@10=%.4f (%.2f ms/pair)",
            rr_tag,
            curve["top30"]["macro"]["ndcg@10"],
            ms_per_pair,
        )

    # --- RET-6: Embedder Fine-Tuning (Sanity Check + Optuna HPO + KazParC Ablation + 3 Seeds) ---
    base_emb_model = params["retrieval"]["embedder"]
    train_pairs_all, pmap_tr, _tr_stats = build_balanced_pairs(
        retrieval_dir, split="train", max_per_direction=1500, kazparc_max_share=0.20, seed=seed
    )
    val_pairs, pmap_val, _ = build_balanced_pairs(
        retrieval_dir, split="validation", max_per_direction=150, kazparc_max_share=0.20, seed=seed
    )
    pmap_tr.update(pmap_val)

    # 1. Mandatory 64-example overfit sanity check
    sanity = train_biencoder(
        base_emb_model,
        train_pairs_all[:64],
        pmap_tr,
        out_dir / "_sanity_ret6",
        lr=3e-5,
        batch_size=16,
        max_length=192,
        sanity_check_steps=20,
        seed=seed,
    )
    summary["ret6"]["sanity_check"] = sanity
    _clear_gpu()

    # 2. Positive-aware hard-negative mining on pairs lacking negatives
    base_emb = MultilingualEmbedder(
        base_emb_model, max_length=256, batch_size=64, use_fp16=True, use_bge_sparse=False
    )
    mining_stats = mine_positive_aware_negatives(
        train_pairs_all[:4000], pmap_tr, base_emb, top_k=20, margin_ratio=0.95, max_negatives=2
    )
    summary["ret6"]["hard_negative_mining"] = mining_stats
    del base_emb
    _clear_gpu()

    # 3. Optuna HPO (4 trials on validation: lr & kazparc_cap)
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    optuna_db = f"sqlite:///{(out_dir / 'optuna_ret6.db').as_posix()}"
    study = optuna.create_study(
        study_name="ret6_bge_m3_hpo",
        storage=optuna_db,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    def _objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 5e-6, 3e-5, log=True)
        temp = trial.suggest_float("temperature", 0.015, 0.05)
        kp_cap = trial.suggest_categorical("kazparc_cap", [0.0, 0.20, 0.50])
        tr_p, pm, _ = build_balanced_pairs(
            retrieval_dir, split="train", max_per_direction=400, kazparc_max_share=kp_cap, seed=seed
        )
        pm.update(pmap_val)
        res = train_biencoder(
            base_emb_model,
            tr_p,
            pm,
            out_dir / f"_hpo_trial_{trial.number}",
            val_pairs=val_pairs,
            lr=lr,
            epochs=1,
            batch_size=16,
            grad_accum=2,
            max_length=192,
            temperature=temp,
            seed=seed,
        )
        shutil.rmtree(out_dir / f"_hpo_trial_{trial.number}", ignore_errors=True)
        _clear_gpu()
        return float(res["val_loss"] if res["val_loss"] is not None else 99.0)

    if len(study.trials) < 4:
        study.optimize(_objective, n_trials=4 - len(study.trials))
    best_hp = study.best_params
    summary["ret6"]["hpo"] = {
        "n_trials": len(study.trials),
        "best_val_loss": round(float(study.best_value), 5),
        "best_params": best_hp,
        "trials": [{"number": t.number, "params": t.params, "val_loss": t.value} for t in study.trials],
    }

    # 4. 3-seed RET-6 training & frozen test comparison against zero-shot BGE-M3 dense
    base_dense_pq = [
        json.loads(line)
        for line in (out_dir / "RET-2_bge_m3_dense_per_query.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sub_qid_set = set(sub_qids)
    base_dense_sub_pq = [r for r in base_dense_pq if r["query_id"] in sub_qid_set]

    ret6_seed_pqs: list[list[dict[str, Any]]] = []
    ret6_seed_reports: list[dict[str, Any]] = []
    best_seed_ndcg = -1.0
    best_seed_dir: Path | None = None

    for s in (42, 43, 44):
        s_dir = out_dir / f"ret6_seed_{s}"
        tr_meta = train_biencoder(
            base_emb_model,
            train_pairs_all,
            pmap_tr,
            s_dir,
            val_pairs=val_pairs,
            lr=float(best_hp["lr"]),
            epochs=1,
            batch_size=16,
            grad_accum=2,
            max_length=256,
            temperature=float(best_hp["temperature"]),
            seed=s,
        )
        _clear_gpu()
        ft_emb = MultilingualEmbedder(
            str(s_dir), family="bge_m3", max_length=512, batch_size=64, use_fp16=True, use_bge_sparse=False
        )
        d_v = ft_emb.embed_passages_np(doc_texts)
        q_v = ft_emb.embed_queries_np([sub_qtexts[q] for q in sub_qids])
        rk_s = dict(zip(sub_qids, dense_exact_search(q_v, d_v, doc_ids, top_k=100), strict=True))
        ev_s = evaluate_rankings(
            queries_sub, qrels_sub, rk_s, first_stage_rankings=rk_s, n_bootstrap=n_boot, seed=seed
        )
        pq_s = ev_s.pop("per_query")
        ret6_seed_pqs.append(pq_s)
        ret6_seed_reports.append(
            {
                "seed": s,
                "train": tr_meta,
                "macro": ev_s["macro"],
                "worst_direction_gap": ev_s["worst_direction_gap"],
            }
        )
        if ev_s["macro"]["ndcg@10"] > best_seed_ndcg:
            best_seed_ndcg = ev_s["macro"]["ndcg@10"]
            best_seed_dir = s_dir
        del ft_emb, d_v, q_v
        _clear_gpu()

    ret6_ms_cmp = compare_multi_seed_retrieval(
        [base_dense_sub_pq], ret6_seed_pqs, n_bootstrap=n_boot, seed=seed
    )
    ret6_best_ev = evaluate_rankings(
        queries_sub,
        qrels_sub,
        {r["query_id"]: [(d, 1.0 / (i + 1)) for i, d in enumerate(r["top10_ids"])] for r in ret6_seed_pqs[0]},
        n_bootstrap=200,
        seed=seed,
    )
    ret6_best_ev.pop("per_query")
    ret6_gate = evaluate_retrieval_gate(
        ret6_ms_cmp,
        ret6_best_ev,
        params["promotion"],
        candidate_name="RET-6_ft_bge_m3",
        baseline_name="RET-2_bge_m3_dense",
        component="retrieval",
    )
    summary["ret6"]["seeds"] = ret6_seed_reports
    summary["ret6"]["multi_seed_comparison"] = ret6_ms_cmp
    summary["ret6"]["gate"] = ret6_gate
    _write_json(out_dir / "RET-6_report.json", summary["ret6"])

    # Clean up non-best seed weights to save disk
    for s in (42, 43, 44):
        s_dir = out_dir / f"ret6_seed_{s}"
        if s_dir != best_seed_dir and s_dir.exists():
            shutil.rmtree(s_dir, ignore_errors=True)

    # --- RET-7: Reranker Fine-Tuning (3 seeds + Platt calibration) ---
    base_rr_model = params["retrieval"]["reranker"]
    rr_train, rr_pmap, _ = build_balanced_pairs(
        retrieval_dir, split="train", max_per_direction=400, kazparc_max_share=0.15, seed=seed
    )
    rr_pmap.update(pmap_val)
    base_rr_pq = [
        json.loads(line)
        for line in (out_dir / "RET-5_bge_reranker_v2_m3_top30_per_query.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    ret7_seed_pqs: list[list[dict[str, Any]]] = []
    ret7_seed_reports: list[dict[str, Any]] = []
    for s in (42, 43, 44):
        s_dir = out_dir / f"ret7_seed_{s}"
        tr_meta = train_cross_encoder(
            base_rr_model,
            rr_train,
            rr_pmap,
            s_dir,
            val_pairs=val_pairs[:200],
            lr=1e-5,
            epochs=1,
            batch_size=4,
            grad_accum=4,
            max_length=320,
            num_negatives=2,
            seed=s,
            calibrate=True,
        )
        _clear_gpu()
        ft_rr = CrossEncoderReranker(str(s_dir), batch_size=64, max_length=384, use_fp16=True)
        pairs_eval: list[tuple[str, str]] = []
        offs: list[tuple[str, int, int]] = []
        for qid in sub_qids:
            cs = ret4_sub_rk[qid][:30]
            si = len(pairs_eval)
            for d, _ in cs:
                pairs_eval.append((sub_qtexts[qid], doc_text_map.get(d, "")))
            offs.append((qid, si, len(pairs_eval)))
        sc = ft_rr.score_pairs(pairs_eval)
        rk_s: dict[str, list[tuple[str, float]]] = {}
        for qid, si, ei in offs:
            cs = ret4_sub_rk[qid][:30]
            rk_s[qid] = sorted(
                zip((d for d, _ in cs), sc[si:ei], strict=True), key=lambda kv: (-kv[1], kv[0])
            )
        ev_s = evaluate_rankings(
            queries_sub, qrels_sub, rk_s, first_stage_rankings=ret4_sub_rk, n_bootstrap=n_boot, seed=seed
        )
        pq_s = ev_s.pop("per_query")
        ret7_seed_pqs.append(pq_s)
        ret7_seed_reports.append(
            {
                "seed": s,
                "train": tr_meta,
                "macro": ev_s["macro"],
                "worst_direction_gap": ev_s["worst_direction_gap"],
            }
        )
        del ft_rr
        shutil.rmtree(s_dir, ignore_errors=True)
        _clear_gpu()

    ret7_ms_cmp = compare_multi_seed_retrieval([base_rr_pq], ret7_seed_pqs, n_bootstrap=n_boot, seed=seed)
    ret7_gate = evaluate_retrieval_gate(
        ret7_ms_cmp,
        {"worst_direction_gap": ret7_seed_reports[0]["worst_direction_gap"], "worst_direction": ""},
        params["promotion"],
        candidate_name="RET-7_ft_bge_reranker",
        baseline_name="RET-5_bge_reranker_v2_m3",
        component="reranker",
    )
    summary["ret7"]["seeds"] = ret7_seed_reports
    summary["ret7"]["multi_seed_comparison"] = ret7_ms_cmp
    summary["ret7"]["gate"] = ret7_gate
    _write_json(out_dir / "RET-7_report.json", summary["ret7"])

    # --- RET-C: Chunking variants (s350, s200, s512, flat350) x contextual headers ---
    tok_cfg = params["curation"]["stats"]["tokenizers"]["xlmr"]
    from huggingface_hub import hf_hub_download

    tok_path = hf_hub_download(tok_cfg["hf_id"], tok_cfg["file"], revision=tok_cfg["revision"])
    counter = hf_counter(tok_path)

    # Select long eval docs (> 200 words) and their queries so chunking boundaries actually trigger
    long_docs = docs.filter(pl.col("text").str.len_chars() >= 800)
    long_doc_ids = set(long_docs["doc_id"].to_list())
    qrels_long = qrels_full.filter(pl.col("doc_id").is_in(list(long_doc_ids)) & (pl.col("relevance") > 0))
    long_qids = set(qrels_long["query_id"].to_list())
    queries_retc = queries_full.filter(pl.col("query_id").is_in(list(long_qids)))
    # Cap to 150/dir for fast RET-C sweep
    retc_parts = [
        g.sample(n=min(150, g.height), seed=seed)
        for _, g in queries_retc.group_by("direction", maintain_order=True)
    ]
    queries_retc = pl.concat(retc_parts) if retc_parts else queries_sub
    qrels_retc = qrels_full.filter(pl.col("query_id").is_in(queries_retc["query_id"].to_list()))

    emb_retc = MultilingualEmbedder(
        base_emb_model, max_length=512, batch_size=64, use_fp16=True, use_bge_sparse=False
    )
    q_retc_ids = queries_retc["query_id"].to_list()
    q_retc_vecs = emb_retc.embed_queries_np(queries_retc["query"].to_list())

    retc_results: dict[str, Any] = {}
    for vname, vcfg in params["chunking"]["variants"].items():
        for use_header in (True, False):
            if vname != "s350" and not use_header:
                continue  # Full header ablation on s350 + header=True across all 4 variants
            tag = f"{vname}_{'header' if use_header else 'noheader'}"
            chunk_texts: list[str] = []
            chunk_to_doc: list[str] = []
            for r in long_docs.iter_rows(named=True):
                did = str(r["doc_id"])
                title = str(r["title"] or "").strip()
                text = str(r["text"] or "")
                spans = chunk_text(
                    text,
                    counter,
                    max_tokens=int(vcfg["max_tokens"]),
                    overlap_tokens=int(vcfg["overlap_tokens"]),
                )
                for sp in spans:
                    piece = text[sp.start : sp.end]
                    chunk_texts.append(f"{title}\n{piece}".strip() if (use_header and title) else piece)
                    chunk_to_doc.append(did)
            c_vecs = emb_retc.embed_passages_np(chunk_texts)
            # Max-passage score aggregation back to doc_id so relevance mapping is 100% comparable across variants
            raw_res = dense_exact_search(
                q_retc_vecs, c_vecs, [str(i) for i in range(len(chunk_texts))], top_k=100
            )
            doc_rk: dict[str, list[tuple[str, float]]] = {}
            for qid, hits in zip(q_retc_ids, raw_res, strict=True):
                best_by_doc: dict[str, float] = {}
                for cid_str, sc in hits:
                    did = chunk_to_doc[int(cid_str)]
                    if sc > best_by_doc.get(did, -1e9):
                        best_by_doc[did] = sc
                doc_rk[qid] = sorted(best_by_doc.items(), key=lambda kv: (-kv[1], kv[0]))[:100]
            ev_c = evaluate_rankings(
                queries_retc, qrels_retc, doc_rk, first_stage_rankings=doc_rk, n_bootstrap=500, seed=seed
            )
            ev_c.pop("per_query")
            retc_results[tag] = {
                "variant": vname,
                "max_tokens": vcfg["max_tokens"],
                "overlap_tokens": vcfg["overlap_tokens"],
                "contextual_header": use_header,
                "n_docs": long_docs.height,
                "n_chunks": len(chunk_texts),
                "chunks_per_doc": round(len(chunk_texts) / max(long_docs.height, 1), 3),
                "macro": ev_c["macro"],
                "worst_direction_gap": ev_c["worst_direction_gap"],
            }
            log.info(
                "RET-C %s: chunks=%d (%.2f/doc), macro nDCG@10=%.4f",
                tag,
                len(chunk_texts),
                retc_results[tag]["chunks_per_doc"],
                ev_c["macro"]["ndcg@10"],
            )

    summary["ret_c"] = retc_results
    _write_json(out_dir / "RET-C_report.json", retc_results)
    _write_json(out_dir / "phase2_summary.json", summary)

    if drive_dir is not None:
        drive_dir.mkdir(parents=True, exist_ok=True)
        for f in out_dir.glob("*.json"):
            shutil.copy2(f, drive_dir / f.name)
    return summary


# ---------------------------------------------------------------- Phase 3: RET-S + GPU Annotate


def run_phase3(
    out_dir: Path,
    qdrant_url: str | None,
    qdrant_path: str | None,
    scale_n: int,
    drive_dir: Path | None,
) -> dict[str, Any]:
    """Execute RET-S (Qdrant scale & quantization vs exact benchmark) and produce 35M capacity analysis."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    # Load real BGE-M3 vectors from phase1 if present and tile with small unit-norm perturbations to reach scale_n
    real_doc_path = out_dir / "bge_m3_doc_vecs_f16.npy"
    real_q_path = out_dir / "bge_m3_q_vecs_f16.npy"
    if real_doc_path.is_file() and real_q_path.is_file():
        base_vecs = np.load(real_doc_path).astype(np.float32)
        q_vecs_all = np.load(real_q_path).astype(np.float32)[:200]
        dim = int(base_vecs.shape[1])
        source_kind = "real_bge_m3_augmented"
    else:
        dim = 1024
        base_vecs = rng.standard_normal((10_000, dim), dtype=np.float32)
        base_vecs /= np.linalg.norm(base_vecs, axis=1, keepdims=True)
        q_vecs_all = rng.standard_normal((200, dim), dtype=np.float32)
        q_vecs_all /= np.linalg.norm(q_vecs_all, axis=1, keepdims=True)
        source_kind = "synthetic_unit_gaussian"

    store = QdrantStore(
        url=qdrant_url, path=qdrant_path, in_memory=(qdrant_url is None and qdrant_path is None)
    )
    from qdrant_client import models

    grid_results: list[dict[str, Any]] = []
    # Quantization x oversampling x MRL grid
    configs = [
        ("none", 1024, False, 1.0, False),
        ("scalar", 1024, True, 2.0, False),
        ("binary", 1024, False, 1.0, False),
        ("binary", 1024, True, 2.0, False),
        ("binary", 1024, True, 3.0, False),
        ("turbo", 1024, True, 3.0, False),
        ("binary", 512, True, 3.0, False),
    ]

    for quant, d_dim, rescore, oversamp, hnsw_disk in configs:
        col = f"rets_{scale_n}_{quant}_d{d_dim}_r{int(rescore)}_o{int(oversamp)}"
        cfg = CorpusIndexConfig(
            dim=d_dim,
            datatype="float16",
            vectors_on_disk=True,
            quantization=quant,
            rescore=rescore,
            oversampling=oversamp,
            hnsw_m=16,
            hnsw_ef_construct=128,
            hnsw_on_disk=hnsw_disk,
        )
        t0 = time.perf_counter()
        store.create_corpus_collection(col, cfg, disable_hnsw_for_bulk=True, recreate=True)

        # Stream upload in 5,000-vector batches to keep driver RAM flat
        batch = 5_000
        n_base = base_vecs.shape[0]
        for start in range(0, scale_n, batch):
            end = min(start + batch, scale_n)
            cnt = end - start
            idx = np.arange(start, end) % n_base
            sub = base_vecs[idx, :d_dim].copy()
            if start >= n_base:
                noise = rng.normal(0.0, 0.05, size=sub.shape).astype(np.float32)
                sub = sub + noise
            norms = np.linalg.norm(sub, axis=1, keepdims=True)
            sub /= np.where(norms > 1e-12, norms, 1.0)

            pts = [
                models.PointStruct(
                    id=int(start + i), vector={"dense": sub[i].tolist()}, payload={"lang": "kk"}
                )
                for i in range(cnt)
            ]
            store.client.upsert(collection_name=col, points=pts, wait=True)
        upload_s = time.perf_counter() - t0

        t_hnsw = time.perf_counter()
        store.enable_hnsw(col, cfg)
        hnsw_s = time.perf_counter() - t_hnsw

        # Query subsample: exact vs ANN
        qv = apply_mrl(q_vecs_all, d_dim if d_dim < dim else None)
        from trifact.batch.index_corpus import benchmark_ann_vs_exact

        bench = benchmark_ann_vs_exact(
            store,
            col,
            qv[:100],
            top_k=10,
            candidate_k=100,
            rescore=rescore,
            oversampling=oversamp,
            hnsw_ef=128,
        )
        rss_mb = round(psutil.Process().memory_info().rss / (1024**2), 1)
        row = {
            "scale_n": scale_n,
            "vector_source": source_kind,
            "quantization": quant,
            "dim": d_dim,
            "rescore": rescore,
            "oversampling": oversamp,
            "hnsw_on_disk": hnsw_disk,
            "upload_seconds": round(upload_s, 2),
            "hnsw_seconds": round(hnsw_s, 2),
            "points_per_s": round(scale_n / max(upload_s, 1e-6), 1),
            "driver_rss_mb": rss_mb,
            "analytical_35m": estimate_vector_bytes(35_000_000, d_dim, quant),
            **bench,
        }
        grid_results.append(row)
        log.info(
            "RET-S N=%d %s d=%d rescore=%s os=%.1f: ANN vs exact R@10=%.4f R@100=%.4f p95=%.2fms",
            scale_n,
            quant,
            d_dim,
            rescore,
            oversamp,
            bench["ann_vs_exact_recall@10"],
            bench["ann_vs_exact_recall@100"],
            bench["ann_p95_ms"],
        )
        store.client.delete_collection(col)

    report = {
        "phase": "phase3_ret_s",
        "git_sha": git_sha(ROOT),
        "measured_scale_n": scale_n,
        "unexecuted_scales_note": (
            "10M and 35M full Qdrant server builds exceed free 12.7 GB RAM / 100 GB disk cloud VMs; "
            "35M figures are analytical projections anchored to measured build/query curves."
        ),
        "grid": grid_results,
        "analytical_scale_ladder_1024d": {
            "1M": {
                q: estimate_vector_bytes(1_000_000, 1024, q) for q in ("none", "scalar", "binary", "turbo")
            },
            "10M": {
                q: estimate_vector_bytes(10_000_000, 1024, q) for q in ("none", "scalar", "binary", "turbo")
            },
            "35M": {
                q: estimate_vector_bytes(35_000_000, 1024, q) for q in ("none", "scalar", "binary", "turbo")
            },
        },
        "serving_vm_decision_adr0013": {
            "local_dev_15gb_ram": "Subset corpus (up to ~1-2M chunks, binary+rescore) + full workspace_chunks",
            "production_35m_corpus": (
                "Dedicated Serving VM required (32-64 GB RAM, NVMe SSD >= 250 GB, 16-24 GB GPU): "
                "35M x 1024-d fp16 on-disk = 66.8 GB + payload ~50 GB SSD; binary quant in RAM = 4.2 GB "
                "+ HNSW level-0 (m=16) ~4.5 GB + sparse IDF ~6-8 GB + OS page cache for fp16 rescoring."
            ),
        },
    }
    _write_json(out_dir / "RET-S_report.json", report)
    if drive_dir is not None:
        drive_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_dir / "RET-S_report.json", drive_dir / "RET-S_report.json")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["phase1", "phase2", "phase3"])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "stage6")
    ap.add_argument("--eval-dir", type=Path, default=ROOT / "data" / "eval" / "retrieval")
    ap.add_argument("--retrieval-dir", type=Path, default=ROOT / "data" / "task" / "retrieval")
    ap.add_argument("--mlflow-db", type=Path, default=Path("/content/trifact_stage6.db"))
    ap.add_argument("--drive-dir", type=Path, default=None)
    ap.add_argument("--qdrant-url", default=None)
    ap.add_argument("--qdrant-path", default=None)
    ap.add_argument("--scale-n", type=int, default=100_000)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.environ["TRIFACT_CLOUD_RUNTIME"] = "1"
    args.mlflow_db.parent.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{args.mlflow_db.as_posix()}"

    if args.phase == "phase1":
        run_phase1(args.out_dir, args.eval_dir, tracking_uri, args.drive_dir)
    elif args.phase == "phase2":
        run_phase2(args.out_dir, args.eval_dir, args.retrieval_dir, tracking_uri, args.drive_dir)
    else:
        run_phase3(args.out_dir, args.qdrant_url, args.qdrant_path, args.scale_n, args.drive_dir)

    _snapshot_mlflow(args.mlflow_db, args.drive_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
