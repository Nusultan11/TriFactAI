"""Multilingual retrieval evaluation across all 9 KK/RU/EN directions (ADR-0022, ROADMAP 6.1–6.6).

Computes exact IR metrics (verified against ``ranx``):
- ``recall@5``, ``recall@10``, ``ndcg@10``, ``mrr``
- First-stage ``candidate_recall@50`` and ``candidate_recall@100`` (reranker ceiling)
- ``judged@10`` (closed-pool bias monitor)
- Per-direction, macro-9-direction, ``worst_direction``, ``worst_direction_gap``
- Slices by ``verification_tier`` (``human_verified`` vs ``machine_cross_validated``),
  ``source`` (``kazqad``, ``miracl``), and Wikipedia vs non-Wikipedia contamination check
- Query-cluster paired bootstrap & multi-seed bootstrap via ``trifact.evaluation.stats``
- Stage-attributed miss analysis (7 failure stages)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from trifact.core.params import ROOT, load_params
from trifact.evaluation.stats import (
    cluster_bootstrap_ci,
    cluster_paired_bootstrap,
    multi_seed_cluster_bootstrap,
    write_per_example_jsonl,
)

log = logging.getLogger("trifact.evaluation.eval_retrieval")

ALL_DIRECTIONS = (
    "kk-kk",
    "kk-ru",
    "kk-en",
    "ru-kk",
    "ru-ru",
    "ru-en",
    "en-kk",
    "en-ru",
    "en-en",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen_retrieval_eval(eval_dir: Path) -> dict[str, Any]:
    """Verify SHA-256 hashes of frozen ``data/eval/retrieval`` shards against ``frozen.json``."""
    frozen_path = eval_dir / "frozen.json"
    if not frozen_path.is_file():
        return {"verified": False, "reason": "frozen.json missing"}
    meta = json.loads(frozen_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for rel, expected in (meta.get("shards") or {}).items():
        p = eval_dir / rel
        if not p.is_file():
            mismatches.append(f"missing:{rel}")
            continue
        actual = _sha256_file(p)
        if actual != expected:
            mismatches.append(f"sha256_mismatch:{rel}")
    return {
        "verified": len(mismatches) == 0,
        "fingerprint": meta.get("fingerprint"),
        "status": meta.get("status"),
        "mismatches": mismatches,
    }


def load_frozen_retrieval_eval(
    eval_dir: Path,
    *,
    verify_frozen: bool = True,
    max_per_direction: int | None = None,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Load ``queries``, ``qrels``, and ``docs`` from ``data/eval/retrieval``."""
    check = verify_frozen_retrieval_eval(eval_dir) if verify_frozen else {"verified": False, "skipped": True}
    if verify_frozen and not check["verified"]:
        raise RuntimeError(f"Frozen retrieval eval integrity check failed: {check}")

    queries = pl.read_parquet(str(eval_dir / "*/queries-*.parquet"))
    qrels = pl.read_parquet(str(eval_dir / "multi/qrels-*.parquet"))
    docs = pl.read_parquet(str(eval_dir / "multi/docs-*.parquet"))

    if max_per_direction is not None and max_per_direction > 0:
        parts = []
        for (d,), grp in queries.group_by("direction", maintain_order=True):
            if grp.height > max_per_direction:
                parts.append(grp.sample(n=max_per_direction, seed=seed))
            else:
                parts.append(grp)
        queries = pl.concat(parts)
        valid_qids = set(queries["query_id"].to_list())
        qrels = qrels.filter(pl.col("query_id").is_in(list(valid_qids)))

    return queries, qrels, docs, check


def build_qrels_map(qrels: pl.DataFrame) -> tuple[dict[str, dict[str, int]], dict[str, set[str]]]:
    """Return ``(positive_qrels[qid][doc_id]=rel, all_judged[qid]=set(doc_ids))``."""
    pos: dict[str, dict[str, int]] = defaultdict(dict)
    judged: dict[str, set[str]] = defaultdict(set)
    for qid, did, rel in qrels.select("query_id", "doc_id", "relevance").iter_rows():
        q_str, d_str, r_int = str(qid), str(did), int(rel)
        judged[q_str].add(d_str)
        if r_int > 0:
            pos[q_str][d_str] = r_int
    return dict(pos), dict(judged)


# ---------------------------------------------------------------- per-query IR metrics


def _dcg_at_k(relevances: Sequence[int], k: int) -> float:
    total = 0.0
    for rank_1, rel in enumerate(relevances[:k], start=1):
        if rel > 0:
            total += (2**rel - 1) / math.log2(rank_1 + 1)
    return total


def score_single_query(
    ranked_doc_ids: Sequence[str],
    pos_qrels: Mapping[str, int],
    judged_docs: set[str] | None = None,
    *,
    first_stage_doc_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    """Compute exact per-query IR metrics (matches ``ranx`` binary/graded definitions)."""
    n_pos = len(pos_qrels)
    if n_pos == 0:
        return {
            "recall@5": 0.0,
            "recall@10": 0.0,
            "ndcg@10": 0.0,
            "mrr": 0.0,
            "candidate_recall@50": 0.0,
            "candidate_recall@100": 0.0,
            "judged@10": 0.0,
        }

    # Deduplicate ranked list preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for d in ranked_doc_ids:
        if d not in seen:
            seen.add(d)
            uniq.append(d)

    hits5 = sum(1 for d in uniq[:5] if d in pos_qrels)
    hits10 = sum(1 for d in uniq[:10] if d in pos_qrels)
    recall5 = hits5 / n_pos
    recall10 = hits10 / n_pos

    rels10 = [pos_qrels.get(d, 0) for d in uniq[:10]]
    ideal_rels = sorted(pos_qrels.values(), reverse=True)
    idcg10 = _dcg_at_k(ideal_rels, 10)
    ndcg10 = (_dcg_at_k(rels10, 10) / idcg10) if idcg10 > 0 else 0.0

    mrr = 0.0
    for rank_1, d in enumerate(uniq, start=1):
        if d in pos_qrels:
            mrr = 1.0 / rank_1
            break

    fs = list(first_stage_doc_ids) if first_stage_doc_ids is not None else uniq
    fs_seen: set[str] = set()
    fs_uniq: list[str] = []
    for d in fs:
        if d not in fs_seen:
            fs_seen.add(d)
            fs_uniq.append(d)
    cand50 = sum(1 for d in fs_uniq[:50] if d in pos_qrels) / n_pos
    cand100 = sum(1 for d in fs_uniq[:100] if d in pos_qrels) / n_pos

    j_set = judged_docs if judged_docs is not None else set(pos_qrels.keys())
    top10 = uniq[:10]
    judged10 = (sum(1 for d in top10 if d in j_set) / len(top10)) if top10 else 0.0

    return {
        "recall@5": float(recall5),
        "recall@10": float(recall10),
        "ndcg@10": float(ndcg10),
        "mrr": float(mrr),
        "candidate_recall@50": float(cand50),
        "candidate_recall@100": float(cand100),
        "judged@10": float(judged10),
    }


def verify_ranx_parity(
    run_scores: Mapping[str, Mapping[str, float]],
    pos_qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """Cross-check our per-query IR metrics against ``ranx`` on identical run & qrels."""
    import os
    import tempfile

    os.environ.setdefault("IR_DATASETS_HOME", str(Path(tempfile.gettempdir()) / "trifact_ir_datasets"))
    from ranx import Qrels, Run, evaluate

    valid_qids = [q for q in pos_qrels if pos_qrels[q]]
    q_dict = {q: {d: int(r) for d, r in pos_qrels[q].items()} for q in valid_qids}
    r_dict = {q: {d: float(s) for d, s in run_scores.get(q, {}).items()} for q in valid_qids}
    rx = evaluate(
        Qrels(q_dict), Run(r_dict), ["ndcg@10", "recall@5", "recall@10", "mrr"], make_comparable=True
    )

    ours_list = []
    for q in valid_qids:
        ranked = [d for d, _ in sorted(r_dict[q].items(), key=lambda kv: (-kv[1], kv[0]))]
        ours_list.append(score_single_query(ranked, q_dict[q]))

    parity: dict[str, Any] = {}
    for m in ("ndcg@10", "recall@5", "recall@10", "mrr"):
        our_val = statistics.fmean(x[m] for x in ours_list) if ours_list else 0.0
        rx_val = float(rx[m])
        parity[m] = {
            "ours": round(our_val, 6),
            "ranx": round(rx_val, 6),
            "abs_diff": round(abs(our_val - rx_val), 8),
        }
    parity["max_abs_diff"] = max(v["abs_diff"] for v in parity.values())
    return parity


# ---------------------------------------------------------------- evaluation & slicing


def _mean_metric(rows: Sequence[dict[str, Any]], key: str) -> float:
    return float(statistics.fmean(float(r[key]) for r in rows)) if rows else 0.0


def evaluate_rankings(
    queries_df: pl.DataFrame,
    qrels_df: pl.DataFrame,
    final_rankings: Mapping[str, Sequence[tuple[str, float]]],
    *,
    first_stage_rankings: Mapping[str, Sequence[tuple[str, float]]] | None = None,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate retrieval rankings across all 9 directions, provenance tiers, and sources."""
    pos_map, judged_map = build_qrels_map(qrels_df)

    per_query_rows: list[dict[str, Any]] = []
    for rec in queries_df.iter_rows(named=True):
        qid = str(rec["query_id"])
        pos = pos_map.get(qid, {})
        if not pos:
            continue
        ranked = [d for d, _ in final_rankings.get(qid, ())]
        fs_ranked = (
            [d for d, _ in first_stage_rankings.get(qid, ())] if first_stage_rankings is not None else None
        )
        metrics = score_single_query(ranked, pos, judged_map.get(qid), first_stage_doc_ids=fs_ranked)
        # Pretraining contamination slice: MIRACL is Wikipedia-derived; KazQAD native kk/nq has non-Wikipedia/QA mix
        source = str(rec.get("source", "unknown"))
        origin = str(rec.get("origin", "unknown"))
        is_wiki = source == "miracl"
        per_query_rows.append(
            {
                "query_id": qid,
                "cluster_id": str(rec.get("cluster_id") or qid),
                "direction": str(rec["direction"]),
                "q_lang": str(rec["q_lang"]),
                "d_lang": str(rec["d_lang"]),
                "source": source,
                "origin": origin,
                "verification_tier": str(rec.get("verification_tier", "unknown")),
                "corpus_slice": "wikipedia" if is_wiki else "non_wikipedia_qa",
                "top10_ids": ranked[:10],
                "gold_ids": sorted(pos.keys()),
                **metrics,
            }
        )

    def summarize_group(rows: Sequence[dict[str, Any]], with_ci: bool = False) -> dict[str, Any]:
        if not rows:
            return {"n_queries": 0, "n_clusters": 0}
        out: dict[str, Any] = {
            "n_queries": len(rows),
            "n_clusters": len({r["cluster_id"] for r in rows}),
        }
        for m in (
            "recall@5",
            "recall@10",
            "ndcg@10",
            "mrr",
            "candidate_recall@50",
            "candidate_recall@100",
            "judged@10",
        ):
            out[m] = round(_mean_metric(rows, m), 6)
        if with_ci:
            cids = [r["cluster_id"] for r in rows]
            ci_res = cluster_bootstrap_ci(
                rows, cids, lambda rs: _mean_metric(rs, "ndcg@10"), n=n_bootstrap, ci=ci, seed=seed
            )
            out["ndcg@10_ci"] = [round(ci_res["ci_low"], 6), round(ci_res["ci_high"], 6)]
        return out

    by_direction: dict[str, dict[str, Any]] = {}
    for d in ALL_DIRECTIONS:
        d_rows = [r for r in per_query_rows if r["direction"] == d]
        if d_rows:
            by_direction[d] = summarize_group(d_rows, with_ci=True)

    present_dirs = [d for d in ALL_DIRECTIONS if d in by_direction]
    macro: dict[str, Any] = {}
    for m in (
        "recall@5",
        "recall@10",
        "ndcg@10",
        "mrr",
        "candidate_recall@50",
        "candidate_recall@100",
        "judged@10",
    ):
        macro[m] = (
            round(statistics.fmean(by_direction[d][m] for d in present_dirs), 6) if present_dirs else 0.0
        )

    if present_dirs:
        worst_dir = min(present_dirs, key=lambda d: by_direction[d]["ndcg@10"])
        worst_ndcg = by_direction[worst_dir]["ndcg@10"]
        worst_gap = round(macro["ndcg@10"] - worst_ndcg, 6)
    else:
        worst_dir, worst_ndcg, worst_gap = "", 0.0, 0.0

    by_tier: dict[str, dict[str, Any]] = {}
    for tier in sorted({r["verification_tier"] for r in per_query_rows}):
        by_tier[tier] = summarize_group(
            [r for r in per_query_rows if r["verification_tier"] == tier], with_ci=False
        )

    by_source: dict[str, dict[str, Any]] = {}
    for src in sorted({r["source"] for r in per_query_rows}):
        by_source[src] = summarize_group([r for r in per_query_rows if r["source"] == src], with_ci=False)

    by_contamination_slice: dict[str, dict[str, Any]] = {}
    for sl in sorted({r["corpus_slice"] for r in per_query_rows}):
        by_contamination_slice[sl] = summarize_group(
            [r for r in per_query_rows if r["corpus_slice"] == sl], with_ci=False
        )

    return {
        "n_queries": len(per_query_rows),
        "n_clusters": len({r["cluster_id"] for r in per_query_rows}),
        "n_directions": len(present_dirs),
        "macro": macro,
        "worst_direction": worst_dir,
        "worst_direction_ndcg@10": worst_ndcg,
        "worst_direction_gap": worst_gap,
        "by_direction": by_direction,
        "by_verification_tier": by_tier,
        "by_source": by_source,
        "by_contamination_slice": by_contamination_slice,
        "per_query": per_query_rows,
    }


def compare_retrieval_runs(
    per_query_a: Sequence[dict[str, Any]],
    per_query_b: Sequence[dict[str, Any]],
    *,
    metric: str = "ndcg@10",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired query-cluster bootstrap comparing system B vs system A across 9 directions and macro mean."""
    map_a = {r["query_id"]: r for r in per_query_a}
    map_b = {r["query_id"]: r for r in per_query_b}
    common_qids = sorted(set(map_a) & set(map_b))
    if not common_qids:
        raise ValueError("No common query_ids between system A and system B")

    rows_a = [map_a[q] for q in common_qids]
    rows_b = [map_b[q] for q in common_qids]
    cids = [r["cluster_id"] for r in rows_a]

    def macro_ndcg(rows: Sequence[dict[str, Any]]) -> float:
        by_d: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            by_d[r["direction"]].append(float(r[metric]))
        if not by_d:
            return 0.0
        return float(statistics.fmean(statistics.fmean(vs) for vs in by_d.values()))

    macro_cmp = cluster_paired_bootstrap(rows_a, rows_b, cids, macro_ndcg, n=n_bootstrap, ci=ci, seed=seed)

    by_dir: dict[str, dict[str, Any]] = {}
    for d in ALL_DIRECTIONS:
        idx = [i for i, r in enumerate(rows_a) if r["direction"] == d]
        if not idx:
            continue
        da = [rows_a[i] for i in idx]
        db = [rows_b[i] for i in idx]
        dc = [cids[i] for i in idx]
        by_dir[d] = cluster_paired_bootstrap(
            da, db, dc, lambda rs: _mean_metric(rs, metric), n=n_bootstrap, ci=ci, seed=seed
        )

    return {
        "metric": metric,
        "n_common_queries": len(common_qids),
        "macro": macro_cmp,
        "by_direction": by_dir,
    }


def compare_multi_seed_retrieval(
    seeds_a: Sequence[Sequence[dict[str, Any]]],
    seeds_b: Sequence[Sequence[dict[str, Any]]],
    *,
    metric: str = "ndcg@10",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Joint seed + query-cluster paired bootstrap for >=3-seed model comparisons (ADR-0022)."""
    common_qids = sorted(set.intersection(*({r["query_id"] for r in s} for s in (*seeds_a, *seeds_b))))
    if not common_qids:
        raise ValueError("No common query_ids across seed runs")

    aligned_a = [[{r["query_id"]: r for r in s}[q] for q in common_qids] for s in seeds_a]
    aligned_b = [[{r["query_id"]: r for r in s}[q] for q in common_qids] for s in seeds_b]
    cids = [r["cluster_id"] for r in aligned_a[0]]

    def macro_metric(rows: Sequence[dict[str, Any]]) -> float:
        by_d: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            by_d[r["direction"]].append(float(r[metric]))
        return float(statistics.fmean(statistics.fmean(vs) for vs in by_d.values())) if by_d else 0.0

    macro_cmp = multi_seed_cluster_bootstrap(
        aligned_a, aligned_b, cids, macro_metric, n=n_bootstrap, ci=ci, seed=seed
    )

    by_dir: dict[str, dict[str, Any]] = {}
    for d in ALL_DIRECTIONS:
        idx = [i for i, r in enumerate(aligned_a[0]) if r["direction"] == d]
        if not idx:
            continue
        da = [[s[i] for i in idx] for s in aligned_a]
        db = [[s[i] for i in idx] for s in aligned_b]
        dc = [cids[i] for i in idx]
        by_dir[d] = multi_seed_cluster_bootstrap(
            da, db, dc, lambda rs: _mean_metric(rs, metric), n=n_bootstrap, ci=ci, seed=seed
        )

    return {
        "metric": metric,
        "n_common_queries": len(common_qids),
        "n_seeds_a": len(seeds_a),
        "n_seeds_b": len(seeds_b),
        "macro": macro_cmp,
        "by_direction": by_dir,
    }


# ---------------------------------------------------------------- stage-attributed miss analysis


def attribute_retrieval_misses(
    queries_df: pl.DataFrame,
    qrels_df: pl.DataFrame,
    indexed_doc_ids: set[str],
    *,
    dense_top100: Mapping[str, Sequence[str]],
    sparse_top100: Mapping[str, Sequence[str]],
    fused_top50: Mapping[str, Sequence[str]],
    final_top10: Mapping[str, Sequence[str]],
    exact_dense_top100: Mapping[str, Sequence[str]] | None = None,
    boundary_split_qids: set[str] | None = None,
) -> dict[str, Any]:
    """Attribute every missed query (zero gold hits in final top-10) to its FIRST failing stage.

    Stages (ordered):
    1. ``missing_from_index``
    2. ``chunk_boundary_split``
    3. ``ann_quantization_loss`` (found by exact dense top-100, missed by ANN dense top-100)
    4. ``representation_miss`` (absent from both dense top-100 and sparse top-100)
    5. ``dropped_by_fusion`` (present in dense or sparse top-100, absent from fused top-50)
    6. ``demoted_by_reranker`` (present in fused top-50, absent from final top-10)
    7. ``unjudged_top_results`` (fallback when top-10 has unjudged docs despite gold in pool)
    """
    pos_map, judged_map = build_qrels_map(qrels_df)
    by_direction: dict[str, Counter[str]] = defaultdict(Counter)
    overall: Counter[str] = Counter()
    examples_by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)

    total_queries = 0
    total_misses = 0

    for rec in queries_df.iter_rows(named=True):
        qid = str(rec["query_id"])
        direction = str(rec["direction"])
        pos = set(pos_map.get(qid, {}).keys())
        if not pos:
            continue
        total_queries += 1
        top10 = list(final_top10.get(qid, ()))[:10]
        if pos & set(top10):
            continue  # hit in top-10

        total_misses += 1
        in_index = bool(pos & indexed_doc_ids)
        d100 = set(list(dense_top100.get(qid, ()))[:100])
        s100 = set(list(sparse_top100.get(qid, ()))[:100])
        f50 = set(list(fused_top50.get(qid, ()))[:50])
        ex100 = set(list(exact_dense_top100.get(qid, ()))[:100]) if exact_dense_top100 is not None else set()

        if not in_index:
            stage = "missing_from_index"
        elif boundary_split_qids and qid in boundary_split_qids:
            stage = "chunk_boundary_split"
        elif exact_dense_top100 is not None and (pos & ex100) and not (pos & d100):
            stage = "ann_quantization_loss"
        elif not (pos & (d100 | s100)):
            stage = "representation_miss"
        elif not (pos & f50):
            stage = "dropped_by_fusion"
        elif pos & f50:
            # Check if top-10 is mostly unjudged vs genuinely demoted
            j_set = judged_map.get(qid, set())
            unjudged_cnt = sum(1 for d in top10 if d not in j_set)
            stage = (
                "unjudged_top_results"
                if (unjudged_cnt >= 8 and not (pos & set(top10)))
                else "demoted_by_reranker"
            )
        else:
            stage = "unjudged_top_results"

        overall[stage] += 1
        by_direction[direction][stage] += 1
        if len(examples_by_stage[stage]) < 15:
            examples_by_stage[stage].append(
                {
                    "query_id": qid,
                    "direction": direction,
                    "query": str(rec["query"])[:200],
                    "gold_ids": sorted(pos)[:3],
                    "top3_retrieved": top10[:3],
                }
            )

    return {
        "total_queries": total_queries,
        "total_misses": total_misses,
        "miss_rate": round(total_misses / max(total_queries, 1), 6),
        "overall_counts": dict(overall),
        "overall_shares": {k: round(v / max(total_misses, 1), 6) for k, v in overall.items()},
        "by_direction": {d: dict(cnt) for d, cnt in sorted(by_direction.items())},
        "sample_misses": dict(examples_by_stage),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, default=ROOT / "data" / "eval" / "retrieval")
    ap.add_argument("--mode", choices=["bm25", "bm25_kk_stem", "dense", "hybrid", "rerank"], default="bm25")
    ap.add_argument("--embedder", default=None)
    ap.add_argument("--reranker", default=None)
    ap.add_argument("--fusion", choices=["rrf", "dbsf", "weighted"], default="rrf")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--dense-weight", type=float, default=1.0)
    ap.add_argument("--sparse-weight", type=float, default=1.0)
    ap.add_argument("--mrl-dim", type=int, default=None)
    ap.add_argument("--max-per-direction", type=int, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "retrieval.json")
    ap.add_argument("--per-query", type=Path, default=None)
    ap.add_argument("--baseline-per-query", type=Path, default=None)
    ap.add_argument("--no-verify-frozen", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    params = load_params()
    t0 = time.perf_counter()

    queries, qrels, docs, frozen_check = load_frozen_retrieval_eval(
        args.eval_dir,
        verify_frozen=not args.no_verify_frozen,
        max_per_direction=args.max_per_direction,
        seed=int(params["seed"]),
    )
    doc_ids = docs["doc_id"].to_list()
    doc_texts = [
        f"{(t or '').strip()}\n{(x or '').strip()}".strip()
        for t, x in zip(docs["title"].to_list(), docs["text"].to_list(), strict=True)
    ]
    q_ids = queries["query_id"].to_list()
    q_texts = queries["query"].to_list()

    from trifact.data.bm25 import BM25
    from trifact.retrieval.embedder import MultilingualEmbedder, dense_exact_search, tokenize_sparse
    from trifact.retrieval.hybrid import fuse_id_rankings
    from trifact.retrieval.reranker import CrossEncoderReranker

    dense_rankings: dict[str, list[tuple[str, float]]] = {}
    sparse_rankings: dict[str, list[tuple[str, float]]] = {}
    first_stage: dict[str, list[tuple[str, float]]] = {}
    final_rankings: dict[str, list[tuple[str, float]]] = {}

    if args.mode in {"bm25", "bm25_kk_stem", "hybrid", "rerank"}:
        stem_kk = args.mode == "bm25_kk_stem"
        proc_docs = (
            [" ".join(tokenize_sparse(t, stem_kk=stem_kk)) for t in doc_texts] if stem_kk else doc_texts
        )
        proc_qs = [" ".join(tokenize_sparse(t, stem_kk=stem_kk)) for t in q_texts] if stem_kk else q_texts
        bm25 = BM25(proc_docs)
        idx_mat, sc_mat = bm25.search(proc_qs, k=100)
        for i, qid in enumerate(q_ids):
            sparse_rankings[qid] = [
                (doc_ids[int(idx_mat[i, j])], float(sc_mat[i, j]))
                for j in range(idx_mat.shape[1])
                if idx_mat[i, j] >= 0
            ]

    if args.mode in {"dense", "hybrid", "rerank"}:
        emb_name = args.embedder or params["retrieval"]["embedder"]
        emb = MultilingualEmbedder(emb_name, mrl_dim=args.mrl_dim)
        d_vecs = emb.embed_passages_np(doc_texts)
        q_vecs = emb.embed_queries_np(q_texts)
        d_res = dense_exact_search(q_vecs, d_vecs, doc_ids, top_k=100)
        for qid, res in zip(q_ids, d_res, strict=True):
            dense_rankings[qid] = res

    if args.mode in {"bm25", "bm25_kk_stem"}:
        first_stage = sparse_rankings
        final_rankings = sparse_rankings
    elif args.mode == "dense":
        first_stage = dense_rankings
        final_rankings = dense_rankings
    else:
        weights = {"dense": args.dense_weight, "sparse": args.sparse_weight}
        for qid in q_ids:
            fused = fuse_id_rankings(
                {"dense": dense_rankings.get(qid, []), "sparse": sparse_rankings.get(qid, [])},
                method=args.fusion,
                rrf_k=args.rrf_k,
                weights=weights,
                top_k=100,
            )
            first_stage[qid] = [(d, s) for d, s, _ in fused]
        final_rankings = dict(first_stage)

        if args.mode == "rerank":
            rr_name = args.reranker or params["retrieval"]["reranker"]
            reranker = CrossEncoderReranker(rr_name)
            doc_text_map = dict(zip(doc_ids, doc_texts, strict=True))
            cand_k = int(params["retrieval"]["candidates_k"])
            for qid, qtext in zip(q_ids, q_texts, strict=True):
                cands = first_stage[qid][:cand_k]
                if not cands:
                    continue
                scores = reranker.score_pairs([(qtext, doc_text_map[d]) for d, _ in cands])
                rescored = sorted(
                    zip((d for d, _ in cands), scores, strict=True), key=lambda kv: (-kv[1], kv[0])
                )
                final_rankings[qid] = rescored

    ev_cfg = params["evaluation"]
    report = evaluate_rankings(
        queries,
        qrels,
        final_rankings,
        first_stage_rankings=first_stage,
        n_bootstrap=int(ev_cfg["bootstrap_samples"]),
        ci=float(ev_cfg["ci"]),
        seed=int(params["seed"]),
    )
    per_query = report.pop("per_query")

    if args.per_query is not None:
        write_per_example_jsonl(args.per_query, per_query)

    if args.baseline_per_query is not None and args.baseline_per_query.is_file():
        base_rows = [
            json.loads(line)
            for line in args.baseline_per_query.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        report["comparison_vs_baseline"] = compare_retrieval_runs(
            base_rows,
            per_query,
            n_bootstrap=int(ev_cfg["bootstrap_samples"]),
            ci=float(ev_cfg["ci"]),
            seed=int(params["seed"]),
        )

    report["meta"] = {
        "mode": args.mode,
        "embedder": args.embedder
        or (params["retrieval"]["embedder"] if args.mode not in {"bm25", "bm25_kk_stem"} else None),
        "reranker": args.reranker or (params["retrieval"]["reranker"] if args.mode == "rerank" else None),
        "fusion": args.fusion if args.mode in {"hybrid", "rerank"} else None,
        "frozen_check": frozen_check,
        "seconds": round(time.perf_counter() - t0, 2),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    log.info(
        "Eval complete (%s): macro nDCG@10=%.4f, worst_dir=%s (gap=%.4f)",
        args.mode,
        report["macro"]["ndcg@10"],
        report["worst_direction"],
        report["worst_direction_gap"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
