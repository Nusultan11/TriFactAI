"""Hybrid retrieval fusion (RRF, DBSF, weighted) and optional entity soft-boosting (ADR-0002, ADR-0004, ADR-0013).

Supports both:
- Application-level ``ScoredChunk`` fusion (preserving per-leg ranks in ``ScoredChunk.ranks``).
- Fast ID-level fusion for 9-direction evaluation and RET-4 fusion ablations.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

from trifact.core.schemas import Chunk, ScoredChunk


def _normalize_scores_minmax(items: Sequence[tuple[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    vals = [s for _, s in items]
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 1e-12:
        return {doc_id: 1.0 for doc_id, _ in items}
    return {doc_id: (s - lo) / span for doc_id, s in items}


def _normalize_scores_dbsf(items: Sequence[tuple[str, float]]) -> dict[str, float]:
    """Distribution-Based Score Fusion normalization: clamp to ``[mean - 3*std, mean + 3*std]`` -> ``[0, 1]``."""
    if not items:
        return {}
    vals = [s for _, s in items]
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / max(n, 1)
    std = math.sqrt(var)
    lo = mean - 3.0 * std
    hi = mean + 3.0 * std
    span = hi - lo
    if span <= 1e-12:
        return {doc_id: 0.5 for doc_id, _ in items}
    return {doc_id: max(0.0, min(1.0, (s - lo) / span)) for doc_id, s in items}


def fuse_id_rankings(
    leg_rankings: Mapping[str, Sequence[tuple[str, float]]],
    *,
    method: str = "rrf",
    rrf_k: int = 60,
    weights: Mapping[str, float] | None = None,
    top_k: int | None = None,
) -> list[tuple[str, float, dict[str, int]]]:
    """Fuse ranked ``(doc_id, score)`` lists from named legs into ``(doc_id, fused_score, per_leg_ranks)``.

    Methods:
    - ``rrf``: ``sum(w_leg / (rrf_k + rank_1based))``
    - ``dbsf``: ``sum(w_leg * dbsf_norm(score))``
    - ``weighted``: ``sum(w_leg * minmax_norm(score))``
    """
    if method not in {"rrf", "dbsf", "weighted"}:
        raise ValueError(f"Unsupported fusion method {method!r}; expected rrf | dbsf | weighted")

    scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)

    for leg_name, ranked in leg_rankings.items():
        w = float(weights.get(leg_name, 1.0)) if weights is not None else 1.0
        if w == 0.0 or not ranked:
            continue

        if method == "rrf":
            for rank_1, (doc_id, _) in enumerate(ranked, start=1):
                scores[doc_id] += w / (rrf_k + rank_1)
                if leg_name not in ranks[doc_id]:
                    ranks[doc_id][leg_name] = rank_1
        else:
            norm_map = (
                _normalize_scores_dbsf(ranked) if method == "dbsf" else _normalize_scores_minmax(ranked)
            )
            for rank_1, (doc_id, _) in enumerate(ranked, start=1):
                scores[doc_id] += w * norm_map.get(doc_id, 0.0)
                if leg_name not in ranks[doc_id]:
                    ranks[doc_id][leg_name] = rank_1

    # Deterministic tie-breaking by score desc, then doc_id asc
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_k is not None:
        ordered = ordered[:top_k]
    return [(doc_id, float(sc), dict(ranks[doc_id])) for doc_id, sc in ordered]


def fuse_scored_chunks(
    leg_results: Mapping[str, Sequence[ScoredChunk]],
    *,
    method: str = "rrf",
    rrf_k: int = 60,
    weights: Mapping[str, float] | None = None,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """Fuse ``ScoredChunk`` sequences from multiple legs or collections, preserving per-leg ranks."""
    chunk_by_id: dict[str, Chunk] = {}
    existing_ranks: dict[str, dict[str, int]] = defaultdict(dict)
    id_rankings: dict[str, list[tuple[str, float]]] = {}

    for leg_name, items in leg_results.items():
        ranking: list[tuple[str, float]] = []
        for sc in items:
            cid = sc.chunk.chunk_id
            if cid not in chunk_by_id:
                chunk_by_id[cid] = sc.chunk
            for rk, rv in sc.ranks.items():
                existing_ranks[cid].setdefault(rk, rv)
            ranking.append((cid, float(sc.score)))
        id_rankings[leg_name] = ranking

    fused = fuse_id_rankings(id_rankings, method=method, rrf_k=rrf_k, weights=weights, top_k=top_k)
    out: list[ScoredChunk] = []
    for cid, score, leg_ranks in fused:
        merged_ranks = dict(existing_ranks[cid])
        merged_ranks.update(leg_ranks)
        out.append(ScoredChunk(chunk=chunk_by_id[cid], score=score, ranks=merged_ranks))
    return out


def rrf_fuse(
    leg_results: Mapping[str, Sequence[ScoredChunk]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """Convenience wrapper for Reciprocal Rank Fusion over ``ScoredChunk`` legs."""
    return fuse_scored_chunks(leg_results, method="rrf", rrf_k=k, weights=weights, top_k=top_k)


def dbsf_fuse(
    leg_results: Mapping[str, Sequence[ScoredChunk]],
    *,
    weights: Mapping[str, float] | None = None,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """Convenience wrapper for Distribution-Based Score Fusion."""
    return fuse_scored_chunks(leg_results, method="dbsf", weights=weights, top_k=top_k)


def weighted_fuse(
    leg_results: Mapping[str, Sequence[ScoredChunk]],
    *,
    weights: Mapping[str, float] | None = None,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """Convenience wrapper for min-max weighted score fusion."""
    return fuse_scored_chunks(leg_results, method="weighted", weights=weights, top_k=top_k)


def apply_entity_soft_boost(
    candidates: Sequence[ScoredChunk],
    chunk_entities: Mapping[str, Sequence[str]],
    query_entity_ids: Sequence[str],
    *,
    boost: float = 0.05,
) -> list[ScoredChunk]:
    """Optional soft boost for overlapping entity QIDs — NEVER drops candidates (ADR-0004)."""
    if not query_entity_ids or boost <= 0.0:
        return list(candidates)
    q_set = set(query_entity_ids)
    boosted: list[ScoredChunk] = []
    for sc in candidates:
        c_ents = set(chunk_entities.get(sc.chunk.chunk_id, ()))
        overlap = len(q_set & c_ents)
        bonus = boost * min(overlap, 3) / 3.0 if overlap else 0.0
        boosted.append(ScoredChunk(chunk=sc.chunk, score=float(sc.score + bonus), ranks=dict(sc.ranks)))
    boosted.sort(key=lambda x: (-x.score, x.chunk.chunk_id))
    return boosted


def merge_adjacent_chunks(
    candidates: Sequence[ScoredChunk],
    *,
    max_ordinal_gap: int = 1,
    max_merged_tokens: int = 750,
) -> list[ScoredChunk]:
    """Serving-time merge of adjacent retrieved chunks from the same document (RET-C ablation)."""
    if len(candidates) <= 1:
        return list(candidates)

    by_doc: dict[str, list[ScoredChunk]] = defaultdict(list)
    for sc in candidates:
        by_doc[sc.chunk.document_id].append(sc)

    merged_all: list[ScoredChunk] = []
    for doc_items in by_doc.values():
        doc_items.sort(key=lambda x: x.chunk.ordinal)
        cur = doc_items[0]
        for nxt in doc_items[1:]:
            gap = nxt.chunk.ordinal - cur.chunk.ordinal
            cur_tok = cur.chunk.token_count or len(cur.chunk.text.split())
            nxt_tok = nxt.chunk.token_count or len(nxt.chunk.text.split())
            if 0 < gap <= max_ordinal_gap and (cur_tok + nxt_tok) <= max_merged_tokens:
                # Merge text respecting character offsets if overlapping
                if nxt.chunk.char_start < cur.chunk.char_end:
                    overlap_chars = cur.chunk.char_end - nxt.chunk.char_start
                    combined_text = cur.chunk.text + nxt.chunk.text[overlap_chars:]
                else:
                    combined_text = cur.chunk.text + "\n" + nxt.chunk.text
                merged_chunk = Chunk(
                    document_id=cur.chunk.document_id,
                    chunk_id=cur.chunk.chunk_id,
                    ordinal=cur.chunk.ordinal,
                    language=cur.chunk.language,
                    page=cur.chunk.page,
                    section=cur.chunk.section,
                    char_start=cur.chunk.char_start,
                    char_end=max(cur.chunk.char_end, nxt.chunk.char_end),
                    text=combined_text,
                    source_type=cur.chunk.source_type,
                    corpus_source=cur.chunk.corpus_source,
                    token_count=cur_tok + nxt_tok,
                )
                cur = ScoredChunk(
                    chunk=merged_chunk,
                    score=max(cur.score, nxt.score),
                    ranks=dict(cur.ranks),
                )
            else:
                merged_all.append(cur)
                cur = nxt
        merged_all.append(cur)

    merged_all.sort(key=lambda x: (-x.score, x.chunk.chunk_id))
    return merged_all
