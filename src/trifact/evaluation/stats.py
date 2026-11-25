"""Shared statistical evaluation utilities (ADR-0022, ROADMAP 5.1).

Provides:
- ``bootstrap_ci``: item-level mean bootstrap CI (extracted from ``eval_docproc``).
- ``cluster_bootstrap_ci``: document/query-cluster bootstrap for nonlinear metrics (entity F1, nDCG).
- ``cluster_paired_bootstrap``: paired cluster bootstrap of ``M(B) - M(A)`` on identical resampled clusters.
- ``multi_seed_cluster_bootstrap``: joint seed + cluster paired bootstrap for >=3-seed model comparisons.
- ``single_system_multi_seed_bootstrap``: multi-seed mean, std, and joint seed+cluster CI for one system.
- ``holm_bonferroni``: step-down multiple-comparisons correction.
- ``equal_mass_ece``: equal-mass binned Expected Calibration Error, Brier score, and reliability bins.
- ``precision_coverage_curve``: threshold sweep for corpus-annotation confidence selection.
- ``write_per_example_jsonl``: deterministic per-example output writer for MLflow traces.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any


def bootstrap_ci(xs: Sequence[float], n: int = 1000, ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``xs``."""
    if len(xs) < 2:
        return (float(xs[0]), float(xs[0])) if xs else (0.0, 0.0)
    rng = random.Random(seed)
    k = len(xs)
    means = sorted(statistics.fmean(rng.choices(xs, k=k)) for _ in range(n))
    lo = means[int((1.0 - ci) / 2.0 * n)]
    hi = means[min(int((1.0 + ci) / 2.0 * n), n - 1)]
    return float(lo), float(hi)


def _group_clusters(cluster_ids: Sequence[str]) -> list[list[int]]:
    """Map each unique cluster id to its list of item indices (deterministic order)."""
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, cid in enumerate(cluster_ids):
        groups[str(cid)].append(idx)
    return list(groups.values())


def _percentiles(values: list[float], ci: float) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    s = sorted(values)
    n = len(s)
    lo_idx = max(0, min(int((1.0 - ci) / 2.0 * n), n - 1))
    hi_idx = max(0, min(int((1.0 + ci) / 2.0 * n), n - 1))
    return float(s[lo_idx]), float(s[hi_idx])


def cluster_bootstrap_ci[T](
    items: Sequence[T],
    cluster_ids: Sequence[str],
    metric_fn: Callable[[Sequence[T]], float],
    n: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Cluster bootstrap CI recomputing ``metric_fn`` from scratch on each cluster resample."""
    if len(items) != len(cluster_ids):
        raise ValueError(f"items ({len(items)}) and cluster_ids ({len(cluster_ids)}) must match in length")
    if not items:
        return {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_items": 0, "n_clusters": 0}
    point = float(metric_fn(items))
    clusters = _group_clusters(cluster_ids)
    k = len(clusters)
    if k < 2:
        return {"point": point, "ci_low": point, "ci_high": point, "n_items": len(items), "n_clusters": k}
    rng = random.Random(seed)
    scores: list[float] = []
    for _ in range(n):
        sampled: list[T] = []
        for c_idx in rng.choices(range(k), k=k):
            for i in clusters[c_idx]:
                sampled.append(items[i])
        scores.append(float(metric_fn(sampled)))
    lo, hi = _percentiles(scores, ci)
    return {
        "point": point,
        "ci_low": lo,
        "ci_high": hi,
        "n_items": len(items),
        "n_clusters": k,
    }


def cluster_paired_bootstrap[T](
    items_a: Sequence[T],
    items_b: Sequence[T],
    cluster_ids: Sequence[str],
    metric_fn: Callable[[Sequence[T]], float],
    n: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired cluster bootstrap for ``delta = metric_fn(B) - metric_fn(A)`` on identical clusters."""
    if not (len(items_a) == len(items_b) == len(cluster_ids)):
        raise ValueError("items_a, items_b, and cluster_ids must have identical lengths")
    if not items_a:
        return {
            "point_a": 0.0,
            "point_b": 0.0,
            "diff": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_value": 1.0,
            "significant": False,
            "n_items": 0,
            "n_clusters": 0,
        }
    pa = float(metric_fn(items_a))
    pb = float(metric_fn(items_b))
    diff = pb - pa
    clusters = _group_clusters(cluster_ids)
    k = len(clusters)
    if k < 2:
        return {
            "point_a": pa,
            "point_b": pb,
            "diff": diff,
            "ci_low": diff,
            "ci_high": diff,
            "p_value": 0.0 if diff > 0 else 1.0,
            "significant": False,
            "n_items": len(items_a),
            "n_clusters": k,
        }
    rng = random.Random(seed)
    diffs: list[float] = []
    non_pos = 0
    for _ in range(n):
        sa: list[T] = []
        sb: list[T] = []
        for c_idx in rng.choices(range(k), k=k):
            for i in clusters[c_idx]:
                sa.append(items_a[i])
                sb.append(items_b[i])
        d = float(metric_fn(sb)) - float(metric_fn(sa))
        diffs.append(d)
        if d <= 0.0:
            non_pos += 1
    lo, hi = _percentiles(diffs, ci)
    return {
        "point_a": pa,
        "point_b": pb,
        "diff": diff,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": float((non_pos + 1) / (n + 1)),
        "significant": bool(lo > 0.0),
        "n_items": len(items_a),
        "n_clusters": k,
    }


def single_system_multi_seed_bootstrap[T](
    seeds_items: Sequence[Sequence[T]],
    cluster_ids: Sequence[str],
    metric_fn: Callable[[Sequence[T]], float],
    n: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Joint seed + cluster bootstrap for a single system trained with multiple seeds."""
    if not seeds_items:
        raise ValueError("seeds_items must contain at least one seed")
    for s_items in seeds_items:
        if len(s_items) != len(cluster_ids):
            raise ValueError("Each seed's items must match cluster_ids length")
    per_seed = [float(metric_fn(s)) for s in seeds_items]
    mean = float(statistics.fmean(per_seed))
    std = float(statistics.stdev(per_seed)) if len(per_seed) > 1 else 0.0
    clusters = _group_clusters(cluster_ids)
    k = len(clusters)
    if k < 2:
        return {
            "mean": mean,
            "std": std,
            "per_seed": per_seed,
            "ci_low": min(per_seed),
            "ci_high": max(per_seed),
            "n_seeds": len(seeds_items),
            "n_clusters": k,
        }
    rng = random.Random(seed)
    n_seeds = len(seeds_items)
    scores: list[float] = []
    for _ in range(n):
        chosen = seeds_items[rng.randrange(n_seeds)]
        sampled: list[T] = []
        for c_idx in rng.choices(range(k), k=k):
            for i in clusters[c_idx]:
                sampled.append(chosen[i])
        scores.append(float(metric_fn(sampled)))
    lo, hi = _percentiles(scores, ci)
    return {
        "mean": mean,
        "std": std,
        "per_seed": per_seed,
        "ci_low": lo,
        "ci_high": hi,
        "n_seeds": n_seeds,
        "n_clusters": k,
    }


def multi_seed_cluster_bootstrap[T](
    seeds_a: Sequence[Sequence[T]],
    seeds_b: Sequence[Sequence[T]],
    cluster_ids: Sequence[str],
    metric_fn: Callable[[Sequence[T]], float],
    n: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Joint seed + cluster paired bootstrap of ``M(B) - M(A)`` per ADR-0022.

    Each resample draws one random seed from ``seeds_a``, one random seed from ``seeds_b``,
    and resamples clusters with replacement identically for both systems.
    """
    if not seeds_a or not seeds_b:
        raise ValueError("Both seeds_a and seeds_b must contain at least one seed")
    for seq in (*seeds_a, *seeds_b):
        if len(seq) != len(cluster_ids):
            raise ValueError("All seed item sequences must match cluster_ids length")
    scores_a = [float(metric_fn(s)) for s in seeds_a]
    scores_b = [float(metric_fn(s)) for s in seeds_b]
    mean_a = float(statistics.fmean(scores_a))
    std_a = float(statistics.stdev(scores_a)) if len(scores_a) > 1 else 0.0
    mean_b = float(statistics.fmean(scores_b))
    std_b = float(statistics.stdev(scores_b)) if len(scores_b) > 1 else 0.0
    diff = mean_b - mean_a
    clusters = _group_clusters(cluster_ids)
    k = len(clusters)
    if k < 2:
        return {
            "mean_a": mean_a,
            "std_a": std_a,
            "mean_b": mean_b,
            "std_b": std_b,
            "diff": diff,
            "ci_low": diff,
            "ci_high": diff,
            "p_value": 0.0 if diff > 0 else 1.0,
            "significant": False,
            "n_seeds_a": len(seeds_a),
            "n_seeds_b": len(seeds_b),
            "n_clusters": k,
        }
    rng = random.Random(seed)
    na, nb = len(seeds_a), len(seeds_b)
    diffs: list[float] = []
    non_pos = 0
    for _ in range(n):
        seq_a = seeds_a[rng.randrange(na)]
        seq_b = seeds_b[rng.randrange(nb)]
        sa: list[T] = []
        sb: list[T] = []
        for c_idx in rng.choices(range(k), k=k):
            for i in clusters[c_idx]:
                sa.append(seq_a[i])
                sb.append(seq_b[i])
        d = float(metric_fn(sb)) - float(metric_fn(sa))
        diffs.append(d)
        if d <= 0.0:
            non_pos += 1
    lo, hi = _percentiles(diffs, ci)
    return {
        "mean_a": mean_a,
        "std_a": std_a,
        "mean_b": mean_b,
        "std_b": std_b,
        "diff": diff,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": float((non_pos + 1) / (n + 1)),
        "significant": bool(lo > 0.0),
        "n_seeds_a": na,
        "n_seeds_b": nb,
        "n_clusters": k,
    }


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    """Holm-Bonferroni step-down correction for multiple comparisons."""
    m = len(p_values)
    if m == 0:
        return {}
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    out: dict[str, dict[str, Any]] = {}
    running_max = 0.0
    stopped = False
    for rank, (key, raw_p) in enumerate(ordered):
        factor = m - rank
        adj_p = min(1.0, max(running_max, raw_p * factor))
        running_max = adj_p
        if not stopped and adj_p <= alpha:
            reject = True
        else:
            stopped = True
            reject = False
        out[key] = {"raw_p": float(raw_p), "adj_p": float(adj_p), "reject": reject}
    return out


def equal_mass_ece(probs: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> dict[str, Any]:
    """Equal-mass (quantile) Expected Calibration Error, Brier score, and reliability bins."""
    if len(probs) != len(labels):
        raise ValueError("probs and labels must have equal length")
    n = len(probs)
    if n == 0:
        return {"ece": 0.0, "brier": 0.0, "n": 0, "bins": []}
    pairs = sorted((float(max(0.0, min(1.0, p))), 1 if y else 0) for p, y in zip(probs, labels, strict=True))
    brier = sum((p - y) ** 2 for p, y in pairs) / n
    effective_bins = max(1, min(n_bins, n))
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for b in range(effective_bins):
        lo = (b * n) // effective_bins
        hi = ((b + 1) * n) // effective_bins
        chunk = pairs[lo:hi]
        if not chunk:
            continue
        cnt = len(chunk)
        avg_conf = sum(p for p, _ in chunk) / cnt
        acc = sum(y for _, y in chunk) / cnt
        gap = abs(avg_conf - acc)
        ece += (cnt / n) * gap
        bins.append(
            {
                "bin": b,
                "count": cnt,
                "min_conf": round(chunk[0][0], 4),
                "max_conf": round(chunk[-1][0], 4),
                "avg_conf": round(avg_conf, 4),
                "accuracy": round(acc, 4),
                "gap": round(gap, 4),
            }
        )
    return {
        "ece": round(float(ece), 6),
        "brier": round(float(brier), 6),
        "n": n,
        "bins": bins,
    }


def precision_coverage_curve(
    probs: Sequence[float],
    labels: Sequence[int],
    thresholds: Sequence[float] | None = None,
) -> list[dict[str, float]]:
    """Compute precision, recall-of-predicted, coverage, and F1 across confidence thresholds."""
    if len(probs) != len(labels):
        raise ValueError("probs and labels must have equal length")
    n = len(probs)
    total_pos = sum(1 for y in labels if y)
    if thresholds is None:
        thresholds = [round(i / 20.0, 2) for i in range(20)]
    curve: list[dict[str, float]] = []
    for thr in thresholds:
        kept = [(p, y) for p, y in zip(probs, labels, strict=True) if p >= thr]
        k = len(kept)
        tp = sum(1 for _, y in kept if y)
        prec = tp / k if k else 1.0
        rec = tp / total_pos if total_pos else 0.0
        cov = k / n if n else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        curve.append(
            {
                "threshold": float(thr),
                "kept": float(k),
                "coverage": round(cov, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            }
        )
    return curve


def select_annotation_threshold(
    curve: Sequence[dict[str, float]],
    min_precision: float = 0.85,
    min_coverage: float = 0.50,
) -> dict[str, float]:
    """Select the lowest confidence threshold meeting ``min_precision`` (preferring high precision
    for Qdrant payload filters while keeping usable coverage). Falls back to max-F1 point."""
    eligible = [pt for pt in curve if pt["precision"] >= min_precision and pt["coverage"] >= min_coverage]
    if eligible:
        return min(eligible, key=lambda pt: pt["threshold"])
    prec_only = [pt for pt in curve if pt["precision"] >= min_precision and pt["kept"] > 0]
    if prec_only:
        return max(prec_only, key=lambda pt: pt["coverage"])
    if not curve:
        return {"threshold": 0.5, "kept": 0.0, "coverage": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    return max(curve, key=lambda pt: (pt["f1"], -abs(pt["threshold"] - 0.5)))


def write_per_example_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> Path:
    """Write per-example evaluation records as UTF-8 LF JSONL (ADR-0022)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return max(0.0, center - half), min(1.0, center + half)
