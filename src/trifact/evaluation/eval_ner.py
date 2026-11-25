"""Strict IOB2 NER evaluation, error taxonomy, unseen-entity F1, and cluster bootstrap (ADR-0006, ADR-0022).

Usage:
    python -m trifact.evaluation.eval_ner --preds <preds.jsonl> [--baseline-preds <base.jsonl>]
    python -m trifact.evaluation.eval_ner --model <model_dir> [--backend hf|onnx|gliner]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
from seqeval.metrics import f1_score as seqeval_f1_score
from seqeval.metrics import precision_score as seqeval_precision_score
from seqeval.metrics import recall_score as seqeval_recall_score
from seqeval.scheme import IOB2

from trifact.core.params import ROOT, load_params
from trifact.evaluation.stats import (
    cluster_bootstrap_ci,
    multi_seed_cluster_bootstrap,
    single_system_multi_seed_bootstrap,
    write_per_example_jsonl,
)

log = logging.getLogger("trifact.eval_ner")

CORE_TYPES = ("PER", "ORG", "LOC")
PUNCT_ONLY_RE = re.compile(r"^[\W_]+$")
CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁәғқңөұүһіӘҒҚҢӨҰҮҺІ]")
LATIN_RE = re.compile(r"[a-zA-Z]")
INITIAL_RE = re.compile(r"^[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ]\.$")
KK_SUFFIX_RE = re.compile(
    r"(?:дар|дер|тар|тер|лар|лер)?(?:дың|дің|тың|тің|ның|нің|дан|ден|тан|тен|нан|нен|да|де|та|те|ға|ге|қа|ке|ны|ні|ды|ді|ты|ті|мен|бен|пен)$",
    re.IGNORECASE,
)


def verify_frozen_eval(eval_dir: Path) -> dict[str, Any]:
    """Verify SHA-256 digests of frozen NER eval shards against ``frozen.json`` (ADR-0007)."""
    frozen_path = eval_dir / "frozen.json"
    if not frozen_path.is_file():
        raise FileNotFoundError(f"Frozen eval manifest missing: {frozen_path}")
    manifest = json.loads(frozen_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen":
        raise ValueError(f"Eval suite at {eval_dir} is not frozen (status={manifest.get('status')!r})")
    mismatches: list[str] = []
    for rel, expected in manifest.get("shards", {}).items():
        p = eval_dir / rel
        if not p.is_file():
            mismatches.append(f"missing:{rel}")
            continue
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != expected:
            mismatches.append(f"sha256_mismatch:{rel}")
    if mismatches:
        raise ValueError(f"Frozen NER eval integrity check failed: {mismatches}")
    return manifest


def extract_iob2_entities(tags: Sequence[str]) -> list[tuple[int, int, str]]:
    """Extract strict IOB2 entity spans ``(start, end_exclusive, label)`` matching ``seqeval`` IOB2 strict mode.

    In strict IOB2:
    - ``B-X`` starts a new entity of type ``X`` (closing any previous entity).
    - ``I-X`` continues an active entity of type ``X``; an ``I-X`` after ``O`` or ``*-Y`` (Y != X)
      is invalid in strict IOB2 and treated as starting an entity in seqeval's ``IOB2`` strict parser
      only when preceded by matching prefix — wait: let's match ``seqeval.scheme.Entities(..., IOB2)``
      exact semantics by delegating or mirroring its state machine.
    """
    spans: list[tuple[int, int, str]] = []
    start: int | None = None
    label: str | None = None
    for i, tag in enumerate(tags):
        if not tag or tag == "O":
            if start is not None and label is not None:
                spans.append((start, i, label))
                start = None
                label = None
            continue
        if "-" not in tag:
            if start is not None and label is not None:
                spans.append((start, i, label))
                start = None
                label = None
            continue
        prefix, ent_type = tag.split("-", 1)
        if prefix == "B":
            if start is not None and label is not None:
                spans.append((start, i, label))
            start = i
            label = ent_type
        elif prefix == "I":
            if start is not None and label == ent_type:
                continue
            # Strict IOB2 (matches seqeval mode="strict", scheme=IOB2):
            # an I-X not preceded by B-X/I-X closes any open span and does NOT start a valid entity.
            if start is not None and label is not None:
                spans.append((start, i, label))
                start = None
                label = None
        else:
            if start is not None and label is not None:
                spans.append((start, i, label))
                start = None
                label = None
    if start is not None and label is not None:
        spans.append((start, len(tags), label))
    return spans


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {
        "precision": round(p, 6),
        "recall": round(r, 6),
        "f1": round(f1, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": tp + fn,
    }


def strict_entity_metrics(
    y_true: Sequence[Sequence[str]],
    y_pred: Sequence[Sequence[str]],
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute strict IOB2 entity-level P/R/F1 (micro, macro, and per-type).

    Verified to match ``seqeval.metrics.{precision,recall,f1}_score(..., mode='strict', scheme=IOB2)``.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) length mismatch")
    tp_by: Counter[str] = Counter()
    fp_by: Counter[str] = Counter()
    fn_by: Counter[str] = Counter()
    seen_types: set[str] = set(labels) if labels is not None else set()

    for gt, pr in zip(y_true, y_pred, strict=True):
        gt_set = set(extract_iob2_entities(gt))
        pr_set = set(extract_iob2_entities(pr))
        for _, _, t in gt_set:
            seen_types.add(t)
        for _, _, t in pr_set:
            seen_types.add(t)
        for span in gt_set & pr_set:
            tp_by[span[2]] += 1
        for span in pr_set - gt_set:
            fp_by[span[2]] += 1
        for span in gt_set - pr_set:
            fn_by[span[2]] += 1

    per_type: dict[str, dict[str, Any]] = {}
    ordered_types = list(labels) if labels is not None else sorted(seen_types)
    for t in ordered_types:
        per_type[t] = _prf(tp_by[t], fp_by[t], fn_by[t])

    total_tp = sum(tp_by[t] for t in ordered_types)
    total_fp = sum(fp_by[t] for t in ordered_types)
    total_fn = sum(fn_by[t] for t in ordered_types)
    micro = _prf(total_tp, total_fp, total_fn)

    active = [per_type[t] for t in ordered_types if per_type[t]["support"] > 0]
    if active:
        macro_p = sum(x["precision"] for x in active) / len(active)
        macro_r = sum(x["recall"] for x in active) / len(active)
        macro_f1 = sum(x["f1"] for x in active) / len(active)
    else:
        macro_p = macro_r = macro_f1 = 0.0

    return {
        "micro": micro,
        "macro": {
            "precision": round(macro_p, 6),
            "recall": round(macro_r, 6),
            "f1": round(macro_f1, 6),
            "support": micro["support"],
        },
        "per_type": per_type,
    }


def seqeval_strict_reference(
    y_true: Sequence[Sequence[str]],
    y_pred: Sequence[Sequence[str]],
) -> dict[str, float]:
    """Direct ``seqeval`` reference calculation in strict IOB2 mode."""
    yt = [list(s) for s in y_true]
    yp = [list(s) for s in y_pred]
    return {
        "precision": float(seqeval_precision_score(yt, yp, mode="strict", scheme=IOB2, zero_division=0)),
        "recall": float(seqeval_recall_score(yt, yp, mode="strict", scheme=IOB2, zero_division=0)),
        "f1": float(seqeval_f1_score(yt, yp, mode="strict", scheme=IOB2, zero_division=0)),
    }


def _spans_overlap(a: tuple[int, int, str], b: tuple[int, int, str]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def span_error_taxonomy(
    y_true: Sequence[Sequence[str]],
    y_pred: Sequence[Sequence[str]],
    tokens_list: Sequence[Sequence[str]] | None = None,
    langs: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Decompose entity errors into strict/exact/partial/ent_type and TriFact error categories."""
    counts: Counter[str] = Counter()
    confusion: Counter[str] = Counter()
    categories: Counter[str] = Counter()

    for idx, (gt_tags, pr_tags) in enumerate(zip(y_true, y_pred, strict=True)):
        gt_spans = extract_iob2_entities(gt_tags)
        pr_spans = extract_iob2_entities(pr_tags)
        toks = tokens_list[idx] if tokens_list is not None else None
        lang = langs[idx] if langs is not None else ""
        sent_text = " ".join(toks) if toks else ""
        is_mixed_script = bool(CYRILLIC_RE.search(sent_text) and LATIN_RE.search(sent_text))

        matched_gt: set[int] = set()
        matched_pr: set[int] = set()

        # 1. Exact boundary matches
        for pi, ps in enumerate(pr_spans):
            for gi, gs in enumerate(gt_spans):
                if gi in matched_gt:
                    continue
                if ps[0] == gs[0] and ps[1] == gs[1]:
                    matched_gt.add(gi)
                    matched_pr.add(pi)
                    if ps[2] == gs[2]:
                        counts["strict_correct"] += 1
                    else:
                        counts["type_error"] += 1
                        confusion[f"{gs[2]}->{ps[2]}"] += 1
                        if {gs[2], ps[2]} == {"ORG", "LOC"}:
                            categories["org_loc_metonymy"] += 1
                        if is_mixed_script:
                            categories["mixed_script_error"] += 1
                    break

        # 2. Overlapping boundary matches among remaining
        for pi, ps in enumerate(pr_spans):
            if pi in matched_pr:
                continue
            for gi, gs in enumerate(gt_spans):
                if gi in matched_gt:
                    continue
                if _spans_overlap(ps, gs):
                    matched_gt.add(gi)
                    matched_pr.add(pi)
                    if ps[2] == gs[2]:
                        counts["boundary_error"] += 1
                    else:
                        counts["boundary_and_type_error"] += 1
                        confusion[f"{gs[2]}->{ps[2]}"] += 1
                        if {gs[2], ps[2]} == {"ORG", "LOC"}:
                            categories["org_loc_metonymy"] += 1
                    if (ps[0] >= gs[0] and ps[1] <= gs[1]) or (gs[0] >= ps[0] and gs[1] <= ps[1]):
                        categories["nested_or_compound_span"] += 1
                    if toks is not None:
                        if ps[2] == "PER" or gs[2] == "PER":
                            span_toks = toks[min(ps[0], gs[0]) : max(ps[1], gs[1])]
                            if any(INITIAL_RE.match(t) for t in span_toks):
                                categories["initials_or_title_error"] += 1
                        if lang == "kk":
                            span_toks = toks[min(ps[0], gs[0]) : max(ps[1], gs[1])]
                            if any(KK_SUFFIX_RE.search(t) for t in span_toks):
                                categories["kk_case_suffix_error"] += 1
                    if is_mixed_script:
                        categories["mixed_script_error"] += 1
                    break

        # 3. Remaining predicted = spurious (false positive)
        for pi, ps in enumerate(pr_spans):
            if pi in matched_pr:
                continue
            counts["spurious"] += 1
            if ps[0] == 0 and ps[2] == "PER":
                categories["sentence_initial_false_per"] += 1
            if toks is not None:
                surf = " ".join(toks[ps[0] : ps[1]]).strip()
                if PUNCT_ONLY_RE.match(surf):
                    categories["punctuation_only_spurious"] += 1
            if is_mixed_script:
                categories["mixed_script_error"] += 1

        # 4. Remaining gold = missed (false negative)
        for gi, gs in enumerate(gt_spans):
            if gi in matched_gt:
                continue
            counts["missed"] += 1
            if toks is not None and lang == "kk":
                span_toks = toks[gs[0] : gs[1]]
                if any(KK_SUFFIX_RE.search(t) for t in span_toks):
                    categories["kk_case_suffix_error"] += 1
            if is_mixed_script:
                categories["mixed_script_error"] += 1

    # Compute nervaluate-style summary F1s
    cor = counts["strict_correct"]
    typ = counts["type_error"]
    bnd = counts["boundary_error"]
    bt = counts["boundary_and_type_error"]
    spu = counts["spurious"]
    mis = counts["missed"]
    possible = cor + typ + bnd + bt + mis
    actual = cor + typ + bnd + bt + spu

    def _f1_from(num: float) -> float:
        p = num / actual if actual > 0 else 0.0
        r = num / possible if possible > 0 else 0.0
        return round((2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0, 6)

    return {
        "counts": {
            "strict_correct": cor,
            "type_error": typ,
            "boundary_error": bnd,
            "boundary_and_type_error": bt,
            "spurious": spu,
            "missed": mis,
            "possible": possible,
            "actual": actual,
        },
        "scores": {
            "strict_f1": _f1_from(cor),
            "exact_boundary_f1": _f1_from(cor + typ),
            "ent_type_f1": _f1_from(cor + bnd),
            "partial_f1": _f1_from(cor + typ + 0.5 * (bnd + bt)),
        },
        "type_confusion": dict(confusion.most_common()),
        "error_categories": dict(categories.most_common()),
    }


def normalize_surface(tokens: Sequence[str], start: int, end: int) -> str:
    return " ".join(tokens[start:end]).strip().lower()


def extract_seen_surfaces(train_dir: Path, langs: Sequence[str] = ("kk", "ru", "en")) -> dict[str, set[str]]:
    """Collect lowercase entity surfaces seen in ``gold_train`` per language for unseen-entity F1."""
    seen: dict[str, set[str]] = {lg: set() for lg in langs}
    for lg in langs:
        files = sorted((train_dir / lg).glob("gold_train-*.parquet"))
        for fp in files:
            df = pl.read_parquet(fp, columns=["tokens", "tags"])
            for toks, tags in zip(df["tokens"].to_list(), df["tags"].to_list(), strict=True):
                for s, e, _ in extract_iob2_entities(tags):
                    seen[lg].add(normalize_surface(toks, s, e))
    return seen


def unseen_entity_metrics(
    y_true: Sequence[Sequence[str]],
    y_pred: Sequence[Sequence[str]],
    tokens_list: Sequence[Sequence[str]],
    seen_surfaces: set[str],
) -> dict[str, Any]:
    """Strict entity P/R/F1 split by whether the entity surface appeared in training data."""
    u_tp = u_fp = u_fn = 0
    s_tp = s_fp = s_fn = 0
    for gt, pr, toks in zip(y_true, y_pred, tokens_list, strict=True):
        gt_ent = {(s, e, t): normalize_surface(toks, s, e) for s, e, t in extract_iob2_entities(gt)}
        pr_ent = {(s, e, t): normalize_surface(toks, s, e) for s, e, t in extract_iob2_entities(pr)}
        for span, surf in gt_ent.items():
            is_seen = surf in seen_surfaces
            if span in pr_ent:
                if is_seen:
                    s_tp += 1
                else:
                    u_tp += 1
            else:
                if is_seen:
                    s_fn += 1
                else:
                    u_fn += 1
        for span, surf in pr_ent.items():
            if span not in gt_ent:
                if surf in seen_surfaces:
                    s_fp += 1
                else:
                    u_fp += 1
    return {
        "unseen": _prf(u_tp, u_fp, u_fn),
        "seen": _prf(s_tp, s_fp, s_fn),
    }


def _pair_strict_micro_f1(pairs: Sequence[tuple[Sequence[str], Sequence[str]]]) -> float:
    tp = fp = fn = 0
    for gt, pr in pairs:
        gt_s = set(extract_iob2_entities(gt))
        pr_s = set(extract_iob2_entities(pr))
        tp += len(gt_s & pr_s)
        fp += len(pr_s - gt_s)
        fn += len(gt_s - pr_s)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0


def evaluate_ner_predictions(
    rows: Sequence[dict[str, Any]],
    pred_tags_list: Sequence[Sequence[str]],
    seen_surfaces_by_lang: dict[str, set[str]] | None = None,
    labels: Sequence[str] = CORE_TYPES,
    bootstrap_samples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    per_example_out: Path | None = None,
) -> dict[str, Any]:
    """Full NER evaluation: per-language, per-type, per-source, non-Wikipedia slice,
    cluster bootstrap CI (by ``doc_key``), unseen-entity F1, and error taxonomy."""
    if len(rows) != len(pred_tags_list):
        raise ValueError(f"rows ({len(rows)}) != pred_tags_list ({len(pred_tags_list)})")

    by_lang: dict[str, list[int]] = defaultdict(list)
    by_source: dict[str, list[int]] = defaultdict(list)
    non_wiki_idx: list[int] = []
    per_example_records: list[dict[str, Any]] = []

    for i, (row, pr) in enumerate(zip(rows, pred_tags_list, strict=True)):
        lg = str(row["lang"])
        src = str(row.get("source", "unknown"))
        by_lang[lg].append(i)
        by_source[f"{lg}/{src}"].append(i)
        if src not in {"wikiann", "wikineural"}:
            non_wiki_idx.append(i)
        if per_example_out is not None:
            gt_s = set(extract_iob2_entities(row["tags"]))
            pr_s = set(extract_iob2_entities(pr))
            per_example_records.append(
                {
                    "sent_id": row.get("sent_id", str(i)),
                    "doc_key": row.get("doc_key", str(i)),
                    "lang": lg,
                    "source": src,
                    "n_tokens": len(row["tokens"]),
                    "strict_tp": len(gt_s & pr_s),
                    "strict_fp": len(pr_s - gt_s),
                    "strict_fn": len(gt_s - pr_s),
                    "gold_entities": [list(x) for x in sorted(gt_s)],
                    "pred_entities": [list(x) for x in sorted(pr_s)],
                }
            )

    if per_example_out is not None:
        write_per_example_jsonl(per_example_out, per_example_records)

    def _eval_subset(indices: list[int], lg_for_seen: str | None = None) -> dict[str, Any]:
        y_t = [rows[i]["tags"] for i in indices]
        y_p = [pred_tags_list[i] for i in indices]
        toks = [rows[i]["tokens"] for i in indices]
        lgs = [str(rows[i]["lang"]) for i in indices]
        cids = [str(rows[i].get("doc_key") or rows[i].get("sent_id") or i) for i in indices]
        metrics = strict_entity_metrics(y_t, y_p, labels=labels)
        pairs = list(zip(y_t, y_p, strict=True))
        boot = cluster_bootstrap_ci(pairs, cids, _pair_strict_micro_f1, n=bootstrap_samples, ci=ci, seed=seed)
        tax = span_error_taxonomy(y_t, y_p, tokens_list=toks, langs=lgs)
        out: dict[str, Any] = {
            "n_sentences": len(indices),
            "n_clusters": boot["n_clusters"],
            "micro": metrics["micro"],
            "macro": metrics["macro"],
            "per_type": metrics["per_type"],
            "strict_f1_ci": [round(boot["ci_low"], 6), round(boot["ci_high"], 6)],
            "error_taxonomy": tax,
        }
        if seen_surfaces_by_lang is not None:
            if lg_for_seen is not None and lg_for_seen in seen_surfaces_by_lang:
                seen = seen_surfaces_by_lang[lg_for_seen]
            else:
                seen = set().union(*seen_surfaces_by_lang.values()) if seen_surfaces_by_lang else set()
            out["unseen_eval"] = unseen_entity_metrics(y_t, y_p, toks, seen)
        return out

    lang_results: dict[str, Any] = {}
    for lg in sorted(by_lang):
        lang_results[lg] = _eval_subset(by_lang[lg], lg_for_seen=lg)

    source_results: dict[str, Any] = {}
    for src_key in sorted(by_source):
        lg = src_key.split("/", 1)[0]
        source_results[src_key] = _eval_subset(by_source[src_key], lg_for_seen=lg)

    overall = _eval_subset(list(range(len(rows))))
    macro_across_langs = (
        round(sum(lang_results[lg]["micro"]["f1"] for lg in lang_results) / len(lang_results), 6)
        if lang_results
        else 0.0
    )

    result: dict[str, Any] = {
        "by_lang": lang_results,
        "by_source": source_results,
        "overall": overall,
        "macro_lang_f1": macro_across_langs,
    }
    if non_wiki_idx:
        result["non_wikipedia_slice"] = _eval_subset(non_wiki_idx)
    return result


def compare_ner_systems(
    rows: Sequence[dict[str, Any]],
    seeds_a_preds: Sequence[Sequence[Sequence[str]]],
    seeds_b_preds: Sequence[Sequence[Sequence[str]]],
    bootstrap_samples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare system B against baseline system A across languages using joint seed + cluster bootstrap."""
    by_lang: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_lang[str(row["lang"])].append(i)

    out: dict[str, Any] = {}
    for lg in sorted(by_lang):
        idxs = by_lang[lg]
        cids = [str(rows[i].get("doc_key") or rows[i].get("sent_id") or i) for i in idxs]
        sa = [[(rows[i]["tags"], preds[i]) for i in idxs] for preds in seeds_a_preds]
        sb = [[(rows[i]["tags"], preds[i]) for i in idxs] for preds in seeds_b_preds]
        out[lg] = multi_seed_cluster_bootstrap(
            sa, sb, cids, _pair_strict_micro_f1, n=bootstrap_samples, ci=ci, seed=seed
        )

    # Also summarize single-system multi-seed CIs for B
    b_summary: dict[str, Any] = {}
    for lg in sorted(by_lang):
        idxs = by_lang[lg]
        cids = [str(rows[i].get("doc_key") or rows[i].get("sent_id") or i) for i in idxs]
        sb = [[(rows[i]["tags"], preds[i]) for i in idxs] for preds in seeds_b_preds]
        b_summary[lg] = single_system_multi_seed_bootstrap(
            sb, cids, _pair_strict_micro_f1, n=bootstrap_samples, ci=ci, seed=seed
        )

    return {"paired_diff_by_lang": out, "candidate_multi_seed": b_summary}


def load_eval_ner_rows(
    eval_dir: Path,
    langs: Sequence[str] = ("kk", "ru", "en"),
    verify_frozen: bool = True,
    limit_per_lang: int | None = None,
) -> list[dict[str, Any]]:
    """Load frozen NER test rows from ``data/eval/ner``."""
    if verify_frozen:
        verify_frozen_eval(eval_dir)
    rows: list[dict[str, Any]] = []
    for lg in langs:
        files = sorted((eval_dir / lg).glob("test-*.parquet"))
        if not files:
            continue
        df = pl.concat([pl.read_parquet(fp) for fp in files])
        if limit_per_lang is not None:
            df = df.head(limit_per_lang)
        rows.extend(df.to_dicts())
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, default=ROOT / "data" / "eval" / "ner")
    ap.add_argument("--train-dir", type=Path, default=ROOT / "data" / "task" / "ner")
    ap.add_argument("--preds", type=Path, help="JSONL file with {sent_id, pred_tags}")
    ap.add_argument("--baseline-preds", type=Path, help="Optional baseline JSONL for paired bootstrap")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "ner.json")
    ap.add_argument("--per-example", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-verify-frozen", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    params = load_params()
    ev = params["evaluation"]
    seed = int(params["seed"])

    t0 = time.perf_counter()
    rows = load_eval_ner_rows(
        args.eval_dir,
        verify_frozen=not args.no_verify_frozen,
        limit_per_lang=args.limit,
    )
    if not args.preds or not args.preds.is_file():
        raise FileNotFoundError(f"Prediction file required via --preds: {args.preds}")

    pred_map: dict[str, list[str]] = {}
    with args.preds.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pred_map[str(obj["sent_id"])] = list(obj["pred_tags"])

    preds = [pred_map[str(r["sent_id"])] for r in rows]
    seen = extract_seen_surfaces(args.train_dir) if args.train_dir.is_dir() else None
    report = evaluate_ner_predictions(
        rows,
        preds,
        seen_surfaces_by_lang=seen,
        bootstrap_samples=int(ev["bootstrap_samples"]),
        ci=float(ev["ci"]),
        seed=seed,
        per_example_out=args.per_example,
    )

    if args.baseline_preds and args.baseline_preds.is_file():
        base_map: dict[str, list[str]] = {}
        with args.baseline_preds.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                base_map[str(obj["sent_id"])] = list(obj["pred_tags"])
        base_preds = [base_map[str(r["sent_id"])] for r in rows]
        report["comparison_vs_baseline"] = compare_ner_systems(
            rows,
            [base_preds],
            [preds],
            bootstrap_samples=int(ev["bootstrap_samples"]),
            ci=float(ev["ci"]),
            seed=seed,
        )

    report["elapsed_s"] = round(time.perf_counter() - t0, 2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    log.info("Wrote NER evaluation report to %s (macro_lang_f1=%.4f)", args.out, report["macro_lang_f1"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
