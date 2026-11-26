"""Multilingual grounded QA generation evaluation harness (Stage 7, ADR-0022, ROADMAP 7.2–7.5).

Computes per-language, per-direction, per-provenance, and contamination-sensitive slices:
- Answer EM, token F1, and character n-gram F-score (``chrF``, critical for agglutinative ``kk``).
- Claim-level faithfulness via ``VER-0`` 3-sentence evidence windows (ADR-0022), plus export of the
  mandatory 200 claims/language human faithfulness audit packet (``human_verified: false``).
- Citation Precision / Recall / F1 and ``invalid_citation_rate``.
- Refusal Precision / Recall / F1 and ``false_refusal_rate`` on answerable queries.
- Answer language accuracy and cross-lingual performance.
- Adapter regression checks (ROADMAP 7.3) and systematic error taxonomy.
- Context-cluster paired bootstrap & multi-seed bootstrap via ``trifact.evaluation.stats``.
- Multilingual ``imatrix`` calibration corpus builder for ``GEN-Q``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from trifact.core.interfaces import EvidenceVerifier, Generator
from trifact.core.params import ROOT, load_params
from trifact.core.schemas import Chunk, Lang, ScoredChunk, SourceType, Verdict
from trifact.evaluation.stats import (
    cluster_bootstrap_ci,
    cluster_paired_bootstrap,
    multi_seed_cluster_bootstrap,
    write_per_example_jsonl,
)
from trifact.generation.prompt import (
    detect_repetition_loop,
    extract_citation_ids,
    is_refusal,
    strip_thinking_tags,
)

log = logging.getLogger("trifact.evaluation.eval_generation")

_KAZAKH_LETTERS = frozenset("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
_CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁәғқңөұүһіӘҒҚҢӨҰҮҺІ]")
_LATIN_RE = re.compile(r"[a-zA-Z]")
_BRACKET_RE = re.compile(r"\[[^\[\]\n]{1,64}\]")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_EN_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")

# Sources derived from Wikipedia (ADR-0022 pretraining contamination slice)
WIKIPEDIA_SOURCES: frozenset[str] = frozenset({"squad_v2", "sberquad", "kazqad", "wikipedia"})


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen_qa_eval(eval_dir: Path) -> dict[str, Any]:
    """Verify SHA-256 checksums of frozen ``data/eval/qa`` shards against ``frozen.json``."""
    frozen_path = eval_dir / "frozen.json"
    if not frozen_path.is_file():
        return {"verified": False, "reason": "frozen.json missing"}
    meta = json.loads(frozen_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for rel, expected in (meta.get("shards") or {}).items():
        p = eval_dir / rel
        if not p.is_file():
            mismatches.append(f"missing:{rel}")
        elif _sha256_file(p) != expected:
            mismatches.append(f"sha256_mismatch:{rel}")
    return {
        "verified": len(mismatches) == 0,
        "fingerprint": meta.get("fingerprint"),
        "status": meta.get("status"),
        "mismatches": mismatches,
    }


def _context_cluster_id(context: str) -> str:
    norm = _WS_RE.sub(" ", unicodedata.normalize("NFC", context).lower()).strip()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


def load_frozen_qa_eval(
    eval_dir: Path,
    *,
    verify_frozen: bool = True,
    max_per_lang: int | None = None,
    seed: int = 42,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Load frozen ``data/eval/qa`` with deterministic context ``cluster_id`` (ADR-0022)."""
    check = verify_frozen_qa_eval(eval_dir) if verify_frozen else {"verified": False, "skipped": True}
    if verify_frozen and not check["verified"]:
        raise RuntimeError(f"Frozen QA eval integrity check failed: {check}")

    df = pl.read_parquet(str(eval_dir / "*/test-*.parquet"))
    cluster_ids = [_context_cluster_id(c) for c in df["context"].to_list()]
    df = df.with_columns(
        pl.Series("cluster_id", cluster_ids),
        pl.lit("human_verified").alias("verification_tier"),
        pl.when(pl.col("source").is_in(list(WIKIPEDIA_SOURCES)))
        .then(pl.lit("wikipedia"))
        .otherwise(pl.lit("non_wikipedia"))
        .alias("contamination_slice"),
    )

    if max_per_lang is not None and max_per_lang > 0:
        parts: list[pl.DataFrame] = []
        for lg in ("kk", "ru", "en"):
            sub = df.filter(pl.col("lang") == lg)
            if sub.height == 0:
                continue
            # Stratify answerable vs unanswerable if both exist
            ans = sub.filter(pl.col("answerable"))
            unans = sub.filter(~pl.col("answerable"))
            if ans.height > 0 and unans.height > 0:
                u_cap = min(unans.height, max(1, max_per_lang // 3))
                a_cap = max_per_lang - u_cap
                parts.append(
                    ans.with_columns(_r=pl.col("example_id").hash(seed=seed))
                    .sort("_r")
                    .drop("_r")
                    .head(a_cap)
                )
                parts.append(
                    unans.with_columns(_r=pl.col("example_id").hash(seed=seed))
                    .sort("_r")
                    .drop("_r")
                    .head(u_cap)
                )
            else:
                parts.append(
                    sub.with_columns(_r=pl.col("example_id").hash(seed=seed))
                    .sort("_r")
                    .drop("_r")
                    .head(max_per_lang)
                )
        df = pl.concat(parts)

    return df, check


def normalize_answer(text: str, lang: str = "en") -> str:
    """Language-aware answer normalization (NFC, strip citations, lowercase, strip punct)."""
    s, _ = strip_thinking_tags(text)
    s = _BRACKET_RE.sub(" ", s)
    s = unicodedata.normalize("NFC", s).lower()
    s = _PUNCT_RE.sub(" ", s)
    if lang == "en":
        s = _EN_ARTICLES_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def compute_em(prediction: str, references: Sequence[str], lang: str = "en") -> float:
    """Exact Match over normalized reference answers."""
    if not references:
        return 0.0
    pred_n = normalize_answer(prediction, lang)
    return 1.0 if any(pred_n == normalize_answer(ref, lang) for ref in references) else 0.0


def compute_token_f1(prediction: str, references: Sequence[str], lang: str = "en") -> float:
    """Maximum token-overlap F1 against any reference answer."""
    if not references:
        return 0.0
    pred_toks = normalize_answer(prediction, lang).split()
    best = 0.0
    for ref in references:
        ref_toks = normalize_answer(ref, lang).split()
        if not pred_toks and not ref_toks:
            best = max(best, 1.0)
            continue
        if not pred_toks or not ref_toks:
            continue
        common = Counter(pred_toks) & Counter(ref_toks)
        n_same = sum(common.values())
        if n_same == 0:
            continue
        prec = n_same / len(pred_toks)
        rec = n_same / len(ref_toks)
        best = max(best, 2.0 * prec * rec / (prec + rec))
    return float(best)


def _char_ngrams(text: str, n: int) -> Counter[str]:
    if len(text) < n:
        return Counter()
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def compute_chrf(
    prediction: str,
    references: Sequence[str],
    lang: str = "kk",
    *,
    max_n: int = 6,
    beta: float = 2.0,
) -> float:
    """Character n-gram F-score (chrF, Popovic 2015) — robust to Kazakh agglutinative suffixes."""
    if not references:
        return 0.0
    pred_n = normalize_answer(prediction, lang)
    if not pred_n:
        return 0.0
    beta2 = beta * beta
    best = 0.0
    for ref in references:
        ref_n = normalize_answer(ref, lang)
        if not ref_n:
            continue
        precs: list[float] = []
        recs: list[float] = []
        for n in range(1, max_n + 1):
            p_ng = _char_ngrams(pred_n, n)
            r_ng = _char_ngrams(ref_n, n)
            if not p_ng or not r_ng:
                continue
            ov = sum((p_ng & r_ng).values())
            precs.append(ov / sum(p_ng.values()))
            recs.append(ov / sum(r_ng.values()))
        if not precs:
            continue
        avg_p = sum(precs) / len(precs)
        avg_r = sum(recs) / len(recs)
        if avg_p + avg_r > 0:
            score = (1.0 + beta2) * avg_p * avg_r / (beta2 * avg_p + avg_r)
            best = max(best, score)
    return float(best)


def detect_answer_language(text: str) -> str:
    """Lightweight deterministic script + Kazakh-letter language detector for ``kk``/``ru``/``en``."""
    clean = _BRACKET_RE.sub(" ", text).strip()
    if not clean:
        return "unknown"
    kk_chars = sum(1 for ch in clean if ch in _KAZAKH_LETTERS)
    cyr = len(_CYRILLIC_RE.findall(clean))
    lat = len(_LATIN_RE.findall(clean))
    if cyr == 0 and lat == 0:
        return "unknown"
    if lat > cyr * 1.5:
        return "en"
    if kk_chars >= 1:
        return "kk"
    return "ru"


def compute_citation_metrics(
    cited_valid_ids: Sequence[str],
    invalid_ids: Sequence[str],
    gold_chunk_ids: Sequence[str],
) -> dict[str, float]:
    """Compute citation precision, recall, F1, and invalid citation rate for one answer."""
    valid_set = set(cited_valid_ids)
    gold_set = set(gold_chunk_ids)
    total_cited = len(cited_valid_ids) + len(invalid_ids)
    invalid_rate = (len(invalid_ids) / total_cited) if total_cited > 0 else 0.0

    if not gold_set:
        return {
            "citation_precision": 1.0 if total_cited == 0 else 0.0,
            "citation_recall": 1.0,
            "citation_f1": 1.0 if total_cited == 0 else 0.0,
            "invalid_citation_rate": invalid_rate,
        }

    tp = len(valid_set & gold_set)
    prec = tp / (len(valid_set) + len(invalid_ids)) if (len(valid_set) + len(invalid_ids)) > 0 else 0.0
    rec = tp / len(gold_set)
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {
        "citation_precision": float(prec),
        "citation_recall": float(rec),
        "citation_f1": float(f1),
        "invalid_citation_rate": float(invalid_rate),
    }


def classify_generation_error(rec: Mapping[str, Any]) -> str:
    """Assign each example to its primary error category for Stage 7 error analysis."""
    if rec.get("retrieval_gold_present") is False:
        return "retrieval_miss"
    if rec.get("prompt_gold_present") is False:
        return "context_truncation"
    if rec.get("repetition_loop"):
        return "repetition_loop"
    answerable = bool(rec.get("answerable", True))
    refused = bool(rec.get("refused", False))
    if answerable and refused:
        return "false_refusal"
    if (not answerable) and (not refused):
        return "missed_refusal"
    if not answerable and refused:
        return "ok"
    if not rec.get("lang_match", True):
        return "language_switch"
    if float(rec.get("invalid_citation_rate", 0.0)) > 0.0 or float(rec.get("citation_recall", 1.0)) == 0.0:
        return "incorrect_citation"
    if rec.get("faithfulness") is not None and float(rec["faithfulness"]) < 1.0:
        return "unsupported_claim"
    if float(rec.get("answer_f1", 0.0)) < 0.3 and float(rec.get("chrf", 0.0)) < 0.35:
        return "extractive_mismatch"
    return "ok"


def build_eval_qa_contexts(
    df: pl.DataFrame,
    *,
    num_distractors: int = 3,
    seed: int = 42,
    add_distractor_unanswerables_per_lang: int = 0,
) -> list[dict[str, Any]]:
    """Prepare evaluation items with gold + same-language distractor passages (plus optional refusal probes)."""
    rows = df.to_dicts()
    by_lang_ctx: dict[str, list[tuple[str, str]]] = defaultdict(list)
    seen_ctx: set[tuple[str, str]] = set()
    for r in rows:
        cid = f"c_{r['cluster_id'][:8]}"
        key = (str(r["lang"]), cid)
        if key not in seen_ctx:
            seen_ctx.add(key)
            by_lang_ctx[str(r["lang"])].append((cid, str(r["context"])))

    items: list[dict[str, Any]] = []
    for r in rows:
        lg = str(r["lang"])
        gold_cid = f"c_{r['cluster_id'][:8]}"
        rnd = random.Random(hashlib.sha1(f"{seed}:{r['example_id']}".encode()).hexdigest())
        pool = [pair for pair in by_lang_ctx[lg] if pair[0] != gold_cid]
        negs = rnd.sample(pool, k=min(num_distractors, len(pool))) if pool else []

        ev_list: list[ScoredChunk] = []
        all_pairs = [(gold_cid, str(r["context"]), 0.95)] + [
            (cid, txt, 0.70 - 0.05 * i) for i, (cid, txt) in enumerate(negs)
        ]
        rnd.shuffle(all_pairs)
        for idx, (cid, txt, sc) in enumerate(all_pairs):
            ev_list.append(
                ScoredChunk(
                    chunk=Chunk(
                        document_id=cid,
                        chunk_id=cid,
                        ordinal=idx,
                        language=Lang(lg),
                        char_start=0,
                        char_end=len(txt),
                        text=txt,
                        source_type=SourceType.CORPUS,
                        corpus_source=str(r.get("source", "eval")),
                    ),
                    score=sc,
                    ranks={"rerank": idx + 1},
                )
            )

        items.append(
            {
                "example_id": str(r["example_id"]),
                "cluster_id": str(r["cluster_id"]),
                "lang": lg,
                "direction": f"{lg}-{lg}",
                "source": str(r.get("source", "unknown")),
                "verification_tier": str(r.get("verification_tier", "human_verified")),
                "contamination_slice": str(r.get("contamination_slice", "wikipedia")),
                "question": str(r["question"]),
                "answers": list(r.get("answers") or []),
                "answerable": bool(r["answerable"]),
                "gold_chunk_ids": [gold_cid] if bool(r["answerable"]) else [],
                "context_chunks": ev_list,
            }
        )

    # Supplement kk/ru refusal evaluation with distractor-only unanswerable probes (explicitly tagged)
    if add_distractor_unanswerables_per_lang > 0:
        for lg in ("kk", "ru"):
            ans_rows = [r for r in rows if str(r["lang"]) == lg and bool(r["answerable"])]
            rnd = random.Random(seed + hash(lg) % 10000)
            chosen = rnd.sample(ans_rows, k=min(add_distractor_unanswerables_per_lang, len(ans_rows)))
            for r in chosen:
                gold_cid = f"c_{r['cluster_id'][:8]}"
                pool = [pair for pair in by_lang_ctx[lg] if pair[0] != gold_cid]
                negs = rnd.sample(pool, k=min(max(num_distractors, 2), len(pool)))
                ev_list = [
                    ScoredChunk(
                        chunk=Chunk(
                            document_id=cid,
                            chunk_id=cid,
                            ordinal=i,
                            language=Lang(lg),
                            char_start=0,
                            char_end=len(txt),
                            text=txt,
                            source_type=SourceType.CORPUS,
                            corpus_source=str(r.get("source", "eval")),
                        ),
                        score=0.65 - 0.05 * i,
                        ranks={"rerank": i + 1},
                    )
                    for i, (cid, txt) in enumerate(negs)
                ]
                items.append(
                    {
                        "example_id": f"{r['example_id']}#distractor_unans",
                        "cluster_id": str(r["cluster_id"]),
                        "lang": lg,
                        "direction": f"{lg}-{lg}",
                        "source": str(r.get("source", "unknown")),
                        "verification_tier": "distractor_unanswerable",
                        "contamination_slice": str(r.get("contamination_slice", "wikipedia")),
                        "question": str(r["question"]),
                        "answers": [],
                        "answerable": False,
                        "gold_chunk_ids": [],
                        "context_chunks": ev_list,
                    }
                )

    return items


def _aggregate_slice(
    records: Sequence[dict[str, Any]], *, n_boot: int = 1000, seed: int = 42
) -> dict[str, Any]:
    """Aggregate all Stage 7 metrics for a slice of per-example evaluation records."""
    if not records:
        return {"n": 0}

    ans_recs = [r for r in records if r["answerable"]]
    unans_recs = [r for r in records if not r["answerable"]]

    # Refusal P/R/F1
    tp_ref = sum(1 for r in unans_recs if r["refused"])
    fp_ref = sum(1 for r in ans_recs if r["refused"])
    fn_ref = sum(1 for r in unans_recs if not r["refused"])
    ref_p = tp_ref / (tp_ref + fp_ref) if (tp_ref + fp_ref) > 0 else 1.0
    ref_r = tp_ref / (tp_ref + fn_ref) if (tp_ref + fn_ref) > 0 else 1.0
    ref_f1 = (2.0 * ref_p * ref_r / (ref_p + ref_r)) if (ref_p + ref_r) > 0 else 0.0
    false_refusal_rate = fp_ref / len(ans_recs) if ans_recs else 0.0

    # Answerable quality metrics
    em_vals = [float(r["answer_em"]) for r in ans_recs]
    f1_vals = [float(r["answer_f1"]) for r in ans_recs]
    chrf_vals = [float(r["chrf"]) for r in ans_recs]
    cit_p_vals = [float(r["citation_precision"]) for r in ans_recs]
    cit_r_vals = [float(r["citation_recall"]) for r in ans_recs]
    cit_f1_vals = [float(r["citation_f1"]) for r in ans_recs]

    # Faithfulness (ADR-0022: non-refused claims marked supported / total claims)
    total_claims = sum(int(r.get("n_claims", 0)) for r in records if not r["refused"])
    supported_claims = sum(int(r.get("n_supported_claims", 0)) for r in records if not r["refused"])
    contradicted_claims = sum(int(r.get("n_contradicted_claims", 0)) for r in records if not r["refused"])
    faithfulness = (supported_claims / total_claims) if total_claims > 0 else None

    # Cluster bootstrap CI for answer_f1 on answerable examples
    f1_ci = (
        cluster_bootstrap_ci(
            ans_recs,
            [str(r["cluster_id"]) for r in ans_recs],
            lambda xs: statistics.fmean(float(x["answer_f1"]) for x in xs) if xs else 0.0,
            n=n_boot,
            seed=seed,
        )
        if ans_recs
        else {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_items": 0, "n_clusters": 0}
    )

    # Cluster bootstrap CI for faithfulness across non-refused records with claims
    claim_recs = [r for r in records if not r["refused"] and int(r.get("n_claims", 0)) > 0]
    faith_ci = (
        cluster_bootstrap_ci(
            claim_recs,
            [str(r["cluster_id"]) for r in claim_recs],
            lambda xs: (
                sum(int(x["n_supported_claims"]) for x in xs) / max(sum(int(x["n_claims"]) for x in xs), 1)
            ),
            n=n_boot,
            seed=seed,
        )
        if claim_recs
        else None
    )

    lat_ms = [float(r.get("total_ms", 0.0)) for r in records if float(r.get("total_ms", 0.0)) > 0]
    ttft_ms = [float(r.get("ttft_ms", 0.0)) for r in records if float(r.get("ttft_ms", 0.0)) > 0]
    tok_s = [float(r.get("tokens_per_s", 0.0)) for r in records if float(r.get("tokens_per_s", 0.0)) > 0]

    def _p95(xs: list[float]) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        return round(s[min(int(0.95 * len(s)), len(s) - 1)], 2)

    err_counts = Counter(str(r.get("error_category", "ok")) for r in records)

    return {
        "n": len(records),
        "n_answerable": len(ans_recs),
        "n_unanswerable": len(unans_recs),
        "answer_em": round(statistics.fmean(em_vals), 4) if em_vals else 0.0,
        "answer_f1": round(statistics.fmean(f1_vals), 4) if f1_vals else 0.0,
        "answer_f1_ci": {
            "ci_low": round(float(f1_ci["ci_low"]), 4),
            "ci_high": round(float(f1_ci["ci_high"]), 4),
            "n_clusters": int(f1_ci["n_clusters"]),
        },
        "chrf": round(statistics.fmean(chrf_vals), 4) if chrf_vals else 0.0,
        "faithfulness": round(faithfulness, 4) if faithfulness is not None else None,
        "faithfulness_ci": (
            {
                "ci_low": round(float(faith_ci["ci_low"]), 4),
                "ci_high": round(float(faith_ci["ci_high"]), 4),
                "n_clusters": int(faith_ci["n_clusters"]),
            }
            if faith_ci is not None
            else None
        ),
        "total_claims": total_claims,
        "supported_claims": supported_claims,
        "contradicted_claims": contradicted_claims,
        "citation_precision": round(statistics.fmean(cit_p_vals), 4) if cit_p_vals else 0.0,
        "citation_recall": round(statistics.fmean(cit_r_vals), 4) if cit_r_vals else 0.0,
        "citation_f1": round(statistics.fmean(cit_f1_vals), 4) if cit_f1_vals else 0.0,
        "invalid_citation_rate": round(
            statistics.fmean(float(r.get("invalid_citation_rate", 0.0)) for r in records), 4
        ),
        "refusal_precision": round(ref_p, 4),
        "refusal_recall": round(ref_r, 4),
        "refusal_f1": round(ref_f1, 4),
        "false_refusal_rate": round(false_refusal_rate, 4),
        "answer_lang_accuracy": round(
            statistics.fmean(1.0 if r.get("lang_match", False) else 0.0 for r in records), 4
        ),
        "repetition_loop_rate": round(
            statistics.fmean(1.0 if r.get("repetition_loop", False) else 0.0 for r in records), 4
        ),
        "think_leak_rate": round(
            statistics.fmean(1.0 if r.get("had_think_tags", False) else 0.0 for r in records), 4
        ),
        "latency_p50_ms": round(statistics.median(lat_ms), 2) if lat_ms else 0.0,
        "latency_p95_ms": _p95(lat_ms),
        "ttft_p50_ms": round(statistics.median(ttft_ms), 2) if ttft_ms else 0.0,
        "ttft_p95_ms": _p95(ttft_ms),
        "mean_tokens_per_s": round(statistics.fmean(tok_s), 2) if tok_s else 0.0,
        "error_taxonomy": dict(sorted(err_counts.items())),
    }


def evaluate_generator(
    generator: Generator,
    items: Sequence[dict[str, Any]],
    *,
    verifier: EvidenceVerifier | None = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run end-to-end generation evaluation and optional VER-0 faithfulness verification.

    Returns ``(report, per_example_records, audit_claim_rows)``.
    """
    records: list[dict[str, Any]] = []
    audit_claims: list[dict[str, Any]] = []

    for item in items:
        lg = str(item["lang"])
        ctx: list[ScoredChunk] = item["context_chunks"]
        if hasattr(generator, "generate_detailed"):
            gres = generator.generate_detailed(item["question"], ctx, Lang(lg))
            ans_text = gres.answer
            refused = gres.refused
            valid_cids = gres.valid_chunk_ids
            invalid_cids = gres.invalid_citations
            had_think = gres.had_think_tags
            rep_loop = gres.repetition_loop
            ttft_ms = gres.ttft_ms
            total_ms = gres.total_ms
            tok_s = gres.tokens_per_s
            prompt_tokens = gres.prompt_tokens
            completion_tokens = gres.completion_tokens
            prompt_cids = (
                [c.chunk.chunk_id for c in gres.prompt_info.selected_chunks]
                if gres.prompt_info is not None
                else [c.chunk.chunk_id for c in ctx]
            )
            id_map = gres.prompt_info.id_to_chunk_id if gres.prompt_info is not None else None
        else:
            ans_text = generator.generate(item["question"], ctx, Lang(lg))
            refused = is_refusal(ans_text, lg)
            raw_cids = extract_citation_ids(ans_text)
            ctx_ids = {c.chunk.chunk_id for c in ctx}
            valid_cids = [c for c in raw_cids if c in ctx_ids]
            invalid_cids = [c for c in raw_cids if c not in ctx_ids]
            had_think = False
            rep_loop = detect_repetition_loop(ans_text)
            ttft_ms = total_ms = tok_s = 0.0
            prompt_tokens = completion_tokens = 0
            prompt_cids = [c.chunk.chunk_id for c in ctx]
            id_map = None

        golds = list(item.get("answers") or [])
        answerable = bool(item["answerable"])
        em = compute_em(ans_text, golds, lg) if (answerable and not refused) else 0.0
        f1 = compute_token_f1(ans_text, golds, lg) if (answerable and not refused) else 0.0
        chrf = compute_chrf(ans_text, golds, lg) if (answerable and not refused) else 0.0
        cit = compute_citation_metrics(valid_cids, invalid_cids, item.get("gold_chunk_ids") or [])

        pred_lang = detect_answer_language(ans_text)
        lang_match = (pred_lang == lg) or (pred_lang == "unknown" and len(ans_text.strip()) <= 4)

        n_claims = n_sup = n_con = 0
        faith: float | None = None
        if verifier is not None and not refused:
            if hasattr(verifier, "verify"):
                try:
                    claims = verifier.verify(ans_text, ctx, id_to_chunk_id=id_map)  # type: ignore[call-arg]
                except TypeError:
                    claims = verifier.verify(ans_text, ctx)
            else:
                claims = []
            n_claims = len(claims)
            n_sup = sum(1 for c in claims if c.verdict == Verdict.SUPPORTED)
            n_con = sum(1 for c in claims if c.verdict == Verdict.CONTRADICTED)
            faith = (n_sup / n_claims) if n_claims > 0 else 0.0
            for c_idx, cl in enumerate(claims):
                audit_claims.append(
                    {
                        "claim_id": f"{item['example_id']}#c{c_idx}",
                        "example_id": item["example_id"],
                        "lang": lg,
                        "direction": item.get("direction", f"{lg}-{lg}"),
                        "question": item["question"],
                        "claim_text": cl.text,
                        "cited_chunk_ids": "|".join(ci.chunk_id for ci in cl.citations),
                        "winning_evidence_window": cl.citations[0].quote if cl.citations else "",
                        "ver0_verdict": cl.verdict.value if cl.verdict else "unverified",
                        "ver0_score": cl.nli_score if cl.nli_score is not None else "",
                        "human_verdict": "",
                        "reviewer": "",
                        "note": "",
                    }
                )

        gold_cids = set(item.get("gold_chunk_ids") or [])
        rec: dict[str, Any] = {
            "example_id": item["example_id"],
            "cluster_id": item["cluster_id"],
            "lang": lg,
            "direction": item.get("direction", f"{lg}-{lg}"),
            "source": item.get("source", "unknown"),
            "verification_tier": item.get("verification_tier", "human_verified"),
            "contamination_slice": item.get("contamination_slice", "wikipedia"),
            "answerable": answerable,
            "refused": refused,
            "prediction": ans_text,
            "references": golds,
            "answer_em": round(em, 4),
            "answer_f1": round(f1, 4),
            "chrf": round(chrf, 4),
            "faithfulness": round(faith, 4) if faith is not None else None,
            "n_claims": n_claims,
            "n_supported_claims": n_sup,
            "n_contradicted_claims": n_con,
            **{k: round(v, 4) for k, v in cit.items()},
            "pred_lang": pred_lang,
            "lang_match": lang_match,
            "had_think_tags": had_think,
            "repetition_loop": rep_loop,
            "retrieval_gold_present": (
                bool(gold_cids & {c.chunk.chunk_id for c in ctx}) if gold_cids else True
            ),
            "prompt_gold_present": (bool(gold_cids & set(prompt_cids)) if gold_cids else True),
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_per_s": tok_s,
        }
        rec["error_category"] = classify_generation_error(rec)
        records.append(rec)

    by_lang = {
        lg: _aggregate_slice([r for r in records if r["lang"] == lg], n_boot=n_boot, seed=seed)
        for lg in ("kk", "ru", "en")
        if any(r["lang"] == lg for r in records)
    }
    by_direction = {
        d: _aggregate_slice([r for r in records if r["direction"] == d], n_boot=n_boot, seed=seed)
        for d in sorted({str(r["direction"]) for r in records})
    }
    by_provenance = {
        t: _aggregate_slice([r for r in records if r["verification_tier"] == t], n_boot=n_boot, seed=seed)
        for t in sorted({str(r["verification_tier"]) for r in records})
    }
    by_contamination = {
        s: _aggregate_slice([r for r in records if r["contamination_slice"] == s], n_boot=n_boot, seed=seed)
        for s in sorted({str(r["contamination_slice"]) for r in records})
    }
    report = {
        "model_version": getattr(generator, "version", "unknown"),
        "overall": _aggregate_slice(records, n_boot=n_boot, seed=seed),
        "by_lang": by_lang,
        "by_direction": by_direction,
        "by_provenance": by_provenance,
        "by_contamination_slice": by_contamination,
        "faithfulness_audit_status": {
            "human_verified": False,
            "ver0_automated_only": verifier is not None,
            "claims_extracted": len(audit_claims),
        },
    }
    return report, records, audit_claims


def evaluate_adapter_regressions(
    baseline_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
    *,
    max_false_refusal_increase: float = 0.03,
    max_lang_drop: float = 0.02,
    max_invalid_cit_increase: float = 0.02,
    max_repetition_increase: float = 0.02,
) -> dict[str, Any]:
    """ROADMAP 7.3 adapter regression suite comparing candidate vs baseline across kk/ru/en."""
    checks: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []

    for lg in ("kk", "ru", "en"):
        b = (baseline_report.get("by_lang") or {}).get(lg)
        c = (candidate_report.get("by_lang") or {}).get(lg)
        if not b or not c:
            continue
        fr_delta = float(c["false_refusal_rate"]) - float(b["false_refusal_rate"])
        lg_delta = float(c["answer_lang_accuracy"]) - float(b["answer_lang_accuracy"])
        ic_delta = float(c["invalid_citation_rate"]) - float(b["invalid_citation_rate"])
        rp_delta = float(c["repetition_loop_rate"]) - float(b["repetition_loop_rate"])
        ok = (
            fr_delta <= max_false_refusal_increase
            and lg_delta >= -max_lang_drop
            and ic_delta <= max_invalid_cit_increase
            and rp_delta <= max_repetition_increase
        )
        checks[lg] = {
            "passed": ok,
            "false_refusal_delta": round(fr_delta, 4),
            "answer_lang_delta": round(lg_delta, 4),
            "invalid_citation_delta": round(ic_delta, 4),
            "repetition_loop_delta": round(rp_delta, 4),
        }
        if not ok:
            reasons.append(f"{lg} regression check failed: {checks[lg]}")

    return {
        "passed": all(v["passed"] for v in checks.values()) if checks else False,
        "checks": checks,
        "reasons": reasons,
    }


def compare_generation_runs(
    baseline_records: Sequence[dict[str, Any]],
    candidate_records: Sequence[dict[str, Any]],
    *,
    n_boot: int = 1000,
    seed: int = 42,
    candidate_seed_runs: Sequence[Sequence[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Cluster paired bootstrap (or multi-seed cluster bootstrap) comparing candidate vs baseline."""
    base_by_id = {str(r["example_id"]): r for r in baseline_records}
    cand_by_id = {str(r["example_id"]): r for r in candidate_records}
    common_ids = sorted(set(base_by_id) & set(cand_by_id))

    by_lang: dict[str, dict[str, Any]] = {}
    for lg in ("kk", "ru", "en"):
        ans_ids = [
            eid for eid in common_ids if base_by_id[eid]["lang"] == lg and base_by_id[eid]["answerable"]
        ]
        if not ans_ids:
            continue
        clusters = [str(base_by_id[eid]["cluster_id"]) for eid in ans_ids]
        a_list = [base_by_id[eid] for eid in ans_ids]
        b_list = [cand_by_id[eid] for eid in ans_ids]

        def _f1(xs: Sequence[dict[str, Any]]) -> float:
            return statistics.fmean(float(x["answer_f1"]) for x in xs) if xs else 0.0

        def _chrf(xs: Sequence[dict[str, Any]]) -> float:
            return statistics.fmean(float(x["chrf"]) for x in xs) if xs else 0.0

        def _faith(xs: Sequence[dict[str, Any]]) -> float:
            tot = sum(int(x.get("n_claims", 0)) for x in xs if not x["refused"])
            sup = sum(int(x.get("n_supported_claims", 0)) for x in xs if not x["refused"])
            return (sup / tot) if tot > 0 else 0.0

        if candidate_seed_runs and len(candidate_seed_runs) >= 2:
            seed_maps = [{str(r["example_id"]): r for r in run} for run in candidate_seed_runs]
            valid_ids = [eid for eid in ans_ids if all(eid in sm for sm in seed_maps)]
            c_clusters = [str(base_by_id[eid]["cluster_id"]) for eid in valid_ids]
            seeds_a = [[base_by_id[eid] for eid in valid_ids]]
            seeds_b = [[sm[eid] for eid in valid_ids] for sm in seed_maps]
            f1_cmp = multi_seed_cluster_bootstrap(seeds_a, seeds_b, c_clusters, _f1, n=n_boot, seed=seed)
            chrf_cmp = multi_seed_cluster_bootstrap(seeds_a, seeds_b, c_clusters, _chrf, n=n_boot, seed=seed)
            faith_cmp = multi_seed_cluster_bootstrap(
                seeds_a, seeds_b, c_clusters, _faith, n=n_boot, seed=seed
            )
        else:
            f1_cmp = cluster_paired_bootstrap(a_list, b_list, clusters, _f1, n=n_boot, seed=seed)
            chrf_cmp = cluster_paired_bootstrap(a_list, b_list, clusters, _chrf, n=n_boot, seed=seed)
            faith_cmp = cluster_paired_bootstrap(a_list, b_list, clusters, _faith, n=n_boot, seed=seed)

        by_lang[lg] = {
            "answer_f1": f1_cmp,
            "chrf": chrf_cmp,
            "faithfulness": faith_cmp,
        }

    return {"n_common_examples": len(common_ids), "by_lang": by_lang}


def export_human_faithfulness_audit(
    audit_claims: Sequence[dict[str, Any]],
    out_dir: Path,
    *,
    claims_per_lang: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Export 200 claims/language human faithfulness audit packet (ADR-0022, human_verified=False)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, Any]] = []
    counts_by_lang: dict[str, int] = {}

    for lg in ("kk", "ru", "en"):
        pool = [c for c in audit_claims if c["lang"] == lg]
        rnd = random.Random(seed + hash(lg) % 1000)
        if len(pool) > claims_per_lang:
            chosen = rnd.sample(pool, k=claims_per_lang)
        else:
            chosen = list(pool)
        counts_by_lang[lg] = len(chosen)
        selected.extend(chosen)

    csv_path = out_dir / "to_label.csv"
    fieldnames = [
        "claim_id",
        "example_id",
        "lang",
        "direction",
        "question",
        "claim_text",
        "cited_chunk_ids",
        "winning_evidence_window",
        "ver0_verdict",
        "ver0_score",
        "human_verdict",
        "reviewer",
        "note",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in selected:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    manifest = {
        "protocol": "ADR-0022 Stage 7 human faithfulness audit",
        "target_claims_per_lang": claims_per_lang,
        "exported_by_lang": counts_by_lang,
        "total_exported": len(selected),
        "human_verified": False,
        "human_labeled_count": 0,
        "csv_path": str(csv_path),
        "allowed_human_verdicts": ["supported", "contradicted", "not_enough_info"],
    }
    (out_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def build_multilingual_imatrix_corpus(
    sft_dir: Path,
    out_path: Path,
    *,
    per_lang: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Build balanced ``kk``+``ru``+``en`` SFT validation calibration text for ``llama-imatrix`` (GEN-Q)."""
    lines: list[str] = []
    counts: dict[str, int] = {}
    for lg in ("kk", "ru", "en"):
        files = sorted((sft_dir / lg).glob("validation-*.parquet"))
        if not files:
            counts[lg] = 0
            continue
        df = pl.read_parquet([str(p) for p in files])
        sub = df.with_columns(_r=pl.col("example_id").hash(seed=seed)).sort("_r").drop("_r").head(per_lang)
        counts[lg] = sub.height
        for row in sub.select("prompt", "answer").iter_rows():
            lines.append(f"{row[0]}\n\nAnswer: {row[1]}\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n---\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8", newline="\n")
    return {
        "out_path": str(out_path),
        "counts_by_lang": counts,
        "total_examples": sum(counts.values()),
        "total_chars": len(text),
        "sha256": _sha256_file(out_path),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["extractive", "openai", "transformers"], default="extractive")
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--eval-dir", type=Path, default=ROOT / "data" / "eval" / "qa")
    ap.add_argument("--max-per-lang", type=int, default=100)
    ap.add_argument("--distractor-unans", type=int, default=30)
    ap.add_argument("--verifier", choices=["none", "lexical", "ver0"], default="lexical")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "generation.json")
    ap.add_argument("--per-example", type=Path, default=ROOT / "reports" / "generation_per_example.jsonl")
    ap.add_argument("--audit-dir", type=Path, default=ROOT / "data" / "review" / "faithfulness_human")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    params = load_params()

    df, frozen_check = load_frozen_qa_eval(
        args.eval_dir, verify_frozen=True, max_per_lang=args.max_per_lang, seed=args.seed
    )
    items = build_eval_qa_contexts(
        df,
        num_distractors=int(params["sft"]["distractors"]),
        seed=args.seed,
        add_distractor_unanswerables_per_lang=args.distractor_unans,
    )

    generator: Generator
    if args.backend == "extractive":
        from trifact.generation.llm_client import ExtractiveBaselineGenerator

        generator = ExtractiveBaselineGenerator()
    elif args.backend == "openai":
        from trifact.generation.llm_client import OpenAICompatibleLLMClient

        generator = OpenAICompatibleLLMClient(base_url=args.base_url, model=args.model)
    else:
        from trifact.generation.llm_client import TransformersGenerator

        generator = TransformersGenerator(model_name_or_path=args.model, adapter_path=args.adapter)

    verifier: EvidenceVerifier | None = None
    if args.verifier == "lexical":
        from trifact.verification.nli import LexicalOverlapVerifier

        verifier = LexicalOverlapVerifier()
    elif args.verifier == "ver0":
        from trifact.verification.nli import NLIEvidenceVerifier

        verifier = NLIEvidenceVerifier(
            model_name_or_path=params["verification"]["nli_model"],
            revision=params["verification"]["nli_job"].get("revision"),
        )

    report, records, audit_claims = evaluate_generator(
        generator,
        items,
        verifier=verifier,
        n_boot=int(params["evaluation"]["bootstrap_samples"]),
        seed=args.seed,
    )
    report["frozen_eval"] = frozen_check
    if audit_claims:
        report["human_audit_packet"] = export_human_faithfulness_audit(
            audit_claims,
            args.audit_dir,
            claims_per_lang=int(params["verification"]["faithfulness_audit_claims"]),
            seed=args.seed,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    write_per_example_jsonl(args.per_example, records)
    log.info("Saved generation evaluation report to %s (%d examples)", args.out, len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
