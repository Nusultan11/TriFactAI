"""Multilingual (KK/RU/EN) atomic claim splitter and citation attachment (ADR-0008, ADR-0022).

Splits a raw generated answer into sentence-level claims with validated ``[chunk_id]`` citations.
Rejects any citation ID outside the supplied evidence context (ADR-0023).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from trifact.core.schemas import Citation, Claim, ScoredChunk
from trifact.generation.prompt import extract_citation_ids, is_refusal, strip_thinking_tags

# Abbreviations that should not trigger sentence splits (en / ru / kk)
_ABBREV_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|Inc|Ltd|vs|etc|e\.g|i\.e|Fig|No|Vol|т\.д|т\.п|т\.е|гг|г|ж|жж|қ|б|с)\.$",
    re.IGNORECASE,
)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[^\s])")
_BRACKET_STRIP_RE = re.compile(r"\s*\[[^\[\]\n]{1,64}\]")
_WS_RE = re.compile(r"\s+")


@dataclass
class SplitClaim:
    """Intermediate claim representation before NLI verification."""

    text: str
    raw_citation_ids: list[str]
    valid_chunk_ids: list[str]
    invalid_citation_ids: list[str]


def split_sentences(text: str) -> list[str]:
    """Split multilingual (KK/RU/EN) text into clean sentences, respecting common abbreviations."""
    if not text or not text.strip():
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out: list[str] = []
    for ln in lines:
        # Remove leading list bullets while preserving factual text
        ln = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", ln).strip()
        if not ln:
            continue
        parts = _SENT_SPLIT_RE.split(ln)
        buf = ""
        for part in parts:
            p = part.strip()
            if not p:
                continue
            if buf:
                buf = f"{buf} {p}"
            else:
                buf = p
            if not _ABBREV_RE.search(buf):
                out.append(buf)
                buf = ""
        if buf:
            out.append(buf)
    return out


def make_sentence_windows(text: str, window_size: int = 3, stride: int = 1) -> list[str]:
    """Build sliding ``window_size``-sentence windows over an evidence passage (ADR-0022)."""
    sents = split_sentences(text)
    if not sents:
        return [text.strip()] if text.strip() else []
    if len(sents) <= window_size:
        return [" ".join(sents)]
    windows: list[str] = []
    for i in range(0, len(sents) - window_size + 1, max(1, stride)):
        windows.append(" ".join(sents[i : i + window_size]))
    if (len(sents) - window_size) % max(1, stride) != 0:
        windows.append(" ".join(sents[-window_size:]))
    return windows


def split_claims(
    raw_answer: str,
    context: Sequence[ScoredChunk],
    *,
    id_to_chunk_id: Mapping[str, str] | None = None,
) -> list[SplitClaim]:
    """Split ``raw_answer`` into atomic claims with validated supplied ``chunk_id``s.

    Returns an empty list for canonical refusals (tracked separately via refusal P/R per ADR-0022).
    """
    cleaned, _ = strip_thinking_tags(raw_answer)
    if not cleaned or is_refusal(cleaned):
        return []

    chunk_map: dict[str, ScoredChunk] = {sc.chunk.chunk_id: sc for sc in context}
    lookup: dict[str, str] = {cid: cid for cid in chunk_map}
    if id_to_chunk_id:
        for k, v in id_to_chunk_id.items():
            if v in chunk_map:
                lookup[str(k)] = v

    claims: list[SplitClaim] = []
    for sent in split_sentences(cleaned):
        raw_ids = extract_citation_ids(sent)
        valid_ids: list[str] = []
        valid_seen: set[str] = set()
        invalid_ids: list[str] = []
        for rid in raw_ids:
            mapped = lookup.get(rid)
            if mapped is not None:
                if mapped not in valid_seen:
                    valid_seen.add(mapped)
                    valid_ids.append(mapped)
            else:
                invalid_ids.append(rid)

        # Strip citation markers from hypothesis text before NLI scoring
        claim_text = _WS_RE.sub(" ", _BRACKET_STRIP_RE.sub("", sent)).strip()
        claim_text = re.sub(r"\s+([.,;:!?])", r"\1", claim_text).strip()
        if not claim_text:
            continue
        claims.append(
            SplitClaim(
                text=claim_text,
                raw_citation_ids=raw_ids,
                valid_chunk_ids=valid_ids,
                invalid_citation_ids=invalid_ids,
            )
        )
    return claims


def build_unverified_claims(
    raw_answer: str,
    context: Sequence[ScoredChunk],
    *,
    id_to_chunk_id: Mapping[str, str] | None = None,
) -> list[Claim]:
    """Convert ``raw_answer`` into schema ``Claim`` objects with populated ``Citation``s (pre-NLI)."""
    chunk_map: dict[str, ScoredChunk] = {sc.chunk.chunk_id: sc for sc in context}
    out: list[Claim] = []
    for sc in split_claims(raw_answer, context, id_to_chunk_id=id_to_chunk_id):
        citations = [
            Citation(
                chunk_id=cid,
                document_id=chunk_map[cid].chunk.document_id,
                page=chunk_map[cid].chunk.page,
                quote=chunk_map[cid].chunk.text[:240],
            )
            for cid in sc.valid_chunk_ids
            if cid in chunk_map
        ]
        out.append(Claim(text=sc.text, citations=citations, verdict=None, nli_score=None))
    return out
