"""Tokenization, sentence splitting and BIO helpers shared by gold and silver NER (ADR-0006, ADR-0015)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from trifact.ingestion.segment import SENT_END, TOKEN, sentences, tokenize

__all__ = ["SENT_END", "TOKEN", "entities", "map_tags", "repair_bio", "sentences", "spans_to_bio", "tokenize"]


def spans_to_bio(tokens: Sequence[tuple[str, int, int]], spans: Iterable[tuple[int, int, str]]) -> list[str]:
    """BIO tags of tokens from character spans (start, end, label); spans are applied outermost first,
    a token already tagged is not overwritten (nested inner spans lose)."""
    tags = ["O"] * len(tokens)
    for s, e, label in sorted(spans, key=lambda x: (x[0], -(x[1] - x[0]))):
        idx = [i for i, (_, ts, te) in enumerate(tokens) if ts >= s and te <= e]
        if not idx or any(tags[i] != "O" for i in idx):
            continue
        tags[idx[0]] = f"B-{label}"
        for i in idx[1:]:
            tags[i] = f"I-{label}"
    return tags


def repair_bio(tags: Sequence[str]) -> tuple[list[str], int]:
    """`I-X` not continuing an `X` entity becomes `B-X`; returns tags and the number of repairs."""
    out, fixed, prev = [], 0, "O"
    for t in tags:
        if t.startswith("I-") and (prev == "O" or prev[2:] != t[2:]):
            t = "B-" + t[2:]
            fixed += 1
        out.append(t)
        prev = t
    return out, fixed


def map_tags(tags: Sequence[str], mapping: dict[str, str]) -> list[str]:
    """Source BIO tags -> core BIO tags (`mapping`: source type -> core label; unmapped -> O)."""
    out = []
    for t in tags:
        if t == "O":
            out.append("O")
            continue
        core = mapping.get(t[2:])
        out.append(f"{t[:2]}{core}" if core else "O")
    return repair_bio(out)[0]


def entities(tags: Sequence[str]) -> list[tuple[int, int, str]]:
    """(first token, last token + 1, label) of the BIO entities."""
    out, start, label = [], None, None
    for i, t in enumerate([*tags, "O"]):
        if start is not None and not (t.startswith("I-") and t[2:] == label):
            out.append((start, i, label))
            start = None
        if t.startswith("B-") or (t.startswith("I-") and start is None):
            start, label = i, t[2:]
    return out  # type: ignore[return-value]
