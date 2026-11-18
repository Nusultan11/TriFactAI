"""Word tokens and sentence spans for kk/ru/en text (chunking, NER projection, silver NER)."""

from __future__ import annotations

import re

TOKEN = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]", re.UNICODE)
# sentence end: terminal punctuation (+ closing quotes/brackets) followed by space and an upper-case
# letter, digit or opening quote; or a blank line
SENT_END = re.compile(r"(?<=[.!?…])[»\"')\]]*\s+(?=[«\"(\[]?[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ0-9])|\n\s*\n|\n(?=\S)")


def tokenize(text: str, offset: int = 0) -> list[tuple[str, int, int]]:
    """(token, start, end) with character offsets into the original text."""
    return [(m.group(), m.start() + offset, m.end() + offset) for m in TOKEN.finditer(text)]


def sentences(text: str) -> list[tuple[int, int]]:
    """Character spans of sentences (heuristic splitter for kk/ru/en)."""
    out, start = [], 0
    for m in SENT_END.finditer(text):
        if text[start : m.start()].strip():
            out.append((start, m.start()))
        start = m.end()
    if text[start:].strip():
        out.append((start, len(text)))
    return out
