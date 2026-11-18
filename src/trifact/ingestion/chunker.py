"""Section- and sentence-aware chunking (~350 tokens, ~50 overlap) keeping page/section/offsets.

A chunk is a run of whole sentences of one paragraph block holding up to `max_tokens` tokens of the
retrieval tokenizer (XLM-R / BGE-M3 sentencepiece); the next chunk starts with the trailing sentences
of the previous one that fit into `overlap_tokens`. A sentence longer than `max_tokens` is cut into
token windows. Character offsets always point into the original text, so citations are exact.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from trifact.ingestion.segment import sentences


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    tokens: int


def _units(text: str, count: Callable[[list[str]], list[int]], max_tokens: int) -> list[Span]:
    """Sentences (with their token counts); over-long sentences cut at whitespace into windows."""
    sents = sentences(text)
    n_tok = count([text[a:b] for a, b in sents]) if sents else []
    out: list[Span] = []
    for (a, b), n in zip(sents, n_tok, strict=True):
        if n <= max_tokens:
            out.append(Span(a, b, n))
            continue
        # proportional character windows, cut at the nearest whitespace
        pieces = -(-n // max_tokens)
        step = (b - a) / pieces
        s = a
        for k in range(1, pieces + 1):
            e = b if k == pieces else text.rfind(" ", s + 1, int(a + k * step)) + 1 or int(a + k * step)
            if e > s:
                out.append(Span(s, e, -(-n // pieces)))
            s = e
    return out


def chunk_text(
    text: str, count: Callable[[list[str]], list[int]], max_tokens: int = 350, overlap_tokens: int = 50
) -> list[Span]:
    """Chunks of a document as (start, end, tokens) spans of `text`."""
    units = _units(text, count, max_tokens)
    chunks: list[Span] = []
    cur: list[Span] = []
    tok = 0
    for u in units:
        if cur and tok + u.tokens > max_tokens:
            chunks.append(Span(cur[0].start, cur[-1].end, tok))
            keep: list[Span] = []
            t = 0
            for v in reversed(cur):
                if t + v.tokens > overlap_tokens:
                    break
                keep.insert(0, v)
                t += v.tokens
            cur, tok = keep, t
        cur.append(u)
        tok += u.tokens
    if cur and (not chunks or cur[-1].end > chunks[-1].end):
        chunks.append(Span(cur[0].start, cur[-1].end, tok))
    return chunks


def hf_counter(tokenizer_json: str) -> Callable[[list[str]], list[int]]:
    """Token counter over a `tokenizers` tokenizer.json (no special tokens)."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer_json)

    def count(texts: Sequence[str]) -> list[int]:
        return [len(e.ids) for e in tok.encode_batch(list(texts), add_special_tokens=False)]

    return count
