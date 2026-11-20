"""Section- and sentence-aware chunking (~350 tokens, ~50 overlap) keeping page/section/offsets.

A chunk is a run of whole sentences of one paragraph block holding up to `max_tokens` tokens of the
retrieval tokenizer (XLM-R / BGE-M3 sentencepiece); the next chunk starts with the trailing sentences
of the previous one that fit into `overlap_tokens`. A sentence longer than `max_tokens` is cut into
token windows. Character offsets always point into the original text, so citations are exact.
"""

from __future__ import annotations

import bisect
import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from trifact.core.interfaces import ParsedDocument
from trifact.core.schemas import Chunk, Lang, SourceType
from trifact.ingestion.segment import sentences


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    tokens: int


def _units(
    text: str, count: Callable[[list[str]], list[int]], max_tokens: int, n_tok: list[int] | None = None
) -> list[Span]:
    """Sentences (with their token counts); over-long sentences cut at whitespace into windows."""
    sents = sentences(text)
    if n_tok is None:
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


def chunk_many(
    texts: Sequence[str],
    count: Callable[[list[str]], list[int]],
    max_tokens: int = 350,
    overlap_tokens: int = 50,
) -> list[list[Span]]:
    """`chunk_text` for many documents with one tokenizer call for all their sentences (batching matters:
    a per-document call costs ~1 ms of overhead)."""
    sents = [sentences(t) for t in texts]
    flat = [t[a:b] for t, ss in zip(texts, sents, strict=True) for a, b in ss]
    counts = count(flat) if flat else []
    out, i = [], 0
    for t, ss in zip(texts, sents, strict=True):
        out.append(_chunk(_units(t, count, max_tokens, counts[i : i + len(ss)]), max_tokens, overlap_tokens))
        i += len(ss)
    return out


def chunk_text(
    text: str, count: Callable[[list[str]], list[int]], max_tokens: int = 350, overlap_tokens: int = 50
) -> list[Span]:
    """Chunks of a document as (start, end, tokens) spans of `text`."""
    return _chunk(_units(text, count, max_tokens), max_tokens, overlap_tokens)


def _chunk(units: list[Span], max_tokens: int, overlap_tokens: int) -> list[Span]:
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


_SOURCE_TYPE = {
    "pdf": SourceType.UPLOADED_PDF,
    "docx": SourceType.UPLOADED_DOCX,
    "html": SourceType.UPLOADED_HTML,
    "txt": SourceType.UPLOADED_TXT,
}


class DocumentChunker:
    """`trifact.core.interfaces.Chunker` for parsed uploads: sections -> sentences -> token windows.

    With `sections=True` a chunk never crosses a heading: every region between two consecutive section starts
    is chunked on its own (the heading line opens its region, so it stays with its text). `page` is the page
    the chunk starts on, `section` the innermost section containing its start. All regions of a document are
    tokenized in one call.
    """

    def __init__(
        self,
        count: Callable[[list[str]], list[int]],
        detect: Callable[[str], tuple[Lang, float]],
        max_tokens: int = 350,
        overlap_tokens: int = 50,
        sections: bool = True,
    ):
        self.count, self.detect = count, detect
        self.max_tokens, self.overlap_tokens, self.sections = max_tokens, overlap_tokens, sections

    def regions(self, parsed: ParsedDocument) -> list[tuple[int, int]]:
        n = len(parsed.text)
        cuts = {0, n}
        if self.sections:
            cuts |= {s for _, s, _ in parsed.sections if 0 < s < n}
        b = sorted(cuts)
        return [(a, c) for a, c in itertools.pairwise(b) if parsed.text[a:c].strip()]

    def split(self, document_id: str, parsed: ParsedDocument) -> list[Chunk]:
        text = parsed.text
        regs = self.regions(parsed)
        spans = chunk_many([text[a:b] for a, b in regs], self.count, self.max_tokens, self.overlap_tokens)
        meta = getattr(parsed, "meta", {}) or {}
        stype = _SOURCE_TYPE.get(meta.get("format", ""), SourceType.UPLOADED_TXT)
        page_starts = [p[1] for p in parsed.pages]
        out: list[Chunk] = []
        for (a, _), ss in zip(regs, spans, strict=True):
            for s in ss:
                start, end = a + s.start, a + s.end
                # leading/trailing whitespace is not part of a citation
                while start < end and text[start].isspace():
                    start += 1
                while end > start and text[end - 1].isspace():
                    end -= 1
                if start == end:
                    continue
                piece = text[start:end]
                lang, _ = self.detect(piece)
                page = (
                    parsed.pages[max(bisect.bisect_right(page_starts, start) - 1, 0)][0]
                    if parsed.pages
                    else None
                )
                section = None
                best = -1
                for title, cs, ce in parsed.sections:
                    if cs <= start < max(ce, cs + 1) and cs > best:
                        section, best = title, cs
                out.append(
                    Chunk(
                        document_id=document_id,
                        chunk_id=f"{document_id}#{len(out)}",
                        ordinal=len(out),
                        language=lang,
                        page=page,
                        section=section,
                        char_start=start,
                        char_end=end,
                        text=piece,
                        source_type=stype,
                        token_count=s.tokens,
                    )
                )
        return out


def hf_counter(tokenizer_json: str) -> Callable[[list[str]], list[int]]:
    """Token counter over a `tokenizers` tokenizer.json (no special tokens)."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer_json)

    def count(texts: Sequence[str]) -> list[int]:
        return [len(e.ids) for e in tok.encode_batch(list(texts), add_special_tokens=False)]

    return count
