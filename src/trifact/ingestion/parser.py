"""Document parsing: PDF / DOCX / HTML / TXT bytes -> one text with exact page, section and table spans.

Every engine produces a list of `Block`s in reading order (heading, text, table) with their page; `assemble`
normalizes each block (`trifact.ingestion.normalize`), joins them with blank lines and records the character
span of every page, section and table in that text. Offsets therefore always point into `ParsedDoc.text`, the
text that is stored and chunked, so a citation `(char_start, char_end, page)` is exact.

Engines (architecture §7.1, ADR-0021): `pymupdf` (DOC-0, text layer + Tesseract on pages without one),
`docling` (DOC-1/DOC-2, layout and table models), `router` (DOC-3: PyMuPDF per page, Docling only for pages
with columns or tables). DOCX, HTML and TXT have native parsers (the Docling backends are the comparison).
A file that cannot be parsed raises `ParseError` with a reason code; `trifact.ingestion.sandbox` runs parsing
in a separate process so crashes, hangs and memory blow-ups of the native libraries never reach the worker.
"""

from __future__ import annotations

import bisect
import codecs
import io
import re
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from trifact.ingestion.normalize import normalize_text

FORMATS = ("pdf", "docx", "html", "txt")
REASONS = (
    "empty", "too_large", "unsupported_format", "corrupted", "encrypted", "too_many_pages",
    "too_long_text", "no_text", "timeout", "memory", "crash",
)  # fmt: skip


class ParseError(Exception):
    """A document that cannot be parsed; `reason` is one of `REASONS` (stored as `documents.error`)."""

    def __init__(self, reason: str, detail: str = ""):
        assert reason in REASONS, reason
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(slots=True)
class Block:
    kind: str  # heading | text | table
    text: str = ""
    page: int | None = None  # 1-based
    level: int = 0  # headings: 1 = top
    rows: list[list[str]] | None = None  # tables


@dataclass(slots=True)
class Table:
    page: int | None
    char_start: int
    char_end: int
    rows: list[list[str]]


@dataclass
class ParsedDoc:
    """Satisfies `trifact.core.interfaces.ParsedDocument`; `tables`, `levels` and `meta` are extras."""

    text: str
    pages: list[tuple[int, int, int]]  # (page_no, char_start, char_end), contiguous, every page listed
    sections: list[tuple[str, int, int]]  # (title, char_start, char_end), nested by heading level
    tables: list[Table] = field(default_factory=list)
    levels: list[int] = field(default_factory=list)  # heading level of each section
    meta: dict[str, Any] = field(default_factory=dict)

    def page_at(self, offset: int) -> int | None:
        if not self.pages:
            return None
        i = bisect.bisect_right([p[1] for p in self.pages], offset) - 1
        return self.pages[max(i, 0)][0]

    def section_at(self, offset: int) -> str | None:
        """Innermost section containing `offset`."""
        best: tuple[str, int, int] | None = None
        for s in self.sections:
            if s[1] <= offset < s[2] and (best is None or s[1] >= best[1]):
                best = s
        return best[0] if best else None


def render_table(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(c for c in r) for r in rows)


def _clean_rows(rows: list[list[str]]) -> list[list[str]]:
    out = [[normalize_text(c).replace("\n", " ") for c in r] for r in rows]
    return [r for r in out if any(r)]


def assemble(
    blocks: Iterable[Block], n_pages: int | None = None, meta: dict[str, Any] | None = None
) -> ParsedDoc:
    """Join normalized blocks with blank lines; record page, section and table spans."""
    parts: list[str] = []
    spans: list[tuple[Block, int, int]] = []
    pos = 0
    for b in blocks:
        if b.kind == "table":
            b.rows = _clean_rows(b.rows or [])
            t = render_table(b.rows)
        else:
            t = normalize_text(b.text)
            if b.kind == "heading":
                t = " ".join(t.split())
        if not t:
            continue
        if parts:
            parts.append("\n\n")
            pos += 2
        parts.append(t)
        spans.append((b, pos, pos + len(t)))
        pos += len(t)
    text = "".join(parts)

    pages: list[tuple[int, int, int]] = []
    if n_pages:
        first: dict[int, int] = {}
        for b, s, _ in spans:
            if b.page is not None:
                first.setdefault(b.page, s)
        # a page starts at its first block; an empty page is zero-length at the start of the next non-empty
        # one; starts are kept monotone so the spans tile the text
        starts = [0] * n_pages
        nxt = len(text)
        for i in reversed(range(n_pages)):
            nxt = min(first.get(i + 1, nxt), nxt)
            starts[i] = nxt
        starts[0] = 0
        for i in range(n_pages):
            pages.append((i + 1, starts[i], starts[i + 1] if i + 1 < n_pages else len(text)))

    heads = [(b, s, e) for b, s, e in spans if b.kind == "heading"]
    sections: list[tuple[str, int, int]] = []
    levels: list[int] = []
    for i, (b, s, e) in enumerate(heads):
        lvl = max(b.level, 1)
        end = len(text)
        for nb, ns, _ in heads[i + 1 :]:
            if max(nb.level, 1) <= lvl:
                end = ns - 2
                break
        sections.append((text[s:e], s, max(end, e)))
        levels.append(lvl)
    tables = [Table(b.page, s, e, b.rows or []) for b, s, e in spans if b.kind == "table"]
    m = dict(meta or {})
    m.setdefault("n_pages", n_pages)
    return ParsedDoc(text, pages, sections, tables, levels, m)


# --------------------------------------------------------------------------- format sniffing


def sniff(raw: bytes, filename: str = "") -> str:
    """Format by content (magic bytes), the extension only breaks ties between html and txt."""
    head = raw[:4096]
    if b"%PDF-" in raw[:1024]:
        return "pdf"
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                if "word/document.xml" in z.namelist():
                    return "docx"
        except zipfile.BadZipFile:
            pass
        if filename.lower().endswith(".docx"):
            raise ParseError("corrupted", "docx is not a valid zip package")
        raise ParseError("unsupported_format", "zip archive")
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        raise ParseError("unsupported_format", "OLE2 (legacy .doc/.xls)")
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext == "pdf":
        raise ParseError("corrupted", "no %PDF header")
    if ext == "docx":
        raise ParseError("corrupted", "docx is not a zip package")
    text = _decode(head, probe=True)
    if text is None:
        raise ParseError("unsupported_format", "binary content")
    low = text.lstrip("﻿ \t\r\n").lower()
    if (
        ext in ("html", "htm", "xhtml")
        or low.startswith(("<!doctype html", "<html"))
        or "<body" in low[:2048]
    ):
        return "html"
    return "txt"


_BOMS = (
    (b"\xef\xbb\xbf", "utf-8-sig"), (b"\xff\xfe\x00\x00", "utf-32"), (b"\x00\x00\xfe\xff", "utf-32"),
    (b"\xff\xfe", "utf-16"), (b"\xfe\xff", "utf-16"),
)  # fmt: skip


def _decode(raw: bytes, probe: bool = False) -> str | None:
    """Text of `raw`: BOM, then strict UTF-8, then charset detection; None for binary data."""
    for bom, enc in _BOMS:
        if raw.startswith(bom):
            return raw.decode(enc, errors="replace")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        if probe and e.start >= len(raw) - 4:  # a multi-byte character cut by the probe window
            text = raw[: e.start].decode("utf-8")
        else:
            from charset_normalizer import from_bytes

            best = from_bytes(raw).best()
            if best is None:
                return None
            text = str(best)
    sample = text[:20000]
    ctrl = sum(1 for c in sample if ord(c) < 32 and c not in "\n\r\t\f")
    if sample and ctrl / len(sample) > 0.05:
        return None
    return text


# --------------------------------------------------------------------------- TXT


_MD_HEAD = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")


def parse_txt(raw: bytes) -> ParsedDoc:
    text = _decode(raw)
    if text is None:
        raise ParseError("unsupported_format", "binary content")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks: list[Block] = []
    for para in re.split(r"\n\s*\n", text):
        lines = para.strip("\n").split("\n")
        m = _MD_HEAD.match(lines[0].strip()) if len(lines) == 1 else None
        if m:
            blocks.append(Block("heading", m.group(2), level=len(m.group(1))))
        elif para.strip():
            blocks.append(Block("text", para))
    return assemble(blocks, None, {"format": "txt", "engine": "native"})


# --------------------------------------------------------------------------- HTML

_SKIP_TAGS = {"script", "style", "noscript", "template", "head", "nav", "footer", "aside", "form", "button",
              "svg", "iframe", "select", "option", "header"}  # fmt: skip
_BLOCK_TAGS = {"p", "li", "dd", "dt", "blockquote", "pre", "figcaption", "caption", "address", "summary"}
_CONTAINER_TAGS = {
    "div",
    "section",
    "article",
    "main",
    "body",
    "ul",
    "ol",
    "dl",
    "figure",
    "details",
    "center",
}


_META_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?([\w-]+)""", re.IGNORECASE)


def _html_encoding(raw: bytes) -> str:
    """BOM, then <meta charset>, then strict UTF-8, then charset detection."""
    for bom, enc in _BOMS:
        if raw.startswith(bom):
            return "utf-8" if enc == "utf-8-sig" else enc
    m = _META_CHARSET.search(raw[:4096])
    if m:
        try:
            return codecs.lookup(m.group(1).decode("ascii")).name
        except LookupError:
            pass
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        from charset_normalizer import from_bytes

        best = from_bytes(raw[:200_000]).best()
        return best.encoding if best is not None else "utf-8"


def parse_html(raw: bytes) -> ParsedDoc:
    import lxml.html
    from lxml import etree

    enc = _html_encoding(raw)
    try:
        root = lxml.html.document_fromstring(raw, parser=lxml.html.HTMLParser(encoding=enc, huge_tree=True))
    except (etree.ParserError, ValueError) as e:
        if not raw.strip():
            raise ParseError("empty") from e
        raise ParseError("corrupted", str(e)) from e
    blocks: list[Block] = []
    # <header> of the page is boilerplate, but <header> inside <article> holds the title
    for el in root.iter("header"):
        if any(a.tag in ("article", "main") for a in el.iterancestors()):
            el.tag = "div"

    def text_of(el: Any) -> str:
        parts: list[str] = []

        def walk(e: Any) -> None:
            if not isinstance(e.tag, str) or e.tag in _SKIP_TAGS:
                if e.tail:
                    parts.append(e.tail)
                return
            if e.tag == "br":
                parts.append("\n")
            if e.text:
                parts.append(e.text)
            for c in e:
                walk(c)
            if e.tail and e is not el:
                parts.append(e.tail)

        walk(el)
        return "".join(parts) if el.tag == "pre" else " ".join("".join(parts).split())

    def visit(el: Any) -> None:
        tag = el.tag if isinstance(el.tag, str) else ""
        if not tag or tag in _SKIP_TAGS or el.get("hidden") is not None or el.get("aria-hidden") == "true":
            return
        if re.fullmatch(r"h[1-6]", tag):
            blocks.append(Block("heading", text_of(el), level=int(tag[1])))
            return
        if tag == "title":
            return
        if tag == "table":
            rows = []
            for tr in el.iter("tr"):
                if next(tr.iterancestors("table"), None) is not el:  # rows of a nested table
                    continue
                rows.append([text_of(c) for c in tr if isinstance(c.tag, str) and c.tag in ("td", "th")])
            cap = el.find("caption")
            if cap is not None and text_of(cap):
                blocks.append(Block("text", text_of(cap)))
            if rows:
                blocks.append(Block("table", rows=rows))
            return
        if tag in _BLOCK_TAGS and not any(isinstance(c.tag, str) and (c.tag in _BLOCK_TAGS or c.tag == "table"
                                                                       or re.fullmatch(r"h[1-6]", c.tag))
                                           for c in el.iterdescendants()):  # fmt: skip
            blocks.append(Block("text", text_of(el)))
            return
        # mixed container: loose text runs become paragraphs, child blocks are visited in order
        loose: list[str] = [el.text or ""] if tag in _CONTAINER_TAGS or tag in _BLOCK_TAGS else []
        for c in el:
            ctag = c.tag if isinstance(c.tag, str) else ""
            if (
                ctag in _BLOCK_TAGS
                or ctag in _CONTAINER_TAGS
                or ctag == "table"
                or re.fullmatch(r"h[1-6]", ctag)
            ):
                if "".join(loose).strip():
                    blocks.append(Block("text", " ".join("".join(loose).split())))
                loose = []
                visit(c)
                loose.append(c.tail or "")
            elif ctag in _SKIP_TAGS:
                loose.append(c.tail or "")
            elif ctag:
                loose.append(text_of(c) + (c.tail or ""))
            else:
                loose.append(c.tail or "")
        if "".join(loose).strip():
            blocks.append(Block("text", " ".join("".join(loose).split())))

    body = root.find("body")
    try:
        visit(body if body is not None else root)
    except RecursionError:  # pathological nesting: fall back to visible text lines
        for el in list(root.iter(*_SKIP_TAGS)):
            if el.getparent() is not None:
                el.drop_tree()
        blocks = [Block("text", ln) for ln in (body if body is not None else root).text_content().split("\n")]
    title = root.findtext(".//title")
    return assemble(
        blocks, None, {"format": "html", "engine": "native", "title": (title or "").strip() or None}
    )


# --------------------------------------------------------------------------- DOCX

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_HEADING_STYLE = re.compile(r"^(?:heading|заголовок|título|titre|überschrift)\s*(\d)$", re.IGNORECASE)


def parse_docx(raw: bytes, max_xml_bytes: int = 500 << 20) -> ParsedDoc:
    """Body paragraphs and tables in order; headings from paragraph styles (Title / Heading N or outline
    level); pages from explicit and last-rendered page breaks (Word's own pagination at last save)."""
    import docx
    from docx.table import Table as DocxTable
    from docx.text.paragraph import Paragraph

    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            info = z.getinfo("word/document.xml")
            if info.file_size > max_xml_bytes:  # zip bomb guard
                raise ParseError("too_large", f"document.xml {info.file_size} bytes")
        d = docx.Document(io.BytesIO(raw))
    except ParseError:
        raise
    except Exception as e:
        raise ParseError("corrupted", f"{type(e).__name__}: {e}") from e

    blocks: list[Block] = []
    page = 1
    seen_break = False

    def para_level(p: Any) -> int:
        name = (p.style.name if p.style is not None else "") or ""
        if name.lower() == "title":
            return 1
        m = _HEADING_STYLE.match(name.strip())
        if m:
            return int(m.group(1)) + (0 if not has_title else 1)
        ppr = p._p.pPr
        ol = ppr.find(f"{_W}outlineLvl") if ppr is not None else None
        if ol is not None:
            return int(ol.get(f"{_W}val")) + 1 + (0 if not has_title else 1)
        return 0

    has_title = any(
        (p.style is not None and (p.style.name or "").lower() == "title") for p in d.paragraphs[:20]
    )
    # Word writes both an explicit break and, at the next save, a rendered break for the same page boundary:
    # rendered breaks are used when the file has them, explicit ones otherwise
    body = d.element.body
    rendered = next(body.iter(f"{_W}lastRenderedPageBreak"), None) is not None

    def is_break(el: Any) -> bool:
        if rendered:
            return bool(el.tag == f"{_W}lastRenderedPageBreak")
        return bool(el.tag == f"{_W}br" and el.get(f"{_W}type") == "page")

    def breaks(p: Any) -> tuple[int, int]:
        """(breaks before the first text of the paragraph, breaks after it)."""
        before = after = 0
        seen_text = False
        ppr = p._p.pPr
        if not rendered and ppr is not None and ppr.find(f"{_W}pageBreakBefore") is not None:
            before += 1
        for el in p._p.iter(f"{_W}t", f"{_W}br", f"{_W}lastRenderedPageBreak"):
            if el.tag == f"{_W}t":
                seen_text = seen_text or bool((el.text or "").strip())
            elif is_break(el):
                if seen_text:
                    after += 1
                else:
                    before += 1
        return before, after

    for item in d.iter_inner_content():
        if isinstance(item, Paragraph):
            before, after = breaks(item)
            if before and blocks:
                page += before
            seen_break = seen_break or bool(before or after)
            lvl = para_level(item)
            if item.text.strip():
                blocks.append(Block("heading" if lvl else "text", item.text, page, lvl))
            page += after
        elif isinstance(item, DocxTable):
            rows = []
            for r in item.rows:
                cells, prev = [], None
                for c in r.cells:
                    if c._tc is prev:  # horizontally merged cell repeats
                        continue
                    prev = c._tc
                    cells.append(c.text)
                rows.append(cells)
            blocks.append(Block("table", page=page, rows=rows))
    n_pages = max((b.page or 1 for b in blocks), default=1) if seen_break else None
    if not n_pages:
        for b in blocks:
            b.page = None
    title = (d.core_properties.title or "").strip() or None
    return assemble(blocks, n_pages, {"format": "docx", "engine": "native", "title": title})
