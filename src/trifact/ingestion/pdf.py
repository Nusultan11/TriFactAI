"""PyMuPDF fast path for PDF (DOC-0) and the page features of the DOC-3 router.

Per page: text blocks of the text layer (`page.get_text("dict")`), or of a Tesseract OCR text page when the
layer is missing or broken (`trifact.ingestion.ocr`). Reading order comes from a recursive XY-cut over block
boxes: horizontal bands are merged while their union still has a vertical gutter, so the paragraphs of two
columns are read column by column and full-width titles split the bands. Headings are short blocks set in a
larger (or bold) font than the body text, levels by font size rank. Running headers, footers and page
numbers (lines in the top/bottom band repeated on many pages) are dropped. Lines of a block are joined with
spaces and words hyphenated at line ends are rejoined.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from trifact.ingestion.parser import Block, ParseError

_BAD = re.compile("[�-]")
_PAGE_NO = re.compile(
    r"^[\W_]*(?:page|стр\.?|страница|бет|б\.)?\s*\d{1,4}(?:\s*(?:/|of|из|-)\s*\d{1,4})?[\W_]*$", re.IGNORECASE
)


@dataclass(slots=True)
class Line:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    size: float
    bold: bool


@dataclass(slots=True)
class TBlock:
    x0: float
    y0: float
    x1: float
    y1: float
    lines: list[Line]
    kind: str = "text"
    level: int = 0

    @property
    def chars(self) -> int:
        return sum(len(ln.text) for ln in self.lines)

    @property
    def size(self) -> float:
        n = self.chars or 1
        return sum(ln.size * len(ln.text) for ln in self.lines) / n

    @property
    def bold(self) -> bool:
        return all(ln.bold for ln in self.lines if ln.text.strip())


@dataclass
class PageInfo:
    page: int
    width: float
    height: float
    blocks: list[TBlock] = field(default_factory=list)
    n_chars: int = 0
    bad_share: float = 0.0
    image_cover: float = 0.0
    columns: bool = False
    table: bool = False
    ocr: bool = False

    @property
    def complex(self) -> bool:
        return self.columns or self.table


def _blocks_of(d: dict[str, Any]) -> list[TBlock]:
    out = []
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        lines = []
        for ln in b["lines"]:
            spans = [s for s in ln["spans"] if s["text"]]
            if not spans:
                continue
            text = "".join(s["text"] for s in spans)
            if not text.strip():
                continue
            n = sum(len(s["text"].strip()) for s in spans) or 1
            size = sum(s["size"] * len(s["text"].strip()) for s in spans) / n
            bold = all((s["flags"] & 16) or "bold" in s["font"].lower() for s in spans if s["text"].strip())
            x0, y0, x1, y1 = ln["bbox"]
            lines.append(Line(x0, y0, x1, y1, text, round(size, 1), bool(bold)))
        if lines:
            out.append(TBlock(min(ln.x0 for ln in lines), min(ln.y0 for ln in lines),
                              max(ln.x1 for ln in lines), max(ln.y1 for ln in lines), lines))  # fmt: skip
    return out


def _split_wide_blocks(blocks: list[TBlock]) -> list[TBlock]:
    """MuPDF sometimes puts the lines of two columns into one block: split such a block into line groups by
    x-overlap so the XY-cut can see the gutter."""
    out = []
    for b in blocks:
        xs = sorted({(round(ln.x0), round(ln.x1)) for ln in b.lines})
        if len(b.lines) < 2 or len(xs) < 2:
            out.append(b)
            continue
        groups: list[list[Line]] = []
        for ln in sorted(b.lines, key=lambda ln: ln.x0):
            for g in groups:
                if min(ln.x1, max(x.x1 for x in g)) - max(ln.x0, min(x.x0 for x in g)) > 0:
                    g.append(ln)
                    break
            else:
                groups.append([ln])
        if len(groups) == 1:
            out.append(b)
            continue
        for g in groups:
            g.sort(key=lambda ln: (ln.y0, ln.x0))
            out.append(TBlock(min(x.x0 for x in g), min(x.y0 for x in g), max(x.x1 for x in g),
                              max(x.y1 for x in g), g))  # fmt: skip
    return out


def _gaps(iv: list[tuple[float, float]], min_gap: float) -> list[tuple[float, float]]:
    """Uncovered intervals (>= min_gap) between the union of intervals."""
    iv = sorted(iv)
    out, end = [], iv[0][1]
    for a, b in iv[1:]:
        if a - end >= min_gap:
            out.append((end, a))
        end = max(end, b)
    return out


def xy_cut(blocks: list[TBlock], min_vgap: float, depth: int = 0) -> tuple[list[TBlock], bool]:
    """Reading order of blocks; the flag says whether a column split was made."""
    if len(blocks) <= 1 or depth > 30:
        return sorted(blocks, key=lambda b: (b.y0, b.x0)), False
    vg = _gaps([(b.x0, b.x1) for b in blocks], min_vgap)
    if vg:
        a, c = max(vg, key=lambda g: g[1] - g[0])
        cut = (a + c) / 2
        left = [b for b in blocks if b.x1 <= cut]
        right = [b for b in blocks if b.x1 > cut]
        lo, lc = xy_cut(left, min_vgap, depth + 1)
        ro, rc = xy_cut(right, min_vgap, depth + 1)
        cols = (sum(b.chars for b in left) > 200 and sum(b.chars for b in right) > 200) or lc or rc
        return lo + ro, cols
    # horizontal bands; consecutive bands are merged while their union has a vertical gutter
    order = sorted(blocks, key=lambda b: b.y0)
    bands: list[list[TBlock]] = [[order[0]]]
    bottom = order[0].y1
    for b in order[1:]:
        if b.y0 >= bottom - 1:
            bands.append([b])
        else:
            bands[-1].append(b)
        bottom = max(bottom, b.y1)
    if len(bands) == 1:
        return sorted(blocks, key=lambda b: (round(b.y0), b.x0)), False
    groups: list[list[TBlock]] = [bands[0]]
    for band in bands[1:]:
        merged = groups[-1] + band
        if _gaps([(b.x0, b.x1) for b in merged], min_vgap):
            groups[-1] = merged
        else:
            groups.append(band)
    out: list[TBlock] = []
    cols = False
    for g in groups:
        if len(groups) == 1:  # no band split possible: plain top-down order
            return sorted(g, key=lambda b: (round(b.y0), b.x0)), False
        o, c = xy_cut(g, min_vgap, depth + 1)
        out += o
        cols = cols or c
    return out, cols


def _table_signal(page: Any, lines: list[Line], min_rows: int) -> bool:
    """Ruled grid (>= 3 horizontal and >= 2 vertical rules) or >= min_rows rows of >= 3 separate cells."""
    h = v = 0
    try:
        for d in page.get_drawings():
            for it in d["items"]:
                if it[0] == "l":
                    p1, p2 = it[1], it[2]
                    if abs(p1.y - p2.y) < 1 and abs(p1.x - p2.x) > 20:
                        h += 1
                    elif abs(p1.x - p2.x) < 1 and abs(p1.y - p2.y) > 8:
                        v += 1
                elif it[0] == "re":
                    r = it[1]
                    if r.height < 2 and r.width > 20:
                        h += 1
                    elif r.width < 2 and r.height > 8:
                        v += 1
                    elif r.width > 20 and r.height > 8 and d.get("fill") is None:
                        h += 2
                        v += 2
    except Exception:  # noqa: BLE001, S110 - broken vector content must not stop text extraction
        pass
    if h >= 3 and v >= 2:
        return True
    rows: dict[int, list[Line]] = {}
    for ln in lines:
        rows.setdefault(round((ln.y0 + ln.y1) / 4), []).append(ln)  # 2-pt buckets of the line centre
    cell_rows = 0
    for r in rows.values():
        r.sort(key=lambda ln: ln.x0)
        segs = 1 + sum(1 for a, b in itertools.pairwise(r) if b.x0 - a.x1 > 8)
        if segs >= 3 and max(len(ln.text) for ln in r) < 60:
            cell_rows += 1
    return cell_rows >= min_rows


def text_flags() -> int:
    import pymupdf

    return int(pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP)


def analyze_page(page: Any, cfg: dict[str, Any], textpage: Any = None, table_min_rows: int = 3) -> PageInfo:
    FLAGS = text_flags()
    w, h = page.rect.width, page.rect.height
    d = page.get_text("dict", flags=FLAGS, textpage=textpage)
    blocks = _split_wide_blocks(_blocks_of(d))
    info = PageInfo(page.number + 1, w, h)
    text = "".join(ln.text for b in blocks for ln in b.lines)
    info.n_chars = len(text.strip())
    info.bad_share = len(_BAD.findall(text)) / max(len(text), 1)
    try:
        area = sum(abs(r["bbox"][2] - r["bbox"][0]) * abs(r["bbox"][3] - r["bbox"][1])
                   for r in page.get_image_info())  # fmt: skip
        info.image_cover = min(area / max(w * h, 1), 1.0)
    except Exception:  # noqa: BLE001
        info.image_cover = 0.0
    info.blocks, info.columns = xy_cut(blocks, max(8.0, 0.015 * w)) if blocks else ([], False)
    info.table = (
        _table_signal(page, [ln for b in blocks for ln in b.lines], table_min_rows) if blocks else False
    )
    return info


def needs_ocr(info: PageInfo, cfg: dict[str, Any]) -> bool:
    return info.n_chars < cfg["min_page_chars"] or info.bad_share > cfg["max_bad_char_share"]


def _furniture(pages: list[PageInfo], cfg: dict[str, Any]) -> set[tuple[int, int]]:
    """(page, block index) of running headers/footers and page numbers."""
    band, out = cfg["furniture_band"], set()
    keyed: list[tuple[int, int, str]] = []
    for p in pages:
        for i, b in enumerate(p.blocks):
            if b.y1 > p.height * band and b.y0 < p.height * (1 - band):
                continue
            txt = " ".join(ln.text for ln in b.lines).strip()
            if _PAGE_NO.match(txt):
                out.add((p.page, i))
            elif len(txt) <= 150:
                keyed.append((p.page, i, re.sub(r"\d+", "#", txt.lower())))
    if len(pages) >= 3:
        on_pages: dict[str, set[int]] = {}
        for pg, _, k in keyed:
            on_pages.setdefault(k, set()).add(pg)
        for pg, i, k in keyed:
            if len(on_pages[k]) >= max(3, cfg["furniture_min_share"] * len(pages)):
                out.add((pg, i))
    return out


def _join_lines(lines: list[Line]) -> str:
    out = ""
    for ln in lines:
        t = ln.text.strip()
        if not out:
            out = t
        elif out.endswith(("-", "­")) and len(out) > 1 and out[-2].isalpha() and t[:1].islower():
            out = out[:-1] + t
        else:
            out += " " + t
    return out.replace("­", "")


def _headings(pages: list[PageInfo], cfg: dict[str, Any]) -> None:
    sizes: Counter[float] = Counter()
    bold_chars = all_chars = 0
    for p in pages:
        for b in p.blocks:
            for ln in b.lines:
                n = len(ln.text.strip())
                sizes[round(ln.size * 2) / 2] += n
                all_chars += n
                bold_chars += n if ln.bold else 0
    if not sizes:
        return
    body = sizes.most_common(1)[0][0]
    body_bold = bold_chars > all_chars / 2
    cands: list[tuple[TBlock, float]] = []
    for p in pages:
        for b in p.blocks:
            txt = _join_lines(b.lines)
            if (
                not txt
                or len(txt) > cfg["heading_max_chars"]
                or len(b.lines) > 3
                or not any(c.isalpha() for c in txt)
            ):
                continue
            # bold table header cells have short neighbours in their row; a heading in a column does not
            if any(
                o is not b and min(o.y1, b.y1) - max(o.y0, b.y0) > 1 and len(o.lines) <= 2 and o.chars < 60
                for o in p.blocks
            ):
                continue
            big = b.size >= body * cfg["heading_size_ratio"]
            boldh = b.bold and not body_bold and len(txt) <= 120 and not txt.endswith((".", ",", ";", ":"))
            if big or boldh:
                cands.append((b, b.size if big else 0.0))
    # heading sizes within 1 pt are one level (OCR and font rounding jitter)
    groups: list[float] = []
    for s in sorted({s for _, s in cands if s > 0}, reverse=True):
        if not groups or groups[-1] - s > 1.0:
            groups.append(s)
    for b, s in cands:
        b.kind = "heading"
        b.level = next(i for i, g in enumerate(groups, 1) if g - s <= 1.0) if s > 0 else len(groups) + 1


def extract(doc: Any, cfg: dict[str, Any], ocr: Any = None, pages: list[int] | None = None,
            table_min_rows: int = 3) -> tuple[list[PageInfo], list[int]]:  # fmt: skip
    """Analyze `pages` (0-based, default all); pages without a usable text layer are OCRed with `ocr`
    (a `trifact.ingestion.ocr.PageOcr`) when given. Returns page infos and the OCRed page numbers (1-based)."""
    infos, ocred = [], []
    for i in pages if pages is not None else range(doc.page_count):
        try:
            page = doc[i]
            info = analyze_page(page, cfg, table_min_rows=table_min_rows)
            if needs_ocr(info, cfg) and ocr is not None:
                tp = ocr.textpage(page, text_flags())
                info = analyze_page(page, cfg, textpage=tp, table_min_rows=table_min_rows)
                info.ocr = True
                ocred.append(i + 1)
        except ParseError:
            raise
        except Exception as e:  # noqa: BLE001 - one broken page is skipped, not the document
            info = PageInfo(i + 1, 0, 0)
            info.n_chars = -1
            info.bad_share = 1.0
            info.error = f"{type(e).__name__}: {e}"
        infos.append(info)
    return infos, ocred


def to_blocks(infos: list[PageInfo], cfg: dict[str, Any]) -> list[Block]:
    _headings(infos, cfg)
    furn = _furniture(infos, cfg)
    out = []
    for p in infos:
        for i, b in enumerate(p.blocks):
            if (p.page, i) in furn:
                continue
            out.append(Block(b.kind, _join_lines(b.lines), p.page, b.level))
    return out
