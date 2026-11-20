"""Stage build_docproc: the document-processing benchmark with reference text (ROADMAP 4.1) -> data/eval/docproc.

python -m trifact.data.build_docproc [--no-word]

Real kk/ru/en paragraphs and section titles of Wikipedia articles (`docproc.bench.source`) are laid out into
documents whose structure is known exactly:

| kind        | how it is made                                                                          |
|-------------|-----------------------------------------------------------------------------------------|
| pdf_text    | one column, headings, running header/footer and page numbers (PyMuPDF, own line layout) |
| pdf_columns | full-width title and lead, then two columns; half of the files write the content stream |
|             | in shuffled block order, so the text-layer order is not the reading order               |
| pdf_tables  | one column with ruled and borderless tables between paragraphs                          |
| pdf_scan    | pdf_text pages rasterized (200-300 dpi, blur, noise, rotation, JPEG) into image-only PDF |
| pdf_word    | DOCX laid out and exported to PDF by Microsoft Word (real producer; page of every        |
|             | paragraph read back from Word)                                                          |
| docx        | python-docx file re-saved by Word (rendered page breaks) or with explicit page breaks    |
| html        | article with h1-h3, lists, a table, inside navigation/header/footer/script boilerplate   |
| txt         | UTF-8 / UTF-8-BOM / UTF-16 / cp1251, LF or CRLF, some hard-wrapped at 80 columns          |
| mixed       | kk, ru and en sections in one PDF whose second page is a scan                           |

Every document gets `gold/<doc_id>.json`: blocks in reading order (`heading` with level, `text`, `table` with
rows) and the page each block starts on. `big/` holds one long text PDF per language (throughput and RAM),
`robust/` broken, encrypted, oversized and hostile files with the expected outcome. `manifest.yaml` lists
the sha256 of every file; the set is frozen after the first build (ADR-0007).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import random
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from trifact.core.params import ROOT, load_params

log = logging.getLogger("trifact.build_docproc")
LANGS = ("kk", "ru", "en")
KINDS = ("pdf_text", "pdf_columns", "pdf_tables", "pdf_scan", "pdf_word", "docx", "html", "txt")
W, H, MARGIN, TOP, BOTTOM = 595.0, 842.0, 56.0, 64.0, 64.0
TABLE_HEAD = {
    "kk": ["Атауы", "Жылы", "Саны", "Ескерту"],
    "ru": ["Название", "Год", "Количество", "Примечание"],
    "en": ["Name", "Year", "Count", "Note"],
}
PAGE_WORD = {"kk": "Бет", "ru": "Стр.", "en": "Page"}


@dataclass
class GBlock:
    kind: str  # heading | text | table
    text: str = ""
    level: int = 0
    rows: list[list[str]] | None = None
    page: int | None = None


@dataclass
class Gold:
    doc_id: str
    kind: str
    lang: str
    file: str
    format: str
    producer: str
    n_pages: int | None = None
    pages_known: bool = True
    headings_known: bool = True
    stream_order: str = "reading"
    extra: dict[str, Any] = field(default_factory=dict)
    blocks: list[GBlock] = field(default_factory=list)


# --------------------------------------------------------------------------- source articles


@dataclass
class Article:
    lang: str
    title: str
    sections: list[tuple[str | None, list[str]]]


_HEAD_END = re.compile(r"[.!?:;,…»)\]]$")


def _clip(p: str, limit: int) -> str:
    if len(p) <= limit:
        return p
    cut = max(p.rfind(". ", 0, limit), p.rfind("! ", 0, limit), p.rfind("? ", 0, limit))
    return p[: cut + 1] if cut > limit // 3 else p[:limit].rsplit(" ", 1)[0]


def parse_article(lang: str, title: str, text: str) -> Article:
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for chunk in text.split("\n\n"):
        lines = [ln.strip() for ln in chunk.split("\n") if ln.strip()]
        if not lines:
            continue
        if len(lines[0]) < 80 and not _HEAD_END.search(lines[0]) and lines[0][:1].isupper():
            sections.append((lines[0], []))
            lines = lines[1:]
        for ln in lines:
            if len(ln) >= 60 and re.search(r"[.!?…]$", ln) and "|" not in ln:
                sections[-1][1].append(_clip(ln, 900))
    return Article(lang, title, [(h, ps) for h, ps in sections if ps])


def load_articles(lake: Path, source: str, lang: str, n: int, seed: int) -> list[Article]:
    # the two largest shards: a partition also holds a few documents re-labelled from other sources
    files = sorted((lake / source / lang).glob("part-*.parquet"), key=lambda f: -f.stat().st_size)[:2]
    df = (
        pl.scan_parquet(files)
        .select("doc_id", "title", "text", "n_chars", "lang")
        .filter(pl.col("n_chars").is_between(3000, 40000) & (pl.col("lang") == lang))
        .with_columns(h=pl.col("doc_id").hash(seed))
        .sort("h")
        .head(n * 4)
        .collect()
    )
    out = []
    for r in df.iter_rows(named=True):
        a = parse_article(lang, r["title"] or "", r["text"])
        if a.title and len(a.sections) >= 3 and sum(len(ps) for _, ps in a.sections) >= 6:
            out.append(a)
        if len(out) == n:
            break
    if len(out) < n:
        raise RuntimeError(f"{lang}: only {len(out)} usable articles, {n} needed")
    return out


def table_rows(a: Article, rng: random.Random) -> list[list[str]]:
    words = sorted({w for _, ps in a.sections for p in ps for w in re.findall(r"\b[^\W\d_]{4,14}\b", p)
                    if w[:1].isupper()})  # fmt: skip
    rng.shuffle(words)
    notes = [w.lower() for w in words[::-1]]
    ncols = rng.choice([3, 4])
    rows = [TABLE_HEAD[a.lang][:ncols]]
    for i in range(rng.randint(3, 7)):
        r = [words[i % max(len(words), 1)] if words else f"X{i}", str(rng.randint(1900, 2024)),
             f"{rng.randint(1, 99999):,}".replace(",", " ")]  # fmt: skip
        if ncols == 4:
            r.append(
                " ".join(
                    notes[(2 * i) % max(len(notes), 1) : (2 * i) % max(len(notes), 1) + rng.randint(1, 2)]
                )
            )
        rows.append(r)
    return rows


def doc_blocks(a: Article, rng: random.Random, tables: bool, max_blocks: int) -> list[GBlock]:
    blocks = [GBlock("heading", a.title, 1)]
    for h, ps in a.sections:
        if h:
            blocks.append(GBlock("heading", h, 2))
        for i, p in enumerate(ps):
            blocks.append(GBlock("text", p))
            if tables and i == 0 and rng.random() < 0.7:
                blocks.append(GBlock("table", rows=table_rows(a, rng)))
        if len(blocks) >= max_blocks:
            break
    return blocks


# --------------------------------------------------------------------------- PDF layout


def _font_file(cands: list[str]) -> str:
    for c in cands:
        if Path(c).is_file():
            return c
    raise FileNotFoundError(f"none of the fonts exists: {cands}")


class Fonts:
    def __init__(self, cfg: dict[str, list[str]], family: str):
        import pymupdf

        self.reg_file = _font_file(cfg[family])
        self.bold_file = _font_file(cfg[f"{family}_bold"])
        self.reg = pymupdf.Font(fontfile=self.reg_file)
        self.bold = pymupdf.Font(fontfile=self.bold_file)


def wrap(text: str, font: Any, size: float, width: float) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        cand = f"{cur} {w}" if cur else w
        if font.text_length(cand, fontsize=size) <= width or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


class PdfLayout:
    """Line layout of blocks over pages and columns; draw operations are kept per page and block so the
    content stream can be written in reading order or shuffled."""

    def __init__(self, fonts: Fonts, columns: int, body: float, rng: random.Random):
        self.f, self.cols, self.body, self.rng = fonts, columns, body, rng
        self.gutter = 18.0
        self.colw = (W - 2 * MARGIN - self.gutter * (columns - 1)) / columns
        self.pages: list[list[tuple[int, list[tuple[str, Any]]]]] = []
        self._new_page()

    def _new_page(self) -> None:
        self.pages.append([])
        self.col, self.y, self.col_top = 0, TOP, TOP

    def _next_col(self) -> None:
        self.col += 1
        if self.col >= self.cols:
            self._new_page()
        self.y = self.col_top

    def _x(self, full: bool) -> tuple[float, float]:
        if full:
            return MARGIN, W - 2 * MARGIN
        return MARGIN + self.col * (self.colw + self.gutter), self.colw

    def place(self, bi: int, b: GBlock, full: bool = False) -> None:
        if b.kind == "table":
            return self._table(bi, b)
        size = (
            {1: self.body + 7, 2: self.body + 2.5, 3: self.body + 1}.get(b.level, self.body)
            if b.kind == "heading"
            else self.body
        )
        font = self.f.bold if b.kind == "heading" else self.f.reg
        x, width = self._x(full)
        lines = wrap(b.text, font, size, width)
        lh = size * 1.3
        gap = size * (1.2 if b.kind == "heading" else 0.5)
        if self.y > self.col_top:
            self.y += gap
        if b.kind == "heading" and self.y + lh * (len(lines) + 2) > H - BOTTOM:  # no orphan headings
            self._next_col()
            x, width = self._x(full)
        ops: list[tuple[str, Any]] = []
        first_page = None
        for ln in lines:
            if self.y + lh > H - BOTTOM:
                if ops:
                    self.pages[-1].append((bi, ops))
                    ops = []
                self._next_col()
                x, width = self._x(full)
            ops.append(("text", (x, self.y + size, ln, font, size)))
            first_page = first_page or len(self.pages)
            self.y += lh
        if ops:
            self.pages[-1].append((bi, ops))
        b.page = first_page
        if full:
            self.y = self.col_top = self.y + self.body

    def _table(self, bi: int, b: GBlock) -> None:
        rows = b.rows or []
        size = self.body - 1
        rh = size * 1.6
        ncol = len(rows[0])
        widths = [max(self.f.reg.text_length(r[c], fontsize=size) for r in rows) + 14 for c in range(ncol)]
        x0, width = self._x(False)
        scale = min(1.0, width / sum(widths))
        widths = [w * scale for w in widths]
        need = rh * len(rows) + self.body
        if self.y + need > H - BOTTOM:
            self._next_col()
            x0, width = self._x(False)
        self.y += self.body * 0.6
        ruled = self.rng.random() < 0.5
        ops: list[tuple[str, Any]] = []
        top = self.y
        for ri, r in enumerate(rows):
            x = x0
            font = self.f.bold if ri == 0 else self.f.reg
            for c, cell in enumerate(r):
                txt = cell
                while txt and font.text_length(txt, fontsize=size) > widths[c] - 8:
                    txt = txt[:-1]
                r[c] = txt.strip()
                ops.append(("text", (x + 4, self.y + size + 2, r[c], font, size)))
                x += widths[c]
            self.y += rh
            if ruled or ri == 0:
                ops.append(("line", (x0, self.y, x0 + sum(widths), self.y)))
        if ruled:
            ops.append(("line", (x0, top, x0 + sum(widths), top)))
            x = x0
            for wdt in [0.0, *widths]:
                x += wdt
                ops.append(("line", (x, top, x, self.y)))
        self.pages[-1].append((bi, ops))
        b.page = len(self.pages)
        self.y += self.body * 0.6

    def render(self, title: str, furniture: dict[str, Any] | None, shuffle: bool) -> Any:
        import pymupdf

        doc = pymupdf.open()
        n = len(self.pages)
        for pi, items in enumerate(self.pages, start=1):
            page = doc.new_page(width=W, height=H)
            order = list(items)
            if shuffle:
                self.rng.shuffle(order)
            fur: list[tuple[str, Any]] = []
            if furniture:
                fs = 8.0
                if furniture.get("header"):
                    fur.append(("text", (MARGIN, 36.0, title[:80], self.f.reg, fs)))
                fmt = furniture["footer"]
                label = {
                    "n": f"{pi}",
                    "dash": f"— {pi} —",
                    "of": f"{PAGE_WORD[furniture['lang']]} {pi} / {n}",
                }[fmt]
                fur.append(
                    (
                        "text",
                        (
                            W / 2 - self.f.reg.text_length(label, fontsize=fs) / 2,
                            H - 30,
                            label,
                            self.f.reg,
                            fs,
                        ),
                    )
                )
            seq = [(-1, fur)] + order if not shuffle else order + [(-1, fur)]
            for _, ops in seq:
                tw = pymupdf.TextWriter(page.rect)
                lines = []
                for kind, a in ops:
                    if kind == "text":
                        x, y, s, font, size = a
                        tw.append((x, y), s, font=font, fontsize=size)
                    else:
                        lines.append(a)
                tw.write_text(page)
                for x1, y1, x2, y2 in lines:
                    page.draw_line((x1, y1), (x2, y2), width=0.6)
        return doc


def layout_pdf(blocks: list[GBlock], fonts: Fonts, rng: random.Random, columns: int, max_pages: int,
               full_first: int = 0) -> tuple[PdfLayout, list[GBlock]]:  # fmt: skip
    body = rng.choice([10.0, 10.5, 11.0]) if columns == 1 else rng.choice([9.0, 9.5])
    lay = PdfLayout(fonts, columns, body, rng)
    kept: list[GBlock] = []
    for i, b in enumerate(blocks):
        snapshot = (len(lay.pages), [list(p) for p in lay.pages], lay.col, lay.y, lay.col_top)
        lay.place(i, b, full=i < full_first)
        if len(lay.pages) > max_pages:
            _, pages, lay.col, lay.y, lay.col_top = snapshot
            lay.pages = pages
            break
        kept.append(b)
    while lay.pages and not lay.pages[-1]:
        lay.pages.pop()
    return lay, kept


def rasterize(
    pdf_bytes: bytes, rng: random.Random, pages: set[int] | None = None
) -> tuple[bytes, dict[str, Any]]:
    """Image-only copy of the given pages (1-based; default all): scan-like noise, blur, skew, JPEG."""
    import numpy as np
    import pymupdf
    from PIL import Image, ImageFilter

    src = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    out = pymupdf.open()
    noisy = rng.random() < 0.5
    dpi = rng.choice([200, 240]) if noisy else 300
    params = {"dpi": dpi, "noisy": noisy}
    for i, page in enumerate(src, start=1):
        if pages is not None and i not in pages:
            out.insert_pdf(src, from_page=i - 1, to_page=i - 1)
            continue
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        angle = rng.uniform(-1.2, 1.2) if noisy else rng.uniform(-0.3, 0.3)
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=255, expand=False)
        if noisy:
            img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 0.9)))
        arr = np.asarray(img, dtype=np.int16)
        arr = arr + np.random.default_rng(rng.randint(0, 2**31)).normal(
            0, 12 if noisy else 4, arr.shape
        ).astype(np.int16)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "L")
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=55 if noisy else 85)
        p = out.new_page(width=page.rect.width, height=page.rect.height)
        p.insert_image(p.rect, stream=buf.getvalue())
    data = out.tobytes(garbage=3, deflate=True)
    return data, params


# --------------------------------------------------------------------------- DOCX / Word


def write_docx(blocks: list[GBlock], path: Path, rng: random.Random, page_breaks: bool) -> None:
    import docx

    d = docx.Document()
    d.core_properties.title = blocks[0].text
    page = 1
    for i, b in enumerate(blocks):
        if b.kind == "heading":
            if page_breaks and b.level == 2 and i > 2 and rng.random() < 0.5:
                d.add_page_break()
                page += 1
            d.add_heading(b.text, level=0 if b.level == 1 else b.level - 1)
        elif b.kind == "text":
            d.add_paragraph(b.text)
        else:
            rows = b.rows or []
            t = d.add_table(rows=len(rows), cols=len(rows[0]))
            t.style = "Table Grid"
            for r, row in enumerate(rows):
                for c, cell in enumerate(row):
                    t.cell(r, c).text = cell
        b.page = page
    d.save(str(path))


_WORD_PS = r"""
$ErrorActionPreference = 'Stop'
$jobs = Get-Content -Raw -Encoding UTF8 $args[0] | ConvertFrom-Json
$w = New-Object -ComObject Word.Application
$w.Visible = $false; $w.DisplayAlerts = 0
$res = @()
try {
  foreach ($j in $jobs) {
    $d = $w.Documents.Open($j.src, $false, $false)
    $d.Repaginate()
    $paras = @()
    foreach ($p in $d.Paragraphs) {
      $t = $p.Range.Text -replace "[\r\a\f\v\x07\x0b\x0c]", ''
      $paras += ,@([int]$p.Range.Information(3), [int]$p.Range.Information(12), $t.Trim().Length)
    }
    $d.SaveAs2([string]$j.out, [int]$j.fmt)
    $d.Close(0)
    $res += ,@{ out = $j.out; paras = $paras }
  }
} finally { $w.Quit() }
[IO.File]::WriteAllText($args[1], (ConvertTo-Json -Depth 5 -InputObject $res), [Text.Encoding]::UTF8)
"""


def word_convert(jobs: list[dict[str, Any]], tmp: Path) -> list[dict[str, Any]] | None:
    """Open each DOCX in Microsoft Word, read the page of every paragraph, save as DOCX (16) or PDF (17)."""
    if sys.platform != "win32":
        return None
    script, jin, jout = tmp / "word.ps1", tmp / "jobs.json", tmp / "pages.json"
    script.write_text(_WORD_PS, encoding="utf-8-sig")
    jin.write_text(json.dumps(jobs), encoding="utf-8")
    try:
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), str(jin),
                        str(jout)], check=True, capture_output=True, timeout=1800)  # fmt: skip
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("Microsoft Word is not usable (%s): pdf_word skipped, docx keeps explicit breaks", e)
        return None
    return list(json.loads(jout.read_text(encoding="utf-8-sig")))


def apply_word_pages(blocks: list[GBlock], paras: list[list[int]]) -> None:
    """Match Word paragraphs (page, in_table, text length) to blocks in order."""
    k = 0
    seq = [p for p in paras if p[2] > 0]
    for b in blocks:
        if b.kind == "table":
            while k < len(seq) and not seq[k][1]:
                k += 1
            if k < len(seq):
                b.page = seq[k][0]
            while k < len(seq) and seq[k][1]:
                k += 1
        else:
            while k < len(seq) and seq[k][1]:
                k += 1
            if k < len(seq):
                b.page = seq[k][0]
                k += 1


# --------------------------------------------------------------------------- HTML / TXT

_HTML = """<!DOCTYPE html>
<html lang="{lang}"><head><meta charset="{charset}"><title>{title}</title>
<style>body{{font-family:sans-serif}} .nav a{{margin:0 8px}}</style>
<script>window.dataLayer=window.dataLayer||[];function gtag(){{dataLayer.push(arguments)}}</script>
</head><body>
<header><div class="logo">Portal</div><nav class="nav"><a href="/">Home</a><a href="/news">News</a><a href="/about">About</a></nav></header>
<main><article>
{body}
</article></main>
<aside><h3>Related</h3><ul><li><a href="#">Link one</a></li><li><a href="#">Link two</a></li></ul></aside>
<footer><p>&copy; 2024 Portal. All rights reserved.</p></footer>
<script>gtag('config','X-1');</script>
</body></html>
"""


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_html(blocks: list[GBlock], lang: str, rng: random.Random) -> tuple[bytes, dict[str, Any]]:
    parts, out_blocks = [], []
    for b in blocks:
        if b.kind == "heading":
            parts.append(f"<h{b.level}>{_esc(b.text)}</h{b.level}>")
        elif b.kind == "table":
            rows = b.rows or []
            head = "".join(f"<th>{_esc(c)}</th>" for c in rows[0])
            body = "".join("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
            parts.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
        elif rng.random() < 0.15 and ". " in b.text:  # a paragraph shown as a list of its sentences
            items = [s.strip() for s in re.split(r"(?<=\.)\s+", b.text) if s.strip()]
            parts.append("<ul>" + "".join(f"<li>{_esc(s)}</li>" for s in items) + "</ul>")
            out_blocks.extend(GBlock("text", s) for s in items)
            continue
        else:
            words = b.text.split(" ")
            if len(words) > 6:  # inline markup inside a paragraph
                j = rng.randint(1, len(words) - 3)
                words[j] = f"<b>{_esc(words[j])}</b>"
                words[j + 1] = f'<a href="#r{j}">{_esc(words[j + 1])}</a>'
                parts.append("<p>" + " ".join(w if w.startswith("<") else _esc(w) for w in words) + "</p>")
            else:
                parts.append(f"<p>{_esc(b.text)}</p>")
        out_blocks.append(b)
    blocks[:] = out_blocks
    charset = "windows-1251" if lang == "ru" and rng.random() < 0.4 else "utf-8"
    html = _HTML.format(lang=lang, charset=charset, title=_esc(blocks[0].text), body="\n".join(parts))
    try:
        return html.encode(charset), {"charset": charset}
    except UnicodeEncodeError:  # text outside the legacy code page: the page stays UTF-8
        return html.replace(f'charset="{charset}"', 'charset="utf-8"').encode(), {"charset": "utf-8"}


def write_txt(blocks: list[GBlock], lang: str, rng: random.Random, i: int) -> tuple[bytes, dict[str, Any]]:
    enc = [
        "utf-8",
        "utf-8-sig",
        "utf-8",
        "utf-16" if lang == "kk" else "cp1251" if lang == "ru" else "ascii",
    ][i % 4]
    if enc == "ascii":
        enc = "utf-8"
    hard = i % 3 == 1
    paras = []
    for b in blocks:
        t = b.text if b.kind != "table" else "\n".join("\t".join(r) for r in b.rows or [])
        if hard and b.kind == "text":
            t = "\n".join(_wrap80(t))
        paras.append(t)
    eol = "\r\n" if i % 2 else "\n"
    text = "\n\n".join(paras).replace("\n", eol)
    try:
        data = text.encode(enc)
    except UnicodeEncodeError:  # text outside the legacy code page
        enc, data = "utf-8", text.encode()
    return data, {"encoding": enc, "hard_wrapped": hard, "eol": "crlf" if eol == "\r\n" else "lf"}


def _wrap80(t: str) -> list[str]:
    out, cur = [], ""
    for w in t.split():
        if cur and len(cur) + 1 + len(w) > 80:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    return [*out, cur] if cur else out


# --------------------------------------------------------------------------- robustness set


def robust_files(
    out: Path, sample_pdf: bytes, sample_docx: bytes, rng: random.Random
) -> list[dict[str, Any]]:
    import pymupdf

    out.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []

    def put(name: str, data: bytes, expect: str, reason: str | None = None) -> None:
        (out / name).write_bytes(data)
        cases.append({"file": f"robust/{name}", "expect": expect, "reason": reason})

    put("empty.pdf", b"", "error", "empty")
    put("garbage.pdf", rng.randbytes(50_000), "error", "corrupted")
    put("png_named.pdf", b"\x89PNG\r\n\x1a\n" + rng.randbytes(2000), "error", "corrupted")
    put("truncated.pdf", sample_pdf[: len(sample_pdf) // 2], "any")
    broken = bytearray(sample_pdf)
    i = broken.rfind(b"startxref")
    broken[i + 10 : i + 16] = b"999999"
    put("broken_xref.pdf", bytes(broken), "ok")
    d = pymupdf.open(stream=sample_pdf, filetype="pdf")
    put("encrypted.pdf", d.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner"),
        "error", "encrypted")  # fmt: skip
    put("owner_only.pdf", d.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner",
                                    permissions=pymupdf.PDF_PERM_PRINT), "ok")  # fmt: skip
    big = pymupdf.open()
    for k in range(2001):
        big.new_page(width=200, height=200).insert_text((20, 50), f"p{k}")
    put("too_many_pages.pdf", big.tobytes(garbage=1, deflate=True), "error", "too_many_pages")
    blank = pymupdf.open()
    blank.new_page()
    put("blank_page.pdf", blank.tobytes(), "error", "no_text")
    put("truncated.docx", sample_docx[: len(sample_docx) // 2], "error", "corrupted")
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as z:
        z.writestr("readme.txt", "not a word document")
    put("not_word.docx", zbuf.getvalue(), "error", "corrupted")
    put(
        "legacy.doc", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + rng.randbytes(4000), "error", "unsupported_format"
    )
    put("nested.html", ("<html><body>" + "<div>" * 20000 + "<p>Deep paragraph text survives nesting.</p>"
                        + "</div>" * 20000 + "</body></html>").encode(), "ok")  # fmt: skip
    put("binary.html", rng.randbytes(30_000), "error", "unsupported_format")
    put("binary.txt", rng.randbytes(30_000), "error", "unsupported_format")
    mixed = ("Қазақстан Республикасы — Орталық Азиядағы мемлекет. Россия — государство в Восточной Европе. "
             "English text follows in the same paragraph.\n\n") * 3  # fmt: skip
    put("mixed_lang.txt", mixed.encode(), "ok")
    line = "Абай Құнанбайұлы — ақын, ағартушы. Абай Кунанбаев — поэт. Abai Kunanbayev was a poet.\n"
    put("huge.txt", (line * (25_000_000 // len(line.encode()))).encode(), "ok")
    return cases


# --------------------------------------------------------------------------- build


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-word", action="store_true", help="skip Microsoft Word (pdf_word, Word-saved docx)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    params = load_params()
    cfg = params["docproc"]["bench"]
    lake = (ROOT / params["lake"]["root"]).resolve()
    out = lake / "eval" / "docproc"
    if (out / "manifest.yaml").is_file():
        log.info("%s is frozen (manifest.yaml exists); delete it deliberately to rebuild", out)
        return 0
    tmp_out = out.with_name("docproc.tmp")
    shutil.rmtree(tmp_out, ignore_errors=True)
    (tmp_out / "files").mkdir(parents=True)
    (tmp_out / "gold").mkdir()
    rng = random.Random(cfg["seed"])
    n = cfg["docs_per_kind"]
    golds: list[Gold] = []
    word_jobs: list[tuple[Gold, list[GBlock], dict[str, Any]]] = []
    tmp = Path(tempfile.mkdtemp(prefix="docproc_"))
    articles = {
        lang: load_articles(lake, cfg["source"], lang, n * len(KINDS) + 8, cfg["seed"]) for lang in LANGS
    }
    sample_pdf = sample_docx = b""

    def fonts() -> Fonts:
        return Fonts(cfg["fonts"], rng.choice(["sans", "serif"]))

    for lang in LANGS:
        pool = iter(articles[lang])
        for kind in KINDS:
            for k in range(n):
                a = next(pool)
                doc_id = f"{kind}-{lang}-{k:02d}"
                pages = rng.randint(cfg["min_pages"], cfg["max_pages"])
                if kind in ("pdf_text", "pdf_columns", "pdf_tables", "pdf_scan"):
                    cols = 2 if kind == "pdf_columns" else 1
                    blocks = doc_blocks(a, rng, tables=kind == "pdf_tables" or (kind == "pdf_scan" and k % 3 == 2),
                                        max_blocks=400)  # fmt: skip
                    if cols == 2:  # the first paragraph is a full-width lead under the title
                        lead = next(b for b in blocks if b.kind == "text")
                        blocks = [blocks[0], lead] + [b for b in blocks[1:] if b is not lead]
                    lay, kept = layout_pdf(
                        blocks, fonts(), rng, cols, pages, full_first=2 if cols == 2 else 0
                    )
                    shuffle = kind == "pdf_columns" and k % 2 == 1
                    fur = {
                        "header": rng.random() < 0.6,
                        "footer": rng.choice(["n", "dash", "of"]),
                        "lang": lang,
                    }
                    data = lay.render(a.title, fur if rng.random() < 0.8 or kind == "pdf_scan" else None, shuffle).tobytes(
                        garbage=3, deflate=True)  # fmt: skip
                    extra: dict[str, Any] = {"furniture": fur}
                    if kind == "pdf_scan":
                        data, sp = rasterize(data, rng)
                        extra.update(sp)
                    elif kind == "pdf_text" and not sample_pdf:
                        sample_pdf = data
                    g = Gold(doc_id, kind, lang, f"files/{doc_id}.pdf", "pdf", "pymupdf", len(lay.pages),
                             stream_order="shuffled" if shuffle else "reading", extra=extra, blocks=kept)  # fmt: skip
                    (tmp_out / g.file).write_bytes(data)
                    golds.append(g)
                elif kind in ("pdf_word", "docx"):
                    blocks = doc_blocks(a, rng, tables=True, max_blocks=10 + pages * 9)
                    src = tmp / f"{doc_id}.src.docx"
                    write_docx(blocks, src, rng, page_breaks=kind == "docx")
                    ext = "pdf" if kind == "pdf_word" else "docx"
                    g = Gold(
                        doc_id,
                        kind,
                        lang,
                        f"files/{doc_id}.{ext}",
                        ext,
                        "python-docx",
                        extra={},
                        blocks=blocks,
                    )
                    golds.append(g)
                    if kind == "docx":
                        shutil.copy(src, tmp_out / g.file)  # replaced by the Word-saved file when Word works
                        g.n_pages = max(b.page or 1 for b in blocks)
                        if not sample_docx:
                            sample_docx = src.read_bytes()
                    word_jobs.append((g, blocks, {"src": str(src), "out": str((tmp_out / g.file).resolve()),
                                                  "fmt": 17 if ext == "pdf" else 16}))  # fmt: skip
                elif kind == "html":
                    blocks = doc_blocks(a, rng, tables=True, max_blocks=10 + pages * 9)
                    data, extra = write_html(blocks, lang, rng)
                    g = Gold(doc_id, kind, lang, f"files/{doc_id}.html", "html", "template", None, False, extra=extra,
                             blocks=blocks)  # fmt: skip
                    (tmp_out / g.file).write_bytes(data)
                    golds.append(g)
                else:  # txt: no heading markup, headings are plain lines of text
                    blocks = doc_blocks(a, rng, tables=False, max_blocks=10 + pages * 9)
                    for b in blocks:
                        b.kind, b.level = ("text", 0) if b.kind == "heading" else (b.kind, b.level)
                    data, extra = write_txt(blocks, lang, rng, k)
                    g = Gold(doc_id, kind, lang, f"files/{doc_id}.txt", "txt", "template", None, False, False,
                             extra=extra, blocks=blocks)  # fmt: skip
                    (tmp_out / g.file).write_bytes(data)
                    golds.append(g)
    # mixed-language PDFs with a scanned second page
    for k in range(n):
        blocks = []
        for lang in LANGS:
            a = articles[lang][-8 + k]
            part = doc_blocks(a, rng, tables=False, max_blocks=7)
            blocks += part if lang == "kk" else part[1:]
        lay, kept = layout_pdf(blocks, fonts(), rng, 1, 3)
        data, sp = rasterize(lay.render(kept[0].text, None, False).tobytes(garbage=3, deflate=True), rng, {2})
        g = Gold(f"mixed-mul-{k:02d}", "mixed", "mul", f"files/mixed-mul-{k:02d}.pdf", "pdf", "pymupdf",
                 len(lay.pages), extra={"scan_pages": [2], **sp}, blocks=kept)  # fmt: skip
        (tmp_out / g.file).write_bytes(data)
        golds.append(g)
    # Word: PDF export and re-saved DOCX with rendered page breaks
    res = None if args.no_word else word_convert([j for _, _, j in word_jobs], tmp)
    for g, blocks, _job in word_jobs:
        if res is None:
            if g.kind == "pdf_word":
                golds.remove(g)
            continue
        r = next(r for r in res if Path(r["out"]).resolve() == (tmp_out / g.file).resolve())
        apply_word_pages(blocks, r["paras"])
        g.producer = "msword"
        g.n_pages = max((b.page or 1) for b in blocks)
    # long documents (throughput, RAM) and the robustness set
    (tmp_out / "big").mkdir()
    bigs = []
    for lang in LANGS:  # the articles repeated until the page budget is filled
        blocks = [
            b
            for _ in range(6)
            for a in articles[lang]
            for b in doc_blocks(a, rng, tables=False, max_blocks=400)
        ]
        lay, kept = layout_pdf(blocks, fonts(), rng, 1, cfg["big_doc_pages"])
        g = Gold(f"big-{lang}", "pdf_big", lang, f"big/big-{lang}.pdf", "pdf", "pymupdf", len(lay.pages),
                 blocks=kept)  # fmt: skip
        (tmp_out / g.file).write_bytes(lay.render(blocks[0].text, {"header": True, "footer": "n", "lang": lang},
                                                  False).tobytes(garbage=3, deflate=True))  # fmt: skip
        bigs.append(g)
    robust = robust_files(tmp_out / "robust", sample_pdf, sample_docx, rng)
    for g in golds + bigs:
        (tmp_out / "gold" / f"{g.doc_id}.json").write_text(json.dumps(asdict(g), ensure_ascii=False, indent=1),
                                                           encoding="utf-8")  # fmt: skip
    files = sorted(p for p in tmp_out.rglob("*") if p.is_file())
    manifest = {
        "suite": "docproc",
        "description": "Document-processing benchmark with reference text (ROADMAP 4.1); frozen (ADR-0007).",
        "source": f"{cfg['source']} (kk/ru/en), CC-BY-SA-3.0",
        "params": cfg,
        "docs": {k: sum(1 for g in golds if g.kind == k) for k in (*KINDS, "mixed")},
        "langs": {lang: sum(1 for g in golds if g.lang == lang) for lang in (*LANGS, "mul")},
        "pages": sum(g.n_pages or 0 for g in golds if g.format == "pdf"),
        "big": [g.file for g in bigs],
        "robust": robust,
        "word_available": res is not None,
        "sha256": {str(p.relative_to(tmp_out)).replace("\\", "/"): _sha(p) for p in files},
    }
    (tmp_out / "manifest.yaml").write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
                                           encoding="utf-8")  # fmt: skip
    shutil.rmtree(out, ignore_errors=True)
    tmp_out.replace(out)
    shutil.rmtree(tmp, ignore_errors=True)
    print(
        json.dumps({k: manifest[k] for k in ("docs", "langs", "pages", "word_available")}, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
