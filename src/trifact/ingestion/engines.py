"""`DocumentParser` implementations: format dispatch, limits and the PDF engines DOC-0 … DOC-3.

    parser = Parser(params["docproc"], ROOT)          # champion configuration from params.yaml
    parsed = parser.parse(raw_bytes, "report.pdf")    # ParsedDoc or ParseError(reason)

`pdf` selects the PDF engine: `pymupdf` (DOC-0), `docling` (DOC-1, + `docling_ocr` = DOC-2), `router`
(DOC-3: every page is analysed by PyMuPDF; pages with the signals of `docproc.router` (tables, optionally
columns, scans without a text layer) go to Docling in contiguous page ranges, scans with Docling's OCR, the
rest stays on the fast path). `ocr` is the fast-path OCR of `pymupdf` (and of the router when
`router.scans: ocr`), `docling_ocr` the OCR engine inside Docling; both default to `params.yaml`. `office`
selects the DOCX/HTML parser: `native` or `docling`. TXT is always native.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from trifact.ingestion import pdf as fast
from trifact.ingestion.parser import (
    Block,
    ParsedDoc,
    ParseError,
    assemble,
    parse_docx,
    parse_html,
    parse_txt,
    sniff,
)


def _ranges(pages: list[int]) -> list[tuple[int, int]]:
    """Contiguous 1-based ranges of sorted page numbers."""
    out: list[tuple[int, int]] = []
    for p in pages:
        if out and p == out[-1][1] + 1:
            out[-1] = (out[-1][0], p)
        else:
            out.append((p, p))
    return out


class Parser:
    def __init__(
        self,
        cfg: dict[str, Any],
        root: Path,
        pdf: str | None = None,
        office: str = "native",
        ocr: str | None = "params",
        docling_ocr: str | None = "params",
    ):
        self.cfg = cfg
        self.root = root
        self.pdf = pdf or cfg["engine"]
        if self.pdf not in ("pymupdf", "docling", "router"):
            raise ValueError(self.pdf)
        self.office = office
        self.ocr_name = cfg["ocr"]["engine"] if ocr == "params" else ocr
        self.docling_ocr = cfg["docling"]["ocr_engine"] if docling_ocr == "params" else docling_ocr
        self.limits = cfg["limits"]
        self._ocr: Any = None
        self._docling: dict[str | None, Any] = {}

    # lazy: Docling loads its models (~1-2 GB) only when a document needs them
    def _page_ocr(self) -> Any:
        if self._ocr is None and self.ocr_name == "tesseract":
            from trifact.ingestion.ocr import PageOcr

            self._ocr = PageOcr(self.cfg["ocr"], self.root)
        return self._ocr

    def _docling_engine(self, ocr: str | None) -> Any:
        if ocr not in self._docling:
            from trifact.ingestion.docling_backend import DoclingEngine

            self._docling[ocr] = DoclingEngine(self.cfg, ocr, self.root)
        return self._docling[ocr]

    def warmup(self) -> None:
        """Load every model this configuration may use (so timings exclude model loading)."""
        if self.ocr_name:
            self._page_ocr()
        if self.pdf == "docling" or self.office == "docling":
            self._docling_engine(self.docling_ocr).warmup()
        if self.pdf == "router":
            self._docling_engine(None).warmup()

    def parse(self, raw: bytes, filename: str) -> ParsedDoc:
        t0 = time.perf_counter()
        if not raw or not raw.strip():
            raise ParseError("empty")
        if len(raw) > self.limits["max_bytes"]:
            raise ParseError("too_large", f"{len(raw)} bytes")
        fmt = sniff(raw, filename)
        if fmt == "pdf":
            doc = self._parse_pdf(raw, filename)
        elif fmt == "docx":
            doc = self._docling_office(raw, filename, "docx") if self.office == "docling" else parse_docx(raw)
        elif fmt == "html":
            doc = self._docling_office(raw, filename, "html") if self.office == "docling" else parse_html(raw)
        else:
            doc = parse_txt(raw)
        if len(doc.text) > self.limits["max_chars"]:
            raise ParseError("too_long_text", f"{len(doc.text)} chars")
        if not doc.text.strip():
            raise ParseError("no_text", f"{fmt}: no extractable text")
        doc.meta["format"] = fmt
        doc.meta["seconds"] = round(time.perf_counter() - t0, 4)
        return doc

    def _docling_office(self, raw: bytes, filename: str, fmt: str) -> ParsedDoc:
        name = filename if filename.lower().endswith((f".{fmt}", ".htm")) else f"{filename}.{fmt}"
        blocks, _ = self._docling_engine(None).convert(raw, name)
        for b in blocks:
            b.page = None  # Docling gives DOCX/HTML no pagination
        return assemble(blocks, None, {"engine": "docling"})

    def _open_pdf(self, raw: bytes) -> Any:
        import pymupdf

        try:
            doc = pymupdf.open(stream=raw, filetype="pdf")
        except Exception as e:
            raise ParseError("corrupted", f"{type(e).__name__}: {e}") from e
        if doc.needs_pass and not doc.authenticate(""):
            raise ParseError("encrypted", "password required")
        if doc.page_count == 0:
            raise ParseError("corrupted", "no pages")
        if doc.page_count > self.limits["max_pages"]:
            raise ParseError("too_many_pages", f"{doc.page_count} > {self.limits['max_pages']}")
        return doc

    def _parse_pdf(self, raw: bytes, filename: str) -> ParsedDoc:
        doc = self._open_pdf(raw)
        n = doc.page_count
        name = filename if filename.lower().endswith(".pdf") else f"{filename}.pdf"
        pc = self.cfg["pymupdf"]
        rc = self.cfg["router"]
        meta: dict[str, Any] = {"engine": self.pdf, "ocr_pages": [], "docling_pages": []}
        if self.pdf == "docling":
            doc.close()
            blocks, _ = self._docling_engine(self.docling_ocr).convert(raw, name)
            meta["docling_pages"] = list(range(1, n + 1))
            return assemble(blocks, n, meta)
        if self.pdf == "pymupdf":
            infos, ocred = fast.extract(doc, pc, self._page_ocr(), table_min_rows=rc["table_min_rows"])
            doc.close()
            meta["ocr_pages"] = ocred
            meta["page_errors"] = [i.page for i in infos if i.n_chars < 0]
            return assemble(fast.to_blocks(infos, pc), n, meta)
        # router (DOC-3)
        infos, _ = fast.extract(doc, pc, None, table_min_rows=rc["table_min_rows"])
        to_docling: list[int] = []
        keep: list[fast.PageInfo] = []
        for info in infos:
            if fast.needs_ocr(info, pc):
                if rc["scans"] == "docling":
                    to_docling.append(info.page)
                    continue
                ocr = self._page_ocr()
                if ocr is not None:
                    redo, ocred = fast.extract(doc, pc, ocr, [info.page - 1], rc["table_min_rows"])
                    info = redo[0]
                    meta["ocr_pages"] += ocred
                    if (rc["columns"] and info.columns) or (rc["tables"] and info.table):
                        to_docling.append(info.page)  # a scan with complex layout: Docling + its OCR
                        meta["ocr_pages"].remove(info.page)
                        continue
            elif (rc["columns"] and info.columns) or (rc["tables"] and info.table):
                to_docling.append(info.page)
                continue
            keep.append(info)
        doc.close()
        by_page: dict[int, list[Block]] = {}
        for b in fast.to_blocks(keep, pc):
            by_page.setdefault(b.page or 0, []).append(b)
        for a, b in _ranges(sorted(to_docling)):
            scanned = any(fast.needs_ocr(infos[p - 1], pc) for p in range(a, b + 1))
            eng = self._docling_engine(
                (self.docling_ocr or self.cfg["docling"]["ocr_engine"]) if scanned else None
            )
            blocks, _ = eng.convert(raw, name, page_range=(a, b))
            for blk in blocks:
                by_page.setdefault(blk.page or a, []).append(blk)
        meta["docling_pages"] = sorted(to_docling)
        meta["page_errors"] = [i.page for i in infos if i.n_chars < 0]
        return assemble([b for p in sorted(by_page) for b in by_page[p]], n, meta)
