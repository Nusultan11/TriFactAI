"""Docling conversion (DOC-1 / DOC-2): layout model, TableFormer, optional OCR -> `Block`s in reading order.

The body layer of the `DoclingDocument` is walked in its reading order; page headers and footers live in the
furniture layer and are skipped. Titles and section headers become headings (title = level 1, section
headers one below), tables keep their cell grid, pictures contribute only the text Docling found in them.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

from trifact.ingestion.parser import Block, ParseError


class DoclingEngine:
    """One configured `DocumentConverter`; models are loaded once and reused for every document."""

    def __init__(self, cfg: dict[str, Any], ocr: str | None, root: Path):
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            AcceleratorOptions,
            EasyOcrOptions,
            PdfPipelineOptions,
            RapidOcrOptions,
            TableFormerMode,
            TesseractCliOcrOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        from trifact.ingestion.ocr import tessdata_dir, tesseract_cmd

        dc, oc = cfg["docling"], cfg["ocr"]
        po = PdfPipelineOptions(do_ocr=ocr is not None, do_table_structure=True)
        po.table_structure_options.mode = (
            TableFormerMode.ACCURATE if dc["table_mode"] == "accurate" else TableFormerMode.FAST
        )
        po.accelerator_options = AcceleratorOptions(num_threads=int(dc["threads"]), device="cpu")
        if ocr == "tesseract":
            os.environ["TESSDATA_PREFIX"] = str(tessdata_dir(oc, root))
            po.ocr_options = TesseractCliOcrOptions(lang=list(oc["langs"]), tesseract_cmd=tesseract_cmd(oc),
                                                    path=os.environ["TESSDATA_PREFIX"])  # fmt: skip
        elif ocr == "easyocr":  # no Kazakh model: the Cyrillic recognizer lacks ә ң ұ һ (models.md)
            po.ocr_options = EasyOcrOptions(lang=["ru", "en"], use_gpu=False)
        elif ocr == "rapidocr":  # PP-OCRv5 Cyrillic recognizer
            po.ocr_options = RapidOcrOptions(lang=["cyrillic"])
        elif ocr is not None:
            raise ValueError(f"unknown OCR engine {ocr}")
        self.ocr = ocr
        self._conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=po)})

    def warmup(self) -> None:
        """Docling loads the layout, table and OCR weights on the first conversion: convert a two-page PDF
        (text with a ruled table, then a rasterized page) so that later timings exclude loading."""
        import pymupdf

        d = pymupdf.open()
        p = d.new_page()
        p.insert_text((72, 72), "Warm-up heading", fontsize=16)
        p.insert_text((72, 100), "A paragraph of text that precedes a small table.", fontsize=10)
        for r in range(4):
            for c in range(3):
                p.insert_text((80 + 120 * c, 140 + 20 * r), f"cell {r}.{c}", fontsize=10)
            p.draw_line((72, 126 + 20 * r), (432, 126 + 20 * r))
        for c in range(4):
            p.draw_line((72 + 120 * c, 126), (72 + 120 * c, 206))
        img = d[0].get_pixmap(dpi=150).tobytes("png")
        d.new_page().insert_image(pymupdf.Rect(0, 0, 595, 842), stream=img)
        self.convert(d.tobytes(), "warmup.pdf")

    def convert(
        self, raw: bytes, filename: str, page_range: tuple[int, int] | None = None
    ) -> tuple[list[Block], int]:
        from docling.datamodel.base_models import ConversionStatus, DocumentStream

        kw: dict[str, Any] = {"raises_on_error": False}
        if page_range is not None:
            kw["page_range"] = page_range
        try:
            res = self._conv.convert(DocumentStream(name=filename, stream=io.BytesIO(raw)), **kw)
        except Exception as e:
            raise ParseError("corrupted", f"docling: {type(e).__name__}: {e}") from e
        if res.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
            raise ParseError("corrupted", f"docling: {res.status} {[str(e) for e in res.errors][:3]}")
        return to_blocks(res.document), len(res.document.pages)


def to_blocks(doc: Any) -> list[Block]:
    from docling_core.types.doc import DocItemLabel, SectionHeaderItem, TableItem, TextItem, TitleItem

    out: list[Block] = []
    has_title = any(isinstance(t, TitleItem) for t in doc.texts)
    for item, _ in doc.iterate_items():
        page = item.prov[0].page_no if getattr(item, "prov", None) else None
        if isinstance(item, TableItem):
            grid = item.data.grid if item.data is not None else []
            # a spanned cell is repeated in every grid position it covers: keep it at its origin only
            rows = [[c.text for ci, c in enumerate(r) if c.start_row_offset_idx == ri and c.start_col_offset_idx == ci]
                    for ri, r in enumerate(grid)]  # fmt: skip
            if rows:
                out.append(Block("table", page=page, rows=rows))
        elif isinstance(item, TitleItem):
            out.append(Block("heading", item.text, page, 1))
        elif isinstance(item, SectionHeaderItem):
            out.append(Block("heading", item.text, page, item.level + (1 if has_title else 0)))
        elif isinstance(item, TextItem):
            if item.label in (DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER):
                continue
            out.append(Block("text", item.text, page))
    return out
