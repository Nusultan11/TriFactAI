"""OCR of pages without a usable text layer: Tesseract `kaz+rus+eng` through PyMuPDF (the router's scan path).

The traineddata files (tessdata_best, pinned by URL and sha256 in `params.yaml: docproc.ocr.models`) live in
`docproc.ocr.tessdata`; `tessdata_dir` downloads and checks them once. The same directory is the
`TESSDATA_PREFIX` of Docling's Tesseract CLI engine (DOC-2), so both paths use identical models.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from trifact.ingestion.langdetect import model_path


def tessdata_dir(cfg: dict[str, Any], root: Path) -> Path:
    """Directory with every pinned traineddata file, downloaded and sha256-checked on first use."""
    d = (root / cfg["tessdata"]).resolve()
    for lang, spec in cfg["models"].items():
        path = model_path(d.name, spec, d.parent)
        assert path.name == f"{lang}.traineddata", path
    # the CLI resolves output configs (`... stdout tsv`, used by Docling) inside --tessdata-dir
    tsv = d / "configs" / "tsv"
    if not tsv.is_file():
        tsv.parent.mkdir(exist_ok=True)
        tsv.write_text("tessedit_create_tsv 1\ntessedit_pageseg_mode 1\n", encoding="ascii")
    return d


def tesseract_cmd(cfg: dict[str, Any]) -> str:
    return os.environ.get("TRIFACT_TESSERACT") or cfg["tesseract_cmd"]


class PageOcr:
    """Tesseract OCR text page of a PyMuPDF page (MuPDF's built-in Tesseract, full-page)."""

    def __init__(self, cfg: dict[str, Any], root: Path):
        self.tessdata = str(tessdata_dir(cfg, root))
        self.language = "+".join(cfg["langs"])
        self.dpi = int(cfg["dpi"])

    def textpage(self, page: Any, flags: int) -> Any:
        return page.get_textpage_ocr(
            flags=flags, language=self.language, dpi=self.dpi, full=True, tessdata=self.tessdata
        )
