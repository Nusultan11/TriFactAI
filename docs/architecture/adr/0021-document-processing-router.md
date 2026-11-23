# 21. Document Processing: PyMuPDF → Docling Router, Tesseract OCR, and Parser Sandboxing

Date: 2026-11-20

## Status

Accepted

## Context

Users upload KK/RU/EN PDFs (text, scans, multi-column, tables), DOCX, HTML, and TXT. Citations require exact `page`, `section`, and character offsets into stored text (`01-architecture.md §6.1`), with NFR ≥ 2 pages/s on text PDFs and crash isolation against hostile files.

Stage 4 benchmark (`data/eval/docproc`, `reports/docproc.json`; tables in `PROGRESS.md` Stage 4 & `models.md §6.3`) showed:
- PyMuPDF + custom layout (XY-cut, font-size headings, header/footer stripping) matches Docling on text & multi-column PDFs (CER `0.35–0.36%` vs. `0.38–0.41%`, reading order `1.0`) at 70–80 p/s vs. 0.6 p/s, but misreads tables column-wise (CER 6%, cell F1 `0.00`).
- Docling + TableFormer recovers tables (cell F1 `0.99–1.0`) at 1.5–3.5 s/page on CPU; `FAST` vs. `ACCURATE` gives cell F1 `0.993` vs. `1.000` at 1.4× speed.
- On scans, Docling + Tesseract `kaz+rus+eng` achieves `0.5–0.7%` CER; PyMuPDF + Tesseract gets ~3% and loses headings/tables; EasyOCR/RapidOCR drop Kazakh letters (`kk` CER 36% / 21%).
- Native C/C++/ML libraries can crash, hang, or OOM on hostile inputs.

## Decision

- **Unified output `ParsedDoc`** (`trifact.ingestion.parser`, implements `ParsedDocument`): normalized blocks in reading order, gapless page intervals, nested sections, and table cell grids — all offsets into stored text.
- **PDF `DOC-3` per-page router** (`docproc.engine: router`, `trifact.ingestion.engines`): inspect each page with PyMuPDF → route missing/broken text layers (`min_page_chars`, `max_bad_char_share`) to Docling + Tesseract CLI (`router.scans: docling`), table pages (ruled or ≥3 aligned rows) to Docling TableFormer `fast`, and all other pages (including multi-column, `router.columns: false`) to PyMuPDF fast path; batch contiguous Docling pages together.
- **OCR:** Tesseract 5 `kaz+rus+eng` with SHA-256-pinned `tessdata_best` (`docproc.ocr.models`) shared by PyMuPDF and Docling.
- **DOCX / HTML / TXT:** native parsers (Word styles/`outlineLvl`/rendered page breaks; HTML boilerplate stripping + explicit encoding; TXT BOM/UTF-8/detector). Keep Docling DOCX/HTML backends for comparison only.
- **Sandboxing (`SafeParser`, `trifact.ingestion.sandbox`):** long-lived isolated child process with per-document timeout and RSS cap (`docproc.limits`); crashes/hangs/OOM return `ParseError(reason)` and respawn the child.
- **Chunking (`DocumentChunker`):** section regions → sentences → token windows; select from `chunking.variants` in Stage 6.

## Consequences

Text PDFs run at PyMuPDF speed (165 p/s on 300-page PDFs) with Docling quality on tables/scans (~2 GB RAM/worker once warmed up). Worker image requires Tesseract 5 binary (`TRIFACT_TESSERACT`). PyMuPDF is AGPL-3.0 (acceptable for research; open-source or commercial Artifex license required if hosted publicly, `models.md §6.3`). Validate router heuristics on `real_docs` in Stage 10.
