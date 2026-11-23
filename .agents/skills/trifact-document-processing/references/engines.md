# Parsing & OCR Engine Selection

Current project choice and measurements: [ADR-0021](../../../../docs/architecture/adr/0021-document-processing-router.md), [models.md §6.3](../../../../docs/research/models.md).

## Selection Criteria (Priority Order)

1. **Kazakh accuracy:** Must preserve `ә ғ қ ң ө ұ ү һ і`. Verify via `kk` CER on `pdf_scan` and character confusion matrix; public leaderboards rarely test `kk`.
2. **Citation structure:** Must emit page numbers, blocks, reading order, headings, and tables — not flat Markdown — so `(page, char_start, char_end)` offsets remain exact.
3. **Compute profile:** Online worker targets CPU (~2 GB RAM/process); GPU-only models are restricted to offline cloud batch reprocessing (Colab L4) or an explicit GPU serving profile.
4. **Throughput:** ≥ 2 pages/s on text PDFs; benchmark scans separately.
5. **License:** Record license in `models.md` before adoption (PyMuPDF is AGPL-3.0; check restrictive VLM weights).
6. **Fault isolation:** Wrap in `SafeParser`; add GPU OOM and timeout handling for neural/VLM backends.

## Page-Class Defaults

| Page Class | Default Path | Escalate When |
|---|---|---|
| Text layer, 1–2 columns | PyMuPDF fast path | Broken encoding or missing text layer |
| Ruled / borderless tables | Docling + TableFormer (`fast`) | Cell F1 on `real_docs` below target |
| Scans (200–300 dpi) | Docling + Tesseract `kaz+rus+eng` | High CER on photos, skew, low dpi |
| Phone photos, handwriting, formulas | Cloud GPU VLM-OCR experiment | — |
| DOCX, HTML, TXT | Native parsers | — (beat Docling backends in Stage 4) |

## VLM-OCR Evaluation Protocol (Cloud GPU)

Candidate list and published numbers: [models.md §6.4](../../../../docs/research/models.md). Treat published benchmarks only as a shortlist filter.

1. Shortlist 2–3 candidates by license, VRAM footprint (fits Colab L4 24 GB), and Cyrillic coverage.
2. Benchmark on `data/eval/docproc` (`pdf_scan`, `mixed`) and `real_docs` in isolated cloud GPU processes: record per-language CER, Kazakh confusion matrix, heading/cell F1, `s/page`, and peak VRAM.
3. Verify page/bbox extraction supports `ParsedDoc` offset invariants.
4. Enable only for page classes where CI shows significant gain over Tesseract, as a new `docproc.router` branch documented by ADR.

## Sources

- *OmniDocBench* (arXiv:2412.07626); *olmOCR* (arXiv:2502.18443); *MinerU* (arXiv:2409.18839).
