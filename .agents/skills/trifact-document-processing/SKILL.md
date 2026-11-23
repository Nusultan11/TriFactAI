---
name: trifact-document-processing
description: Cloud-ready document intelligence for TriFact — parsing PDF (text, scans, columns, tables), DOCX, HTML, TXT into ParsedDoc with exact page/section/table offsets; Tesseract kaz+rus+eng and cloud GPU VLM-OCR, layout and reading order, tables, headings, encodings, per-page routing, sandboxing, error detection and benchmarks (CER/WER, reading order, heading F1, cell F1/TEDS, pages/s). Use when changing trifact.ingestion, adding a format/engine, debugging garbled text or lost Kazakh letters, or running large-scale/cloud OCR and parsing evaluation.
---

# TriFact AI — Document Processing

## Scope

- Load when editing `trifact.ingestion` (`parser`, `pdf`, `engines`, `docling_backend`, `ocr`, `sandbox`, `normalize`, `chunker`), adding formats/engines, debugging Kazakh OCR/table/layout failures, or evaluating `real_docs` (Stage 10).
- Retrieval-driven chunking choice → `trifact-retrieval`.
- Cloud CLI execution & DVC/MLflow persistence → [cloud-compute.md](../trifact-mlops/references/cloud-compute.md).

## Owner Documents

| Topic | Owner |
|---|---|
| Router & architecture | [ADR-0021](../../../docs/architecture/adr/0021-document-processing-router.md), [01-architecture.md §6.1, §7.1](../../../docs/architecture/docs/01-architecture.md) |
| Engine/OCR measurements | [models.md §6.3–6.4](../../../docs/research/models.md), [PROGRESS Stage 4](../../../docs/PROGRESS.md) |
| Limits, thresholds, OCR models | `params.yaml`: `docproc`, `chunking` |
| Benchmark | `trifact.data.build_docproc`, `trifact.evaluation.eval_docproc`, `reports/docproc.json` |

## Non-Negotiable Invariants

1. Every span offset points into `ParsedDoc.text`; page intervals cover the text without gaps.
2. Text normalization uses `trifact.ingestion.normalize` identically online and offline.
3. Native parsers run strictly inside `SafeParser` (isolated process, timeout, RSS cap); failures return `ParseError(reason)` without crashing the worker.
4. Detect file format by content (`sniff`), never by file extension.
5. Promote a new engine or routing threshold only after benchmark improvement by `kind × lang` with bootstrap CI.

## Escalation Ladder & Cloud Execution

Escalate each page only when the cheaper path fails:
1. **PyMuPDF text layer** (XY-cut columns, font-ratio headings, header/footer removal).
2. **Docling layout + TableFormer** for table pages.
3. **Docling + Tesseract CLI (`kaz+rus+eng`)** for scans or broken text layers.
4. **VLM-OCR on cloud GPU (Colab L4)** only as a gated experiment for page classes where 1–3 measurably fail (phone photos, handwriting, formulas).

- **Cloud batch execution (`[REMOTE]`):** Run large-scale `eval_docproc`, `real_docs` sweeps, or VLM-OCR benchmarks on Colab CPU (Docling/Tesseract) or Colab L4 (VLM-OCR). On Linux cloud VMs, install `tesseract-ocr`, set `TRIFACT_TESSERACT`, and restore `tessdata_best` (`kaz`, `rus`, `eng`, `osd`).
- **Process isolation:** Measure speed and peak RSS in a dedicated process per engine after warmup; never mix engine models in one RSS reading.
- Engine matrix & VLM-OCR criteria → [references/engines.md](references/engines.md).

## Online Reference-Free Error Signals

Use for page re-routing and production drift monitoring:
- Share of `U+FFFD`, private-use, and non-letter chars (`docproc.pymupdf.max_bad_char_share`).
- Kazakh-letter share in `kk` pages (sharp drop signals lost `ә ң ұ һ` or `ru` misread).
- Page langid confidence and OOV rate against curated-corpus frequency lists.
- Mean Tesseract word confidence (TSV) on OCR pages.
- Page char count vs. neighboring pages (blank page amid text pages = missing layer).
- Broken sentence boundaries at column edges or small-LM perplexity spikes across block joins.
- Leaked running headers/footers in top/bottom bands.

## Debugging Procedure

1. Reproduce on a single file with each engine in an isolated process (`eval_docproc.isolated`).
2. Dump `(type, page, bbox, text)` blocks and align to reference (`rapidfuzz` partial ratio).
3. Classify defect: `missing_text`, `reading_order`, `merge_split`, `columnar_table`, `heading`, `encoding`, `ocr_substitution` (e.g., `қ→к`), `furniture_leak`.
4. Verify fix across the full `eval_docproc` benchmark, not just the failing file; add regression fixture to `tests/ingestion/` or hostile set.

## Evaluation Protocol

- Metrics: canonicalized CER/WER, reading-order pairwise accuracy, block recall, page accuracy, heading P/R/F1 + level accuracy, table detection + cell F1 (add TEDS/GriTS when topology matters), `s/page`, peak RSS.
- Aggregate by `kind × lang` with bootstrap CI; track tail quality (`p90 CER`), not just mean.
- Validate real-world transfer on `real_docs` (`evaluation.real_docs_min`), especially router thresholds and borderless tables.

## Common Failures

| Symptom | Likely Cause | Check |
|---|---|---|
| `kk` CER >> `ru`/`en` on scans | OCR model lacks Kazakh glyphs | Confusion matrix, `kaz.traineddata` in tessdata (not EasyOCR/RapidOCR) |
| Table collapses into single column | Page missed Docling route | `docproc.router.table_min_rows`, borderless table heuristic |
| Multi-column text interleaved | XY-cut missed gutter | Gutter width, multi-band shared gutter |
| Low reading-order on repeated text | Alignment matched earlier occurrence | Search strictly after previous matched block |
| UTF-8 HTML decoded as latin-1 | Missing `<meta charset>` | Explicit encoding detection |
| Docling Tesseract `KeyError: 'text'` | Missing `configs/tsv` in custom tessdata | Ensure `tessdata_dir` writes TSV config |
| Worker OOM on single file | Hostile/oversized file | Enforce `docproc.limits` in `SafeParser` |

## Done Criteria

- CER and structure ≥ champion across every `kind × lang` (CI); speed and RSS within NFRs (`01-architecture.md §1`).
- Hostile suite produces expected `ParseError` outcomes; worker survives all cases.
- Zero offset-invariant violations; `tests/ingestion/` passes.
- Logged to MLflow `docproc` and `reports/docproc.json`; decision changes recorded in a new ADR.

## Sources

- Auer et al., *Docling* (arXiv:2408.09869); Nassar et al., *TableFormer* (arXiv:2203.01017).
- Zhong et al., *PubTabNet/TEDS* (arXiv:1911.10683); Smock et al., *GriTS* (arXiv:2203.12555).
- Ouyang et al., *OmniDocBench* (arXiv:2412.07626); Poznanski et al., *olmOCR* (arXiv:2502.18443); `tesseract-ocr/tessdata_best`.
