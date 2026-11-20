"""Document-processing benchmark DOC-0 … DOC-3 on data/eval/docproc (ROADMAP 4.2, architecture §7.1).

python -m trifact.evaluation.eval_docproc [--configs DOC-0,DOC-1,...] [--limit N] [--no-big] [--no-robust]
                                          [--report-only]

Per document: CER and WER of the whole text, reading order (pairwise order of the located reference blocks),
block recall, page accuracy (the page of every located block), heading P/R/F1 and level accuracy, table
detection P/R, cell F1 and shape accuracy, seconds and pages/s. Texts are compared after NFC, whitespace
collapsing and dropping the `|` table separators. Results per configuration are cached in
`reports/docproc/<config>.json` (so slow configurations run once), aggregated by kind × language with
bootstrap CI into `reports/docproc.json`, and logged as one MLflow run per configuration (experiment `docproc`).
The champion configuration additionally runs the robustness set through `SafeParser`, the long documents
(NFR: >= 2 pages/s, RAM) and the chunking variants of ROADMAP 4.3.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from trifact.core.params import ROOT, load_params
from trifact.ingestion.normalize import normalize_text
from trifact.ingestion.parser import ParsedDoc, ParseError
from trifact.lake.stats import ResourceMonitor, git_sha, log_mlflow

log = logging.getLogger("trifact.eval_docproc")
PDF_KINDS = ("pdf_text", "pdf_columns", "pdf_tables", "pdf_scan", "pdf_word", "mixed")
SCAN_KINDS = ("pdf_scan", "mixed")
OFFICE_KINDS = ("docx", "html")
ROUTER = {"pdf": "router", "ocr": "tesseract", "docling_ocr": "tesseract"}
# name -> (Parser keyword arguments, document kinds it is measured on, overrides of params.yaml: docproc)
CONFIGS: dict[str, tuple[dict[str, Any], tuple[str, ...], dict[str, Any]]] = {
    "DOC-0": ({"pdf": "pymupdf", "ocr": None}, PDF_KINDS, {}),
    "DOC-1": ({"pdf": "docling", "ocr": None, "docling_ocr": None}, PDF_KINDS, {}),
    "DOC-2-tesseract": ({"pdf": "docling", "docling_ocr": "tesseract"}, SCAN_KINDS, {}),
    "DOC-2-easyocr": ({"pdf": "docling", "docling_ocr": "easyocr"}, SCAN_KINDS, {}),
    "DOC-2-rapidocr": ({"pdf": "docling", "docling_ocr": "rapidocr"}, SCAN_KINDS, {}),
    "DOC-2-mupdf-tesseract": ({"pdf": "pymupdf", "ocr": "tesseract"}, SCAN_KINDS, {}),
    # DOC-3 as in params.yaml; v0 = first router (columns to Docling, scans through PyMuPDF + Tesseract)
    "DOC-3": (ROUTER, PDF_KINDS, {}),
    "DOC-3-v0": (
        ROUTER,
        PDF_KINDS,
        {"router": {"columns": True, "scans": "ocr"}, "docling": {"table_mode": "accurate"}},
    ),
    "DOC-3-tableaccurate": (ROUTER, PDF_KINDS, {"docling": {"table_mode": "accurate"}}),
    "OFFICE-native": ({"office": "native"}, (*OFFICE_KINDS, "txt"), {}),
    "OFFICE-docling": ({"office": "docling"}, OFFICE_KINDS, {}),
}
CHAMPION = "DOC-3"


def config_params(dp: dict[str, Any], name: str) -> dict[str, Any]:
    """params.yaml: docproc with the configuration's overrides merged one level deep."""
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in dp.items()}
    for k, v in CONFIGS[name][2].items():
        out[k] = {**out[k], **v}
    return out


def isolated(fn: Any, *args: Any) -> Any:
    """Run `fn(*args)` in a fresh process: peak RSS and timings of one configuration exclude the models
    other configurations loaded."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
        return ex.submit(fn, *args).result()


def _child_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# --------------------------------------------------------------------------- metrics


def canon(text: str) -> tuple[str, list[int]]:
    """Comparison form of a text and, for every character of it, its offset in `text`."""
    out: list[str] = []
    idx: list[int] = []
    space = True
    for i, c in enumerate(text):
        if c == "|" or c == "­":
            c = " "
        if c.isspace():
            if not space:
                out.append(" ")
                idx.append(i)
            space = True
            continue
        out.append(c)
        idx.append(i)
        space = False
    while out and out[-1] == " ":
        out.pop()
        idx.pop()
    return unicodedata.normalize("NFC", "".join(out)), idx


def block_text(b: dict[str, Any]) -> str:
    if b["kind"] == "table":
        return " ".join(" ".join(normalize_text(c) for c in r) for r in b["rows"] or [])
    return normalize_text(b["text"])


def _locate(needle: str, hyp: str, min_score: float, after: int = 0) -> int | None:
    """Position of a reference block in the output: exact match after the previous block, then anywhere
    (repeated sentences resolve to the next occurrence), then the best fuzzy alignment."""
    from rapidfuzz import fuzz

    i = hyp.find(needle, after)
    if i < 0:
        i = hyp.find(needle)
    if i >= 0:
        return i
    al = fuzz.partial_ratio_alignment(needle, hyp, score_cutoff=min_score)
    return None if al is None else int(al.dest_start)


def _cells(rows: list[list[str]]) -> list[str]:
    return [canon(normalize_text(c))[0] for r in rows for c in r if canon(normalize_text(c))[0]]


def _cell_f1(gold: list[list[str]], pred: list[list[str]]) -> float:
    from collections import Counter

    g, p = Counter(_cells(gold)), Counter(_cells(pred))
    tp = sum((g & p).values())
    if not tp:
        return 0.0
    prec, rec = tp / sum(p.values()), tp / sum(g.values())
    return 2 * prec * rec / (prec + rec)


def score_doc(gold: dict[str, Any], doc: ParsedDoc | None, cfg: dict[str, Any]) -> dict[str, Any]:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein

    blocks = gold["blocks"]
    ref = canon(" ".join(block_text(b) for b in blocks))[0]
    hyp, idx = canon(doc.text) if doc is not None else ("", [])
    m: dict[str, Any] = {
        "cer": min(Levenshtein.distance(ref, hyp) / max(len(ref), 1), 1.0),
        "wer": min(Levenshtein.distance(ref.split(), hyp.split()) / max(len(ref.split()), 1), 1.0),
        "ref_chars": len(ref),
    }
    # reading order, block recall and page accuracy over located blocks
    pos: list[int] = []
    page_ok = page_n = 0
    for b in blocks:
        needle = canon(block_text(b))[0][:120]
        if len(needle) < 15:
            continue
        at = (
            _locate(needle, hyp, cfg["block_match_min"], next((q for q in reversed(pos) if q >= 0), 0))
            if hyp
            else None
        )
        if at is None:
            pos.append(-1)
            continue
        pos.append(at)
        if gold["pages_known"] and doc is not None and doc.pages and b.get("page"):
            page_n += 1
            page_ok += doc.page_at(idx[min(at, len(idx) - 1)]) == b["page"]
    found = [p for p in pos if p >= 0]
    m["block_recall"] = len(found) / max(len(pos), 1)
    pairs = conc = 0
    for i in range(len(found)):
        for j in range(i + 1, len(found)):
            pairs += 1
            conc += found[i] < found[j]
    m["order"] = conc / pairs if pairs else (1.0 if found else 0.0)
    if gold["pages_known"] and gold["format"] in ("pdf", "docx"):
        m["page_acc"] = page_ok / page_n if page_n else 0.0
    # headings
    if gold["headings_known"]:
        gh = [(canon(normalize_text(b["text"]))[0], b["level"]) for b in blocks if b["kind"] == "heading"]
        ph = (
            [(canon(t)[0], lv) for (t, _, _), lv in zip(doc.sections, doc.levels, strict=False)]
            if doc
            else []
        )
        used: set[int] = set()
        tp = lvl_ok = 0
        for t, lv in gh:
            best, bi = 0.0, -1
            for k, (pt, _) in enumerate(ph):
                if k not in used:
                    s = fuzz.ratio(t, pt)
                    if s > best:
                        best, bi = s, k
            if best >= cfg["heading_match_min"]:
                used.add(bi)
                tp += 1
                lvl_ok += ph[bi][1] == lv
        p = tp / len(ph) if ph else 0.0
        r = tp / len(gh) if gh else 1.0
        m.update({"head_p": p, "head_r": r, "head_f1": 2 * p * r / (p + r) if p + r else 0.0,
                  "head_level_acc": lvl_ok / tp if tp else 0.0})  # fmt: skip
    # tables
    gt = [b["rows"] for b in blocks if b["kind"] == "table"]
    if gt:
        pt = [t.rows for t in doc.tables] if doc else []
        f1s, shapes, matched = [], 0, set()
        for g in gt:
            scores = [(_cell_f1(g, p), k) for k, p in enumerate(pt) if k not in matched]
            best, k = max(scores, default=(0.0, -1))
            f1s.append(best)
            if best >= 0.5:
                matched.add(k)
                shapes += (len(pt[k]), max(map(len, pt[k]))) == (len(g), max(map(len, g)))
        m.update({"table_r": len(matched) / len(gt), "table_p": len(matched) / len(pt) if pt else 0.0,
                  "table_cell_f1": statistics.fmean(f1s), "table_shape_acc": shapes / len(gt)})  # fmt: skip
    return m


# --------------------------------------------------------------------------- runs


def run_config(name: str, golds: list[dict[str, Any]], dp: dict[str, Any], suite: Path) -> dict[str, Any]:
    from trifact.ingestion.engines import Parser

    _child_logging()
    kw, kinds, over = CONFIGS[name]
    docs = [g for g in golds if g["kind"] in kinds]
    parser = Parser(config_params(dp, name), ROOT, **kw)
    t0 = time.perf_counter()
    parser.warmup()
    warm = time.perf_counter() - t0
    rows = []
    with ResourceMonitor() as mon:
        for g in docs:
            raw = (suite / g["file"]).read_bytes()
            t = time.perf_counter()
            err = None
            try:
                doc: ParsedDoc | None = parser.parse(raw, Path(g["file"]).name)
            except ParseError as e:
                doc, err = None, e.reason
            secs = time.perf_counter() - t
            m = score_doc(g, doc, dp["bench"])
            m.update({"doc_id": g["doc_id"], "kind": g["kind"], "lang": g["lang"], "seconds": round(secs, 4),
                      "pages": g["n_pages"] or 1, "error": err,
                      "ocr_pages": len(doc.meta.get("ocr_pages", [])) if doc else 0,
                      "docling_pages": len(doc.meta.get("docling_pages", [])) if doc else 0})  # fmt: skip
            rows.append(m)
            log.info(
                "%s %s cer=%.3f order=%.2f %.2fs %s", name, g["doc_id"], m["cer"], m["order"], secs, err or ""
            )
    return {"config": name, "parser": kw, "overrides": over, "kinds": list(kinds), "warmup_s": round(warm, 1),
            "peak_rss_mb": round(mon.peak_rss_mb, 1), "docs": rows}  # fmt: skip


METRICS = ("cer", "wer", "order", "block_recall", "page_acc", "head_p", "head_r", "head_f1", "head_level_acc",
           "table_r", "table_p", "table_cell_f1", "table_shape_acc")  # fmt: skip


def bootstrap_ci(xs: list[float], n: int, ci: float, seed: int) -> tuple[float, float]:
    if len(xs) < 2:
        return (xs[0], xs[0]) if xs else (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(n))
    lo = means[int((1 - ci) / 2 * n)]
    hi = means[min(int((1 + ci) / 2 * n), n - 1)]
    return lo, hi


def aggregate(rows: list[dict[str, Any]], ev: dict[str, Any], seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r["kind"]].append(r)
        groups[f"{r['kind']}/{r['lang']}"].append(r)
        groups["all"].append(r)
        groups[f"all/{r['lang']}"].append(r)
    for key, rs in sorted(groups.items()):
        a: dict[str, Any] = {"docs": len(rs), "errors": sum(1 for r in rs if r["error"])}
        for k in METRICS:
            xs = [r[k] for r in rs if k in r]
            if xs:
                a[k] = round(statistics.fmean(xs), 4)
        cer = [r["cer"] for r in rs]
        a["cer_ci"] = [round(x, 4) for x in bootstrap_ci(cer, ev["bootstrap_samples"], ev["ci"], seed)]
        secs, pages = sum(r["seconds"] for r in rs), sum(r["pages"] for r in rs)
        a["pages_per_s"] = round(pages / secs, 2) if secs else None
        a["s_per_page"] = round(secs / pages, 3) if pages else None
        a["docling_page_share"] = round(sum(r["docling_pages"] for r in rs) / max(pages, 1), 3)
        a["ocr_page_share"] = round(sum(r["ocr_pages"] for r in rs) / max(pages, 1), 3)
        out[key] = a
    return out


def recommend(summary: dict[str, dict[str, Any]], kinds: list[str]) -> dict[str, Any]:
    """Per document kind: among configurations whose CER CI overlaps the best one (not significantly worse)
    and whose reading order and table F1 are within 2 points of the best, the fastest."""
    out = {}
    for kind in kinds:
        cands = {c: s[kind] for c, s in summary.items() if kind in s}
        if not cands:
            continue
        best = min(cands, key=lambda c: cands[c]["cer"])
        b = cands[best]
        ok = [c for c, s in cands.items()
              if s["cer_ci"][0] <= b["cer_ci"][1] and s.get("order", 1) >= b.get("order", 1) - 0.02
              and s.get("table_cell_f1", 1) >= b.get("table_cell_f1", 1) - 0.02]  # fmt: skip
        pick = max(ok, key=lambda c: cands[c]["pages_per_s"] or 0)
        out[kind] = {"best_quality": best, "chosen": pick, "candidates": ok}
    return out


def run_big(name: str, dp: dict[str, Any], suite: Path, golds: list[dict[str, Any]]) -> dict[str, Any]:
    """Long text PDFs: pages/s and peak RSS (NFR: >= 2 pages/s per worker, <= peak_ram_gb)."""
    from trifact.ingestion.engines import Parser

    _child_logging()
    parser = Parser(config_params(dp, name), ROOT, **CONFIGS[name][0])
    parser.warmup()
    res = []
    with ResourceMonitor() as mon:
        for g in golds:
            raw = (suite / g["file"]).read_bytes()
            t = time.perf_counter()
            doc = parser.parse(raw, Path(g["file"]).name)
            secs = time.perf_counter() - t
            res.append({"doc_id": g["doc_id"], "pages": g["n_pages"], "seconds": round(secs, 2),
                        "pages_per_s": round(g["n_pages"] / secs, 2), "chars": len(doc.text),
                        "docling_pages": len(doc.meta.get("docling_pages", []))})  # fmt: skip
            log.info("big %s %s: %.1f pages/s", name, g["doc_id"], g["n_pages"] / secs)
    return {"docs": res, "peak_rss_mb": round(mon.peak_rss_mb, 1),
            "pages_per_s": round(sum(r["pages"] for r in res) / sum(r["seconds"] for r in res), 2)}  # fmt: skip


def run_robust(dp: dict[str, Any], suite: Path, cases: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Hostile files through the isolated parser: expected outcome, no crash of the caller."""
    from trifact.ingestion.sandbox import SafeParser

    res = []
    with SafeParser.from_params(config_params(dp, name), ROOT, **CONFIGS[name][0]) as sp:
        for c in cases:
            raw = (suite / c["file"]).read_bytes()
            t = time.perf_counter()
            try:
                doc = sp.parse(raw, Path(c["file"]).name)
                got, reason, chars = "ok", None, len(doc.text)
            except ParseError as e:
                got, reason, chars = "error", e.reason, 0
            passed = c["expect"] == "any" or (
                got == c["expect"] and (c["reason"] is None or c["reason"] == reason)
            )
            res.append({**c, "got": got, "got_reason": reason, "chars": chars, "passed": passed,
                        "seconds": round(time.perf_counter() - t, 2)})  # fmt: skip
            log.info("robust %s: %s %s (%s)", c["file"], got, reason or "", "pass" if passed else "FAIL")
        restarts = sp.restarts
    return {"cases": res, "passed": sum(r["passed"] for r in res), "total": len(res), "restarts": restarts}


def run_chunking(dp: dict[str, Any], params: dict[str, Any], suite: Path, golds: list[dict[str, Any]],
                 name: str) -> dict[str, Any]:  # fmt: skip
    """ROADMAP 4.3: chunking variants over the champion's parses (the choice is made in stage 6)."""
    from huggingface_hub import hf_hub_download

    from trifact.ingestion.chunker import DocumentChunker, hf_counter
    from trifact.ingestion.engines import Parser
    from trifact.ingestion.langdetect import LangId, TextLanguageDetector, model_path

    tok = params["curation"]["stats"]["tokenizers"]["xlmr"]
    cache = ROOT / params["curation"]["langid"]["cache"]
    count = hf_counter(
        hf_hub_download(tok["hf_id"], tok["file"], revision=tok["revision"], cache_dir=str(cache / "hf"))
    )
    lid_name = params["curation"]["langid"]["model"]
    det = TextLanguageDetector(
        LangId(model_path(lid_name, params["curation"]["langid"]["models"][lid_name], cache))
    )
    parser = Parser(config_params(dp, name), ROOT, **CONFIGS[name][0])
    parsed = []
    for g in golds:
        try:
            parsed.append((g, parser.parse((suite / g["file"]).read_bytes(), Path(g["file"]).name)))
        except ParseError:
            continue
    out = {}
    for vname, v in params["chunking"]["variants"].items():
        ch = DocumentChunker(count, det.detect, v["max_tokens"], v["overlap_tokens"], v["sections"])
        n = toks = cross = with_sec = bad = lang_ok = lang_n = 0
        t = time.perf_counter()
        for g, doc in parsed:
            cs = ch.split(g["doc_id"], doc)
            for c in cs:
                n += 1
                toks += c.token_count or 0
                bad += doc.text[c.char_start : c.char_end] != c.text
                if doc.pages:
                    cross += doc.page_at(c.char_end - 1) != c.page
                with_sec += c.section is not None
                if g["lang"] in ("kk", "ru", "en"):
                    lang_n += 1
                    lang_ok += c.language.value == g["lang"]
        out[vname] = {**v, "chunks": n, "mean_tokens": round(toks / max(n, 1), 1),
                      "chunks_per_doc": round(n / max(len(parsed), 1), 2), "cross_page_share": round(cross / max(n, 1), 3),
                      "with_section_share": round(with_sec / max(n, 1), 3), "offset_violations": bad,
                      "lang_acc": round(lang_ok / max(lang_n, 1), 4), "seconds": round(time.perf_counter() - t, 2)}  # fmt: skip
        log.info("chunking %s: %s", vname, out[vname])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--limit", type=int, help="documents per kind (smoke runs)")
    ap.add_argument("--no-big", action="store_true")
    ap.add_argument("--no-robust", action="store_true")
    ap.add_argument("--rerun", action="store_true", help="ignore cached per-configuration results")
    ap.add_argument("--report-only", action="store_true", help="aggregate cached results only")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    params = load_params()
    dp, ev = params["docproc"], params["evaluation"]
    suite = ROOT / params["lake"]["root"] / "eval" / "docproc"
    manifest = yaml.safe_load((suite / "manifest.yaml").read_text(encoding="utf-8"))
    golds = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((suite / "gold").glob("*.json"))]
    bigs = [g for g in golds if g["kind"] == "pdf_big"]
    golds = [g for g in golds if g["kind"] != "pdf_big"]
    if args.limit:
        seen: dict[str, int] = defaultdict(int)
        lim = []
        for g in golds:
            key = f"{g['kind']}/{g['lang']}"
            if seen[key] < args.limit:
                lim.append(g)
                seen[key] += 1
        golds = lim
    cache = ROOT / "reports" / "docproc"
    cache.mkdir(parents=True, exist_ok=True)
    names = [c for c in args.configs.split(",") if c]
    for name in names:
        path = cache / f"{name}.json"
        if args.report_only or (path.is_file() and not args.rerun and not args.limit):
            continue
        res = isolated(run_config, name, golds, dp, suite)
        res.update({"git_sha": git_sha(ROOT), "suite_sha": manifest["sha256"].get("manifest.yaml")})
        if not args.limit:
            path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        summary = aggregate(res["docs"], ev, params["seed"])
        flat = {f"{m}.{k.replace('/', '.')}": v for k, s in summary.items() for m, v in s.items()
                if isinstance(v, int | float) and not isinstance(v, bool)}  # fmt: skip
        flat.update({"peak_rss_mb": res["peak_rss_mb"], "warmup_s": res["warmup_s"]})
        if not args.limit:
            log_mlflow("docproc", name, {**{f"parser.{k}": v for k, v in res["parser"].items()},
                                         "overrides": json.dumps(res["overrides"]),
                    "kinds": ",".join(res["kinds"]), "suite": str(suite.relative_to(ROOT)),
                                         "git_sha": res["git_sha"], "docproc": json.dumps(dp)}, flat,
                       [path])  # fmt: skip
    if args.limit:
        return 0
    results = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(cache.glob("*.json"))
               if p.stem in CONFIGS}  # fmt: skip
    summary = {n: aggregate(r["docs"], ev, params["seed"]) for n, r in results.items()}
    report: dict[str, Any] = {
        "suite": {k: manifest[k] for k in ("docs", "langs", "pages", "word_available")},
        "configs": {
            n: {
                "parser": r["parser"],
                "overrides": r.get("overrides", {}),
                "peak_rss_mb": r["peak_rss_mb"],
                "warmup_s": r["warmup_s"],
                "summary": summary[n],
            }
            for n, r in results.items()
        },
        "recommendation": recommend(summary, [*PDF_KINDS, *OFFICE_KINDS, "txt"]),
    }
    extra = cache / "extra.json"
    ext: dict[str, Any] = json.loads(extra.read_text(encoding="utf-8")) if extra.is_file() else {}
    if not args.report_only:
        if not args.no_big:
            ext["big"] = {n: isolated(run_big, n, dp, suite, bigs) for n in ("DOC-0", CHAMPION)}
        if not args.no_robust:
            ext["robustness"] = run_robust(dp, suite, manifest["robust"], CHAMPION)
            ext["chunking"] = run_chunking(dp, params, suite, golds, CHAMPION)
        extra.write_text(json.dumps(ext, ensure_ascii=False, indent=1), encoding="utf-8")
        if ext:
            m = {"robust_passed": ext.get("robustness", {}).get("passed", 0),
                 **{f"big_pages_per_s.{k}": v["pages_per_s"] for k, v in ext.get("big", {}).items()},
                 **{f"big_peak_rss_mb.{k}": v["peak_rss_mb"] for k, v in ext.get("big", {}).items()},
                 **{f"chunks.{k}": v["chunks"] for k, v in ext.get("chunking", {}).items()}}  # fmt: skip
            log_mlflow(
                "docproc", f"{CHAMPION}-system", {"champion": CHAMPION, "git_sha": git_sha(ROOT)}, m, [extra]
            )
    report.update(ext)
    out = ROOT / "reports" / "docproc.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for n, s in summary.items():
        for k in ("all", *PDF_KINDS, *OFFICE_KINDS, "txt"):
            if k in s:
                a = s[k]
                print(f"{n:24s} {k:12s} cer={a['cer']:.4f} wer={a['wer']:.4f} order={a.get('order', 0):.3f} "
                      f"head_f1={a.get('head_f1', float('nan')):.3f} tab_f1={a.get('table_cell_f1', float('nan')):.3f} "
                      f"pages/s={a['pages_per_s']}")  # fmt: skip
    print(json.dumps(report["recommendation"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
