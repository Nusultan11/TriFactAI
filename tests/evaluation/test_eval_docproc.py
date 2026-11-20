from __future__ import annotations

from trifact.evaluation.eval_docproc import aggregate, canon, recommend, score_doc
from trifact.ingestion.parser import Block, assemble

CFG = {"block_match_min": 80, "heading_match_min": 90}
GOLD = {
    "format": "pdf", "pages_known": True, "headings_known": True,
    "blocks": [
        {"kind": "heading", "text": "Алматы", "level": 1, "page": 1, "rows": None},
        {"kind": "text", "text": "Алматы — Қазақстандағы ең ірі қала, мәдени орталық.", "level": 0, "page": 1, "rows": None},
        {"kind": "text", "text": "Almaty is the largest city of Kazakhstan by population.", "level": 0, "page": 2, "rows": None},
        {"kind": "table", "text": "", "level": 0, "page": 2, "rows": [["Жыл", "Халық"], ["2020", "1 900 000"]]},
    ],
}  # fmt: skip


def _doc(order: list[int]) -> object:
    src = [Block("heading", "Алматы", 1, 1), Block("text", GOLD["blocks"][1]["text"], 1),
           Block("text", GOLD["blocks"][2]["text"], 2), Block("table", page=2, rows=[["Жыл", "Халық"], ["2020", "1 900 000"]])]  # fmt: skip
    return assemble([src[i] for i in order], 2)


def test_canon_maps_offsets() -> None:
    s, idx = canon("  a |  b\n\nc ")
    assert s == "a b c" and idx == [2, 3, 7, 8, 10]


def test_perfect_parse_scores_perfectly() -> None:
    m = score_doc(GOLD, _doc([0, 1, 2, 3]), CFG)
    assert m["cer"] == 0 and m["wer"] == 0 and m["order"] == 1 and m["page_acc"] == 1
    assert m["head_f1"] == 1 and m["table_cell_f1"] == 1 and m["table_shape_acc"] == 1


def test_swapped_blocks_lower_order_not_recall() -> None:
    m = score_doc(GOLD, _doc([0, 2, 1, 3]), CFG)
    assert m["block_recall"] == 1 and m["order"] < 1 and m["cer"] > 0


def test_failed_parse_scores_zero_and_recommend_prefers_fast_equivalent() -> None:
    m = score_doc(GOLD, None, CFG)
    assert m["cer"] == 1 and m["block_recall"] == 0
    rows = []
    for cfg, secs, cer in [("fast", 0.1, 0.01), ("slow", 2.0, 0.009)]:
        for i in range(5):
            rows.append({"kind": "pdf_text", "lang": "kk", "cer": cer + i * 1e-3, "wer": 0, "order": 1.0, "seconds": secs,
                         "pages": 1, "error": None, "docling_pages": 0, "ocr_pages": 0})  # fmt: skip
    ev = {"bootstrap_samples": 200, "ci": 0.95}
    summary = {
        c: aggregate([r for r in rows if r["seconds"] == s], ev, 1) for c, s in [("fast", 0.1), ("slow", 2.0)]
    }
    assert recommend(summary, ["pdf_text"])["pdf_text"]["chosen"] == "fast"
