from __future__ import annotations

from trifact.data.ner_common import entities, map_tags, repair_bio, spans_to_bio, tokenize
from trifact.data.wikimarkup import normalize_title, prose


def test_repair_and_map() -> None:
    tags, fixed = repair_bio(["O", "I-PER", "I-PER", "B-LOC", "I-ORG"])
    assert tags == ["O", "B-PER", "I-PER", "B-LOC", "B-ORG"] and fixed == 2
    assert map_tags(["B-PERSON", "I-PERSON", "B-DATE", "B-GPE"], {"PERSON": "PER", "GPE": "LOC"}) == [
        "B-PER",
        "I-PER",
        "O",
        "B-LOC",
    ]
    assert entities(["B-PER", "I-PER", "O", "B-LOC"]) == [(0, 2, "PER"), (3, 4, "LOC")]


def test_spans_to_bio_outermost_wins() -> None:
    text = "Университет имени Абая в Алматы"
    toks = tokenize(text)
    tags = spans_to_bio(toks, [(0, 22, "ORG"), (18, 22, "PER"), (25, 31, "LOC")])
    assert tags == ["B-ORG", "I-ORG", "I-ORG", "O", "B-LOC"]


def test_wikitext_prose_links() -> None:
    wt = (
        "{{Infobox}}\n'''Абай''' ([[1845 жыл|1845]]) — [[ақын]]ы.<ref>x</ref> Ол [[Семей облысы|Семейде]] туған.\n"
        "== Өмірі ==\n* тізім\n[[Сурет:A.jpg|thumb|[[1890]]]]\n{| class=x\n| a\n|}\n"
    )
    out = prose(wt)
    assert len(out) == 1
    para, links = out[0]
    assert para == "Абай (1845) — ақыны. Ол Семейде туған."
    assert [(para[a:b], t) for a, b, t in links] == [
        ("1845", "1845 жыл"),
        ("ақыны", "Ақын"),
        ("Семейде", "Семей облысы"),
    ]
    assert normalize_title("ресей_империясы#Тарихы") == "Ресей империясы"
