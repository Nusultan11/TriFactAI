from __future__ import annotations

from pathlib import Path

import polars as pl

from trifact.data.build_ner import gazetteer_tag, prf, tag_unit

CFG = {"capital_filter": True}


def test_silver_tags_links_propagates_and_filters(tmp_path: Path) -> None:
    rows = [
        # linked PER + LOC; second sentence mentions "Абай" unlinked -> propagated from the article
        {"page_id": 1, "title": "Абай Құнанбайұлы", "lang": "kk", "para": 0,
         "tokens": ["Абай", "Семейде", "туған", "ақын", "болды", "."],
         "links": [{"s": 0, "e": 1, "target": "Абай Құнанбайұлы", "ner_type": "PER"},
                   {"s": 1, "e": 2, "target": "Семей", "ner_type": "LOC"}], "title_type": "PER"},
        {"page_id": 1, "title": "Абай Құнанбайұлы", "lang": "kk", "para": 1,
         "tokens": ["Кейін", "Абай", "өлең", "жазды", "және", "аударды", "."], "links": None, "title_type": "PER"},
        # capitalized unlinked word in the middle -> likely a missed entity -> dropped
        {"page_id": 1, "title": "Абай Құнанбайұлы", "lang": "kk", "para": 2,
         "tokens": ["Абай", "Мәскеуге", "барған", "жоқ", "еді", "."], "links": None, "title_type": "PER"},
        # no entity at all
        {"page_id": 2, "title": "Өлең", "lang": "kk", "para": 0,
         "tokens": ["бұл", "сөз", "өлең", "туралы", "."], "links": None, "title_type": None},
    ]  # fmt: skip
    src = tmp_path / "res.parquet"
    pl.DataFrame(rows).write_parquet(src)
    out = tmp_path / "tag.parquet"
    r = tag_unit((str(src), str(out), CFG))
    assert r["reasons"] == {"kept": 2, "capitalized_untagged": 1, "no_entity": 1}
    df = pl.read_parquet(out)
    assert df["tags"].to_list() == [
        ["B-PER", "B-LOC", "O", "O", "O", "O"],
        ["O", "B-PER", "O", "O", "O", "O", "O"],
    ]


def test_silver_untags_spans_without_a_capital_letter(tmp_path: Path) -> None:
    rows = [
        {"page_id": 3, "title": "Ислам", "lang": "kk", "para": 0,
         "tokens": ["Абай", "ислам", "туралы", "жазды", "."],
         "links": [{"s": 0, "e": 1, "target": "Абай", "ner_type": "PER"},
                   {"s": 1, "e": 2, "target": "Ислам", "ner_type": "ORG"}], "title_type": None},
        {"page_id": 4, "title": "X", "lang": "en", "para": 0,
         "tokens": ["the", "band", "played", "in", "Leeds", "."],
         "links": [{"s": 0, "e": 1, "target": "The Who", "ner_type": "ORG"},
                   {"s": 4, "e": 5, "target": "Leeds", "ner_type": "LOC"}], "title_type": None},
    ]  # fmt: skip
    src = tmp_path / "res.parquet"
    pl.DataFrame(rows).write_parquet(src)
    out = tmp_path / "tag.parquet"
    r = tag_unit((str(src), str(out), CFG))
    assert r["reasons"]["span_without_capital"] == 2
    assert pl.read_parquet(out)["tags"].to_list() == [
        ["B-PER", "O", "O", "O", "O"],
        ["O", "O", "O", "O", "B-LOC", "O"],
    ]


def test_gazetteer_tag_longest_match_and_prf() -> None:
    gaz = {"нұр-сұлтан": "LOC", "қазақстан республикасы": "LOC", "қазақстан": "LOC"}
    tags = gazetteer_tag(["Қазақстан", "Республикасы", "және", "Нұр-Сұлтан"], gaz, 3)
    assert tags == ["B-LOC", "I-LOC", "O", "B-LOC"]
    m = prf([["B-LOC", "I-LOC", "O", "B-PER"]], [tags])
    assert m["LOC"]["precision"] == 0.5 and m["PER"]["recall"] == 0.0
