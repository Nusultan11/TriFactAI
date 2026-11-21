from __future__ import annotations

from pathlib import Path

import polars as pl

from trifact.data.build_kb import class_closure


def test_class_closure_walks_p279_and_prefers_the_nearest_label(tmp_path: Path) -> None:
    # Q100 -> Q43229 (organization); Q101 -> Q100; Q200 -> Q515 (city); Q300 subclass of both ORG and LOC roots
    df = pl.DataFrame(
        {
            "qid": ["Q100", "Q101", "Q200", "Q300", "Q400", "Q500"],
            "p279": [["Q43229"], ["Q100"], ["Q515"], ["Q43229", "Q515"], [], ["Q101", "Q515"]],
        }
    )
    f = tmp_path / "e.parquet"
    df.write_parquet(f)
    c = class_closure([str(f)], {"PER": ["Q5"], "ORG": ["Q43229"], "LOC": ["Q515"]}, depth=3)
    got = dict(c.select("cls", "ner_type").iter_rows())
    assert (
        got["Q101"] == "ORG" and got["Q200"] == "LOC" and got["Q300"] == "ORG"
    )  # same depth: ORG before LOC
    assert got["Q500"] == "LOC"  # LOC at depth 1 beats ORG at depth 3
    assert dict(c.select("cls", "depth").iter_rows())["Q101"] == 2
    assert "Q400" not in got


def test_excluded_classes_win_over_core_labels_only_when_nearer(tmp_path: Path) -> None:
    # Q10 religion-like class: under the O root (Q9174) at depth 1 and under organization at depth 2
    df = pl.DataFrame(
        {
            "qid": ["Q10", "Q11", "Q12"],
            "p279": [["Q9174", "Q11"], ["Q43229"], ["Q9174", "Q43229"]],
        }
    )
    f = tmp_path / "e.parquet"
    df.write_parquet(f)
    roots = {"PER": ["Q5"], "ORG": ["Q43229"], "LOC": ["Q515"], "O": ["Q9174"]}
    got = dict(class_closure([str(f)], roots, depth=3).select("cls", "ner_type").iter_rows())
    assert got["Q10"] == "O"  # O at depth 1 beats ORG at depth 2
    assert got["Q12"] == "ORG"  # same depth: the core label (a country is also a `nation`)
    assert got["Q11"] == "ORG"
