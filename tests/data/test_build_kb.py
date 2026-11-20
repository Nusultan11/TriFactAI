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
