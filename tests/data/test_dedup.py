from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from trifact.data.dedup import clusters, link_equal


def test_link_equal_points_every_member_to_the_first() -> None:
    a, b = link_equal(np.array([7, 3, 7, 3, 7, 9], dtype=np.uint64))
    edges = sorted(zip(a.tolist(), b.tolist(), strict=True))
    assert edges == [(2, 0), (3, 1), (4, 0)]


def test_clusters_join_bands_transitively(tmp_path: Path) -> None:
    # doc 0~1 by band 0, 1~2 by band 1, 3 alone, 4 is an exact copy of 3
    sig = pl.DataFrame(
        {
            "doc_id": [f"d{i}" for i in range(5)],
            "exact": np.array([1, 2, 3, 4, 4], dtype=np.uint64),
            "b00": np.array([10, 10, 11, 12, 13], dtype=np.uint64),
            "b01": np.array([20, 21, 21, 22, 23], dtype=np.uint64),
        }
    )
    sig.head(2).write_parquet(tmp_path / "sig-00000.parquet")
    sig.tail(3).write_parquet(tmp_path / "sig-00001.parquet")
    files = [tmp_path / "sig-00000.parquet", tmp_path / "sig-00001.parquet"]
    root, size, exact, edges = clusters(files, 5, 2)
    assert root.tolist() == [0, 0, 0, 3, 3]
    assert size.tolist() == [3, 3, 3, 2, 2]
    assert exact.tolist() == [1, 2, 3, 4, 4]
    assert edges == {"exact": 1, "b00": 1, "b01": 1}
