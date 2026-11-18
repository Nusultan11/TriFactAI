from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trifact.lake.layout import LakeConfig, ShardWriter, sha256_file, shard_name, source_dir


def _batch(n: int, start: int = 0, text: str = "x") -> pa.RecordBatch:
    return pa.RecordBatch.from_pydict({"id": list(range(start, start + n)), "text": [text] * n})


def test_paths_follow_adr_0011(tmp_path: Path) -> None:
    assert source_dir(tmp_path, "raw", "wikipedia") == tmp_path / "raw" / "wikipedia"
    assert shard_name("part", 7) == "part-00007.parquet"
    with pytest.raises(ValueError):
        source_dir(tmp_path, "bronze", "x")


def test_rotation_by_rows_keeps_every_row_in_order(tmp_path: Path) -> None:
    cfg = LakeConfig(root=tmp_path, shard_rows=250, row_group_rows=64)
    w = ShardWriter(tmp_path, lambda i: f"s-{i}", cfg)
    for k in range(10):
        w.write(_batch(97, start=k * 97))
    shards = w.close()
    assert [s.rows for s in shards] == [250, 250, 250, 220]
    ids = pa.concat_tables([pq.read_table(s.path) for s in shards])["id"].to_pylist()
    assert ids == list(range(970))
    for s in shards:
        assert s.sha256 == sha256_file(s.path)
        assert s.bytes == s.path.stat().st_size
        assert pq.ParquetFile(s.path).metadata.row_group(0).column(0).compression == "ZSTD"
    assert not list(tmp_path.glob("*.tmp"))


def test_rotation_by_bytes(tmp_path: Path) -> None:
    cfg = LakeConfig(root=tmp_path, shard_rows=10**9, shard_max_mb=1, row_group_rows=1000)
    w = ShardWriter(tmp_path, lambda i: f"s-{i}", cfg)
    import os

    for k in range(20):  # incompressible text so the file really grows
        w.write(pa.RecordBatch.from_pydict({"t": [os.urandom(200).hex() for _ in range(1000)]}))
    shards = w.close()
    assert len(shards) > 1
    assert all(s.bytes < 3 << 20 for s in shards)
    assert sum(s.rows for s in shards) == 20_000


def test_abort_leaves_no_partial_file(tmp_path: Path) -> None:
    cfg = LakeConfig(root=tmp_path, shard_rows=1000, row_group_rows=10)
    w = ShardWriter(tmp_path, lambda i: f"s-{i}", cfg)
    w.write(_batch(50))
    assert list(tmp_path.glob("*.tmp"))
    w.abort()
    assert not list(tmp_path.iterdir())


def test_batches_are_cast_to_the_first_schema(tmp_path: Path) -> None:
    cfg = LakeConfig(root=tmp_path)
    w = ShardWriter(tmp_path, lambda i: f"s-{i}", cfg)
    w.write(pa.RecordBatch.from_pydict({"a": pa.array([1], pa.int64()), "b": pa.array(["x"])}))
    w.write(pa.RecordBatch.from_pydict({"a": pa.array([2], pa.int32()), "b": pa.array([None], pa.string())}))
    (s,) = w.close()
    assert pq.read_table(s.path).to_pydict() == {"a": [1, 2], "b": ["x", None]}
