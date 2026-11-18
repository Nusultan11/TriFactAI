"""Layer paths data/<layer>/<source>/<lang>/<table>-NNNNN.parquet and shard writing (ADR-0011, ADR-0017).

A shard is written to `<name>.tmp`, then fsynced and atomically renamed, so a crash never leaves a
half-written file under its final name. Rotation happens at `max_rows` rows or `max_bytes` bytes.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

LAYERS = ("raw", "clean", "curated", "task", "eval")
DEFAULT_TABLE = "part"
SHARD_SUFFIX = ".parquet"


@dataclass(frozen=True)
class LakeConfig:
    root: Path
    shard_rows: int = 1_000_000
    shard_max_mb: int = 512
    compression: str = "zstd"
    row_group_rows: int = 65_536

    @classmethod
    def from_params(cls, lake: dict, root: Path | None = None) -> LakeConfig:
        return cls(
            root=Path(root if root is not None else lake["root"]),
            shard_rows=int(lake["shard_rows"]),
            shard_max_mb=int(lake["shard_max_mb"]),
            compression=str(lake["compression"]),
        )


def source_dir(root: Path, layer: str, source: str) -> Path:
    if layer not in LAYERS:
        raise ValueError(f"unknown layer {layer!r}, expected one of {LAYERS}")
    return Path(root) / layer / source


def shard_name(table: str, index: int) -> str:
    return f"{table}-{index:05d}{SHARD_SUFFIX}"


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@dataclass(frozen=True)
class ShardInfo:
    path: Path
    rows: int
    bytes: int
    sha256: str


class ShardWriter:
    """Streams record batches into size-bounded Parquet shards `<name(i)>.parquet` in `out_dir`.

    Batches are buffered up to `row_group_rows` (or ~64 MB) so row groups stay readable in a
    streaming fashion regardless of how the input was chunked. All batches must share one schema.
    """

    def __init__(
        self,
        out_dir: Path,
        name: Callable[[int], str],
        cfg: LakeConfig,
        schema: pa.Schema | None = None,
    ):
        self.out_dir = Path(out_dir)
        self.name = name
        self.cfg = cfg
        self.schema = schema
        self.shards: list[ShardInfo] = []
        self._writer: pq.ParquetWriter | None = None
        self._sink: pa.NativeFile | None = None
        self._tmp: Path | None = None
        self._final: Path | None = None
        self._rows_in_shard = 0
        self._buffer: list[pa.RecordBatch] = []
        self._buffer_rows = 0
        self._buffer_bytes = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def rows(self) -> int:
        return sum(s.rows for s in self.shards) + self._rows_in_shard + self._buffer_rows

    def write(self, batch: pa.RecordBatch | pa.Table) -> None:
        batches = batch.to_batches() if isinstance(batch, pa.Table) else [batch]
        for b in batches:
            if b.num_rows == 0:
                continue
            if self.schema is None:
                self.schema = b.schema
            elif not b.schema.equals(self.schema):
                b = pa.RecordBatch.from_arrays(
                    [b.column(n).cast(self.schema.field(n).type) for n in self.schema.names],
                    schema=self.schema,
                )
            offset = 0
            while offset < b.num_rows:
                room = self.cfg.shard_rows - self._rows_in_shard - self._buffer_rows
                if room <= 0:
                    self._rotate()
                    continue
                piece = b.slice(offset, room)
                offset += piece.num_rows
                self._buffer.append(piece)
                self._buffer_rows += piece.num_rows
                self._buffer_bytes += piece.nbytes
                if self._buffer_rows >= self.cfg.row_group_rows or self._buffer_bytes >= 64 << 20:
                    self._flush()
                if self._rows_in_shard + self._buffer_rows >= self.cfg.shard_rows:
                    self._rotate()

    def _open(self) -> None:
        assert self.schema is not None
        self._final = self.out_dir / f"{self.name(len(self.shards))}{SHARD_SUFFIX}"
        self._tmp = self._final.with_name(self._final.name + ".tmp")
        self._sink = pa.OSFile(str(self._tmp), "wb")
        self._writer = pq.ParquetWriter(self._sink, self.schema, compression=self.cfg.compression)
        self._rows_in_shard = 0

    def _flush(self) -> None:
        if not self._buffer:
            return
        if self._writer is None:
            self._open()
        assert self._writer is not None and self._sink is not None
        table = pa.Table.from_batches(self._buffer, schema=self.schema)
        self._writer.write_table(table, row_group_size=max(table.num_rows, 1))
        self._rows_in_shard += table.num_rows
        self._buffer, self._buffer_rows, self._buffer_bytes = [], 0, 0
        if self._sink.tell() >= self.cfg.shard_max_mb << 20:
            self._rotate()

    def _rotate(self) -> None:
        self._flush()
        if self._writer is None:
            return
        assert self._sink is not None and self._tmp is not None and self._final is not None
        self._writer.close()
        self._sink.close()
        with open(self._tmp, "rb+") as f:
            os.fsync(f.fileno())
        os.replace(self._tmp, self._final)
        self.shards.append(
            ShardInfo(self._final, self._rows_in_shard, self._final.stat().st_size, sha256_file(self._final))
        )
        self._writer, self._sink, self._tmp, self._final = None, None, None, None
        self._rows_in_shard = 0

    def close(self) -> list[ShardInfo]:
        self._rotate()
        return self.shards

    def abort(self) -> None:
        if self._writer is not None:
            self._writer.close()
        if self._sink is not None:
            self._sink.close()
        if self._tmp is not None:
            self._tmp.unlink(missing_ok=True)
        self._writer, self._sink, self._tmp = None, None, None
