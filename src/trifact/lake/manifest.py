"""source.yaml manifests: origin, pinned revision, license decision, rows, shard hashes (ADR-0011, ADR-0017).

`verify_source` re-reads every shard listed in a manifest and checks its sha256, row count, and
that the source directory holds no shard the manifest does not know about.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, Field

from trifact.lake.layout import SHARD_SUFFIX, atomic_write_bytes, sha256_file

MANIFEST_NAME = "source.yaml"


class LicenseUse(StrEnum):
    allowed = "allowed"  # terms permit our use, incl. released models/derived data within the license
    research_only = "research_only"  # non-commercial / unconfirmed terms: train & eval only, release: false
    blocked = "blocked"  # must not be downloaded or used


class LicenseDecision(BaseModel):
    license: str
    use: LicenseUse
    release: bool  # may derived datasets / models trained on it be released (ADR-0015 lineage)
    evidence: str  # where the terms were checked, with date
    note: str = ""


class InputFile(BaseModel):
    path: str
    size: int
    sha256: str | None = None
    config: str | None = None
    split: str | None = None
    lang: str
    table: str
    rows: int


class ShardEntry(BaseModel):
    path: str  # relative to the source directory
    lang: str
    table: str
    rows: int
    bytes: int
    sha256: str


class RowCheck(BaseModel):
    scope: str  # e.g. "config=20231101.kk split=train"
    method: Literal[
        "parquet_footer",
        "hf_dataset_info",
        "hf_datasets_server",
        "line_count",
        "csv_record_count",
        "entity_count",
    ]
    expected: int
    actual: int

    @property
    def ok(self) -> bool:
        return self.expected == self.actual


class RunStats(BaseModel):
    seconds: float
    rows_per_s: float
    mb_per_s: float
    peak_rss_mb: float
    workers: int


class SourceManifest(BaseModel):
    name: str
    role: str
    origin: Literal["hf", "url"]
    hf_id: str | None = None
    url: str | None = None
    revision: str
    license: LicenseDecision
    downloaded_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    code_version: str
    partial: bool = False  # True for smoke runs with a row limit; such data never goes to DVC
    inputs: list[InputFile] = []
    shards: list[ShardEntry] = []
    row_checks: list[RowCheck] = []
    stats: RunStats | None = None

    @property
    def rows(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.shards:
            out[s.lang] = out.get(s.lang, 0) + s.rows
        return dict(sorted(out.items()))

    @property
    def bytes(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.shards:
            out[s.lang] = out.get(s.lang, 0) + s.bytes
        return dict(sorted(out.items()))

    def to_yaml(self) -> str:
        data = self.model_dump(mode="json")
        data["rows"] = self.rows
        data["bytes"] = self.bytes
        data["row_checks_ok"] = all(c.ok for c in self.row_checks)
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120)

    def write(self, src_dir: Path) -> Path:
        path = Path(src_dir) / MANIFEST_NAME
        atomic_write_bytes(path, self.to_yaml().encode("utf-8"))
        return path

    @classmethod
    def read(cls, src_dir: Path) -> SourceManifest:
        data = yaml.safe_load((Path(src_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))
        for derived in ("rows", "bytes", "row_checks_ok"):
            data.pop(derived, None)
        return cls.model_validate(data)


def verify_source(src_dir: Path, deep: bool = True) -> list[str]:
    """Return a list of integrity problems (empty = OK). `deep` re-hashes every shard."""
    src_dir = Path(src_dir)
    problems: list[str] = []
    try:
        m = SourceManifest.read(src_dir)
    except FileNotFoundError:
        return [f"{src_dir}: no {MANIFEST_NAME}"]
    listed = {s.path for s in m.shards}
    on_disk = {p.relative_to(src_dir).as_posix() for p in src_dir.rglob(f"*{SHARD_SUFFIX}")}
    problems += [f"{src_dir.name}: shard missing on disk: {p}" for p in sorted(listed - on_disk)]
    problems += [f"{src_dir.name}: shard not in manifest: {p}" for p in sorted(on_disk - listed)]
    stray = [p for p in src_dir.rglob("*") if p.is_file() and p.name.endswith(".tmp")]
    problems += [f"{src_dir.name}: leftover temp file {p.name}" for p in stray]
    for s in m.shards:
        path = src_dir / s.path
        if not path.is_file():
            continue
        if path.stat().st_size != s.bytes:
            problems.append(f"{s.path}: size {path.stat().st_size} != {s.bytes}")
            continue
        rows = pq.ParquetFile(path).metadata.num_rows
        if rows != s.rows:
            problems.append(f"{s.path}: rows {rows} != {s.rows}")
        if deep and sha256_file(path) != s.sha256:
            problems.append(f"{s.path}: sha256 mismatch")
    problems += [
        f"{m.name}: row check failed {c.scope}: {c.actual} != {c.expected}" for c in m.row_checks if not c.ok
    ]
    if m.license.use == LicenseUse.blocked:
        problems.append(f"{m.name}: license decision is 'blocked' but data is present")
    return problems
