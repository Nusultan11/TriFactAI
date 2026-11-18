"""Manifests of the lake (ADR-0011, ADR-0017).

`source.yaml` (raw layer): origin, pinned revision, license decision, rows, shard hashes.
`dataset.yaml` (derived layers): inputs with their manifest hashes, code version, parameters and
fingerprint, rows by language, drops by reason, inherited licenses and `release`, shard hashes,
run stats and stage metrics; `card.md` next to it is the human-readable data card, rendered from
the manifest so the two never disagree.

`verify_source` / `verify_dataset` re-read every listed shard and check its sha256, row count, and
that the directory holds no shard the manifest does not know about.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, Field

from trifact.lake.layout import SHARD_SUFFIX, atomic_write_bytes, sha256_file

MANIFEST_NAME = "source.yaml"
DATASET_NAME = "dataset.yaml"
CARD_NAME = "card.md"


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


def _verify_shards(src_dir: Path, shards: list[ShardEntry], deep: bool) -> list[str]:
    problems: list[str] = []
    listed = {s.path for s in shards}
    on_disk = {p.relative_to(src_dir).as_posix() for p in src_dir.rglob(f"*{SHARD_SUFFIX}")}
    problems += [f"{src_dir.name}: shard missing on disk: {p}" for p in sorted(listed - on_disk)]
    problems += [f"{src_dir.name}: shard not in manifest: {p}" for p in sorted(on_disk - listed)]
    stray = [p for p in src_dir.rglob("*") if p.is_file() and p.name.endswith(".tmp")]
    problems += [f"{src_dir.name}: leftover temp file {p.name}" for p in stray]
    for s in shards:
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
    return problems


def verify_source(src_dir: Path, deep: bool = True) -> list[str]:
    """Return a list of integrity problems (empty = OK). `deep` re-hashes every shard."""
    src_dir = Path(src_dir)
    try:
        m = SourceManifest.read(src_dir)
    except FileNotFoundError:
        return [f"{src_dir}: no {MANIFEST_NAME}"]
    problems = _verify_shards(src_dir, m.shards, deep)
    problems += [
        f"{m.name}: row check failed {c.scope}: {c.actual} != {c.expected}" for c in m.row_checks if not c.ok
    ]
    if m.license.use == LicenseUse.blocked:
        problems.append(f"{m.name}: license decision is 'blocked' but data is present")
    return problems


# ----------------------------------------------------------------------------- derived datasets


class DatasetInput(BaseModel):
    path: str  # manifest path relative to the lake root
    sha256: str  # of the manifest file: any change of an input changes the fingerprint
    rows: dict[str, int]


class InheritedLicense(BaseModel):
    source: str
    license: str
    use: LicenseUse
    release: bool


class DatasetManifest(BaseModel):
    name: str  # e.g. "clean/wikipedia"
    layer: str
    stage: str
    source: str
    description: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    code_version: str
    fingerprint: str  # hash of code, parameters and input manifests: equal fingerprint = same output
    params: dict[str, Any]
    inputs: list[DatasetInput]
    licenses: list[InheritedLicense]
    rows_in: dict[str, int]  # input documents by language
    drops: dict[str, dict[str, int]] = {}  # reason -> language -> documents
    shards: list[ShardEntry] = []
    stats: RunStats | None = None
    metrics: dict[str, Any] = {}  # stage-specific measurements (filters, dedup, contamination)
    partial: bool = False  # pilot / limited runs; never versioned in DVC

    @property
    def release(self) -> bool:
        return all(lic.release for lic in self.licenses)

    def rows_of(self, table: str = "part") -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.shards:
            if s.table == table:
                out[s.lang] = out.get(s.lang, 0) + s.rows
        return dict(sorted(out.items()))

    def bytes_of(self, table: str = "part") -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.shards:
            if s.table == table:
                out[s.lang] = out.get(s.lang, 0) + s.bytes
        return dict(sorted(out.items()))

    def to_yaml(self) -> str:
        data = self.model_dump(mode="json")
        data["rows"] = self.rows_of()
        data["bytes"] = self.bytes_of()
        data["release"] = self.release
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120)

    def write(self, out_dir: Path) -> Path:
        path = Path(out_dir) / DATASET_NAME
        atomic_write_bytes(path, self.to_yaml().encode("utf-8"))
        atomic_write_bytes(Path(out_dir) / CARD_NAME, render_card(self).encode("utf-8"))
        return path

    @classmethod
    def read(cls, out_dir: Path) -> DatasetManifest:
        data = yaml.safe_load((Path(out_dir) / DATASET_NAME).read_text(encoding="utf-8"))
        for derived in ("rows", "bytes", "release"):
            data.pop(derived, None)
        return cls.model_validate(data)


def _fmt(n: float) -> str:
    return f"{n:,}".replace(",", " ") if isinstance(n, int) else f"{n:.4g}"


def render_card(m: DatasetManifest) -> str:
    """Data card (markdown) rendered from the manifest."""
    rows, size = m.rows_of(), m.bytes_of()
    langs = sorted(set(rows) | set(m.rows_in))
    out = [
        f"# Data card: `{m.name}`",
        "",
        m.description,
        "",
        "| | |",
        "|---|---|",
        f"| Layer / stage | `{m.layer}` / `{m.stage}` |",
        f"| Source | `{m.source}` |",
        f"| Created | {m.created_at} |",
        f"| Code version | `{m.code_version}` |",
        f"| Fingerprint | `{m.fingerprint}` |",
        f"| Release allowed | {'yes' if m.release else '**no**'} (inherited from inputs, ADR-0015) |",
        f"| Partial (pilot) | {'yes' if m.partial else 'no'} |",
        "",
        "## Volumes",
        "",
        "| Language | Input docs | Output docs | Kept, % | Parquet, MB |",
        "|---|---|---|---|---|",
    ]
    for lang in langs:
        rin, rout = m.rows_in.get(lang, 0), rows.get(lang, 0)
        pct = f"{100 * rout / rin:.1f}" if rin else "-"
        out.append(f"| {lang} | {_fmt(rin)} | {_fmt(rout)} | {pct} | {size.get(lang, 0) / 2**20:.1f} |")
    out += ["", "Input languages are those of the input layer; output languages are re-detected.", ""]
    out += ["## Removed documents by reason", ""]
    if m.drops:
        dl = sorted({lg for v in m.drops.values() for lg in v})
        out += ["| Reason | " + " | ".join(dl) + " | Total |", "|---" * (len(dl) + 2) + "|"]
        for reason, by_lang in m.drops.items():
            cells = " | ".join(_fmt(by_lang.get(lg, 0)) for lg in dl)
            out.append(f"| {reason} | {cells} | {_fmt(sum(by_lang.values()))} |")
    else:
        out.append("None.")
    out += ["", "## Provenance", "", "| Input manifest | sha256 | Rows |", "|---|---|---|"]
    out += [f"| `{i.path}` | `{i.sha256[:16]}` | {_fmt(sum(i.rows.values()))} |" for i in m.inputs]
    out += ["", "## Licenses of inputs", "", "| Source | License | Use | Release |", "|---|---|---|---|"]
    out += [
        f"| {lic.source} | {lic.license} | {lic.use.value} | {'yes' if lic.release else 'no'} |"
        for lic in m.licenses
    ]
    dump = yaml.safe_dump(m.params, sort_keys=False, allow_unicode=True, width=110).rstrip()
    out += ["", "## Processing parameters", "", "```yaml", dump, "```"]
    if m.metrics:
        dump = yaml.safe_dump(m.metrics, sort_keys=False, allow_unicode=True, width=110).rstrip()
        out += ["", "## Stage metrics", "", "```yaml", dump, "```"]
    if m.stats:
        st = m.stats
        out += [
            "",
            "## Run",
            "",
            (
                f"{st.seconds} s, {st.rows_per_s} docs/s, {st.mb_per_s} MB/s, peak RSS {st.peak_rss_mb} MB, "
                f"{st.workers} workers."
            ),
        ]
    out += ["", "## Shards", "", "| Path | Rows | Bytes | sha256 |", "|---|---|---|---|"]
    out += [f"| `{s.path}` | {_fmt(s.rows)} | {_fmt(s.bytes)} | `{s.sha256[:16]}` |" for s in m.shards]
    return "\n".join(out) + "\n"


def verify_dataset(out_dir: Path, deep: bool = True) -> list[str]:
    """Integrity problems of a derived dataset (empty = OK), including a stale data card."""
    out_dir = Path(out_dir)
    try:
        m = DatasetManifest.read(out_dir)
    except FileNotFoundError:
        return [f"{out_dir}: no {DATASET_NAME}"]
    problems = _verify_shards(out_dir, m.shards, deep)
    card = out_dir / CARD_NAME
    if not card.is_file() or card.read_text(encoding="utf-8") != render_card(m):
        problems.append(f"{m.name}: {CARD_NAME} missing or out of date")
    return problems
