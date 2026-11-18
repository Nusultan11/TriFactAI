"""Shared runner of the curation stages clean / dedup / decontaminate (ADR-0011, ADR-0014).

Work unit = one input Parquet shard. A unit writes its outputs straight under their final names
`<lang>/<table>-<unit:05d>-<k:02d>.parquet` (each file atomically, see `ShardWriter`) and then a
marker `_work/<unit>.json` with the shard list and counters. A restart deletes the files of
unfinished units and skips marked ones, so a crash costs at most the units in flight.

The output directory carries a fingerprint = sha256 over the stage name, its parameters, the
source of the code it runs and the sha256 of every input manifest. A finished dataset with the
same fingerprint is only verified on re-run; a different fingerprint wipes the directory first.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from trifact.lake.layout import (
    SHARD_SUFFIX,
    LakeConfig,
    ShardInfo,
    ShardWriter,
    atomic_write_bytes,
    sha256_file,
)
from trifact.lake.manifest import (
    DATASET_NAME,
    DatasetInput,
    DatasetManifest,
    InheritedLicense,
    ShardEntry,
    SourceManifest,
    verify_dataset,
)

log = logging.getLogger("trifact.curation")
WORK = "_work"
BATCH_BYTES = 32 << 20


@dataclass(frozen=True)
class Unit:
    idx: int
    source: str
    path: str  # input shard, relative to the lake root
    lang: str  # language of the input partition
    rows: int
    domain: str
    limit: int | None = None  # pilot: read only the first N rows


def fingerprint(stage: str, params: dict[str, Any], code: Iterable[Path], inputs: Iterable[Path]) -> str:
    h = hashlib.sha256()
    h.update(stage.encode())
    h.update(json.dumps(params, sort_keys=True, default=str).encode())
    for p in sorted(Path(c) for c in code):
        files = sorted(p.rglob("*.py")) if p.is_dir() else [p]
        for f in files:
            h.update(f.name.encode())
            h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    for p in sorted(Path(i) for i in inputs):
        h.update(sha256_file(p).encode())
    return h.hexdigest()[:32]


def dataset_input(root: Path, manifest: Path, rows: dict[str, int]) -> DatasetInput:
    return DatasetInput(path=manifest.relative_to(root).as_posix(), sha256=sha256_file(manifest), rows=rows)


def licenses_of(manifests: Iterable[SourceManifest | DatasetManifest]) -> list[InheritedLicense]:
    out: dict[str, InheritedLicense] = {}
    for m in manifests:
        if isinstance(m, SourceManifest):
            lic = m.license
            out[m.name] = InheritedLicense(
                source=m.name, license=lic.license, use=lic.use, release=lic.release
            )
        else:
            out.update({lic.source: lic for lic in m.licenses})
    return sorted(out.values(), key=lambda x: x.source)


def rows_per_batch(pf: pq.ParquetFile, target_bytes: int = BATCH_BYTES) -> int:
    md = pf.metadata
    if md.num_rows == 0:
        return 1024
    raw = sum(md.row_group(i).total_byte_size for i in range(md.num_row_groups))
    return max(16, min(65_536, int(target_bytes / max(raw / md.num_rows, 1))))


def iter_batches(path: Path, columns: list[str] | None, limit: int | None) -> Iterator[pa.RecordBatch]:
    pf = pq.ParquetFile(path)
    seen = 0
    for b in pf.iter_batches(batch_size=rows_per_batch(pf), columns=columns):
        if limit is not None and seen + b.num_rows > limit:
            b = b.slice(0, limit - seen)
        seen += b.num_rows
        if b.num_rows:
            yield b
        if limit is not None and seen >= limit:
            return


class UnitWriter:
    """Per-language ShardWriters for one unit: `<out>/<lang>/<table>-<unit>-<k>.parquet`."""

    def __init__(
        self, out_dir: Path, unit: int, table: str, cfg: LakeConfig, schema: pa.Schema | None = None
    ):
        self.out_dir, self.unit, self.table, self.cfg, self.schema = out_dir, unit, table, cfg, schema
        self.writers: dict[str, ShardWriter] = {}

    def write(self, lang: str, batch: pa.Table) -> None:
        if batch.num_rows == 0:
            return
        w = self.writers.get(lang)
        if w is None:
            name = f"{self.table}-{self.unit:05d}"
            w = self.writers[lang] = ShardWriter(
                self.out_dir / lang, lambda k, n=name: f"{n}-{k:02d}", self.cfg, self.schema
            )
        w.write(batch)

    def close(self) -> list[dict[str, Any]]:
        out = []
        for lang, w in sorted(self.writers.items()):
            for s in w.close():
                out.append(shard_entry(self.out_dir, s, lang, self.table).model_dump())
        return out

    def abort(self) -> None:
        for w in self.writers.values():
            w.abort()


def shard_entry(out_dir: Path, s: ShardInfo, lang: str, table: str) -> ShardEntry:
    return ShardEntry(
        path=s.path.relative_to(out_dir).as_posix(),
        lang=lang,
        table=table,
        rows=s.rows,
        bytes=s.bytes,
        sha256=s.sha256,
    )


def write_single(
    out_dir: Path, rel: str, table: pa.Table, lang: str, name: str, cfg: LakeConfig
) -> ShardEntry:
    """One small side table (e.g. a boilerplate list) as a single shard listed in the manifest."""
    path = out_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression=cfg.compression)
    tmp.replace(path)
    return ShardEntry(
        path=rel,
        lang=lang,
        table=name,
        rows=table.num_rows,
        bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )


class StageDir:
    """Fingerprinted, resumable output directory of one dataset."""

    def __init__(self, out_dir: Path, fp: str):
        self.out = Path(out_dir)
        self.fp = fp
        self.work = self.out / WORK

    def complete(self) -> bool:
        """True when a finished dataset with this fingerprint is present and verifies."""
        if not (self.out / DATASET_NAME).is_file():
            return False
        m = DatasetManifest.read(self.out)
        if m.fingerprint != self.fp:
            return False
        problems = verify_dataset(self.out, deep=False)
        for p in problems:
            log.warning("%s: %s", self.out, p)
        return not problems

    def prepare(self) -> None:
        plan = self.work / "plan.json"
        same = plan.is_file() and json.loads(plan.read_text())["fingerprint"] == self.fp
        if not same and self.out.exists():
            log.info("%s: fingerprint changed or no plan, starting over", self.out)
            shutil.rmtree(self.out)
        self.work.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(plan, json.dumps({"fingerprint": self.fp}).encode())

    def done(self) -> dict[int, dict[str, Any]]:
        out = {}
        for p in self.work.glob("unit-*.json"):
            out[int(p.stem.split("-")[1])] = json.loads(p.read_text(encoding="utf-8"))
        return out

    def clear_unit(self, idx: int) -> None:
        for p in self.out.rglob(f"*-{idx:05d}-*{SHARD_SUFFIX}*"):
            p.unlink()

    def mark(self, idx: int, result: dict[str, Any]) -> None:
        atomic_write_bytes(
            self.work / f"unit-{idx:05d}.json", json.dumps(result, ensure_ascii=False).encode()
        )

    def finish(self, manifest: DatasetManifest) -> None:
        manifest.write(self.out)
        shutil.rmtree(self.work, ignore_errors=True)


def run_units(
    units: list[Unit],
    sd: StageDir,
    fn: Callable[[Unit], dict[str, Any]],
    workers: int,
    init: Callable[..., None] | None = None,
    initargs: tuple[Any, ...] = (),
) -> dict[int, dict[str, Any]]:
    """Run `fn` over units not yet marked in `sd`, in a process pool; returns every unit's result."""
    results = sd.done()
    todo = [u for u in units if u.idx not in results]
    log.info(
        "%s: %d units, %d done, %d to run on %d workers", sd.out, len(units), len(results), len(todo), workers
    )
    for u in todo:
        sd.clear_unit(u.idx)
    if not todo:
        return results
    # every worker's Polars pool defaults to all cores and keeps per-thread arenas: N workers x 16
    # threads oversubscribe the CPU and commit ~3x the resident memory (Windows commit limit)
    os.environ["POLARS_MAX_THREADS"] = str(max(1, (os.cpu_count() or 1) // workers))
    with ProcessPoolExecutor(max_workers=workers, initializer=init, initargs=initargs) as pool:
        pending: dict[Future[dict[str, Any]], Unit] = {}
        queue = list(todo)
        while queue or pending:
            while queue and len(pending) < workers:
                u = queue.pop(0)
                pending[pool.submit(fn, u)] = u
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for f in finished:
                u = pending.pop(f)
                res = f.result()
                sd.mark(u.idx, res)
                results[u.idx] = res
                log.info("%s: unit %d done (%d/%d)", sd.out.name, u.idx, len(results), len(units))
    return results


def merge_counts(results: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in results:
        for k, by_lang in r.get(key, {}).items():
            d = out.setdefault(k, {})
            for lang, n in by_lang.items():
                d[lang] = d.get(lang, 0) + n
    return {k: dict(sorted(v.items())) for k, v in out.items()}


def sum_by_lang(results: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in results:
        for lang, n in r.get(key, {}).items():
            out[lang] = out.get(lang, 0) + n
    return dict(sorted(out.items()))


def unit_dict(u: Unit) -> dict[str, Any]:
    return asdict(u)


def stage_report(m: DatasetManifest) -> dict[str, Any]:
    """Metrics JSON of one dataset (DVC `metrics`): volumes, drops by reason, stage metrics, run resources."""
    st = m.stats
    return {
        "name": m.name, "fingerprint": m.fingerprint, "partial": m.partial,
        "rows_in": m.rows_in, "rows_out": m.rows_of(), "bytes_out": m.bytes_of(), "drops": m.drops,
        "drops_total": {k: sum(v.values()) for k, v in m.drops.items()}, **m.metrics,
        "seconds": st.seconds if st else None, "docs_per_s": st.rows_per_s if st else None,
        "mb_per_s": st.mb_per_s if st else None, "peak_rss_mb": st.peak_rss_mb if st else None,
        "workers": st.workers if st else None,
    }  # fmt: skip


def write_report(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
