"""Stage acquire: copy a pinned HF dataset into raw Parquet shards and write its source.yaml (ADR-0011, ADR-0017).

Work unit = one repository file. Files are downloaded and converted in parallel (threads: download
is I/O, Arrow decoding and zstd release the GIL). Each finished file leaves `_work/<idx>.json`; a
restart skips those files, so an interruption costs at most the files that were in flight.
When every file is done, shards are renamed into `<lang>/<table>-NNNNN.parquet` in file order and
the manifest is written; then rows are checked against the repository metadata.

Usage: python -m trifact.data.acquire --source NAME [--root DIR --limit-rows N] [--workers N]
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import gzip
import hashlib
import io
import json
import logging
import os
import re
import shutil
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any

import httpx
import orjson
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import yaml

from trifact.core.params import ROOT, load_params
from trifact.lake.layout import (
    DEFAULT_TABLE,
    LakeConfig,
    ShardWriter,
    atomic_write_bytes,
    sha256_file,
    shard_name,
    source_dir,
)
from trifact.lake.manifest import (
    InputFile,
    LicenseDecision,
    LicenseUse,
    RowCheck,
    ShardEntry,
    SourceManifest,
    verify_source,
)
from trifact.lake.stats import ResourceMonitor, git_sha, log_mlflow

log = logging.getLogger("trifact.acquire")

SPLIT_RE = re.compile(r"(?:^|[/_.-])(train|validation|valid|val|dev|test(?:-[ab])?)(?=[/_.-]|$)")
SPLIT_ALIASES = {"valid": "validation", "val": "validation", "dev": "validation"}
BATCH_BYTES = 32 << 20  # target uncompressed bytes per Arrow batch
LINE_BATCH = 20_000


# ----------------------------------------------------------------------------- planning


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    sha256: str | None


@dataclass
class Planned:
    idx: int
    path: str
    size: int
    sha256: str | None
    lang: str
    table: str
    config: str | None
    split: str | None  # normalised: train / validation / test / test-a ...
    meta_split: str | None  # split name as the repository metadata calls it
    fmt: str
    columns: list[str] | None = None


@dataclass
class Done:
    """Result of one input file, persisted as _work/<idx>.json."""

    idx: int
    rows: int
    expected: int | None  # parquet footer, independent line/record count
    method: str | None
    sha256: str | None
    shards: list[dict[str, Any]] = field(default_factory=list)


def file_format(path: str) -> str:
    name = path.lower().removesuffix(".gz")
    for ext in ("parquet", "arrow", "csv", "tsv", "jsonl", "json"):
        if name.endswith("." + ext):
            return "jsonl" if ext == "json" else ext
    raise ValueError(f"unsupported file format: {path}")


def natural_key(s: str) -> list[Any]:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def split_of(path: str) -> tuple[str | None, str | None]:
    m = None
    for m in SPLIT_RE.finditer(path):
        pass  # the last match is the most specific (file name over directory)
    if m is None:
        return None, None
    raw = m.group(1)
    return raw, SPLIT_ALIASES.get(raw, raw)


def plan(spec: dict[str, Any], files: list[RemoteFile]) -> list[Planned]:
    out: list[Planned] = []
    taken: set[str] = set()
    for entry in spec["files"]:
        matched = sorted(
            (f for f in files if fnmatch.fnmatchcase(f.path, entry["glob"])),
            key=lambda f: natural_key(f.path),
        )
        matched = [f for f in matched if f.path not in taken]
        if not matched:
            raise ValueError(f"glob {entry['glob']!r} matches no file in the repository")
        for f in matched:
            taken.add(f.path)
            raw, split = split_of(f.path)
            if "split" in entry:
                raw = split = entry["split"]
            meta_split = entry["meta_split"].format(split=raw) if "meta_split" in entry and raw else raw
            out.append(
                Planned(
                    idx=len(out),
                    path=f.path,
                    size=f.size,
                    sha256=f.sha256,
                    lang=entry["lang"],
                    table=entry.get("table", DEFAULT_TABLE),
                    config=entry.get("config"),
                    split=split,
                    meta_split=meta_split,
                    fmt=file_format(f.path),
                    columns=entry.get("columns"),
                )
            )
    return out


# ----------------------------------------------------------------------------- readers


def _open(path: Path) -> IO[bytes]:
    return gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb")


def _rows_per_batch(pf: pq.ParquetFile) -> int:
    md = pf.metadata
    if md.num_rows == 0 or md.num_row_groups == 0:
        return 65_536
    avg = sum(md.row_group(i).total_byte_size for i in range(md.num_row_groups)) / md.num_rows
    return int(max(256, min(65_536, BATCH_BYTES / max(avg, 1))))


def read_parquet(path: Path) -> tuple[Iterator[pa.RecordBatch], int]:
    pf = pq.ParquetFile(path)
    return pf.iter_batches(batch_size=_rows_per_batch(pf)), pf.metadata.num_rows


def read_arrow(path: Path) -> Iterator[pa.RecordBatch]:
    try:
        reader: Any = pa.ipc.open_stream(pa.memory_map(str(path)))
        yield from reader
    except pa.ArrowInvalid:
        f = pa.ipc.open_file(pa.memory_map(str(path)))
        for i in range(f.num_record_batches):
            yield f.get_batch(i)


def read_csv(path: Path) -> Iterator[pa.RecordBatch]:
    with _open(path) as f:
        header = next(csv.reader(io.TextIOWrapper(f, encoding="utf-8", newline="")))
    with _open(path) as f:
        reader = pacsv.open_csv(
            f,
            read_options=pacsv.ReadOptions(block_size=BATCH_BYTES, use_threads=True),
            parse_options=pacsv.ParseOptions(newlines_in_values=True),
            convert_options=pacsv.ConvertOptions(
                column_types={c: pa.string() for c in header}, strings_can_be_null=False
            ),
        )
        yield from reader


def count_csv_records(path: Path) -> int:
    """Independent record count with the stdlib parser (handles quoted newlines)."""
    csv.field_size_limit(2**31 - 1)  # whole books are single fields in some corpora
    with _open(path) as f:
        r = csv.reader(io.TextIOWrapper(f, encoding="utf-8", newline=""))
        next(r)
        return sum(1 for _ in r)


def read_tsv(path: Path, columns: list[str]) -> Iterator[pa.RecordBatch]:
    """Headerless TSV without quoting (MS MARCO style): split each line on the first n-1 tabs."""
    schema = pa.schema([(c, pa.string()) for c in columns])
    cols: list[list[str]] = [[] for _ in columns]
    n = len(columns)
    with _open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.decode("utf-8").rstrip("\r\n")
            parts = line.split("\t", n - 1)
            if len(parts) != n:
                parts = line.split(None, n - 1) if n > 1 else parts  # TREC qrels may use spaces
            if len(parts) != n:
                raise ValueError(f"{path.name}:{lineno}: expected {n} fields, got {len(parts)}")
            for c, v in zip(cols, parts, strict=True):
                c.append(v)
            if len(cols[0]) >= LINE_BATCH:
                yield pa.RecordBatch.from_arrays([pa.array(c, pa.string()) for c in cols], schema=schema)
                cols = [[] for _ in columns]
    if cols[0]:
        yield pa.RecordBatch.from_arrays([pa.array(c, pa.string()) for c in cols], schema=schema)


def count_lines(path: Path) -> int:
    n, last = 0, b"\n"
    with _open(path) as f:
        while block := f.read(16 << 20):
            n += block.count(b"\n")
            last = block[-1:]
    return n + (last != b"\n")


def _json_batches(path: Path) -> Iterator[list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    with _open(path) as f:
        for raw in f:
            if raw.strip():
                rows.append(orjson.loads(raw))
            if len(rows) >= LINE_BATCH:
                yield rows
                rows = []
    if rows:
        yield rows


def jsonl_schema(path: Path) -> pa.Schema:
    # pa.array over all rows infers the union of keys (from_pylist would look at the first row only)
    schemas = [pa.schema(list(pa.array(rows).type)) for rows in _json_batches(path)]
    return pa.unify_schemas(schemas, promote_options="permissive") if schemas else pa.schema([])


def read_jsonl(path: Path, schema: pa.Schema) -> Iterator[pa.RecordBatch]:
    """Two passes over a local file: the unified schema first, then typed batches."""
    for rows in _json_batches(path):
        yield from pa.Table.from_pylist(rows, schema=schema).to_batches()


# ----------------------------------------------------------------------------- conversion


def with_lineage(batch: pa.RecordBatch, config: str | None, split: str | None) -> pa.RecordBatch:
    n = batch.num_rows
    return batch.append_column("_config", pa.array([config] * n, pa.string())).append_column(
        "_split", pa.array([split] * n, pa.string())
    )


def convert(p: Planned, local: Path, work: Path, cfg: LakeConfig, limit: int | None) -> Done:
    expected: int | None = None
    method: str | None = None
    if p.fmt == "parquet":
        batches, expected = read_parquet(local)
        method = "parquet_footer"
    elif p.fmt == "arrow":
        batches = read_arrow(local)
    elif p.fmt == "csv":
        batches = read_csv(local)
    elif p.fmt == "tsv":
        if not p.columns:
            raise ValueError(f"{p.path}: headerless TSV needs `columns` in params")
        batches = read_tsv(local, p.columns)
    else:
        batches = read_jsonl(local, jsonl_schema(local))

    out = work / p.table / p.lang
    writer = ShardWriter(out, lambda j: f"{p.idx:05d}-{j:03d}", cfg)
    rows = 0
    try:
        for b in batches:
            if limit is not None and rows + b.num_rows > limit:
                b = b.slice(0, limit - rows)
            writer.write(with_lineage(b, p.config, p.split))
            rows += b.num_rows
            if limit is not None and rows >= limit:
                break
        infos = writer.close()
    except BaseException:
        writer.abort()
        raise
    if expected is None and limit is None:
        if p.fmt in ("tsv", "jsonl"):
            expected, method = count_lines(local), "line_count"
        elif p.fmt == "csv":
            expected, method = count_csv_records(local), "csv_record_count"
    if limit is not None and expected is not None:
        expected = min(expected, limit)
    return Done(
        idx=p.idx,
        rows=rows,
        expected=expected,
        method=method,
        sha256=p.sha256,
        shards=[
            {
                "path": i.path.relative_to(work).as_posix(),
                "rows": i.rows,
                "bytes": i.bytes,
                "sha256": i.sha256,
            }
            for i in infos
        ],
    )


# ----------------------------------------------------------------------------- metadata


def hf_files(repo: str, revision: str) -> list[RemoteFile]:
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    tree = HfApi().list_repo_tree(repo, revision=revision, repo_type="dataset", recursive=True)
    return [
        RemoteFile(f.path, f.size, f.lfs.sha256 if f.lfs else None) for f in tree if isinstance(f, RepoFile)
    ]


def readme_counts(repo: str, revision: str) -> dict[tuple[str, str], int]:
    """num_examples per (config, split) from the README dataset_info at the pinned revision."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    try:
        text = Path(hf_hub_download(repo, "README.md", repo_type="dataset", revision=revision)).read_text(
            encoding="utf-8"
        )
    except (OSError, HfHubHTTPError):  # a repository without README is a normal case
        return {}
    m = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    info = (yaml.safe_load(m.group(1)) or {}).get("dataset_info") if m else None
    info = [info] if isinstance(info, dict) else info or []
    return {
        (str(d.get("config_name", "default")), s["name"]): int(s["num_examples"])
        for d in info
        for s in d.get("splits", [])
        if "num_examples" in s
    }


def server_counts(repo: str) -> dict[tuple[str, str], int]:
    """Exact (non-partial) counts from the HF dataset viewer; it describes the current main revision."""
    token = os.environ.get("HF_TOKEN")
    try:
        r = httpx.get(
            "https://datasets-server.huggingface.co/size",
            params={"dataset": repo},
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=60,
        )
        data = r.json()
    except (httpx.HTTPError, ValueError):
        return {}
    if "size" not in data or data.get("partial"):
        return {}
    return {(s["config"], s["split"]): int(s["num_rows"]) for s in data["size"]["splits"]}


def row_checks(
    planned: list[Planned],
    done: dict[int, Done],
    meta: dict[tuple[str, str], int],
    meta_method: str,
    partial: bool,
) -> list[RowCheck]:
    checks: list[RowCheck] = []
    groups: dict[tuple[str, str, str], list[Planned]] = {}
    for p in planned:
        groups.setdefault((p.table, p.lang, p.split or "-"), []).append(p)
    for (table, lang, split), ps in groups.items():
        by_method: dict[str, list[Planned]] = {}
        for p in ps:
            if done[p.idx].method:
                by_method.setdefault(done[p.idx].method or "", []).append(p)
        for method, mp in by_method.items():
            checks.append(
                RowCheck(
                    scope=f"table={table} lang={lang} split={split} files={len(mp)}",
                    method=method,  # type: ignore[arg-type]
                    expected=sum(done[p.idx].expected or 0 for p in mp),
                    actual=sum(done[p.idx].rows for p in mp),
                )
            )
    if partial or not meta:
        return checks
    by_meta: dict[tuple[str, str], int] = {}
    for p in planned:
        if p.config is not None and p.meta_split is not None:
            key = (p.config, p.meta_split)
            if key not in meta and (p.config, p.split or "") in meta:
                key = (p.config, p.split or "")
            by_meta[key] = by_meta.get(key, 0) + done[p.idx].rows
    for key, actual in sorted(by_meta.items()):
        if key in meta:
            checks.append(
                RowCheck(
                    scope=f"config={key[0]} split={key[1]}",
                    method=meta_method,
                    expected=meta[key],
                    actual=actual,
                )  # type: ignore[arg-type]
            )
        else:
            log.warning("no metadata count for config=%s split=%s", *key)
    return checks


# ----------------------------------------------------------------------------- stage


def plan_fingerprint(spec: dict[str, Any], planned: list[Planned], limit: int | None) -> str:
    blob = json.dumps(
        {"revision": spec["revision"], "files": [asdict(p) for p in planned], "limit": limit}, sort_keys=True
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def finalize(src: Path, work: Path, planned: list[Planned], done: dict[int, Done]) -> list[ShardEntry]:
    """Rename work shards into <lang>/<table>-NNNNN.parquet; resumable through _work/finalize.json."""
    fin = work / "finalize.json"
    if fin.exists():
        moves = json.loads(fin.read_text())
    else:
        moves, counters = [], {}
        for p in planned:  # planned is in file order
            for s in done[p.idx].shards:
                key = (p.lang, p.table)
                counters[key] = counters.get(key, -1) + 1
                dst = f"{p.lang}/{shard_name(p.table, counters[key])}"
                moves.append(
                    {
                        "src": s["path"],
                        "dst": dst,
                        "lang": p.lang,
                        "table": p.table,
                        "rows": s["rows"],
                        "bytes": s["bytes"],
                        "sha256": s["sha256"],
                    }
                )
        atomic_write_bytes(fin, json.dumps(moves, indent=1).encode())
    for mv in moves:
        a, b = work / mv["src"], src / mv["dst"]
        if a.exists():
            b.parent.mkdir(parents=True, exist_ok=True)
            os.replace(a, b)
        elif not b.exists():
            raise RuntimeError(f"finalize: neither {a} nor {b} exists")
    return [
        ShardEntry(
            path=m["dst"],
            lang=m["lang"],
            table=m["table"],
            rows=m["rows"],
            bytes=m["bytes"],
            sha256=m["sha256"],
        )
        for m in moves
    ]


def check_schemas(src: Path, shards: list[ShardEntry]) -> list[str]:
    first: dict[tuple[str, str], pa.Schema] = {}
    problems = []
    for s in shards:
        schema = pq.read_schema(src / s.path).remove_metadata()
        key = (s.lang, s.table)
        if key not in first:
            first[key] = schema
        elif not schema.equals(first[key]):
            problems.append(f"{s.path}: schema differs from other {s.table} shards in {s.lang}")
    return problems


def acquire(
    name: str, spec: dict[str, Any], cfg: LakeConfig, staging: Path, workers: int, limit: int | None = None
) -> tuple[SourceManifest, list[str]]:
    lic = LicenseDecision.model_validate(spec["license"])
    if lic.use == LicenseUse.blocked:
        raise PermissionError(f"{name}: license decision is 'blocked' ({lic.license}); not acquired")
    repo, revision = spec["hf_id"], spec["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{name}: revision must be a commit SHA, got {revision!r}")
    planned = plan(spec, hf_files(repo, revision))
    src = source_dir(cfg.root, "raw", name)
    work = src / "_work"
    fp = plan_fingerprint(spec, planned, limit)

    manifest_path = src / "source.yaml"
    if manifest_path.exists() and not (work / "plan.json").exists():
        m = SourceManifest.read(src)
        same_files = [i.path for i in m.inputs] == [p.path for p in planned]
        if m.revision == revision and same_files and not m.partial and limit is None:
            problems = verify_source(src, deep=False)
            if not problems:
                if m.license != lic:  # a revised decision needs no re-download
                    log.info("%s: license decision changed to %s, manifest updated", name, lic.use.value)
                    m.license = lic
                    m.write(src)
                log.info("%s: already acquired at %s, manifest verified", name, revision[:12])
                return m, []
        log.info("%s: existing data does not match the plan, re-acquiring", name)
    if (work / "plan.json").exists() and json.loads((work / "plan.json").read_text())["fingerprint"] != fp:
        log.info("%s: plan changed, discarding previous partial work", name)
        shutil.rmtree(src)
    if not (work / "plan.json").exists():
        if src.exists():
            shutil.rmtree(src)
        atomic_write_bytes(work / "plan.json", json.dumps({"fingerprint": fp}).encode())

    done: dict[int, Done] = {}
    for p in planned:
        marker = work / f"{p.idx:05d}.json"
        if marker.exists():
            done[p.idx] = Done(**json.loads(marker.read_text()))
    todo = [p for p in planned if p.idx not in done]
    total_bytes = sum(p.size for p in planned)
    log.info(
        "%s: %d files (%.2f GB), %d to do, %d workers",
        name,
        len(planned),
        total_bytes / 1e9,
        len(todo),
        workers,
    )

    from huggingface_hub import hf_hub_download

    def run(p: Planned) -> Done:
        for stale in (work / p.table / p.lang).glob(f"{p.idx:05d}-*"):
            stale.unlink()
        local = Path(
            hf_hub_download(repo, p.path, repo_type="dataset", revision=revision, local_dir=staging / name)
        )
        if p.sha256 and limit is None and sha256_file(local) != p.sha256:
            local.unlink()
            raise RuntimeError(f"{p.path}: sha256 differs from the repository LFS pointer")
        d = convert(p, local, work, cfg, limit)
        atomic_write_bytes(work / f"{p.idx:05d}.json", json.dumps(asdict(d)).encode())
        log.info("%s: [%d/%d] %s -> %d rows", name, p.idx + 1, len(planned), p.path, d.rows)
        if limit is None:  # kept on failure, so a re-run does not download it again
            local.unlink(missing_ok=True)
        return d

    with ResourceMonitor() as mon:
        with ThreadPoolExecutor(workers) as pool:
            futures = [pool.submit(run, p) for p in todo]
            try:
                for fut in futures:
                    d = fut.result()
                    done[d.idx] = d
            except BaseException:
                pool.shutdown(wait=True, cancel_futures=True)  # finished files keep their markers
                raise
        shards = finalize(src, work, planned, done)

    meta = readme_counts(repo, revision)
    meta_method = "hf_dataset_info"
    if not meta:
        meta, meta_method = server_counts(repo), "hf_datasets_server"
    manifest = SourceManifest(
        name=name,
        role=spec["role"],
        origin="hf",
        hf_id=repo,
        revision=revision,
        license=lic,
        code_version=git_sha(ROOT),
        partial=limit is not None,
        inputs=[
            InputFile(
                path=p.path,
                size=p.size,
                sha256=p.sha256,
                config=p.config,
                split=p.split,
                lang=p.lang,
                table=p.table,
                rows=done[p.idx].rows,
            )
            for p in planned
        ],
        shards=shards,
        row_checks=row_checks(planned, done, meta, meta_method, limit is not None),
        stats=mon.stats(
            sum(done[p.idx].rows for p in todo), sum(p.size for p in todo), workers
        ),  # this run only
    )
    manifest.write(src)
    shutil.rmtree(work)
    problems = verify_source(src, deep=False) + check_schemas(src, shards)
    if not manifest.row_checks and limit is None:
        problems.append(f"{name}: no row count could be checked")
    return manifest, problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument(
        "--root", type=Path, help="lake root (default params lake.root); use a temp dir for smoke runs"
    )
    ap.add_argument(
        "--limit-rows", type=int, help="smoke run: at most N rows per file, manifest marked partial"
    )
    ap.add_argument("--workers", type=int)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    params = load_params()
    spec = params["sources"].get(args.source)
    if spec is None or "hf_id" not in spec:
        sys.exit(f"{args.source}: not an HF source in params.yaml: sources")
    cfg = LakeConfig.from_params(params["lake"], args.root)
    workers = args.workers or int(params["lake"]["acquire_workers"])
    staging = ROOT / params["lake"]["downloads"] / "hf"
    try:
        m, problems = acquire(args.source, spec, cfg, staging, workers, args.limit_rows)
    except PermissionError as e:
        log.error("%s", e)
        return 2
    peak_limit_mb = float(params["lake"]["peak_ram_gb"]) * 1024
    if m.stats and m.stats.peak_rss_mb > peak_limit_mb:
        problems.append(f"peak RSS {m.stats.peak_rss_mb:.0f} MB exceeds lake.peak_ram_gb")
    log_mlflow(
        "data",
        f"acquire-{args.source}{'-smoke' if m.partial else ''}",
        {
            "source": args.source,
            "revision": m.revision,
            "workers": workers,
            "partial": m.partial,
            "code_version": m.code_version,
            "license_use": m.license.use.value,
        },
        {
            "rows": sum(m.rows.values()),
            "bytes": sum(m.bytes.values()),
            **{f"rows_{k}": v for k, v in m.rows.items()},
            **(m.stats.model_dump(exclude={"workers"}) if m.stats else {}),
        },
    )
    for c in m.row_checks:
        log.info(
            "row check %-55s %-20s expected=%d actual=%d %s",
            c.scope,
            c.method,
            c.expected,
            c.actual,
            "OK" if c.ok else "MISMATCH",
        )
    log.info("%s: rows %s, %s", args.source, m.rows, m.stats)
    for p in problems:
        log.error(p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
