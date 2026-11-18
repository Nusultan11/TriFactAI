"""Stage wikidata: download the pinned Wikidata JSON dump and stream-parse it into raw/wikidata/multi/entities-*.

Download: HTTP range requests in fixed chunks over a few connections; finished chunks are recorded
in a state file, so an interrupted download resumes where it stopped; the result is checked
against the dump's published sha1.

Parse: the dump (one entity per line) is decompressed with parallel bzip2 and cut at line borders
into chunks that a process pool turns into Arrow batches. Shard k holds entities
[k * shard_rows, (k + 1) * shard_rows) in dump order, so the layout is deterministic: finished
shards have a marker in `_progress/` and are skipped on restart without being parsed again.
RAM is bounded by the number of chunks in flight, not by the dump size.

Usage: python -m trifact.data.wikidata [--download-only] [--limit-entities N --root DIR] [--workers N]
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import IO, Any

import httpx
import orjson
import pyarrow as pa

from trifact.core.params import ROOT, load_params
from trifact.lake.layout import LakeConfig, ShardWriter, atomic_write_bytes, source_dir
from trifact.lake.manifest import LicenseDecision, RowCheck, ShardEntry, SourceManifest, verify_source
from trifact.lake.stats import ResourceMonitor, git_sha, log_mlflow

log = logging.getLogger("trifact.wikidata")

SOURCE = "wikidata"
TABLE = "entities"
LANG = "multi"
USER_AGENT = "TriFactAI-data-pipeline/0.1 (research; python-httpx)"
CHUNK_BYTES = 64 << 20  # download chunk
READ_BYTES = 16 << 20  # decompressed bytes per parse task

# Wikidata language codes folded into our three languages; `mul` is the shared multilingual label.
LANG_CODES: dict[str, tuple[str, ...]] = {
    "kk": ("kk", "kk-cyrl", "kk-kz"),
    "ru": ("ru",),
    "en": ("en",),
    "mul": ("mul",),
}
SITES = ("kkwiki", "ruwiki", "enwiki")

SCHEMA = pa.schema(
    [
        ("qid", pa.string()),
        ("type", pa.string()),
        ("p31", pa.list_(pa.string())),
        ("p279", pa.list_(pa.string())),
        *[(f"label_{lang}", pa.string()) for lang in LANG_CODES],
        *[(f"description_{lang}", pa.string()) for lang in LANG_CODES],
        *[(f"aliases_{lang}", pa.list_(pa.string())) for lang in LANG_CODES],
        *[(f"sitelink_{site}", pa.string()) for site in SITES],
        ("n_sitelinks", pa.int32()),
        ("lastrevid", pa.int64()),
    ]
)


# ----------------------------------------------------------------------------- download


def _write_json(path: Path, data: Any) -> None:
    atomic_write_bytes(path, json.dumps(data, indent=1).encode())


def published_sha1(client: httpx.Client, checksum_url: str, filename: str) -> str:
    text = client.get(checksum_url).raise_for_status().text
    for line in text.splitlines():
        digest, _, name = line.partition("  ")
        if name.strip() == filename:
            return digest.strip()
    raise RuntimeError(f"{filename} not listed in {checksum_url}")


def download(
    url: str,
    dest: Path,
    connections: int = 3,
    max_bytes: int | None = None,
    checksum_url: str | None = None,
    retries: int = 20,
    chunk_bytes: int = CHUNK_BYTES,
    backoff: float = 5.0,
    transport: httpx.BaseTransport | None = None,
) -> Path:
    """Resumable chunked download of `url` (or its first `max_bytes`) to `dest`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    state_path = dest.with_name(dest.name + ".state.json")
    done_path = dest.with_name(dest.name + ".verified")
    if done_path.exists() and dest.exists():
        return dest
    with httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True, transport=transport
    ) as client:
        size = int(client.head(url).raise_for_status().headers["content-length"])
        total = min(size, max_bytes) if max_bytes else size
        n_chunks = (total + chunk_bytes - 1) // chunk_bytes
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if (
            state.get("url") != url
            or state.get("total") != total
            or state.get("chunk", CHUNK_BYTES) != chunk_bytes
        ):
            state = {"url": url, "total": total, "chunk": chunk_bytes, "done": []}
            dest.unlink(missing_ok=True)
        done = set(state["done"])
        todo = [i for i in range(n_chunks) if i not in done]
        log.info("download %s: %.1f GB, %d/%d chunks left", dest.name, total / 1e9, len(todo), n_chunks)
        if not dest.exists():
            dest.touch()
        lock = threading.Lock()
        started, fetched = time.perf_counter(), 0

        def fetch(i: int) -> None:
            nonlocal fetched
            lo, hi = i * chunk_bytes, min((i + 1) * chunk_bytes, total) - 1
            for attempt in range(retries):
                try:
                    buf = bytearray()
                    with client.stream("GET", url, headers={"Range": f"bytes={lo}-{hi}"}) as r:
                        if r.status_code != 206:
                            raise RuntimeError(f"HTTP {r.status_code} for range {lo}-{hi}")
                        for part in r.iter_bytes(1 << 20):
                            buf += part
                    if len(buf) != hi - lo + 1:
                        raise RuntimeError(f"short chunk {i}: {len(buf)} of {hi - lo + 1}")
                    with lock:
                        with open(dest, "r+b") as f:
                            f.seek(lo)
                            f.write(buf)
                            f.flush()
                            os.fsync(f.fileno())
                        done.add(i)
                        state["done"] = sorted(done)
                        _write_json(state_path, state)
                        fetched += len(buf)
                        if len(done) % 16 == 0 or len(done) == n_chunks:
                            rate = fetched / 2**20 / (time.perf_counter() - started)
                            log.info("download %d/%d chunks, %.2f MB/s", len(done), n_chunks, rate)
                    return
                except (httpx.HTTPError, RuntimeError) as e:
                    wait = min(300.0, backoff * 2**attempt)
                    log.warning("chunk %d attempt %d failed: %s; retry in %ss", i, attempt + 1, e, wait)
                    time.sleep(wait)
            raise RuntimeError(f"chunk {i} failed after {retries} attempts")

        # Ordered submission keeps writes near the end of the file (no large sparse zero-fill on NTFS).
        with ThreadPoolExecutor(connections) as pool:
            for fut in [pool.submit(fetch, i) for i in todo]:
                fut.result()
        if dest.stat().st_size != total:
            raise RuntimeError(f"{dest} has {dest.stat().st_size} bytes, expected {total}")
        if checksum_url and not max_bytes:
            expected = published_sha1(client, checksum_url, url.rsplit("/", 1)[-1])
            h = hashlib.sha1()
            with open(dest, "rb") as f:
                while block := f.read(16 << 20):
                    h.update(block)
            if h.hexdigest() != expected:
                raise RuntimeError(f"sha1 mismatch for {dest.name}: {h.hexdigest()} != {expected}")
            log.info("sha1 verified: %s", expected)
        done_path.write_text(json.dumps({"bytes": total, "partial": bool(max_bytes)}))
        state_path.unlink(missing_ok=True)
    return dest


# ----------------------------------------------------------------------------- parse


def _values(d: dict[str, Any], codes: tuple[str, ...]) -> str | None:
    for c in codes:
        v = d.get(c)
        if v:
            return str(v["value"])
    return None


def _claim_ids(claims: dict[str, Any], prop: str) -> list[str]:
    out = []
    for c in claims.get(prop, ()):
        snak = c.get("mainsnak", {})
        if c.get("rank") != "deprecated" and snak.get("snaktype") == "value":
            out.append(snak["datavalue"]["value"]["id"])
    return out


def entity_row(e: dict[str, Any]) -> dict[str, Any]:
    labels, descs, aliases = e.get("labels", {}), e.get("descriptions", {}), e.get("aliases", {})
    claims, sitelinks = e.get("claims", {}), e.get("sitelinks", {})
    row: dict[str, Any] = {
        "qid": e["id"],
        "type": e.get("type"),
        "p31": _claim_ids(claims, "P31"),
        "p279": _claim_ids(claims, "P279"),
    }
    for lang, codes in LANG_CODES.items():
        row[f"label_{lang}"] = _values(labels, codes)
        row[f"description_{lang}"] = _values(descs, codes)
        seen: list[str] = []
        for c in codes:
            seen += [a["value"] for a in aliases.get(c, ()) if a["value"] not in seen]
        row[f"aliases_{lang}"] = seen
    for site in SITES:
        row[f"sitelink_{site}"] = sitelinks[site]["title"] if site in sitelinks else None
    row["n_sitelinks"] = len(sitelinks)
    row["lastrevid"] = e.get("lastrevid")
    return row


def count_entities(chunk: bytes) -> int:
    """Entity lines start with '{'; the dump's '[' and ']' lines are not entities."""
    return chunk.count(b"\n{") + (1 if chunk.startswith(b"{") else 0)


def parse_chunk(chunk: bytes) -> pa.RecordBatch:
    rows = []
    for line in chunk.split(b"\n"):
        if not line.startswith(b"{"):
            continue
        rows.append(entity_row(orjson.loads(line.rstrip(b",\r"))))
    return pa.RecordBatch.from_pylist(rows, schema=SCHEMA)


def open_dump(path: Path, threads: int) -> IO[bytes]:
    if path.suffix != ".bz2":
        raise ValueError(f"unsupported dump format: {path.name}")
    try:
        import indexed_bzip2

        return indexed_bzip2.open(str(path), parallelization=threads)  # type: ignore[no-any-return]
    except ImportError:
        log.warning("indexed_bzip2 not installed; falling back to single-threaded bz2")
        return bz2.open(path, "rb")


def line_chunks(f: IO[bytes], size: int | None = None, allow_truncated: bool = False) -> Iterator[bytes]:
    size = size or READ_BYTES
    tail = b""
    while True:
        try:
            block = f.read(size)
        except (EOFError, RuntimeError):  # a dump head ends mid-block (smoke runs only)
            if not allow_truncated:
                raise
            block = b""
        if not block:
            if tail.strip() and not allow_truncated:
                raise RuntimeError("dump does not end with a newline")
            return
        block = tail + block
        cut = block.rfind(b"\n") + 1
        if cut == 0:
            tail = block
            continue
        tail = block[cut:]
        yield block[:cut]


class Progress:
    """Markers of finished shards: _progress/<shard>.json = list of files with rows/bytes/sha256."""

    def __init__(self, out: Path):
        self.dir = out / "_progress"
        self.dir.mkdir(parents=True, exist_ok=True)

    def marker(self, k: int) -> Path:
        return self.dir / f"{TABLE}-{k:05d}.json"

    def done(self, k: int) -> dict[str, Any] | None:
        p = self.marker(k)
        return json.loads(p.read_text()) if p.exists() else None

    def mark(self, k: int, data: dict[str, Any]) -> None:
        _write_json(self.marker(k), data)


def parse_dump(
    dump: Path,
    src: Path,
    cfg: LakeConfig,
    workers: int,
    limit: int | None = None,
    allow_truncated: bool = False,
) -> tuple[list[ShardEntry], int]:
    """Parse into src/multi/entities-*.parquet; returns shards and the independently counted entities."""
    out = src / LANG
    out.mkdir(parents=True, exist_ok=True)
    progress = Progress(src)
    per_shard = cfg.shard_rows
    max_inflight = workers * 2

    def shard_name(k: int) -> Callable[[int], str]:
        return lambda i: f"{TABLE}-{k:05d}" + (f"-{i}" if i else "")

    shards: list[ShardEntry] = []
    seen = 0  # entities counted by the reader (independent of the parser)
    k = 0
    writer: ShardWriter | None = None
    in_shard = 0

    def finish_shard(last: bool) -> None:
        nonlocal writer, in_shard, k
        assert writer is not None
        infos = writer.close()
        entries = [
            ShardEntry(
                path=i.path.relative_to(src).as_posix(),
                lang=LANG,
                table=TABLE,
                rows=i.rows,
                bytes=i.bytes,
                sha256=i.sha256,
            )
            for i in infos
        ]
        progress.mark(k, {"entities": in_shard, "last": last, "files": [e.model_dump() for e in entries]})
        shards.extend(entries)
        log.info("shard %05d done: %d entities, total %d", k, in_shard, seen)
        writer, in_shard, k = None, 0, k + 1

    def start_shard() -> ShardWriter:
        for stale in out.glob(f"{TABLE}-{k:05d}*"):
            stale.unlink()  # unmarked leftovers of an interrupted run
        return ShardWriter(out, shard_name(k), cfg, SCHEMA)

    # Fast path: restore finished shards from markers.
    while (m := progress.done(k)) is not None:
        shards.extend(ShardEntry(**f) for f in m["files"])
        seen += m["entities"]
        k += 1
        if m["last"]:
            return shards, seen
    skip = seen
    log.info("resume: %d shards (%d entities) already done", k, skip)

    pending: deque[tuple[Future[pa.RecordBatch], int]] = deque()

    def drain(block: bool) -> None:
        nonlocal writer, in_shard
        while pending and (block or pending[0][0].done() or len(pending) >= max_inflight):
            fut, expected = pending.popleft()
            batch = fut.result()
            if batch.num_rows != expected:
                raise RuntimeError(f"parser returned {batch.num_rows} entities, reader counted {expected}")
            off = 0
            while off < batch.num_rows:
                if writer is None:
                    writer = start_shard()
                take = min(per_shard - in_shard, batch.num_rows - off)
                writer.write(batch.slice(off, take))
                in_shard += take
                off += take
                if in_shard == per_shard:
                    finish_shard(last=False)

    with open_dump(dump, threads=max(2, workers)) as f, ProcessPoolExecutor(workers) as pool:
        position = 0  # entities before the current chunk
        for chunk in line_chunks(f, allow_truncated=allow_truncated):
            n = count_entities(chunk)
            if limit is not None and position + n > limit:
                chunk = (
                    b"\n".join(chunk.split(b"\n")[: _line_index_of_entity(chunk, limit - position)]) + b"\n"
                )
                n = count_entities(chunk)
            if position + n <= skip:  # whole chunk belongs to finished shards
                position += n
                continue
            if position < skip:  # chunk straddles the resume point: drop the finished head
                chunk = _drop_entities(chunk, skip - position)
                position = skip
                n = count_entities(chunk)
            pending.append((pool.submit(parse_chunk, chunk), n))
            position += n
            seen = position
            drain(block=False)
            if limit is not None and position >= limit:
                break
        drain(block=True)
    if writer is not None:
        finish_shard(last=True)
    elif k > 0 and progress.done(k - 1) and not progress.done(k - 1)["last"]:  # type: ignore[index]
        data = progress.done(k - 1)
        assert data is not None
        data["last"] = True
        progress.mark(k - 1, data)
    return shards, seen


def _line_index_of_entity(chunk: bytes, n: int) -> int:
    """Number of leading lines that contain exactly the first `n` entities."""
    count = 0
    for i, line in enumerate(chunk.split(b"\n")):
        if count == n:
            return i
        if line.startswith(b"{"):
            count += 1
    return len(chunk.split(b"\n"))


def _drop_entities(chunk: bytes, n: int) -> bytes:
    lines = chunk.split(b"\n")
    return b"\n".join(lines[_line_index_of_entity(chunk, n) :])


# ----------------------------------------------------------------------------- stage


def license_of(spec: dict[str, Any]) -> LicenseDecision:
    return LicenseDecision.model_validate(spec["license"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--root", type=Path, help="lake root (default params lake.root); use a temp dir for smoke runs"
    )
    ap.add_argument("--limit-entities", type=int, help="smoke run: parse only the first N entities")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--workers", type=int, help="parse processes (default params sources.wikidata.workers)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    params = load_params()
    spec = params["sources"][SOURCE]
    lic = license_of(spec)
    if lic.use == "blocked":
        sys.exit(f"{SOURCE}: license decision is 'blocked'")
    cfg = LakeConfig.from_params(params["lake"], args.root)
    workers = args.workers or int(spec.get("workers", max(1, (os.cpu_count() or 2) - 2)))
    filename = spec["url"].rsplit("/", 1)[-1]
    downloads = ROOT / params["lake"]["downloads"]
    partial = args.limit_entities is not None

    if partial:
        # Smoke run: only the head of the dump is fetched (enough bytes for the requested entities).
        need = max(64 << 20, args.limit_entities * 5_000)
        dump = download(spec["url"], downloads / f"head-{need}-{filename}", 2, max_bytes=need)
    else:
        dump = download(
            spec["url"], downloads / filename, int(spec["connections"]), checksum_url=spec["checksum_url"]
        )
    if args.download_only:
        return 0

    src = source_dir(cfg.root, "raw", SOURCE)
    if partial and src.exists():
        shutil.rmtree(src)
    with ResourceMonitor() as mon:
        shards, counted = parse_dump(dump, src, cfg, workers, args.limit_entities, allow_truncated=partial)
    rows = sum(s.rows for s in shards)
    manifest = SourceManifest(
        name=SOURCE,
        role=spec["role"],
        origin="url",
        url=spec["url"],
        revision=str(spec["revision"]),
        license=lic,
        code_version=git_sha(ROOT),
        partial=partial,
        shards=shards,
        row_checks=[RowCheck(scope="all entities", method="entity_count", expected=counted, actual=rows)],
        stats=mon.stats(rows, dump.stat().st_size, workers),
    )
    manifest.write(src)
    shutil.rmtree(src / "_progress", ignore_errors=True)
    problems = verify_source(src, deep=False)
    peak_limit_mb = float(params["lake"]["peak_ram_gb"]) * 1024
    if manifest.stats and manifest.stats.peak_rss_mb > peak_limit_mb:
        problems.append(f"peak RSS {manifest.stats.peak_rss_mb:.0f} MB exceeds lake.peak_ram_gb")
    log_mlflow(
        "data",
        f"wikidata{'-smoke' if partial else ''}",
        {
            "revision": spec["revision"],
            "workers": workers,
            "partial": partial,
            "code_version": manifest.code_version,
        },
        {"rows": rows, **(manifest.stats.model_dump(exclude={"workers"}) if manifest.stats else {})},
    )
    log.info("wikidata: %d entities in %d shards, %s", rows, len(shards), manifest.stats)
    for p in problems:
        log.error(p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
