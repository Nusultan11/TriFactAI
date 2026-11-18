"""Stage wikitext: pinned Wikipedia XML dumps (wikitext with links) -> raw/wikipedia_wikitext/<lang>/part-*.

python -m trifact.data.wikitext [--download-only] [--workers N]

`wikimedia/wikipedia` on the Hub is plain text without links, but silver NER (ADR-0015) needs the
link targets of every mention. This source keeps the original markup of the main-namespace articles
(redirects excluded) of the dump files listed in `params.yaml: sources.wikipedia_wikitext.files`.

Download: every file with the resumable ranged downloader of `trifact.data.wikidata`, checked
against the dump's published `sha1sums`. Parse: one process per file streams the bz2 XML
(`xml.etree.iterparse`, elements cleared as they are read) into size-bounded Parquet shards
`<lang>/part-<file:03d>-<k:02d>.parquet` (page_id, title, revision_id, timestamp, wikitext) and the
main-namespace redirects `<lang>/redirects-*` (title, target) that links resolve through; a
marker in `_progress/` per finished file makes a restart skip it. Row check per file: pages
written = main-namespace non-redirect `<page>` elements met by the parser (`xml_page_count`).
"""

from __future__ import annotations

import argparse
import bz2
import json
import logging
import shutil
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow as pa

from trifact.core.params import ROOT, load_params
from trifact.data.wikidata import download, license_of
from trifact.lake.layout import LakeConfig, ShardWriter, atomic_write_bytes, source_dir
from trifact.lake.manifest import InputFile, RowCheck, ShardEntry, SourceManifest, verify_source
from trifact.lake.stats import ResourceMonitor, git_sha, log_mlflow

log = logging.getLogger("trifact.wikitext")
SOURCE = "wikipedia_wikitext"
SCHEMA = pa.schema(
    [
        ("page_id", pa.int64()),
        ("title", pa.string()),
        ("revision_id", pa.int64()),
        ("timestamp", pa.string()),
        ("wikitext", pa.string()),
    ]
)
REDIRECTS = pa.schema([("title", pa.string()), ("target", pa.string())])
BATCH = 2000


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_file(idx: int, path: str, lang: str, out_dir: str, lake: LakeConfig) -> dict[str, Any]:
    """Main-namespace, non-redirect pages of one XML dump file into shards; returns shards and counts."""
    out = Path(out_dir)
    writer = ShardWriter(out / lang, lambda k: f"part-{idx:03d}-{k:02d}", lake, SCHEMA)
    redir_w = ShardWriter(out / lang, lambda k: f"redirects-{idx:03d}-{k:02d}", lake, REDIRECTS)
    rows: dict[str, list[Any]] = {n: [] for n in SCHEMA.names}
    redirects: dict[str, list[Any]] = {"title": [], "target": []}
    seen = kept = n_redirects = 0

    def flush() -> None:
        if rows["page_id"]:
            writer.write(pa.table(rows, schema=SCHEMA))
            for v in rows.values():
                v.clear()
        if redirects["title"]:
            redir_w.write(pa.table(redirects, schema=REDIRECTS))
            for v in redirects.values():
                v.clear()

    try:
        with bz2.open(path, "rb") as f:
            page: dict[str, Any] = {}
            depth_rev = False
            for event, el in ET.iterparse(f, events=("start", "end")):
                tag = _local(el.tag)
                if event == "start":
                    if tag == "page":
                        page = {"redirect": False}
                    elif tag == "revision":
                        depth_rev = True
                    continue
                if tag == "title":
                    page["title"] = el.text
                elif tag == "ns":
                    page["ns"] = el.text
                elif tag == "redirect":
                    page["redirect"] = el.attrib.get("title") or True
                elif tag == "id":
                    page["revision_id" if depth_rev else "page_id"] = page.get(
                        "revision_id" if depth_rev else "page_id"
                    ) or int(el.text or 0)
                elif tag == "timestamp":
                    page["timestamp"] = el.text
                elif tag == "text":
                    page["wikitext"] = el.text or ""
                elif tag == "revision":
                    depth_rev = False
                elif tag == "page":
                    seen += 1
                    if page.get("ns") == "0" and not page["redirect"]:
                        for n in SCHEMA.names:
                            rows[n].append(page.get(n))
                        kept += 1
                    elif page.get("ns") == "0" and isinstance(page["redirect"], str):
                        redirects["title"].append(page.get("title"))
                        redirects["target"].append(page["redirect"])
                        n_redirects += 1
                    if len(rows["page_id"]) >= BATCH or len(redirects["title"]) >= 50 * BATCH:
                        flush()
                    el.clear()
        flush()
        shards = [("part", s) for s in writer.close()] + [("redirects", s) for s in redir_w.close()]
    except BaseException:
        writer.abort()
        redir_w.abort()
        raise
    entries = [
        ShardEntry(
            path=s.path.relative_to(out).as_posix(), lang=lang, table=t, rows=s.rows, bytes=s.bytes,
            sha256=s.sha256,
        ).model_dump()
        for t, s in shards
    ]  # fmt: skip
    return {"shards": entries, "pages": seen, "kept": kept, "redirects": n_redirects}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    params = load_params()
    spec = params["sources"][SOURCE]
    lic = license_of(spec)
    if lic.use == "blocked":
        sys.exit(f"{SOURCE}: license decision is 'blocked'")
    cfg = LakeConfig.from_params(params["lake"])
    downloads = ROOT / params["lake"]["downloads"] / SOURCE
    files = []
    for f in spec["files"]:
        url = f"{spec['base_url']}/{f['wiki']}/{spec['revision']}/{f['name']}"
        sums = (
            f"{spec['base_url']}/{f['wiki']}/{spec['revision']}/{f['wiki']}-{spec['revision']}-sha1sums.txt"
        )
        path = download(url, downloads / f["name"], int(spec.get("connections", 2)), checksum_url=sums)
        files.append((url, path, f["lang"]))
    if args.download_only:
        return 0
    src = source_dir(cfg.root, "raw", SOURCE)
    progress = src / "_progress"
    progress.mkdir(parents=True, exist_ok=True)
    workers = args.workers or int(spec.get("workers", 4))
    results: dict[int, dict[str, Any]] = {}
    for i in range(len(files)):
        mark = progress / f"{i:03d}.json"
        if mark.is_file():
            results[i] = json.loads(mark.read_text(encoding="utf-8"))
        else:
            for p in src.rglob(f"part-{i:03d}-*"):
                p.unlink()
    with ResourceMonitor() as mon, ProcessPoolExecutor(workers) as pool:
        futs = {
            pool.submit(parse_file, i, str(p), lang, str(src), cfg): i
            for i, (_, p, lang) in enumerate(files)
            if i not in results
        }
        for fut, i in futs.items():
            r = fut.result()
            atomic_write_bytes(progress / f"{i:03d}.json", json.dumps(r).encode())
            results[i] = r
            log.info("%s: %d pages, %d articles", files[i][1].name, r["pages"], r["kept"])
    shards = [ShardEntry.model_validate(s) for i in sorted(results) for s in results[i]["shards"]]
    rows = sum(s.rows for s in shards if s.table == "part")
    inputs = [
        InputFile(path=url, size=p.stat().st_size, lang=lang, table="part", rows=results[i]["kept"])
        for i, (url, p, lang) in enumerate(files)
    ]
    checks = [
        RowCheck(
            scope=files[i][1].name, method="xml_page_count", expected=results[i]["kept"],
            actual=sum(s["rows"] for s in results[i]["shards"] if s["table"] == "part"),
        )
        for i in sorted(results)
    ]  # fmt: skip
    manifest = SourceManifest(
        name=SOURCE, role=spec["role"], origin="url", url=spec["base_url"], revision=str(spec["revision"]),
        license=lic, code_version=git_sha(ROOT), inputs=inputs, shards=shards, row_checks=checks,
        stats=mon.stats(rows, sum(p.stat().st_size for _, p, _ in files), workers),
    )  # fmt: skip
    manifest.write(src)
    shutil.rmtree(progress, ignore_errors=True)
    problems = verify_source(src, deep=False)
    log_mlflow(
        "data", SOURCE, {"revision": spec["revision"], "workers": workers, "code_version": manifest.code_version},
        {"rows": rows, **(manifest.stats.model_dump(exclude={"workers"}) if manifest.stats else {})},
    )  # fmt: skip
    for p in problems:
        log.error(p)
    log.info("%s: %s articles %s", SOURCE, manifest.rows, manifest.stats)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
