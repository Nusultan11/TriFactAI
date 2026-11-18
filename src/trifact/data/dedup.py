"""Stage dedup: exact and MinHash-LSH near-duplicate removal within and across sources (clean -> curated/dedup).

python -m trifact.data.dedup [--clean DIR] [--out DIR] [--workers N]

Input: the `part` shards of every `clean/<source>` dataset in `curation.sources` order. Global
document index (gidx) = position in that order (source priority, then shard, then row), so the
smallest gidx of a duplicate cluster belongs to the most trusted source (wikipedia >
fineweb2_kk > multidomain_kk) and is the document kept.

1. Signatures (parallel, one unit per shard): exact hash of the lower-cased, whitespace-collapsed
   text and `bands` LSH band hashes of a `num_perm` MinHash over word `ngram`-grams
   (`trifact.data.hashing`). Only band hashes are stored: 8 bytes x (bands + 1) per document.
2. Clusters (driver): for the exact hash and each band, documents are sorted by hash and every
   document is linked to the first (smallest gidx) document with the same hash; connected
   components of the union of all links are duplicate clusters (scipy). Root = smallest gidx.
3. Filter (parallel): roots and singletons are kept (`part`, with `dup_count` = other members of
   the cluster, `minhash_cluster` = root gidx); so are near duplicates whose root is in the same
   source when the source is listed in `keep_near_within` (Wikipedia: one article per entity, so
   similar articles are templated stubs about different places or people); others go to `removed` with `reason` (`exact` when the text
   hash equals the root's, else `near`), `kept_doc_id` and `kept_source`.
4. Check: true word n-gram Jaccard between a sample of removed near duplicates and their roots
   (precision of LSH plus transitive clustering) is reported with examples.

Output: one dataset `curated/dedup` with shards `<source>/<lang>/{part,removed}-*.parquet`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from trifact.core.params import ROOT, load_params
from trifact.data.curation import (
    StageDir,
    Unit,
    UnitWriter,
    dataset_input,
    fingerprint,
    iter_batches,
    licenses_of,
    run_units,
    stage_report,
    write_report,
)
from trifact.data.hashing import MinHasher, exact_hash, jaccard
from trifact.lake.layout import LakeConfig
from trifact.lake.manifest import DATASET_NAME, DatasetManifest, ShardEntry
from trifact.lake.stats import ResourceMonitor, git_sha, log_mlflow

log = logging.getLogger("trifact.dedup")
CODE = [
    Path(__file__),
    ROOT / "src/trifact/data/hashing.py",
    ROOT / "src/trifact/data/curation.py",
]
SAMPLE_PAIRS = 500
EXAMPLES = 20


def plan(clean_root: Path, sources: list[str]) -> tuple[list[Unit], list[DatasetManifest]]:
    units: list[Unit] = []
    manifests = []
    for source in sources:
        m = DatasetManifest.read(clean_root / source)
        manifests.append(m)
        for s in sorted((s for s in m.shards if s.table == "part"), key=lambda s: s.path):
            units.append(Unit(len(units), source, f"{source}/{s.path}", s.lang, s.rows, ""))
    return units, manifests


# ----------------------------------------------------------------------------- workers

_CTX: dict[str, Any] = {}


def init_worker(ctx: dict[str, Any]) -> None:
    _CTX.clear()
    _CTX.update(ctx)
    m = ctx["minhash"]
    _CTX["hasher"] = MinHasher(m["num_perm"], m["bands"], m["ngram"])
    work = Path(ctx["work"])
    if (work / "root.npy").is_file():
        _CTX["root"] = np.load(work / "root.npy", mmap_mode="r")
        _CTX["size"] = np.load(work / "size.npy", mmap_mode="r")
        _CTX["exact"] = np.load(work / "exact.npy", mmap_mode="r")
        _CTX["roots"] = pl.read_parquet(work / "roots.parquet")


def sig_path(work: Path, idx: int) -> Path:
    return work / "sig" / f"sig-{idx:05d}.parquet"


def signature_unit(u: Unit) -> dict[str, Any]:
    hasher: MinHasher = _CTX["hasher"]
    path = Path(_CTX["clean"]) / u.path
    out = sig_path(Path(_CTX["work"]), u.idx)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    writer = None
    rows = 0
    try:
        for b in iter_batches(path, ["doc_id", "text"], None):
            text = pl.Series(b.column("text"))
            sig, words = hasher.signatures(text)
            bands = hasher.band_hashes(sig)
            # documents without words get unique band values so that they never collide
            empty = np.flatnonzero(words == 0)
            if empty.size:
                tag = (np.uint64(1) << np.uint64(63)) | np.uint64(u.idx << 32)
                bands[empty] = tag | (empty + rows).astype(np.uint64)[:, None]
            cols = {"doc_id": b.column("doc_id"), "exact": pa.array(exact_hash(text))}
            cols.update({f"b{j:02d}": pa.array(bands[:, j]) for j in range(hasher.bands)})
            t = pa.table(cols)
            if writer is None:
                writer = pq.ParquetWriter(tmp, t.schema, compression="zstd")
            writer.write_table(t)
            rows += t.num_rows
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(out)
    if rows != u.rows:
        raise ValueError(f"{u.path}: {rows} rows read, manifest says {u.rows}")
    return {"rows": rows, "bytes_in": path.stat().st_size}


def filter_unit(u: Unit) -> dict[str, Any]:
    offset = _CTX["offsets"][u.idx]
    root, size, exact = _CTX["root"], _CTX["size"], _CTX["exact"]
    roots: pl.DataFrame = _CTX["roots"]
    lake: LakeConfig = _CTX["lake"]
    out = Path(_CTX["out"])
    parts = UnitWriter(out / u.source, u.idx, "part", lake)
    removed = UnitWriter(out / u.source, u.idx, "removed", lake)
    pos = 0
    counts: dict[str, int] = {"kept": 0, "exact": 0, "near": 0, "near_kept_within_source": 0}
    by_kept_source: dict[str, dict[str, int]] = {}
    try:
        for b in iter_batches(Path(_CTX["clean"]) / u.path, None, None):
            n = b.num_rows
            g = np.arange(offset + pos, offset + pos + n, dtype=np.int64)
            pos += n
            rt = np.asarray(root[g], dtype=np.int64)
            keep = rt == g
            df = pl.from_arrow(b)
            assert isinstance(df, pl.DataFrame)
            rem = pl.DataFrame()
            if not keep.all():
                gr, rr = g[~keep], rt[~keep]
                is_exact = np.asarray(exact[gr]) == np.asarray(exact[rr])
                rem = (
                    df.filter(pl.Series(~keep))
                    .select("doc_id", "source", "lang", "n_chars")
                    .with_columns(
                        reason=pl.Series(np.where(is_exact, "exact", "near")),
                        gidx=pl.Series(gr),
                        minhash_cluster=pl.Series(rr),
                    )
                    .join(
                        roots, left_on="minhash_cluster", right_on="gidx", how="left", maintain_order="left"
                    )
                )
                if u.source in _CTX["keep_near_within"]:
                    spare = ((rem["reason"] == "near") & (rem["kept_source"] == u.source)).to_numpy()
                    keep[np.flatnonzero(~keep)[spare]] = True
                    counts["near_kept_within_source"] += int(spare.sum())
                    rem = rem.filter(pl.Series(~spare))
            kept = df.filter(pl.Series(keep)).with_columns(
                minhash_cluster=pl.Series(g[keep]),
                dup_count=pl.Series(np.asarray(size[g[keep]], dtype=np.int32) - 1),
            )
            parts.write(u.lang, kept.to_arrow())
            counts["kept"] += kept.height
            if rem.height == 0:
                continue
            removed.write(u.lang, rem.to_arrow())
            for reason, src, k in rem.group_by("reason", "kept_source").len().iter_rows():
                counts[reason] += k
                d = by_kept_source.setdefault(src, {})
                d[reason] = d.get(reason, 0) + k
    except BaseException:
        parts.abort()
        removed.abort()
        raise
    if pos != u.rows:
        raise ValueError(f"{u.path}: {pos} rows, expected {u.rows}")
    shards = parts.close() + removed.close()
    for s in shards:
        s["path"] = f"{u.source}/{s['path']}"
    return {"shards": shards, "source": u.source, "lang": u.lang, "rows": u.rows, "counts": counts,
            "by_kept_source": by_kept_source}  # fmt: skip


# ----------------------------------------------------------------------------- driver


def link_equal(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Edges (gidx, first gidx with the same hash) for every non-first member of a hash group."""
    order = np.argsort(h, kind="stable")
    hs = h[order]
    start = np.empty(len(hs), dtype=bool)
    start[:1] = True
    start[1:] = hs[1:] != hs[:-1]
    first = order[start][np.cumsum(start) - 1]
    member = ~start
    return order[member].astype(np.int64), first[member].astype(np.int64)


def clusters(
    sig_files: list[Path], n: int, bands: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """root gidx and cluster size of every document, exact hashes, and edge counts by hash."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    keys = np.zeros(0, dtype=np.int64)
    edges_by: dict[str, int] = {}
    exact = np.zeros(0, dtype=np.uint64)
    for col in ["exact"] + [f"b{j:02d}" for j in range(bands)]:
        h = pl.scan_parquet(sig_files).select(col).collect()[col].to_numpy()
        if len(h) != n:
            raise ValueError(f"{col}: {len(h)} signatures for {n} documents")
        if col == "exact":
            exact = h.astype(np.uint64)
        a, b = link_equal(h)
        edges_by[col] = len(a)
        keys = np.unique(np.concatenate([keys, (a << 32) | b]))
        log.info("%s: %d links, %d unique edges so far", col, len(a), len(keys))
    a, b = keys >> 32, keys & 0xFFFFFFFF
    graph = coo_matrix((np.ones(len(keys), dtype=np.int8), (a, b)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    first = np.full(labels.max() + 1 if n else 0, n, dtype=np.int64)
    np.minimum.at(first, labels, np.arange(n, dtype=np.int64))
    size = np.bincount(labels).astype(np.int32)
    return first[labels].astype(np.int64), size[labels], exact, edges_by


def root_ids(sig_files: list[Path], units: list[Unit], offsets: list[int], need: np.ndarray) -> pl.DataFrame:
    """doc_id and source of the root documents `need` (sorted gidx)."""
    parts = []
    for u, f in zip(units, sig_files, strict=True):
        lo, hi = offsets[u.idx], offsets[u.idx] + u.rows
        sel = need[(need >= lo) & (need < hi)]
        if sel.size == 0:
            continue
        ids = pl.read_parquet(f, columns=["doc_id"])["doc_id"].gather(sel - lo)
        parts.append(pl.DataFrame({"gidx": sel, "kept_doc_id": ids, "kept_source": u.source}))
    schema = {"gidx": pl.Int64, "kept_doc_id": pl.String, "kept_source": pl.String}
    return pl.concat(parts) if parts else pl.DataFrame(schema=schema)


def read_rows(path: Path, rows: list[int], column: str) -> dict[int, str]:
    """Values of `column` at the given row positions, reading only the row groups that hold them."""
    pf = pq.ParquetFile(path)
    bounds = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)])
    out: dict[int, str] = {}
    want = sorted(set(rows))
    for rg in sorted({int(np.searchsorted(bounds, r, side="right") - 1) for r in want}):
        col = pf.read_row_group(rg, columns=[column]).column(column)
        for r in want:
            if bounds[rg] <= r < bounds[rg + 1]:
                out[r] = col[r - int(bounds[rg])].as_py()
    return out


def check_pairs(
    clean_root: Path, units: list[Unit], offsets: np.ndarray, removed: pl.DataFrame, ngram: int, seed: int
) -> dict[str, Any]:
    near = removed.filter(pl.col("reason") == "near")
    if near.height == 0:
        return {"pairs": 0}
    sample = near.sample(min(SAMPLE_PAIRS, near.height), seed=seed)
    gids = np.concatenate([sample["gidx"].to_numpy(), sample["minhash_cluster"].to_numpy()])
    unit_of = np.searchsorted(offsets, gids, side="right") - 1
    wanted: dict[int, list[int]] = {}
    for g, ui in zip(gids, unit_of, strict=True):
        wanted.setdefault(int(ui), []).append(int(g - offsets[ui]))
    text: dict[int, str] = {}
    for ui, rows in wanted.items():
        for r, t in read_rows(clean_root / units[ui].path, rows, "text").items():
            text[int(offsets[ui]) + r] = t
    js, examples = [], []
    for row in sample.iter_rows(named=True):
        a, b = text[row["gidx"]], text[row["minhash_cluster"]]
        j = jaccard(a, b, ngram)
        js.append(j)
        if len(examples) < EXAMPLES:
            examples.append({"doc_id": row["doc_id"], "kept_doc_id": row["kept_doc_id"], "jaccard": round(j, 3),
                             "removed": a[:300], "kept": b[:300]})  # fmt: skip
    arr = np.array(js)
    return {
        "pairs": len(arr),
        "jaccard_mean": round(float(arr.mean()), 4),
        "jaccard_median": round(float(np.median(arr)), 4),
        "share_ge": {str(t): round(float((arr >= t).mean()), 4) for t in (0.3, 0.5, 0.7, 0.8, 0.9)},
        "examples": examples,
    }


def run(params: dict[str, Any], clean_root: Path, out: Path, workers: int) -> DatasetManifest | None:
    cur = params["curation"]
    mh = cur["minhash"]
    units, inputs = plan(clean_root, cur["sources"])
    stage_params = {"minhash": mh, "sources": cur["sources"]}
    fp = fingerprint("dedup", stage_params, CODE, [clean_root / s / DATASET_NAME for s in cur["sources"]])
    sd = StageDir(out, fp)
    if sd.complete():
        log.info("%s: up to date (fingerprint %s)", out, fp)
        return None
    sd.prepare()
    lake = LakeConfig.from_params(params["lake"], ROOT / params["lake"]["root"])
    offsets = np.cumsum([0] + [u.rows for u in units]).astype(np.int64)
    n = int(offsets[-1])
    if n >= 2**32:
        raise ValueError(f"{n} documents do not fit the 32-bit edge encoding")
    ctx = {"clean": str(clean_root), "work": str(sd.work), "out": str(out), "minhash": mh, "lake": lake,
           "offsets": offsets.tolist(), "keep_near_within": mh.get("keep_near_within", [])}  # fmt: skip
    hasher = MinHasher(mh["num_perm"], mh["bands"], mh["ngram"])
    sig_files = [sig_path(sd.work, u.idx) for u in units]
    with ResourceMonitor() as mon:
        sig_sd = StageDir(sd.work / "sig", fp)
        sig_sd.work.mkdir(parents=True, exist_ok=True)
        sig_res = run_units(units, sig_sd, signature_unit, workers, init_worker, (ctx,))
        if not (sd.work / "roots.parquet").is_file():
            root, size, exact, edges_by = clusters(sig_files, n, hasher.bands)
            need = np.unique(root[root != np.arange(n)])
            np.save(sd.work / "root.npy", root)
            np.save(sd.work / "size.npy", size)
            np.save(sd.work / "exact.npy", exact)
            (sd.work / "edges.json").write_text(json.dumps(edges_by))
            root_ids(sig_files, units, offsets.tolist(), need).write_parquet(sd.work / "roots.parquet")
            del root, size, exact
        results = run_units(units, sd, filter_unit, workers, init_worker, (ctx,))
    res = [results[u.idx] for u in units]
    shards = [ShardEntry.model_validate(s) for r in res for s in r["shards"]]
    removed_files = [out / s.path for s in shards if s.table == "removed"]
    cols = ["doc_id", "reason", "gidx", "minhash_cluster", "kept_doc_id", "source", "kept_source", "lang"]
    removed = pl.read_parquet(removed_files, columns=cols) if removed_files else pl.DataFrame()
    check = (
        check_pairs(clean_root, units, offsets, removed, mh["ngram"], params["seed"])
        if removed.height
        else {}
    )
    edges_by = json.loads((sd.work / "edges.json").read_text())
    rows_in: dict[str, int] = {}
    drops: dict[str, dict[str, int]] = {"exact": {}, "near": {}}
    by_source: dict[str, dict[str, int]] = {}
    for r in res:
        rows_in[r["lang"]] = rows_in.get(r["lang"], 0) + r["rows"]
        bs = by_source.setdefault(
            r["source"], {"in": 0, "kept": 0, "exact": 0, "near": 0, "near_kept_within_source": 0}
        )
        bs["in"] += r["rows"]
        for k in ("kept", "exact", "near", "near_kept_within_source"):
            bs[k] += r["counts"][k]
        for k in ("exact", "near"):
            drops[k][r["lang"]] = drops[k].get(r["lang"], 0) + r["counts"][k]
    cross = (
        removed.group_by("source", "kept_source", "reason")
        .len()
        .sort("source", "kept_source", "reason")
        .to_dicts()
        if removed.height
        else []
    )
    total_in = sum(rows_in.values())
    total_removed = sum(sum(v.values()) for v in drops.values())
    manifest = DatasetManifest(
        name="curated/dedup",
        layer="curated",
        stage="dedup",
        source="+".join(cur["sources"]),
        description=(
            "Documents of the clean layer after exact and MinHash-LSH near-duplicate removal within and across "
            f"sources (priority {' > '.join(cur['sources'])}); table `part` per `<source>/<lang>`, removed "
            "duplicates with the kept document in table `removed`."
        ),
        code_version=git_sha(ROOT),
        fingerprint=fp,
        params={**stage_params, "lsh_threshold": round(hasher.threshold, 4)},
        inputs=[
            dataset_input(clean_root, clean_root / s / DATASET_NAME, m.rows_of())
            for s, m in zip(cur["sources"], inputs, strict=True)
        ],
        licenses=licenses_of(inputs),
        rows_in=dict(sorted(rows_in.items())),
        drops={k: dict(sorted(v.items())) for k, v in drops.items()},
        shards=shards,
        stats=mon.stats(total_in, sum(r["bytes_in"] for r in sig_res.values()), workers),
        metrics={
            "removed_share": round(total_removed / max(total_in, 1), 4),
            "by_source": by_source,
            "cross_source": cross,
            "links_by_hash": edges_by,
            "pair_check": check,
        },
        partial=any(m.partial for m in inputs),
    )
    sd.finish(manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", type=Path, help="clean layer directory (default <lake root>/clean)")
    ap.add_argument("--out", type=Path, help="output directory (default <lake root>/curated/dedup)")
    ap.add_argument(
        "--report", type=Path, help="metrics JSON (default reports/dedup.json for the lake output)"
    )
    ap.add_argument("--workers", type=int)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    params = load_params()
    root = (ROOT / params["lake"]["root"]).resolve()
    clean_root = (args.clean or root / "clean").resolve()
    out = (args.out or root / "curated" / "dedup").resolve()
    workers = args.workers or params["curation"]["workers"]
    m = run(params, clean_root, out, workers) or DatasetManifest.read(out)
    default_rep = ROOT / "reports/dedup.json" if args.out is None else out.parent / "dedup.json"
    rep_path = args.report or default_rep
    write_report(rep_path, stage_report(m))
    print(f"dedup: in {sum(m.rows_in.values())} out {sum(m.rows_of().values())} removed {m.drops} "
          f"share {m.metrics['removed_share']:.2%}")  # fmt: skip
    if m.stats and not m.partial:
        log_mlflow("curation", "dedup", {**m.params, "fingerprint": m.fingerprint, "code_version": m.code_version}, {
            "docs_in": sum(m.rows_in.values()), "docs_out": sum(m.rows_of().values()),
            "removed_share": m.metrics["removed_share"], "seconds": m.stats.seconds,
            "docs_per_s": m.stats.rows_per_s, "peak_rss_mb": m.stats.peak_rss_mb,
            **{f"removed_{k}": sum(v.values()) for k, v in m.drops.items()},
            "pair_jaccard_ge_0.5": m.metrics["pair_check"].get("share_ge", {}).get("0.5", 0.0),
        })  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
