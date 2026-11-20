"""Shared helpers of the task / eval stages (stage 3): keys, splits, dataset writing, raw readers.

Task datasets live in `data/task/<name>/<lang>/<table>-NNNNN.parquet` (eval in `data/eval/...`)
with a `dataset.yaml` + `card.md` (ADR-0011): inputs are the manifests they were built from, the
licence list is inherited (ADR-0015), metrics carry the quality checks of the stage.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa

from trifact.core.params import ROOT
from trifact.data.curation import dataset_input, licenses_of, shard_entry
from trifact.lake.layout import LakeConfig, ShardWriter, source_dir
from trifact.lake.manifest import (
    DATASET_NAME,
    MANIFEST_NAME,
    DatasetInput,
    DatasetManifest,
    InheritedLicense,
    RunStats,
    ShardEntry,
    SourceManifest,
)
from trifact.lake.stats import git_sha

LANGS = ("kk", "ru", "en")
SPLITS = ("train", "validation", "test")
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def norm_key(text: str | None) -> str:
    """Lower-cased, NFKC, punctuation-free, whitespace-collapsed text: the leakage comparison key."""
    t = unicodedata.normalize("NFKC", text or "").lower()
    return _WS.sub(" ", _PUNCT.sub(" ", t)).strip()


def key_expr(col: str | pl.Expr) -> pl.Expr:
    """Vectorized `norm_key` (NFKC is applied by the normalizer upstream; here case and punctuation)."""
    e = pl.col(col) if isinstance(col, str) else col
    return (
        e.fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def stable_hash(text: str, seed: int = 0) -> int:
    """64-bit hash stable across processes and library versions (blake2b)."""
    return int.from_bytes(hashlib.blake2b(f"{seed}:{text}".encode(), digest_size=8).digest(), "big")


def hash_expr(col: str | pl.Expr, seed: int) -> pl.Expr:
    """Stable per-value hash in [0, 1) for split assignment (blake2b via map_elements on unique keys)."""
    e = pl.col(col) if isinstance(col, str) else col
    return e.map_elements(lambda s: stable_hash(s, seed) / 2**64, return_dtype=pl.Float64)


def assign_split(df: pl.DataFrame, key: str, val: float, test: float, seed: int) -> pl.Series:
    """train/validation/test by hash of `key` (all rows of one document land in one split)."""
    keys = df.select(pl.col(key).unique())
    keys = keys.with_columns(u=hash_expr(key, seed))
    split = (
        pl.when(pl.col("u") < test)
        .then(pl.lit("test"))
        .when(pl.col("u") < test + val)
        .then(pl.lit("validation"))
        .otherwise(pl.lit("train"))
    )
    m = keys.with_columns(split=split).select(key, "split")
    return df.select(key).join(m, on=key, how="left", maintain_order="left")["split"]


# ----------------------------------------------------------------------------- raw access


def raw_manifest(root: Path, source: str) -> SourceManifest:
    return SourceManifest.read(source_dir(root, "raw", source))


def raw_files(root: Path, source: str, table: str = "part", lang: str | None = None) -> list[str]:
    m = raw_manifest(root, source)
    base = source_dir(root, "raw", source)
    return [str(base / s.path) for s in m.shards if s.table == table and (lang is None or s.lang == lang)]


def scan_raw(root: Path, source: str, table: str = "part", lang: str | None = None) -> pl.LazyFrame:
    """Lazy scan of a raw table with the partition language as column `lang`."""
    m = raw_manifest(root, source)
    base = source_dir(root, "raw", source)
    by_lang: dict[str, list[str]] = {}
    for s in m.shards:
        if s.table == table and (lang is None or s.lang == lang):
            by_lang.setdefault(s.lang, []).append(str(base / s.path))
    if not by_lang:
        raise FileNotFoundError(f"raw/{source}: no shards of table {table!r} lang {lang!r}")
    # one multi-file scan per language: a concat of one scan per shard is read by the streaming engine with
    # all shards in flight at once (86 miracl_corpus shards -> out of memory)
    frames = [pl.scan_parquet(files).with_columns(lang=pl.lit(lg)) for lg, files in by_lang.items()]
    return pl.concat(frames, how="diagonal_relaxed")


def raw_files_by_lang(root: Path, source: str, table: str = "part") -> dict[str, list[str]]:
    """Raw shard paths of a table by partition language."""
    base = source_dir(root, "raw", source)
    out: dict[str, list[str]] = {}
    for s in raw_manifest(root, source).shards:
        if s.table == table:
            out.setdefault(s.lang, []).append(str(base / s.path))
    return out


def rows_with_keys(
    files: dict[str, list[str]], key: str, keys: dict[str, Iterable[str]], columns: list[str]
) -> pl.DataFrame:
    """Rows of a big table whose `key` is in `keys[lang]` of their partition language, with column `lang`.

    pyarrow filters every batch while reading and copies only the matching rows. A Polars filter over a lazy
    scan keeps string views into every decoded text buffer (6.9 GB miracl_corpus -> out of memory)."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    parts = []
    for lg, fs in files.items():
        if not (ks := set(keys.get(lg, ()))):
            continue
        want = pa.array(sorted(ks), type=pa.string())
        t = ds.dataset(fs, format="parquet").to_table(
            columns=columns, filter=pc.is_in(ds.field(key).cast(pa.string()), value_set=want)
        )
        parts.append(pl.from_arrow(t).with_columns(lang=pl.lit(lg)))  # type: ignore[union-attr]
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()


def raw_input(root: Path, source: str) -> DatasetInput:
    d = source_dir(root, "raw", source)
    return dataset_input(root, d / MANIFEST_NAME, SourceManifest.read(d).rows)


def dataset_input_of(root: Path, out_dir: Path) -> DatasetInput:
    m = DatasetManifest.read(out_dir)
    return dataset_input(root, out_dir / DATASET_NAME, m.rows_of())


def inherited(root: Path, sources: Iterable[str], datasets: Iterable[Path] = ()) -> list[InheritedLicense]:
    ms: list[SourceManifest | DatasetManifest] = [raw_manifest(root, s) for s in sources]
    ms += [DatasetManifest.read(d) for d in datasets]
    return licenses_of(ms)


# ----------------------------------------------------------------------------- writing


class DatasetWriter:
    """Writes tables of one task dataset as `<lang>/<table>-NNNNN.parquet` and finishes its manifest."""

    def __init__(self, out_dir: Path, lake: LakeConfig):
        self.out = Path(out_dir)
        self.lake = lake
        self.writers: dict[tuple[str, str], ShardWriter] = {}
        self.shards: list[ShardEntry] = []

    def write(
        self, table: str, df: pl.DataFrame | pa.Table, lang_col: str | None = "lang", lang: str = "multi"
    ) -> None:
        t = df.to_arrow() if isinstance(df, pl.DataFrame) else df
        if t.num_rows == 0:
            return
        if lang_col is None:
            self._w(table, lang).write(t)
            return
        pdf = pl.from_arrow(t)
        assert isinstance(pdf, pl.DataFrame)
        for (lg,), part in pdf.group_by(lang_col, maintain_order=True):
            self._w(table, str(lg)).write(part.to_arrow())

    def _w(self, table: str, lang: str) -> ShardWriter:
        w = self.writers.get((table, lang))
        if w is None:
            w = self.writers[(table, lang)] = ShardWriter(
                self.out / lang, lambda k, t=table: f"{t}-{k:05d}", self.lake
            )
        return w

    def close(self) -> list[ShardEntry]:
        for (table, lang), w in sorted(self.writers.items()):
            for s in w.close():
                self.shards.append(shard_entry(self.out, s, lang, table))
        self.writers.clear()
        return self.shards

    def abort(self) -> None:
        for w in self.writers.values():
            w.abort()


def finish(
    out_dir: Path,
    *,
    name: str,
    layer: str,
    stage: str,
    source: str,
    description: str,
    params: dict[str, Any],
    inputs: list[DatasetInput],
    licenses: list[InheritedLicense],
    shards: list[ShardEntry],
    rows_in: dict[str, int],
    metrics: dict[str, Any],
    stats: RunStats | None,
    fingerprint: str,
    drops: dict[str, dict[str, int]] | None = None,
    partial: bool = False,
) -> DatasetManifest:
    m = DatasetManifest(
        name=name, layer=layer, stage=stage, source=source, description=description,
        code_version=git_sha(ROOT), fingerprint=fingerprint, params=params, inputs=inputs,
        licenses=licenses, rows_in=rows_in, drops=drops or {}, shards=shards, stats=stats,
        metrics=metrics, partial=partial,
    )  # fmt: skip
    m.write(out_dir)
    return m


def counts(df: pl.DataFrame, *by: str) -> list[dict[str, Any]]:
    return df.group_by(*by).len().sort(*by).to_dicts()


def by_lang(df: pl.DataFrame, col: str = "lang") -> dict[str, int]:
    return {str(k): int(v) for k, v in df.group_by(col).len().sort(col).iter_rows()}
