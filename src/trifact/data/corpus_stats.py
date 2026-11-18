"""Corpus statistics and tokenizer fertility of the curated corpus (ROADMAP 2.6).

python -m trifact.data.corpus_stats [--corpus DIR] [--report FILE]

1. Funnel: documents per source and language through raw -> clean -> dedup -> corpus, read from
   the manifests (no data scan).
2. Corpus: per (source, lang, domain) documents, characters, words and length quantiles;
   length histogram (`curation.stats.length_bins`, characters); share of documents with
   paragraphs in another target language (`para_lang`). One streaming Polars pass over the
   numeric columns.
3. Tokenizer fertility: up to `sample_docs` documents per (source, lang), chosen by a seeded
   hash of `doc_id` (reproducible, spread over all shards), first `max_chars` characters of each.
   For every pinned tokenizer (`tokenizers` library, `tokenizer.json` at a fixed HF revision):
   tokens per whitespace word (fertility), tokens per 1000 characters, and the unknown-token rate.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl

from trifact.core.params import ROOT, load_params
from trifact.data.curation import write_report
from trifact.data.quality import TARGET_LANGS
from trifact.lake.layout import source_dir
from trifact.lake.manifest import DATASET_NAME, DatasetManifest, SourceManifest

log = logging.getLogger("trifact.corpus_stats")
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def part_files(base: Path, m: DatasetManifest) -> list[str]:
    return [str(base / s.path) for s in m.shards if s.table == "part"]


def funnel(root: Path, corpus: DatasetManifest, dedup: DatasetManifest, clean_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for source in corpus.source.split("+"):
        raw = SourceManifest.read(source_dir(root, "raw", source))
        clean = DatasetManifest.read(clean_root / source)
        out[source] = {
            "raw": raw.rows,
            "clean": clean.rows_of(),
            "dedup": dedup.metrics["by_source"].get(source, {}).get("kept"),
            "corpus": corpus.metrics["by_source"].get(source, {}).get("in", 0)
            - corpus.metrics["by_source"].get(source, {}).get("dropped", 0),
        }
    return out


def volumes(files: list[str], bins: list[int]) -> dict[str, Any]:
    lf = pl.scan_parquet(files).select("source", "lang", "domain", "n_chars", "n_words", "para_lang")
    others = {lang: [x for x in TARGET_LANGS if x != lang] for lang in TARGET_LANGS}
    mixed = pl.lit(False)
    for lang, other in others.items():
        mixed = (
            pl.when(pl.col("lang") == lang)
            .then(pl.col("para_lang").list.eval(pl.element().is_in(other)).list.any())
            .otherwise(mixed)
        )
    lf = lf.with_columns(mixed=mixed.fill_null(False))
    aggs = [
        pl.len().alias("docs"),
        pl.col("n_chars").sum().alias("chars"),
        pl.col("n_words").sum().alias("words"),
        pl.col("n_chars").mean().round(1).alias("chars_mean"),
        *[pl.col("n_chars").quantile(q).alias(f"chars_p{int(q * 100):02d}") for q in QUANTILES],
        pl.col("mixed").mean().round(4).alias("mixed_lang_share"),
    ]
    by = (
        lf.group_by("source", "lang", "domain")
        .agg(aggs)
        .sort("source", "lang", "domain")
        .collect(engine="streaming")
    )
    by_lang = lf.group_by("lang").agg(aggs).sort("lang").collect(engine="streaming")
    labels = [f"{a}-{b}" for a, b in itertools.pairwise(bins)] + [f"{bins[-1]}+"]
    hist = (
        lf.with_columns(bin=pl.col("n_chars").cut(bins[1:], labels=labels, left_closed=True))
        .group_by("lang", "bin")
        .agg(docs=pl.len())
        .collect(engine="streaming")
    )
    histogram = {
        lang: {
            lab: int(hist.filter((pl.col("lang") == lang) & (pl.col("bin") == lab))["docs"].sum())
            for lab in labels
        }
        for lang in sorted(hist["lang"].unique().to_list())
    }
    return {
        "by_source_lang_domain": by.to_dicts(),
        "by_lang": by_lang.to_dicts(),
        "length_histogram": histogram,
    }


def sample(files: list[str], by: pl.DataFrame, n: int, max_chars: int, seed: int) -> pl.DataFrame:
    sizes = by.group_by("source", "lang").agg(pl.col("docs").sum())
    rate = {(s, lang): min(1.0, n / d) for s, lang, d in sizes.iter_rows()}
    keys = pl.concat_str("source", "lang", separator="|")
    threshold = keys.replace_strict(
        {f"{s}|{lang}": min(int(r * 2**64), 2**64 - 1) for (s, lang), r in rate.items()},
        default=0,  # also evaluated on row-group min/max statistics, where pairs can be unseen
        return_dtype=pl.UInt64,
    )
    return (
        pl.scan_parquet(files)
        .select("doc_id", "source", "lang", "text")
        .filter(pl.col("doc_id").hash(seed=seed) <= threshold)
        .with_columns(pl.col("text").str.slice(0, max_chars))
        .collect(engine="streaming")
        .sort("doc_id")
    )


def tokenizer(spec: dict[str, Any], cache: Path) -> Any:
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    path = hf_hub_download(
        spec["hf_id"],
        spec["file"],
        revision=spec["revision"],
        cache_dir=str(cache / "hf"),
        local_files_only=False,
    )
    return Tokenizer.from_file(path)


def fertility(texts: pl.DataFrame, specs: dict[str, dict[str, Any]], cache: Path) -> dict[str, Any]:
    words = pl.Series([len(t.split()) for t in texts["text"].to_list()], dtype=pl.Int64)  # whitespace words
    frame = texts.select("source", "lang", chars=pl.col("text").str.len_chars()).with_columns(words=words)
    out: dict[str, Any] = {}
    for name, spec in specs.items():
        tok = tokenizer(spec, cache)
        unk_id = tok.token_to_id(spec["unk"]) if spec.get("unk") else None
        t0 = time.perf_counter()
        enc = tok.encode_batch(texts["text"].to_list(), add_special_tokens=False)
        seconds = time.perf_counter() - t0
        n_tok = [len(e.ids) for e in enc]
        n_unk = [sum(1 for i in e.ids if i == unk_id) for e in enc] if unk_id is not None else [0] * len(enc)
        f = frame.with_columns(tokens=pl.Series(n_tok), unk=pl.Series(n_unk))
        agg = [
            pl.len().alias("docs"),
            (pl.col("tokens").sum() / pl.col("words").sum()).round(3).alias("tokens_per_word"),
            (1000 * pl.col("tokens").sum() / pl.col("chars").sum()).round(1).alias("tokens_per_1k_chars"),
            (pl.col("unk").sum() / pl.col("tokens").sum()).round(5).alias("unk_rate"),
        ]
        out[name] = {
            "hf_id": spec["hf_id"],
            "revision": spec["revision"],
            "vocab_size": tok.get_vocab_size(),
            "by_lang": f.group_by("lang").agg(agg).sort("lang").to_dicts(),
            "by_source_lang": f.group_by("source", "lang").agg(agg).sort("source", "lang").to_dicts(),
            "chars_per_s": round(int(frame["chars"].sum()) / max(seconds, 1e-9)),
        }
        log.info("%s: %s", name, out[name]["by_lang"])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--corpus", type=Path, help="curated/corpus directory (default <lake root>/curated/corpus)"
    )
    ap.add_argument("--report", type=Path, help="metrics JSON (default reports/corpus_stats.json)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    params = load_params()
    cfg = params["curation"]["stats"]
    root = (ROOT / params["lake"]["root"]).resolve()
    corpus_dir = (args.corpus or root / "curated" / "corpus").resolve()
    dedup_dir = corpus_dir.parent / "dedup"
    clean_root = corpus_dir.parent.parent / "clean"
    m = DatasetManifest.read(corpus_dir)
    files = part_files(corpus_dir, m)
    t0 = time.perf_counter()
    vol = volumes(files, cfg["length_bins"])
    by = pl.DataFrame(vol["by_source_lang_domain"])
    texts = sample(files, by, cfg["sample_docs"], cfg["max_chars"], params["seed"])
    report = {
        "corpus": m.name,
        "fingerprint": m.fingerprint,
        "partial": m.partial,
        "funnel": funnel(root, m, DatasetManifest.read(dedup_dir), clean_root)
        if (dedup_dir / DATASET_NAME).is_file()
        else {},
        **vol,
        "fertility_sample": texts.group_by("source", "lang").len().sort("source", "lang").to_dicts(),
        "fertility": fertility(texts, cfg["tokenizers"], ROOT / params["curation"]["langid"]["cache"]),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    rep_path = args.report or (
        ROOT / "reports/corpus_stats.json" if args.corpus is None else corpus_dir.parent / "corpus_stats.json"
    )
    write_report(rep_path, report)
    for name, f in report["fertility"].items():
        print(name, {r["lang"]: r["tokens_per_word"] for r in f["by_lang"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
