"""Full EDA of the raw, clean and curated layers (ROADMAP 2.6): one streaming pass over every shard.

python -m trifact.data.eda raw|clean|curated|entities [--workers N] [--sources a,b]
python -m trifact.data.eda_report          # figures + reports/eda/report.md from the JSON profiles

Every shard is profiled by a worker into mergeable partial aggregates; the driver merges them per
(source, lang, table[, domain, split]) and writes `reports/eda/<layer>.json`. Full-volume, exact:
schema and per-column nulls / empty strings / constant columns, schema drift between shards,
duplicate ids, documents, characters, words, lines, paragraphs, length histograms (log bins ->
quantiles), script mix (Cyrillic / Latin / other letters, Kazakh-specific letters), broken encoding
(U+FFFD, control characters, mojibake), PII matches (e-mail, phone, IIN with checksum, card numbers
with Luhn, URLs), URL hosts / TLDs and dates, repeated short lines (boilerplate), the shortest and
longest documents, quality signals and drop reasons of the clean layer, language scores and
paragraph-language mixing. Full-volume, approximate: word frequency top list (per-batch counts merged
and truncated to `TOP_WORDS`). Deterministic sample (seeded hash of the row, `sample_rate` per shard,
capped): exact vocabulary statistics (type/token, hapax share, Zipf exponent), n-gram top lists.
Tokenizer fertility and context-length shares come from `corpus_stats` (sample of the curated layer).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from trifact.core.params import ROOT, load_params
from trifact.data.curation import write_report
from trifact.lake.layout import source_dir
from trifact.lake.manifest import DATASET_NAME, MANIFEST_NAME, DatasetManifest, SourceManifest
from trifact.lake.stats import ResourceMonitor, log_mlflow

log = logging.getLogger("trifact.eda")
OUT = ROOT / "reports" / "eda"
BATCH = 20_000
BATCH_BYTES = 16 * 2**20  # uncompressed bytes per batch: words of a batch are exploded into one row each
VOCAB_CHARS = (
    200_000  # vocabulary and line counts read the first N characters of a document (a few are 100+ MB)
)
TOP_WORDS = 50_000
TOP_LINES = 200
SAMPLE_RATE = 0.02
SAMPLE_CAP = 4000  # documents per shard in the vocabulary sample
LOG_BINS = np.concatenate([[0], np.unique(np.round(np.logspace(0, 8, 161)).astype(np.int64))])
KK_LETTERS = "әғқңөұүһіӘҒҚҢӨҰҮҺІ"
MOJIBAKE = r"(?:Ð[\x80-\xbf]|Ñ[\x80-\xbf]|Ã[\x80-\xbf]|â€|Ã©|Â )"
CONTROL = r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
EMAIL = r"[\w.+-]+@[\w-]+\.[\w.-]+"
PHONE = r"(?:\+7|8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}|\+\d{1,3}[\s-]?\(?\d{2,4}\)?[\s-]?\d{3}[\s-]?\d{2,4}"
URL = r"https?://[^\s<>\"']+"
DIGITS12 = r"(?:^|\D)\d{12}(?:\D|$)"  # Rust regex: no look-around, delimiters are stripped later
DIGITS16 = r"(?:^|\D)(?:\d[ -]?){15}\d(?:\D|$)"
WORD = r"[^\W\d_]+(?:[-'’][^\W\d_]+)*"

# text / id / url / date columns of every raw table (role -> columns profiled as text)
RAW_TEXT: dict[str, dict[str, list[str]]] = {
    "wikipedia": {"part": ["text", "title"]},
    "fineweb2_kk": {"part": ["text"]},
    "multidomain_kk": {"part": ["text"]},
    "kazparc": {"entries": ["kk", "ru", "en"], "pairs": ["pair"], "sync_entries": ["kk", "ru", "en"], "sync_pairs": ["pair"]},
    "kaznerd": {"part": []}, "wikiann": {"part": []}, "wikineural": {"part": []},
    "nerel": {"part": ["text"]},
    "squad_v2": {"part": ["context", "question"]}, "sberquad": {"part": ["context", "question"]},
    "kazqad": {"part": ["context", "question"]},
    "ms_marco": {"part": ["query"]},
    "mmarco_ru": {"collection": ["text"], "queries": ["text"]},
    "miracl": {"topics": ["query"]},
    "miracl_corpus": {"part": ["text", "title"]},
    "ru_hnp": {"part": ["query"]},
    "xnli": {"part": ["premise", "hypothesis"]},
    "wikipedia_wikitext": {"part": ["wikitext"]},
}  # fmt: skip
ID_COLS = ["id", "doc_id", "docid", "pid", "qid", "query_id", "sentence_id", "page_id"]


# ----------------------------------------------------------------------------- helpers


def hist(values: np.ndarray) -> list[int]:
    return np.histogram(np.clip(values, 0, LOG_BINS[-1] - 1), bins=LOG_BINS)[0].astype(np.int64).tolist()


def quantiles_from_hist(
    h: list[int], qs: tuple[float, ...] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
) -> dict[str, float]:
    c = np.cumsum(h)
    n = c[-1] if len(c) else 0
    out = {}
    for q in qs:
        if n == 0:
            out[f"p{int(q * 100):02d}"] = 0.0
            continue
        i = int(np.searchsorted(c, q * n))
        out[f"p{int(q * 100):02d}"] = float(LOG_BINS[min(i, len(LOG_BINS) - 1)])  # lower bin edge
    return out


def iin_valid(s: str) -> bool:
    s = re.sub(r"\D", "", s)
    if len(s) != 12:
        return False
    d = [int(x) for x in s]
    w1 = list(range(1, 12))
    c = sum(a * b for a, b in zip(d[:11], w1, strict=True)) % 11
    if c == 10:
        w2 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2]
        c = sum(a * b for a, b in zip(d[:11], w2, strict=True)) % 11
        if c == 10:
            return False
    month, day = int(s[2:4]), int(s[4:6])
    return c == d[11] and 1 <= month <= 12 and 1 <= day <= 31 and s[6] in "123456"


def luhn(s: str) -> bool:
    d = [int(x) for x in re.sub(r"\D", "", s)]
    tot = 0
    for i, x in enumerate(reversed(d)):
        if i % 2:
            x *= 2
            x -= 9 if x > 9 else 0
        tot += x
    return tot % 10 == 0


def merge(a: Any, b: Any) -> Any:
    """Merge partial aggregates: numbers add, lists add element-wise, dicts merge recursively, Counters add."""
    if a is None:
        return b
    if b is None:
        return a
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = merge(out.get(k), v)
        return out
    if (
        isinstance(a, list)
        and isinstance(b, list)
        and len(a) == len(b)
        and all(isinstance(x, (int, float)) for x in a + b)
    ):
        return [x + y for x, y in zip(a, b, strict=True)]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + b
    if isinstance(a, list) and isinstance(b, list):
        return a + b
    return a


# ----------------------------------------------------------------------------- shard profiling


def text_profile(s: pl.Series) -> dict[str, Any]:
    """Exact per-document text statistics of one batch (mergeable)."""
    t = s.fill_null("")
    df = pl.DataFrame({"t": t}).select(
        chars=pl.col("t").str.len_chars(),
        words=pl.col("t").str.count_matches(WORD),
        lines=pl.col("t").str.count_matches("\n") + 1,
        paras=pl.col("t").str.count_matches(r"\n\s*\n") + 1,
        cyr=pl.col("t").str.count_matches(r"[\p{Cyrillic}]"),
        lat=pl.col("t").str.count_matches(r"[A-Za-z]"),
        letters=pl.col("t").str.count_matches(r"\p{L}"),
        kk=pl.col("t").str.count_matches(f"[{KK_LETTERS}]"),
        fffd=pl.col("t").str.count_matches("�"),
        ctrl=pl.col("t").str.count_matches(CONTROL),
        moji=pl.col("t").str.count_matches(MOJIBAKE),
        email=pl.col("t").str.count_matches(EMAIL),
        phone=pl.col("t").str.count_matches(PHONE),
        url=pl.col("t").str.count_matches(URL),
        d12=pl.col("t").str.extract_all(DIGITS12),
        d16=pl.col("t").str.extract_all(DIGITS16),
        empty=pl.col("t").str.strip_chars() == "",
    )
    iin = sum(1 for xs in df["d12"].to_list() for x in xs if iin_valid(x))
    iin_docs = sum(1 for xs in df["d12"].to_list() if any(iin_valid(x) for x in xs))
    cards = sum(1 for xs in df["d16"].to_list() for x in xs if luhn(x))
    letters = df["letters"].to_numpy().astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        cyr_share = np.where(letters > 0, df["cyr"].to_numpy() / letters, np.nan)
        kk_share = np.where(
            df["cyr"].to_numpy() > 0, df["kk"].to_numpy() / np.maximum(df["cyr"].to_numpy(), 1), np.nan
        )
    share_bins = np.linspace(0, 1, 21)
    return {
        "docs": df.height,
        "empty": int(df["empty"].sum()),
        "chars": int(df["chars"].sum()), "words": int(df["words"].sum()),
        "lines": int(df["lines"].sum()), "paragraphs": int(df["paras"].sum()),
        "hist_chars": hist(df["chars"].to_numpy()), "hist_words": hist(df["words"].to_numpy()),
        "hist_paragraphs": hist(df["paras"].to_numpy()),
        "script": {
            "cyrillic_letters": int(df["cyr"].sum()), "latin_letters": int(df["lat"].sum()),
            "all_letters": int(letters.sum()), "kk_letters": int(df["kk"].sum()),
            "docs_mostly_latin": int(np.sum(np.nan_to_num(cyr_share, nan=1.0) < 0.5)),
            "docs_with_kk_letters": int((df["kk"] > 0).sum()),
            "hist_cyrillic_share": np.histogram(cyr_share[np.isfinite(cyr_share)], bins=share_bins)[0].tolist(),
            "hist_kk_letter_share": np.histogram(kk_share[np.isfinite(kk_share)], bins=share_bins)[0].tolist(),
        },
        "encoding": {
            "docs_replacement_char": int((df["fffd"] > 0).sum()), "replacement_chars": int(df["fffd"].sum()),
            "docs_control_chars": int((df["ctrl"] > 0).sum()), "docs_mojibake": int((df["moji"] > 0).sum()),
        },
        "pii": {
            "docs_email": int((df["email"] > 0).sum()), "emails": int(df["email"].sum()),
            "docs_phone": int((df["phone"] > 0).sum()), "phones": int(df["phone"].sum()),
            "docs_url": int((df["url"] > 0).sum()), "urls": int(df["url"].sum()),
            "docs_iin": iin_docs, "iins": iin, "card_numbers": cards,
        },
    }  # fmt: skip


def words_of(s: pl.Series) -> pl.DataFrame:
    return (
        pl.DataFrame({"t": s.fill_null("")})
        .select(w=pl.col("t").str.to_lowercase().str.extract_all(WORD))
        .explode("w")
        .drop_nulls()
        .group_by("w")
        .len()
    )


def vocab_sample(texts: list[str]) -> dict[str, Any]:
    """Exact vocabulary statistics and n-gram top lists of a document sample."""
    if not texts:
        return {}
    s = pl.Series(texts)
    wc = words_of(s).sort("len", descending=True)
    toks = int(wc["len"].sum())
    ranks = np.arange(1, wc.height + 1)
    freq = wc["len"].to_numpy()
    sel = (ranks >= 10) & (ranks <= min(10_000, wc.height))
    zipf = float(-np.polyfit(np.log(ranks[sel]), np.log(freq[sel]), 1)[0]) if sel.sum() > 10 else None
    lists = (
        pl.DataFrame({"t": s}).select(w=pl.col("t").str.to_lowercase().str.extract_all(WORD))["w"].to_list()
    )
    bi: Counter[str] = Counter()
    tri: Counter[str] = Counter()
    for ws in lists:
        bi.update(" ".join(ws[i : i + 2]) for i in range(len(ws) - 1))
        tri.update(" ".join(ws[i : i + 3]) for i in range(len(ws) - 2))
    return {
        "sample_docs": len(texts), "tokens": toks, "types": wc.height,
        "type_token_ratio": round(wc.height / max(toks, 1), 5),
        "hapax_share_of_types": round(float((freq == 1).sum()) / max(wc.height, 1), 4),
        "hapax_share_of_tokens": round(float((freq == 1).sum()) / max(toks, 1), 5),
        "zipf_exponent": round(zipf, 4) if zipf else None,
        "rank_freq": [[int(r), int(f)] for r, f in zip(ranks, freq, strict=True) if r in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000)],
        "top_bigrams": bi.most_common(50), "top_trigrams": tri.most_common(50),
    }  # fmt: skip


def lines_of(s: pl.Series) -> pl.DataFrame:
    return (
        pl.DataFrame({"t": s.fill_null("")})
        .with_row_index("doc")
        .select("doc", line=pl.col("t").str.split("\n"))
        .explode("line")
        .with_columns(pl.col("line").str.strip_chars())
        .filter((pl.col("line").str.len_chars() >= 3) & (pl.col("line").str.len_chars() <= 200))
        .unique(["doc", "line"])
        .group_by("line")
        .len()
    )


def profile_shard(args: tuple[str, str, str, str, list[str], dict[str, Any]]) -> dict[str, Any]:
    """Profile of one Parquet shard: schema, columns, text statistics; key = (source, lang, table)."""
    path, source, lang, table, text_cols, opt = args
    pf = pq.ParquetFile(path)
    schema = {f.name: str(f.type) for f in pf.schema_arrow}
    res: dict[str, Any] = {"shards": 1, "rows": pf.metadata.num_rows, "bytes": os.path.getsize(path),
                           "schema": {json.dumps(schema, sort_keys=True): 1}, "columns": {}, "text": {}}  # fmt: skip
    words: dict[str, pl.DataFrame] = {}
    lines: dict[str, pl.DataFrame] = {}
    sample: dict[str, list[str]] = {}
    extremes: dict[str, list[Any]] = {}
    hosts: Counter[str] = Counter()
    tlds: Counter[str] = Counter()
    months: Counter[str] = Counter()
    group_cols = [c for c in opt.get("group_by", []) if c in schema]
    id_col = next((c for c in ID_COLS if c in schema), None)
    ids = []
    seed = opt.get("seed", 42)
    md = pf.metadata
    raw_bytes = sum(md.row_group(i).total_byte_size for i in range(md.num_row_groups))
    batch = max(500, min(BATCH, int(BATCH_BYTES * md.num_rows / max(raw_bytes, 1))))
    for b in pf.iter_batches(batch_size=batch):
        df = pl.from_arrow(b)
        assert isinstance(df, pl.DataFrame)
        for c in df.columns:
            col = df[c]
            st = {"nulls": col.null_count(), "rows": df.height}
            if col.dtype == pl.String:
                st["empty"] = int((col.fill_null("x").str.strip_chars() == "").sum())
            if col.dtype in (pl.String, pl.Int64, pl.Int32, pl.Float32, pl.Float64, pl.Boolean, pl.Int16):
                # up to two distinct values per batch: a column is constant if the merged set has one value
                st["values"] = [str(v)[:100] for v in col.drop_nulls().unique().head(2).to_list()]
            res["columns"] = merge(res["columns"], {c: st})
        if id_col:
            ids.append(df[id_col].cast(pl.String).hash(seed=7))
        if group_cols:
            res["groups"] = merge(res.get("groups"), {
                "|".join(str(x) for x in k): v for *k, v in df.group_by(group_cols).len().iter_rows()
            })  # fmt: skip
        for c in text_cols:
            if c not in df.columns:
                continue
            s = df[c].cast(pl.String)
            res["text"][c] = merge(res["text"].get(c), text_profile(s))
            if opt.get("vocab", True):
                head = s.str.slice(0, VOCAB_CHARS)
                w = words_of(head)
                words[c] = (
                    w if c not in words else pl.concat([words[c], w]).group_by("w").agg(pl.col("len").sum())
                )
                if words[c].height > 4 * TOP_WORDS:
                    words[c] = words[c].sort("len", descending=True).head(2 * TOP_WORDS)
                ln = lines_of(head)
                lines[c] = (
                    ln
                    if c not in lines
                    else pl.concat([lines[c], ln]).group_by("line").agg(pl.col("len").sum())
                )
                if lines[c].height > 200_000:
                    lines[c] = lines[c].filter(pl.col("len") > 1)
                key = df.with_row_index("r").select(
                    h=(pl.col("r").cast(pl.String) + pl.lit(path)).hash(seed=seed)
                )["h"]
                pick = s.filter(key < int(SAMPLE_RATE * 2**64)).drop_nulls().to_list()
                sample.setdefault(c, []).extend(x[:5000] for x in pick)
            n = s.fill_null("").str.len_chars()
            # indices of the 3 shortest / longest non-empty texts: sorting the batch with its texts doubles it
            nz = n.to_frame("_n").with_row_index("_r").filter(pl.col("_n") > 0)
            short_i = nz.bottom_k(3, by="_n").sort("_n")["_r"]
            long_i = nz.top_k(3, by="_n").sort("_n")["_r"]
            pick_cols = [x for x in [id_col] if x]
            for name, part in (
                ("shortest", df.with_columns(_n=n)[short_i]),
                ("longest", df.with_columns(_n=n)[long_i]),
            ):
                for r in part.iter_rows(named=True):
                    extremes.setdefault(f"{c}:{name}", []).append(
                        {
                            "chars": r["_n"],
                            "id": str(r[pick_cols[0]]) if pick_cols else None,
                            "text": (r[c] or "")[:300],
                        }
                    )
        if "url" in df.columns:
            u = df["url"].cast(pl.String).drop_nulls()
            host = u.str.extract(r"^[a-z]+://([^/:?#]+)").str.to_lowercase().drop_nulls()
            hosts.update(dict(host.value_counts().iter_rows()))
            tlds.update(dict(host.str.extract(r"\.([a-z0-9-]+)$").drop_nulls().value_counts().iter_rows()))
        if "date" in df.columns:
            d = df["date"].cast(pl.String).drop_nulls().str.slice(0, 7)
            months.update(dict(d.value_counts().iter_rows()))
    for c in list(res["text"]):
        if c in words:
            res["text"][c]["top_words"] = words[c].sort("len", descending=True).head(TOP_WORDS).rows()
            res["text"][c]["top_lines"] = lines[c].sort("len", descending=True).head(TOP_LINES * 5).rows()
            res["text"][c]["vocab_sample"] = sample.get(c, [])[:SAMPLE_CAP]
        res["text"][c]["extremes"] = {k: v for k, v in extremes.items() if k.startswith(f"{c}:")}
    if ids:
        h = pl.concat(ids)
        res["id"] = {
            "column": id_col,
            "rows": h.len(),
            "duplicate_rows_in_shard": int(h.len() - h.n_unique()),
        }
        out = Path(opt["work"]) / f"ids-{abs(hash(path)) % 10**12:012d}.parquet"
        pl.DataFrame({"h": h.unique()}).write_parquet(out)
        res["id"]["hash_file"] = str(out)
    if hosts:
        res["hosts"] = dict(hosts.most_common(2000))
        res["tlds"] = dict(tlds.most_common(500))
    if months:
        res["months"] = dict(months)
    if opt.get("quality") and "quality" in schema:
        res["quality"] = quality_profile(path, opt["quality_signals"])
    if opt.get("lang") and "lang_score" in schema:
        res["langid"] = langid_profile(path)
    gc.collect()
    return {"key": (source, lang, table), "res": res}


def quality_profile(path: str, signals: list[str]) -> dict[str, Any]:
    q = pl.read_parquet(
        path, columns=["quality", "n_chars", "n_words", *(["drop_reason"] if "dropped" in path else [])]
    )
    q = q.unnest("quality")
    bins = np.linspace(0, 1, 51)
    out: dict[str, Any] = {"rows": q.height}
    for s in signals:
        if s not in q.columns:
            continue
        v = q[s].cast(pl.Float64).to_numpy()
        v = v[np.isfinite(v)]
        out[s] = {"hist": hist(v).copy() if s in ("n_lines", "boilerplate_lines") else np.histogram(np.clip(v, 0, 1), bins=bins)[0].tolist(),
                  "sum": float(v.sum()), "n": int(v.size)}  # fmt: skip
    if "drop_reason" in q.columns:
        out["reasons"] = dict(q["drop_reason"].value_counts().iter_rows())
    # correlation sample (Spearman via ranks) on up to 5000 rows
    cols = [s for s in signals if s in q.columns]
    sm = q.select(cols).head(5000).to_numpy().astype(np.float64)
    out["corr_sample"] = {"cols": cols, "rows": sm.tolist()}
    return out


def langid_profile(path: str) -> dict[str, Any]:
    q = pl.read_parquet(path, columns=["lang", "src_lang", "lang_score", "para_lang"])
    bins = np.linspace(0, 1, 51)
    mixed = q.select(
        n=pl.col("para_lang").list.len(),
        other=pl.col("para_lang").list.eval(pl.element().is_in(["kk", "ru", "en"]).not_()).list.sum(),
        targets=pl.col("para_lang").list.eval(pl.element().is_in(["kk", "ru", "en"])).list.sum(),
        foreign=pl.struct("lang", "para_lang").map_elements(
            lambda r: sum(1 for x in (r["para_lang"] or []) if x in ("kk", "ru", "en") and x != r["lang"]),
            return_dtype=pl.Int64,
        ),
    )
    return {
        "hist_lang_score": np.histogram(q["lang_score"].fill_null(0).to_numpy(), bins=bins)[0].tolist(),
        "moves": dict(
            q.select(pl.concat_str("src_lang", "lang", separator="->")).to_series().value_counts().iter_rows()
        ),
        "docs_with_other_target_lang_paragraph": int((mixed["foreign"] > 0).sum()),
        "docs_with_non_target_paragraph": int((mixed["other"] > 0).sum()),
        "paragraphs": int(mixed["n"].sum()),
        "paragraphs_by_lang": dict(
            q.select(pl.col("para_lang").explode()).to_series().value_counts().iter_rows()
        ),
    }


# ----------------------------------------------------------------------------- finalize


def _top(parts: list[Any], n: int) -> list[tuple[str, int]]:
    c: Counter[str] = Counter()
    for part in parts:
        c.update({k: v for k, v in part})
    return c.most_common(n)


def finalize(key: tuple[str, str, str], res: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in res.items() if k not in ("text", "schema", "columns", "quality")}
    out["schemas"] = len(res["schema"])
    out["schema"] = json.loads(next(iter(res["schema"])))
    out["schema_drift"] = len(res["schema"]) > 1
    cols = {}
    for c, st in res["columns"].items():
        cols[c] = {"null_share": round(st["nulls"] / max(st["rows"], 1), 5), "nulls": st["nulls"]}
        if "empty" in st:
            cols[c]["empty_share"] = round(st["empty"] / max(st["rows"], 1), 5)
        if "values" in st:
            cols[c]["constant"] = len(set(st["values"])) <= 1 and st["nulls"] < st["rows"]
    out["columns"] = cols
    txt = {}
    for c, t in res["text"].items():
        d = {k: v for k, v in t.items() if k not in ("top_words", "top_lines", "vocab_sample", "extremes")}
        n = max(t["docs"], 1)
        d["mean_chars"] = round(t["chars"] / n, 1)
        d["mean_words"] = round(t["words"] / n, 1)
        d["mean_paragraphs"] = round(t["paragraphs"] / n, 2)
        for h in ("chars", "words", "paragraphs"):
            d[f"q_{h}"] = quantiles_from_hist(t[f"hist_{h}"])
        if "top_words" in t:
            d["top_words"] = _top(t["top_words"], 200)
            d["top_lines"] = [(ln, k) for ln, k in _top(t["top_lines"], TOP_LINES) if k > 1]
            d["vocab"] = vocab_sample([x for part in t["vocab_sample"] for x in part][: 50 * SAMPLE_CAP])
        ex: dict[str, list[Any]] = {}
        for part in t["extremes"]:
            for k, v in part.items():
                ex.setdefault(k, []).extend(v)
        d["extremes"] = {
            k: sorted(v, key=lambda r: r["chars"])[:5]
            if k.endswith("shortest")
            else sorted(v, key=lambda r: -r["chars"])[:5]
            for k, v in ex.items()
        }
        txt[c] = d
    out["text"] = txt
    if "id" in res:
        h = pl.scan_parquet(res["id"]["hash_file"]).select("h").collect()
        out["id"] = {
            "column": res["id"]["column"][0], "rows": res["id"]["rows"],
            "duplicates_within_shards": res["id"]["duplicate_rows_in_shard"],
            "duplicates_across_shards": int(h.height - h["h"].n_unique()),
        }  # fmt: skip
    if "quality" in res:
        q = res["quality"]
        sig = {}
        for name, v in q.items():
            if isinstance(v, dict) and "hist" in v:
                sig[name] = {"mean": round(v["sum"] / max(v["n"], 1), 4), "hist": v["hist"]}
        corr = None
        rows = [r for part in q.get("corr_sample", []) for r in part["rows"]]
        if rows:
            colnames = q["corr_sample"][0]["cols"]
            arr = np.array(rows, dtype=np.float64)
            ok = np.isfinite(arr).all(axis=1)
            if ok.sum() > 10:
                ranks = np.argsort(np.argsort(arr[ok], axis=0), axis=0)
                cm = np.corrcoef(ranks, rowvar=False)
                corr = {
                    "cols": colnames,
                    "spearman": np.round(np.nan_to_num(cm), 3).tolist(),
                    "rows": int(ok.sum()),
                }
        out["quality"] = {"rows": q["rows"], "signals": sig, "reasons": q.get("reasons"), "correlation": corr}
    return out


def accumulate(
    acc: dict[tuple[str, str, str], dict[str, Any]], p: dict[str, Any]
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Merge one shard profile into the running accumulator (avoids holding all shard dicts in RAM)."""
    k = tuple(p["key"])
    r = p["res"]
    # lists that must not be added element-wise: keep as list of parts
    for t in r.get("text", {}).values():
        for f in ("top_words", "top_lines", "vocab_sample"):
            if f in t:
                t[f] = [t[f]]
        t["extremes"] = [t["extremes"]]
    if "id" in r:
        r["id"] = {**r["id"], "hash_file": [r["id"]["hash_file"]], "column": [r["id"]["column"]]}
    if "quality" in r:
        r["quality"]["corr_sample"] = [r["quality"]["corr_sample"]]
    acc[k] = merge(acc.get(k), r)  # type: ignore[index]
    return acc


def combine(parts: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    acc: dict[tuple[str, str, str], dict[str, Any]] = {}
    for p in parts:
        accumulate(acc, p)
    return acc


# ----------------------------------------------------------------------------- layers


def raw_jobs(root: Path, sources: list[str], work: Path, seed: int) -> list[tuple[Any, ...]]:
    jobs = []
    for s in sources:
        d = source_dir(root, "raw", s)
        if not (d / MANIFEST_NAME).is_file():
            continue
        m = SourceManifest.read(d)
        spec = RAW_TEXT.get(s, {})
        for sh in m.shards:
            if sh.table == "entities":
                continue
            jobs.append((str(d / sh.path), s, sh.lang, sh.table, spec.get(sh.table, []),
                         {"work": str(work), "seed": seed, "group_by": ["_config", "_split"], "vocab": s not in ("wikipedia_wikitext",)}))  # fmt: skip
    return jobs


def derived_jobs(root: Path, layer: str, work: Path, seed: int, signals: list[str]) -> list[tuple[Any, ...]]:
    jobs = []
    base = root / layer
    for mpath in sorted({*base.glob(f"*/{DATASET_NAME}"), *base.glob(f"*/*/{DATASET_NAME}")}):
        d = mpath.parent
        m = DatasetManifest.read(d)
        for sh in m.shards:
            if sh.table in ("boilerplate",):
                continue
            name = m.name.split("/", 1)[1] if "/" in m.name else m.name
            jobs.append((str(d / sh.path), name, sh.lang, sh.table, ["text"],
                         {"work": str(work), "seed": seed, "group_by": ["source", "domain"] if sh.table != "contaminated" else [],
                          "quality": True, "quality_signals": signals, "lang": True, "vocab": sh.table == "part"}))  # fmt: skip
    return jobs


def entities_profile(root: Path) -> dict[str, Any]:
    """Wikidata `entities`: types, label/description/alias coverage, sitelinks, class graph connectivity."""
    d = source_dir(root, "raw", "wikidata")
    files = [str(d / s.path) for s in SourceManifest.read(d).shards if s.table == "entities"]
    lf = pl.scan_parquet(files)
    langs = ["kk", "ru", "en", "mul"]
    cov = (
        lf.select(
            entities=pl.len(),
            items=(pl.col("type") == "item").sum(),
            properties=(pl.col("type") == "property").sum(),
            **{f"label_{lg}": pl.col(f"label_{lg}").is_not_null().sum() for lg in langs},
            **{f"description_{lg}": pl.col(f"description_{lg}").is_not_null().sum() for lg in langs},
            **{f"aliases_{lg}": pl.col(f"aliases_{lg}").list.len().sum() for lg in langs},
            **{f"with_aliases_{lg}": (pl.col(f"aliases_{lg}").list.len() > 0).sum() for lg in langs},
            **{
                f"sitelink_{w}": pl.col(f"sitelink_{w}").is_not_null().sum()
                for w in ("kkwiki", "ruwiki", "enwiki")
            },
            label_any_target=(
                pl.col("label_kk").is_not_null()
                | pl.col("label_ru").is_not_null()
                | pl.col("label_en").is_not_null()
            ).sum(),
            label_all_three=(
                pl.col("label_kk").is_not_null()
                & pl.col("label_ru").is_not_null()
                & pl.col("label_en").is_not_null()
            ).sum(),
            no_p31=(pl.col("p31").list.len() == 0).sum(),
            with_p279=(pl.col("p279").list.len() > 0).sum(),
            sitelinks_total=pl.col("n_sitelinks").sum(),
        )
        .collect(engine="streaming")
        .to_dicts()[0]
    )
    sl = lf.select("n_sitelinks").collect(engine="streaming")["n_sitelinks"].to_numpy()
    top_p31 = (
        lf.select(pl.col("p31").explode())
        .drop_nulls()
        .group_by("p31")
        .len()
        .sort("len", descending=True)
        .head(100)
        .collect(engine="streaming")
    )
    labels = (
        lf.filter(pl.col("qid").is_in(top_p31["p31"].implode()))
        .select("qid", "label_en")
        .collect(engine="streaming")
    )
    top_p31 = top_p31.join(labels, left_on="p31", right_on="qid", how="left")
    p31_per = lf.select(n=pl.col("p31").list.len()).group_by("n").len().sort("n").collect(engine="streaming")
    kk_by_type = (
        lf.filter(pl.col("label_kk").is_not_null())
        .select(pl.col("p31").list.first())
        .group_by("p31")
        .len()
        .sort("len", descending=True)
        .head(30)
        .collect(engine="streaming")
    )
    # class graph: P279 edges among classes; weakly connected components and depth to the root
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    edges = (
        lf.select("qid", "p279")
        .filter(pl.col("p279").list.len() > 0)
        .explode("p279")
        .collect(engine="streaming")
    )
    nodes = pl.concat([edges["qid"], edges["p279"]]).unique()
    idx = {q: i for i, q in enumerate(nodes.to_list())}
    a = np.array([idx[x] for x in edges["qid"].to_list()])
    b = np.array([idx[x] for x in edges["p279"].to_list()])
    g = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(len(idx), len(idx)))
    ncomp, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab)
    inst = lf.select(pl.col("p31").explode()).drop_nulls().collect(engine="streaming")["p31"]
    in_graph = inst.is_in(nodes.implode()).sum()
    return {
        "coverage": cov,
        "n_sitelinks_quantiles": {
            f"p{q}": float(np.quantile(sl, q / 100)) for q in (50, 75, 90, 95, 99, 100)
        },
        "n_sitelinks_hist": np.histogram(
            np.clip(sl, 0, 400), bins=[0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 400]
        )[0].tolist(),
        "top_p31": top_p31.to_dicts(),
        "p31_per_entity": p31_per.to_dicts(),
        "kk_labelled_top_p31": kk_by_type.to_dicts(),
        "class_graph": {
            "edges": edges.height,
            "classes": len(idx),
            "components": int(ncomp),
            "largest_component_share": round(float(sizes.max()) / max(len(idx), 1), 4),
            "component_size_hist": np.histogram(sizes, bins=[1, 2, 3, 5, 10, 100, 1000, 10**9])[0].tolist(),
            "instance_links": int(inst.len()),
            "instance_links_to_graph_classes": int(in_graph),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layer", choices=["raw", "clean", "curated", "entities"])
    ap.add_argument("--workers", type=int)
    ap.add_argument("--sources")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    params = load_params()
    root = (ROOT / params["lake"]["root"]).resolve()
    workers = args.workers or min(int(params["curation"]["workers"]), 2)
    work = root / ".cache" / "work" / f"eda_{args.layer}"
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with ResourceMonitor() as mon:
        if args.layer == "entities":
            report: dict[str, Any] = {"wikidata/entities": entities_profile(root)}
        else:
            from trifact.data.quality import SIGNALS

            signals = [s for s in SIGNALS if s not in ("n_chars", "n_words")] + ["boilerplate_lines"]
            if args.layer == "raw":
                srcs = (
                    args.sources.split(",")
                    if args.sources
                    else [p.name for p in sorted((root / "raw").iterdir()) if p.is_dir()]
                )
                jobs = raw_jobs(root, srcs, work, params["seed"])
            else:
                jobs = derived_jobs(root, args.layer, work, params["seed"], signals)
            os.environ["POLARS_MAX_THREADS"] = str(max(1, (os.cpu_count() or 1) // workers))
            acc: dict[tuple[str, str, str], dict[str, Any]] = {}
            jobs.sort(key=lambda j: -os.path.getsize(j[0]))  # big shards first: shorter tail
            with ProcessPoolExecutor(workers) as pool:
                for i, p in enumerate(pool.map(profile_shard, jobs, chunksize=1)):
                    accumulate(acc, p)
                    if i % 20 == 0:
                        log.info("%s: %d/%d shards", args.layer, i + 1, len(jobs))
            report = {"/".join(k): finalize(k, v) for k, v in sorted(acc.items())}
    meta = {"layer": args.layer, "seconds": round(time.perf_counter() - t0, 1), "workers": workers,
            "peak_rss_mb": mon.stats(1, 0, workers).peak_rss_mb}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    write_report(OUT / f"{args.layer}.json", {"meta": meta, "profiles": report})
    log_mlflow("eda", f"eda-{args.layer}", {"layer": args.layer, "workers": workers}, {
        "seconds": meta["seconds"], "peak_rss_mb": meta["peak_rss_mb"], "profiles": len(report)}, )  # fmt: skip
    log.info("%s: %d profiles in %.0f s", args.layer, len(report), meta["seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
