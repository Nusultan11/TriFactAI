"""Stage build_ner: gold NER in the core schema + silver NER from Wikipedia links x Wikidata (ADR-0006, ADR-0015).

python -m trifact.data.build_ner [--profile full|pilot] [--workers N] [--skip-silver]

Output `data/task/ner/<lang>/<table>-*.parquet`, one row per sentence:
`sent_id, doc_key, source, lang, origin (gold|silver), tokens, tags (core BIO: PER/ORG/LOC), tags_full`.
Tables: `gold_train`, `gold_validation`, `gold_test` (split from `task/splits`, `drop` rows left out) and
`silver_train` (silver is train-only, ADR-0015).

Gold: KazNERD / WikiANN / WikiNEuRal ids -> tag names (`configs/labels.yaml: tagsets`) -> core labels
(per-source mapping); invalid BIO (`I-X` after `O`) is repaired to `B-X`. NEREL brat documents are
split into sentences and tokenized; spans are projected onto tokens, the outermost core span wins.
`tags_full` keeps the source tag names (KazNERD-25 for NER-4 and the DATE extractor).

Silver, three passes over `raw/wikipedia_wikitext`:
1. sentences (parallel per shard): prose paragraphs with internal link spans (`trifact.data.wikimarkup`)
   -> sentences -> tokens; a link becomes a token span with its target title;
2. resolve (Polars joins): target -> redirect target -> `task/kb` titles -> QID and NER type;
   the article's own title is resolved the same way;
3. tag (parallel per shard): linked spans of type PER/ORG/LOC become BIO; unlinked occurrences of
   surfaces linked elsewhere in the same article (and of the article title) are tagged too; a
   sentence is kept if it has an entity, `min_tokens..max_tokens` tokens, no link to an untyped
   capitalized target in the middle of a span, and (when `capital_filter`) no capitalized
   non-initial untagged word that is linked nowhere — i.e. likely a missed entity. Sentences equal
   (normalized) to any gold validation/test sentence are dropped (leakage, ADR-0007).

Noise of silver against gold (`metrics.silver_noise`): (a) type agreement of surfaces present in both
silver and gold; (b) a gazetteer built from silver surfaces (majority type, >= `gazetteer_min_count`
occurrences, >= `gazetteer_min_share` agreement) tags the gold test sentences: entity-level P/R/F1
per language and type estimates how well the link-derived labels match human annotation.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from trifact.core.params import ROOT, load_params
from trifact.data.curation import fingerprint, write_report
from trifact.data.ner_common import entities, map_tags, repair_bio, sentences, spans_to_bio, tokenize
from trifact.data.task_common import (
    DatasetWriter,
    by_lang,
    dataset_input_of,
    finish,
    inherited,
    key_expr,
    raw_input,
    scan_raw,
)
from trifact.data.wikimarkup import normalize_title, prose
from trifact.lake.layout import LakeConfig, source_dir
from trifact.lake.manifest import DatasetManifest, SourceManifest
from trifact.lake.stats import ResourceMonitor, log_mlflow

log = logging.getLogger("trifact.build_ner")
LABELS = ROOT / "configs" / "labels.yaml"
CODE = [Path(__file__), ROOT / "src/trifact/data/ner_common.py", ROOT / "src/trifact/data/wikimarkup.py",
        ROOT / "src/trifact/data/task_common.py"]  # fmt: skip
GOLD = ["kaznerd", "wikiann", "wikineural", "nerel"]
CORE = {"PER", "ORG", "LOC"}
CAP = re.compile(r"^[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ]")


# ----------------------------------------------------------------------------- gold


def gold_tokens(root: Path, source: str, labels: dict[str, Any]) -> pl.DataFrame:
    names = labels["tagsets"][source]
    df = scan_raw(root, source).select("lang", "_config", "_split", "tokens", "ner_tags").collect()
    df = df.with_row_index("row").with_columns(
        example_id=pl.concat_str(
            pl.col("_config").fill_null(""), pl.col("_split"), pl.col("row"), separator="/"
        ),
        tags_full=pl.col("ner_tags").list.eval(pl.element().replace_strict(dict(enumerate(names)))),
    )
    mapping = labels[source]
    full = df["tags_full"].to_list()
    fixed = [repair_bio(t) for t in full]
    core = [map_tags(t, mapping) for t, _ in fixed]
    return df.with_columns(
        tags=pl.Series(core, dtype=pl.List(pl.String)),
        tags_full=pl.Series([t for t, _ in fixed], dtype=pl.List(pl.String)),
        bio_repaired=pl.Series([n for _, n in fixed], dtype=pl.Int32),
        doc_key=key_expr(pl.col("tokens").list.join(" ")),
    ).select("lang", "example_id", "doc_key", "tokens", "tags", "tags_full", "bio_repaired")


def gold_nerel(root: Path, mapping: dict[str, str]) -> pl.DataFrame:
    df = scan_raw(root, "nerel").select("lang", "_split", "id", "text", "entities").collect()
    rows = []
    span_re = re.compile(r"^T\d+\t(\S+) (\d+) (\d+)(?:;\d+ \d+)*\t")
    for r in df.iter_rows(named=True):
        spans = []
        full = []
        for e in r["entities"] or []:
            m = span_re.match(e)
            if not m:
                continue
            s, t = int(m.group(2)), int(m.group(3))
            full.append((s, t, m.group(1)))
            if m.group(1) in mapping:
                spans.append((s, t, mapping[m.group(1)]))
        text = r["text"]
        doc_key = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()[:200]
        for k, (a, b) in enumerate(sentences(text)):
            toks = tokenize(text[a:b], a)
            if not toks:
                continue
            inside = [x for x in spans if x[0] >= a and x[1] <= b]
            inside_full = [x for x in full if x[0] >= a and x[1] <= b]
            rows.append({
                "lang": r["lang"], "example_id": f"{r['_split']}/{r['id']}", "sent": k,
                "doc_key": doc_key, "tokens": [t for t, _, _ in toks],
                "tags": spans_to_bio(toks, inside), "tags_full": spans_to_bio(toks, inside_full), "bio_repaired": 0,
            })  # fmt: skip
    out = pl.DataFrame(rows)
    return out.with_columns(example_id=pl.col("example_id")).drop("sent")


def build_gold(root: Path, labels: dict[str, Any]) -> pl.DataFrame:
    asg = (
        pl.scan_parquet(str(root / "task/splits/*/assignments-*.parquet"))
        .select("source", "example_id", "split")
        .collect()
    )
    frames = []
    for s in GOLD:
        df = gold_nerel(root, labels["nerel"]) if s == "nerel" else gold_tokens(root, s, labels)
        df = df.with_columns(source=pl.lit(s)).join(
            asg.filter(pl.col("source") == s).drop("source"), on="example_id", how="left"
        )
        missing = df["split"].null_count()
        if missing:
            raise ValueError(f"{s}: {missing} examples without a split assignment")
        frames.append(df)
    g = pl.concat(frames, how="diagonal_relaxed")
    return g.with_columns(
        sent_id=pl.concat_str(
            "source", "example_id", pl.int_range(pl.len()).over("source", "example_id"), separator="/"
        ),
        origin=pl.lit("gold"),
    )


# ----------------------------------------------------------------------------- silver pass 1: sentences


SENT_SCHEMA = pa.schema(
    [
        ("page_id", pa.int64()), ("title", pa.string()), ("lang", pa.string()), ("para", pa.int32()),
        ("tokens", pa.list_(pa.string())),
        ("links", pa.list_(pa.struct([("s", pa.int32()), ("e", pa.int32()), ("target", pa.string())]))),
    ]
)  # fmt: skip
PART_SENTENCES = 200_000  # sentences per pass-1 part: whole articles, bounded memory in passes 2 and 3


def sentences_unit(args: tuple[str, str, str, int, int]) -> dict[str, Any]:
    """Pass 1 over one wikitext shard: parts `<out>-<k:03d>.parquet` of whole articles, marker `<out>.done`."""
    path, lang, out, min_tok, max_tok = args
    rows: dict[str, list[Any]] = defaultdict(list)
    pf = pq.ParquetFile(path)
    n_art = n_sent = k = 0

    def flush() -> None:
        nonlocal k
        if rows["page_id"]:
            pq.write_table(
                pa.table(dict(rows), schema=SENT_SCHEMA), f"{out}-{k:03d}.parquet", compression="zstd"
            )
            k += 1
            rows.clear()

    for b in pf.iter_batches(batch_size=200, columns=["page_id", "title", "wikitext"]):
        for pid, title, wt in zip(
            *(b.column(c).to_pylist() for c in ("page_id", "title", "wikitext")), strict=True
        ):
            n_art += 1
            for p_idx, (para, links) in enumerate(prose(wt or "")):
                for a, z in sentences(para):
                    toks = tokenize(para[a:z], a)
                    if not (min_tok <= len(toks) <= max_tok):
                        continue
                    spans = []
                    for ls, le, target in links:
                        if ls < a or le > z:
                            continue
                        idx = [i for i, (_, ts, te) in enumerate(toks) if ts >= ls and te <= le]
                        if idx:
                            spans.append({"s": idx[0], "e": idx[-1] + 1, "target": target})
                    rows["page_id"].append(pid)
                    rows["title"].append(normalize_title(title or ""))
                    rows["lang"].append(lang)
                    rows["para"].append(p_idx)
                    rows["tokens"].append([t for t, _, _ in toks])
                    rows["links"].append(spans)
                    n_sent += 1
            if len(rows["page_id"]) >= PART_SENTENCES:  # article border: an article never spans two parts
                flush()
    flush()
    res = {"articles": n_art, "sentences": n_sent, "parts": k}
    Path(f"{out}.done").write_text(json.dumps(res))
    return res


# ----------------------------------------------------------------------------- silver pass 3: tagging


def tag_unit(args: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    path, out, cfg = args
    df = pl.read_parquet(path)
    keep_rows: dict[str, list[Any]] = defaultdict(list)
    reasons: Counter[str] = Counter()
    types: Counter[str] = Counter()
    for (_pid,), art in df.group_by("page_id", maintain_order=True):
        recs = art.to_dicts()
        # surfaces linked to a typed entity anywhere in the article (+ the article title itself)
        surf: dict[tuple[str, ...], str] = {}
        linked_words: set[str] = set()
        for r in recs:
            for ln in r["links"] or []:
                words = tuple(r["tokens"][ln["s"] : ln["e"]])
                linked_words.update(words)
                if ln["ner_type"] in CORE:
                    surf.setdefault(words, ln["ner_type"])
        at = recs[0]["title_type"]
        if at in CORE:
            # the article title without its disambiguation suffix, e.g. "Paris (mythology)" -> "Paris"
            title_words = tuple(t for t, _, _ in tokenize(re.sub(r"\s*\(.*\)$", "", recs[0]["title"])))
            if title_words:
                surf.setdefault(title_words, at)
                linked_words.update(title_words)
        by_len = sorted(surf.items(), key=lambda kv: -len(kv[0]))
        for r in recs:
            toks = r["tokens"]
            tags = ["O"] * len(toks)
            bad_link = False
            for ln in r["links"] or []:
                t = ln["ner_type"]
                if t in CORE:
                    if any(x != "O" for x in tags[ln["s"] : ln["e"]]):
                        continue
                    tags[ln["s"]] = f"B-{t}"
                    for i in range(ln["s"] + 1, ln["e"]):
                        tags[i] = f"I-{t}"
                elif t is None and CAP.match(toks[ln["s"]] or "") and ln["s"] > 0:
                    bad_link = True  # capitalized link target we cannot type: may be a missed entity
            for words, t in by_len:  # propagate unlinked mentions, longest surfaces first
                n = len(words)
                for i in range(len(toks) - n + 1):
                    if tuple(toks[i : i + n]) == words and all(x == "O" for x in tags[i : i + n]):
                        tags[i] = f"B-{t}"
                        for j in range(i + 1, i + n):
                            tags[j] = f"I-{t}"
            if not any(t != "O" for t in tags):
                reasons["no_entity"] += 1
                continue
            if bad_link:
                reasons["untyped_capitalized_link"] += 1
                continue
            if cfg["capital_filter"] and any(
                tags[i] == "O"
                and CAP.match(toks[i])
                and toks[i] not in linked_words
                and toks[i - 1] not in '.:;!?«"('
                for i in range(1, len(toks))
            ):
                reasons["capitalized_untagged"] += 1
                continue
            for _, _, lab in entities(tags):
                types[lab] += 1
            keep_rows["page_id"].append(r["page_id"])
            keep_rows["title"].append(r["title"])
            keep_rows["lang"].append(r["lang"])
            keep_rows["tokens"].append(toks)
            keep_rows["tags"].append(tags)
            keep_rows["n_links"].append(len(r["links"] or []))
    reasons["kept"] = len(keep_rows["tokens"])
    if keep_rows["tokens"]:
        pl.DataFrame(dict(keep_rows)).write_parquet(out, compression="zstd")
    return {"reasons": dict(reasons), "types": dict(types)}


# ----------------------------------------------------------------------------- noise


def surface_counts(df: pl.DataFrame) -> pl.DataFrame:
    """(lang, surface, ner_type, n): entity surfaces (lowercased) of silver sentences with their counts."""
    rows = [
        (lang, " ".join(toks[s:e]).lower(), t)
        for lang, toks, tags in df.select("lang", "tokens", "tags").iter_rows()
        for s, e, t in entities(list(tags))
    ]
    out = pl.DataFrame(rows, schema=["lang", "surface", "ner_type"], orient="row")
    return out.group_by("lang", "surface", "ner_type").agg(n=pl.len())


def gazetteer(counts: list[str], min_count: int, min_share: float) -> pl.DataFrame:
    """(lang, surface, ner_type): majority type of every silver entity surface, from `surface_counts` files."""
    agg = (
        pl.scan_parquet(counts)
        .group_by("lang", "surface", "ner_type")
        .agg(len=pl.col("n").sum())
        .collect(engine="streaming")
        .with_columns(total=pl.col("len").sum().over("lang", "surface"))
    )
    return (
        agg.sort("len", descending=True)
        .unique(["lang", "surface"], keep="first")
        .filter((pl.col("total") >= min_count) & (pl.col("len") / pl.col("total") >= min_share))
        .select("lang", "surface", "ner_type", count="total")
    )


def gazetteer_tag(tokens: list[str], gaz: dict[str, str], max_len: int) -> list[str]:
    tags = ["O"] * len(tokens)
    low = [t.lower() for t in tokens]
    i = 0
    while i < len(tokens):
        for n in range(min(max_len, len(tokens) - i), 0, -1):
            t = gaz.get(" ".join(low[i : i + n]))
            if t:
                tags[i] = f"B-{t}"
                for j in range(i + 1, i + n):
                    tags[j] = f"I-{t}"
                i += n
                break
        else:
            i += 1
    return tags


def prf(gold: list[list[str]], pred: list[list[str]]) -> dict[str, Any]:
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    for g, p in zip(gold, pred, strict=True):
        ge, pe = set(entities(g)), set(entities(p))
        for e in ge & pe:
            tp[e[2]] += 1
        for e in pe - ge:
            fp[e[2]] += 1
        for e in ge - pe:
            fn[e[2]] += 1
    out: dict[str, Any] = {}
    for lab in sorted(CORE) + ["all"]:
        t = sum(tp.values()) if lab == "all" else tp[lab]
        f_p = sum(fp.values()) if lab == "all" else fp[lab]
        f_n = sum(fn.values()) if lab == "all" else fn[lab]
        p = t / (t + f_p) if t + f_p else 0.0
        r = t / (t + f_n) if t + f_n else 0.0
        out[lab] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r) if p + r else 0.0, 4),
                    "support": t + f_n}  # fmt: skip
    return out


def silver_noise(counts: list[str], gold: pl.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    gaz = gazetteer(counts, cfg["gazetteer_min_count"], cfg["gazetteer_min_share"])
    # (a) type agreement on shared surfaces
    gold_ents = []
    for lang, toks, tags in gold.select("lang", "tokens", "tags").iter_rows():
        for s, e, t in entities(tags):
            gold_ents.append((lang, " ".join(toks[s:e]).lower(), t))
    ge = pl.DataFrame(gold_ents, schema=["lang", "surface", "gold_type"], orient="row")
    g_major = (
        ge.group_by("lang", "surface", "gold_type")
        .len()
        .sort("len", descending=True)
        .unique(["lang", "surface"], keep="first")
    )
    shared = gaz.join(g_major, on=["lang", "surface"])
    agree = (
        shared.group_by("lang", "ner_type")
        .agg(surfaces=pl.len(), agree=(pl.col("ner_type") == pl.col("gold_type")).mean().round(4))
        .sort("lang", "ner_type")
    )
    confusion = shared.group_by("lang", "ner_type", "gold_type").len().sort("lang", "ner_type", "gold_type")
    # (b) gazetteer tagging of gold test sentences
    test = gold.filter(pl.col("split") == "test")
    by: dict[str, Any] = {}
    max_len = (
        int(gaz.select(pl.col("surface").str.count_matches(" ").max()).item() or 0) + 1 if gaz.height else 1
    )
    for (lang, source), g in test.group_by("lang", "source", maintain_order=True):
        d = dict(gaz.filter(pl.col("lang") == lang).select("surface", "ner_type").iter_rows())
        toks, gold_tags = g["tokens"].to_list(), g["tags"].to_list()
        pred = [gazetteer_tag(t, d, min(max_len, 8)) for t in toks]
        by[f"{source}/{lang}"] = prf(gold_tags, pred)
    return {
        "gazetteer_surfaces": by_lang(gaz),
        "type_agreement_shared_surfaces": agree.to_dicts(),
        "type_confusion": confusion.to_dicts(),
        "gazetteer_vs_gold_test": by,
    }


# ----------------------------------------------------------------------------- manual review sheet


def render_tags(tokens: list[str], tags: list[str]) -> str:
    """`[Абай|PER] Семейде ...` — entities in brackets with their label."""
    out, i = [], 0
    for s0, e0, lab in entities(tags):
        out += tokens[i:s0] + [f"[{' '.join(tokens[s0:e0])}|{lab}]"]
        i = e0
    return " ".join(out + tokens[i:])


def review_sheet(silver: pl.DataFrame, path: Path, per_lang: int, seed: int) -> int:
    """Blank CSV for human reviewers: per sentence the number of wrong / missed entities (ADR-0015 noise check)."""
    if path.exists():
        return 0
    smp = (
        silver.with_columns(r=pl.col("sent_id").hash(seed=seed))
        .sort("r")
        .group_by("lang", maintain_order=True)
        .head(per_lang)
    )
    rows = smp.select(
        "sent_id", "lang", text=pl.struct("tokens", "tags").map_elements(lambda x: render_tags(x["tokens"], x["tags"]), return_dtype=pl.String),
        entities=pl.col("tags").list.eval(pl.element().str.starts_with("B-")).list.sum(),
        wrong_type=pl.lit(""), wrong_span=pl.lit(""), not_entity=pl.lit(""), missed=pl.lit(""), reviewer=pl.lit(""), note=pl.lit(""),
    )  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.write_csv(path)
    return rows.height


# ----------------------------------------------------------------------------- driver


def build_silver(
    root: Path, work: Path, params: dict[str, Any], workers: int, limit_shards: int | None
) -> tuple[list[str], dict[str, Any]]:
    cfg = params["ner"]["silver"]
    m = SourceManifest.read(source_dir(root, "raw", "wikipedia_wikitext"))
    base = source_dir(root, "raw", "wikipedia_wikitext")
    parts = [s for s in m.shards if s.table == "part"]
    if limit_shards:
        per_lang: dict[str, list[Any]] = defaultdict(list)
        for s in parts:
            per_lang[s.lang].append(s)
        parts = [s for v in per_lang.values() for s in v[:limit_shards]]
    (work / "sent").mkdir(parents=True, exist_ok=True)
    (work / "res").mkdir(parents=True, exist_ok=True)
    (work / "tag").mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, s in enumerate(parts):
        out = work / "sent" / f"{i:05d}"
        if not Path(f"{out}.done").is_file():
            for stale in (work / "sent").glob(f"{i:05d}-*.parquet"):
                stale.unlink()
            jobs.append((str(base / s.path), s.lang, str(out), cfg["min_tokens"], cfg["max_tokens"]))
    stats: dict[str, Any] = {"pass1": Counter()}
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(sentences_unit, jobs):
            stats["pass1"].update(r)
    log.info("silver pass 1: %d shards", len(parts))
    # pass 2: resolve link targets through redirects and the KB title table
    kb = pl.scan_parquet(str(root / "task/kb/multi/titles-*.parquet")).select("lang", "title", "ner_type")
    red = (
        pl.concat(
            [
                pl.scan_parquet(str(base / s.path)).with_columns(lang=pl.lit(s.lang))
                for s in m.shards
                if s.table == "redirects"
            ]
        )
        .select(
            "lang",
            title=pl.col("title").map_elements(normalize_title, return_dtype=pl.String),
            target=pl.col("target").map_elements(normalize_title, return_dtype=pl.String),
        )
        .collect()
    )
    kbd = kb.collect()
    sent_parts = sorted(p.name for p in (work / "sent").glob("*.parquet"))
    for name in sent_parts:
        out = work / "res" / name
        if out.is_file():
            continue
        df = pl.read_parquet(work / "sent" / name).with_row_index("sid")
        links = (
            df.select("sid", "lang", "links")
            .explode("links")
            .unnest("links")
            .filter(pl.col("target").is_not_null())
        )
        links = (
            links.join(red, left_on=["lang", "target"], right_on=["lang", "title"], how="left")
            .with_columns(resolved=pl.coalesce("target_right", "target"))
            .join(kbd, left_on=["lang", "resolved"], right_on=["lang", "title"], how="left")
        )
        agg = links.group_by("sid").agg(links=pl.struct("s", "e", "target", "ner_type"))
        tt = (
            df.select("sid", "lang", "title")
            .join(red, left_on=["lang", "title"], right_on=["lang", "title"], how="left")
            .with_columns(r=pl.coalesce("target", "title"))
            .join(kbd, left_on=["lang", "r"], right_on=["lang", "title"], how="left")
            .select("sid", title_type="ner_type")
        )
        res = (
            df.drop("links")
            .join(agg, on="sid", how="left")
            .join(tt, on="sid", how="left")
            .sort("sid")
            .drop("sid")
        )
        tmp = out.with_name(out.name + ".tmp")
        res.write_parquet(tmp, compression="zstd")
        tmp.replace(out)
    log.info("silver pass 2: links resolved in %d parts", len(sent_parts))
    jobs3 = [
        (str(work / "res" / name), str(work / "tag" / name), cfg)
        for name in sent_parts
        if not (work / "tag" / name).with_suffix(".done").is_file()
    ]
    stats["reasons"] = Counter()
    stats["types"] = Counter()
    with ProcessPoolExecutor(workers) as pool:
        for (_, out, _), r in zip(jobs3, pool.map(tag_unit, jobs3), strict=True):
            Path(out).with_suffix(".done").write_text(json.dumps(r))
    for p in sorted((work / "tag").glob("*.done")):
        r = json.loads(p.read_text())
        stats["reasons"].update(r["reasons"])
        stats["types"].update(r["types"])
    log.info("silver pass 3: %s", dict(stats["reasons"]))
    files = sorted(str(p) for p in (work / "tag").glob("*.parquet"))
    return files, {k: dict(v) for k, v in stats.items()}


def select_silver(
    files: list[str], eval_keys: pl.Series, cap: int, seed: int
) -> tuple[pl.DataFrame, dict[str, int]]:
    """(file, row, lang, rank) of the kept silver sentences: sentences of gold validation/test and duplicates
    (by the 64-bit hash of the normalized text) dropped, at most `cap` per language in hash order. Only the
    keys are held in memory, never the sentences."""
    keys = pl.concat([
        pl.scan_parquet(f).select("lang", kh=key_expr(pl.col("tokens").list.join(" ")).hash(seed))
        .with_row_index("row").with_columns(file=pl.lit(i, dtype=pl.Int32)).collect()
        for i, f in enumerate(files)
    ])  # fmt: skip
    n0 = keys.height
    keys = keys.filter(~pl.col("kh").is_in(eval_keys.implode()))
    n1 = keys.height
    keys = keys.unique("kh", keep="first", maintain_order=True)
    n2 = keys.height
    keys = keys.with_columns(rank=pl.col("kh").hash(seed).rank("ordinal").over("lang")).filter(
        pl.col("rank") <= cap
    )
    stats = {
        "leaked_eval_sentences_dropped": n0 - n1,
        "duplicates_dropped": n1 - n2,
        "over_cap_dropped": n2 - keys.height,
    }
    return keys.select("file", "row", "lang", "rank"), stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=["pilot", "full"], default="full")
    ap.add_argument("--workers", type=int)
    ap.add_argument(
        "--skip-silver", action="store_true", help="gold only (silver needs task/kb and wikitext)"
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    params = load_params()
    root = (ROOT / params["lake"]["root"]).resolve()
    out = root / "task" / "ner"
    work = root / ".cache" / "work" / "ner_silver"
    labels = yaml.safe_load(LABELS.read_text(encoding="utf-8"))
    workers = args.workers or params["curation"]["workers"]
    cfg = params["ner"]["silver"]
    stage_params = {
        "silver": cfg,
        "profile": args.profile,
        "max_per_lang": params["scale"][args.profile]["ner_sentences"],
    }
    inputs_m = [root / f"raw/{s}/source.yaml" for s in GOLD] + [root / "task/splits/dataset.yaml"]
    if not args.skip_silver:
        inputs_m += [root / "raw/wikipedia_wikitext/source.yaml", root / "task/kb/dataset.yaml"]
    fp = fingerprint("build_ner", stage_params, [*CODE, LABELS], inputs_m)
    t0 = time.perf_counter()
    lake = LakeConfig.from_params(params["lake"], root)
    with ResourceMonitor() as mon:
        gold = build_gold(root, labels)
        log.info(
            "gold: %s",
            gold.group_by("source", "lang", "split").len().sort("source", "lang", "split").to_dicts(),
        )
        silver = None
        smeta: dict[str, Any] = {}
        if not args.skip_silver:
            if work.exists() and (work / "fp.txt").is_file() and (work / "fp.txt").read_text() != fp:
                shutil.rmtree(work)
            work.mkdir(parents=True, exist_ok=True)
            (work / "fp.txt").write_text(fp)
            lim = None if args.profile == "full" else 1
            silver, smeta = build_silver(root, work, params, workers, lim)
        if out.exists():
            shutil.rmtree(out)
        w = DatasetWriter(out, lake)
        cols = ["sent_id", "doc_key", "source", "lang", "origin", "tokens", "tags", "tags_full"]
        for split in ("train", "validation", "test"):
            w.write(f"gold_{split}", gold.filter(pl.col("split") == split).select(cols))
        metrics: dict[str, Any] = {
            "gold": gold.filter(pl.col("split") != "drop")
            .group_by("source", "lang", "split")
            .agg(
                sentences=pl.len(),
                tokens=pl.col("tokens").list.len().sum(),
                bio_repaired=pl.col("bio_repaired").sum(),
            )
            .sort("source", "lang", "split")
            .to_dicts(),
        }
        if silver is not None:
            eval_keys = (
                gold.filter(pl.col("split").is_in(["validation", "test"]))
                .select(key_expr(pl.col("tokens").list.join(" ")).hash(params["seed"]))
                .to_series()
            )
            keep, sel = select_silver(silver, eval_keys, stage_params["max_per_lang"], params["seed"])
            gaz_dir = work / "gaz"
            shutil.rmtree(gaz_dir, ignore_errors=True)
            gaz_dir.mkdir()
            ents, review, kept = [], [], []
            # one tagged part at a time: memory is bounded by a part (<= 200k sentences)
            for i, f in enumerate(silver):
                k = keep.filter(pl.col("file") == i).select("row", "rank")
                if k.is_empty():
                    continue
                df = pl.read_parquet(f).with_row_index("row").join(k, on="row").with_columns(
                    sent_id=pl.format("wiki/{}/{}/{}-{}", "lang", "page_id", pl.lit(Path(f).stem), "row"),
                    doc_key=pl.format("wiki/{}/{}", "lang", "title"),
                    source=pl.lit("wikipedia_wikitext"), origin=pl.lit("silver"), tags_full=pl.col("tags"),
                )  # fmt: skip
                w.write("silver_train", df.select(cols))
                surface_counts(df).write_parquet(gaz_dir / f"{i:05d}.parquet")
                ents.append(df.select("lang", e=pl.col("tags").list.eval(pl.element().filter(pl.element().str.starts_with("B-"))))
                            .explode("e").drop_nulls().group_by("lang", "e").len())  # fmt: skip
                kept.append(df.group_by("lang").len())
                review.append(df.filter(pl.col("rank") <= 200).select("sent_id", "lang", "tokens", "tags"))
            metrics["silver"] = {
                **smeta, **sel,
                "kept_by_lang": {str(a): int(b) for a, b in pl.concat(kept).group_by("lang").agg(pl.col("len").sum()).sort("lang").iter_rows()},
                "entities_by_lang_type": pl.concat(ents).group_by("lang", "e").agg(pl.col("len").sum()).sort("lang", "e").to_dicts(),
            }  # fmt: skip
            counts = sorted(str(p) for p in gaz_dir.glob("*.parquet"))
            metrics["silver_noise"] = silver_noise(counts, gold.filter(pl.col("split") != "drop"), cfg)
            metrics["review_sheet"] = review_sheet(
                pl.concat(review), root / "review" / "ner_silver" / "to_label.csv", 200, params["seed"]
            )
        shards = w.close()
    sources = GOLD + ([] if args.skip_silver else ["wikipedia_wikitext", "wikidata"])
    inputs = [raw_input(root, x) for x in GOLD] + [dataset_input_of(root, root / "task/splits")]
    if not args.skip_silver:
        inputs += [raw_input(root, "wikipedia_wikitext"), dataset_input_of(root, root / "task/kb")]
    rows = sum(x.rows for x in shards)
    m = finish(
        out, name="task/ner", layer="task", stage="build_ner", source="+".join(sources),
        description="NER sentences in the core schema PER/ORG/LOC (ADR-0006): gold train/validation/test from "
        "KazNERD, WikiANN, WikiNEuRal, NEREL; silver train from Wikipedia links typed by Wikidata P31 (ADR-0015).",
        params=stage_params, inputs=inputs, licenses=inherited(root, sources), shards=shards,
        rows_in=by_lang(gold), metrics=metrics, stats=mon.stats(rows, 0, workers), fingerprint=fp,
        partial=args.profile != "full",
    )  # fmt: skip
    write_report(ROOT / "reports" / "ner_data.json", metrics)
    log_mlflow("task", "build_ner", {**stage_params, "fingerprint": fp}, {
        "rows": rows, "seconds": round(time.perf_counter() - t0, 1),
        **{f"{t}_{lg}": n for (t, lg), n in _rows(m).items()},
    })  # fmt: skip
    print(json.dumps({k: v for k, v in metrics.items() if k != "silver_noise"}, ensure_ascii=False)[:3000])
    return 0


def _rows(m: DatasetManifest) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for s in m.shards:
        out[(s.table, s.lang)] = out.get((s.table, s.lang), 0) + s.rows
    return out


if __name__ == "__main__":
    raise SystemExit(main())
