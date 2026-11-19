"""Stage build_pairs: retrieval training pairs with hard negatives over the 9 language directions (ROADMAP 3.2).

python -m trifact.data.build_pairs [--profile full|pilot] [--synthetic DIR]

Tables of `data/task/retrieval/<lang>/` (lang = query language, `multi` for passages):
* `pairs` — pair_id, query_id, query, q_lang, doc_id, d_lang, direction (`<q_lang>-<d_lang>`), origin,
  source, split (train | validation), neg_ids (hard negatives, up to `retrieval.hard_negatives_per_query`);
* `passages` — doc_id, lang, source, title, text: every positive and negative passage once.

Origins: `gold` (human queries and judgements: MS MARCO, MIRACL, KazQAD, SQuAD v2, SberQuAD), `mt_published`
(published machine translations: mMARCO ru, KazQAD nq-translate-kk), `parallel` (KazParC human pairs,
query = sentence in one language, positive = its translation), `synthetic_mt` (KazParC `sync`, machine
translated; train only, ADR-0015), `mt_query` (our NLLB translation of a gold question, the passage
stays in its language), `synthetic` (teacher queries, `--synthetic`, after the round-trip filter).

Hard negatives: MS MARCO — the non-selected passages Bing returned for the query; mMARCO — the
published BM25 negatives of the triples; MIRACL — passages judged non-relevant; ru-HNP — its negatives;
QA sources — BM25 (`trifact.data.bm25`) over the contexts of the same source and language, excluding the
positive context and contexts with the same title; KazParC — none (in-batch negatives at training).
Test splits never enter: they are the frozen eval of `build_eval` (ADR-0007).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl

from trifact.core.params import ROOT, load_params
from trifact.data.curation import fingerprint, write_report
from trifact.data.qa_common import QA_SOURCES, bm25_negatives, load_qa
from trifact.data.task_common import DatasetWriter, dataset_input_of, finish, inherited, raw_input, scan_raw
from trifact.data.translations import MT_LICENSE, qc_ok
from trifact.lake.layout import LakeConfig
from trifact.lake.stats import ResourceMonitor, log_mlflow

log = logging.getLogger("trifact.build_pairs")
SOURCES = [
    "ms_marco",
    "mmarco_ru",
    "miracl",
    "miracl_corpus",
    "ru_hnp",
    "squad_v2",
    "sberquad",
    "kazqad",
    "kazparc",
]
PAIR_COLS = [
    "pair_id",
    "query_id",
    "query",
    "q_lang",
    "doc_id",
    "d_lang",
    "direction",
    "origin",
    "source",
    "split",
    "neg_ids",
]


def text_id(prefix: str, col: str = "text") -> pl.Expr:
    return pl.concat_str(pl.lit(prefix + "/"), pl.col(col).hash(seed=0).cast(pl.String))


def finish_pairs(df: pl.DataFrame, source: str, origin: str | pl.Expr) -> pl.DataFrame:
    return df.with_columns(
        source=pl.lit(source), origin=origin if isinstance(origin, pl.Expr) else pl.lit(origin),
        direction=pl.concat_str("q_lang", "d_lang", separator="-"),
        pair_id=pl.concat_str(pl.lit(source), "query_id", "doc_id", separator="|"),
    ).select(PAIR_COLS)  # fmt: skip


def assignments(root: Path, source: str) -> pl.DataFrame:
    return (
        pl.scan_parquet(str(root / "task/splits/*/assignments-*.parquet"))
        .filter(pl.col("source") == source)
        .select("lang", "config", "example_id", "split")
        .collect()
    )


def ms_marco(root: Path, k_neg: int, val_share: float, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    lf = scan_raw(root, "ms_marco").filter(
        (pl.col("_config") == "v2.1") & pl.col("_split").is_in(["train", "validation"])
    )
    df = lf.select("query_id", "query", "passages").collect()
    df = df.unique("query_id", keep="first")
    p = (
        df.select(
            "query_id",
            "query",
            t=pl.col("passages").struct.field("passage_text"),
            s=pl.col("passages").struct.field("is_selected"),
        )
        .explode("t", "s")
        .with_columns(doc_id=text_id("msmarco", "t"))
    )
    pos = p.filter(pl.col("s") == 1).unique("query_id", keep="first").select("query_id", "query", "doc_id")
    neg = p.filter(pl.col("s") == 0).group_by("query_id").agg(neg_ids=pl.col("doc_id").head(k_neg))
    pairs = pos.join(neg, on="query_id", how="left").with_columns(
        query_id=pl.concat_str(pl.lit("msmarco/"), pl.col("query_id").cast(pl.String)), q_lang=pl.lit("en"), d_lang=pl.lit("en"),
        split=pl.when((pl.col("query_id").hash(seed=seed) % 10_000) < int(val_share * 10_000)).then(pl.lit("validation")).otherwise(pl.lit("train")),
        neg_ids=pl.col("neg_ids").fill_null([]),
    )  # fmt: skip
    used = (
        pl.concat([pairs.select("doc_id"), pairs.select(doc_id=pl.col("neg_ids")).explode("doc_id")])
        .drop_nulls()
        .unique()
    )
    passages = (
        p.select("doc_id", text="t")
        .unique("doc_id")
        .join(used, on="doc_id", how="semi")
        .with_columns(lang=pl.lit("en"), source=pl.lit("ms_marco"), title=pl.lit(None, dtype=pl.String))
    )
    return finish_pairs(pairs, "ms_marco", "gold"), passages


def mmarco(root: Path, k_neg: int, val_share: float, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    tri = scan_raw(root, "mmarco_ru", "triples").select("qid", "pos_pid", "neg_pid")
    g = (
        tri.group_by("qid", "pos_pid")
        .agg(neg_ids=pl.col("neg_pid").unique(maintain_order=True).head(k_neg))
        .collect(engine="streaming")
    )
    q = (
        scan_raw(root, "mmarco_ru", "queries")
        .filter(pl.col("_split") == "train")
        .select("qid", query="text")
        .unique("qid")
        .collect()
    )
    pairs = g.join(q, on="qid", how="inner").with_columns(
        query_id=pl.concat_str(pl.lit("mmarco/"), "qid"), doc_id=pl.concat_str(pl.lit("mmarco/"), "pos_pid"),
        neg_ids=pl.col("neg_ids").list.eval(pl.concat_str(pl.lit("mmarco/"), pl.element())),
        q_lang=pl.lit("ru"), d_lang=pl.lit("ru"),
        split=pl.when((pl.col("qid").hash(seed=seed) % 10_000) < int(val_share * 10_000)).then(pl.lit("validation")).otherwise(pl.lit("train")),
    )  # fmt: skip
    used = pl.concat(
        [pairs.select("doc_id"), pairs.select(doc_id=pl.col("neg_ids")).explode("doc_id")]
    ).unique()
    coll = scan_raw(root, "mmarco_ru", "collection").select(
        doc_id=pl.concat_str(pl.lit("mmarco/"), "pid"), text="text"
    )
    passages = (
        coll.join(used.lazy(), on="doc_id", how="semi")
        .collect(engine="streaming")
        .unique("doc_id")
        .with_columns(lang=pl.lit("ru"), source=pl.lit("mmarco_ru"), title=pl.lit(None, dtype=pl.String))
    )
    return finish_pairs(pairs, "mmarco_ru", "mt_published"), passages


def miracl(root: Path, k_neg: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    asg = assignments(root, "miracl").filter(pl.col("split").is_in(["train", "validation"]))
    topics = (
        scan_raw(root, "miracl", "topics")
        .select("lang", "qid", "query")
        .collect()
        .join(asg.select("lang", qid="example_id", split="split"), on=["lang", "qid"])
    )
    qrels = (
        scan_raw(root, "miracl", "qrels")
        .select("lang", "qid", "docid", rel=pl.col("relevance").cast(pl.Int32))
        .collect()
    )
    pos = qrels.filter(pl.col("rel") > 0)
    neg = qrels.filter(pl.col("rel") == 0).group_by("lang", "qid").agg(neg_ids=pl.col("docid").head(k_neg))
    pairs = topics.join(pos, on=["lang", "qid"]).join(neg, on=["lang", "qid"], how="left").with_columns(
        query_id=pl.concat_str(pl.lit("miracl/"), "lang", "qid", separator="/"), doc_id=pl.concat_str(pl.lit("miracl/"), "docid"),
        neg_ids=pl.col("neg_ids").fill_null([]).list.eval(pl.concat_str(pl.lit("miracl/"), pl.element())),
        q_lang="lang", d_lang="lang",
    )  # fmt: skip
    used = (
        pl.concat([pairs.select("doc_id"), pairs.select(doc_id=pl.col("neg_ids")).explode("doc_id")])
        .drop_nulls()
        .unique()
    )
    corpus = scan_raw(root, "miracl_corpus").select(
        "lang", "title", "text", doc_id=pl.concat_str(pl.lit("miracl/"), "docid")
    )
    passages = (
        corpus.join(used.lazy(), on="doc_id", how="semi")
        .collect(engine="streaming")
        .unique("doc_id")
        .with_columns(source=pl.lit("miracl_corpus"))
    )
    return finish_pairs(pairs, "miracl", "gold"), passages


def ru_hnp(root: Path, val_share: float, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    df = (
        scan_raw(root, "ru_hnp")
        .filter(pl.col("_split").is_in(["train", "validation"]))
        .select("_split", "query", "pos", "neg")
        .collect()
    )
    df = df.with_columns(query_id=text_id("ru_hnp", "query"))
    negs = df.select("query_id", n=pl.col("neg")).explode("n").with_columns(doc_id=text_id("ru_hnp", "n"))
    neg_ids = negs.group_by("query_id").agg(neg_ids=pl.col("doc_id"))
    pairs = (
        df.select("query_id", "query", "_split", t=pl.col("pos"))
        .explode("t")
        .with_columns(doc_id=text_id("ru_hnp", "t"))
    )
    pairs = pairs.join(neg_ids, on="query_id", how="left").with_columns(
        q_lang=pl.lit("ru"),
        d_lang=pl.lit("ru"),
        split=pl.col("_split"),
        neg_ids=pl.col("neg_ids").fill_null([]),
    )
    passages = (
        pl.concat([pairs.select("doc_id", text="t"), negs.select("doc_id", text="n")])
        .unique("doc_id")
        .with_columns(lang=pl.lit("ru"), source=pl.lit("ru_hnp"), title=pl.lit(None, dtype=pl.String))
    )
    return finish_pairs(
        pairs, "ru_hnp", "synthetic"
    ), passages  # ru-HNP pairs were generated by gpt-3.5-turbo


def qa_pairs(root: Path, k_neg: int, mt: pl.DataFrame | None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Question -> context pairs of the QA sources (train/validation), BM25 negatives, nq kk<->en, MT queries."""
    frames, passages = [], []
    for s in QA_SOURCES:
        df = load_qa(root, s, ("train", "validation")).filter(pl.col("answerable"))
        df = df.with_columns(neg_ids=bm25_negatives(df, k_neg))
        frames.append(finish_pairs(
            df.with_columns(q_lang=pl.col("lang"), d_lang=pl.col("lang"), query=pl.col("question"),
                            query_id=pl.concat_str(pl.lit(f"{s}/"), "lang", "example_id", separator="/")),
            s, pl.when(pl.col("config") == "nq-translate-kk").then(pl.lit("mt_published")).otherwise(pl.lit("gold")),
        ))  # fmt: skip
        passages.append(
            df.unique("doc_id")
            .select("doc_id", "title", "lang", text="context")
            .with_columns(source=pl.lit(s))
        )
        if s == "kazqad":  # parallel nq: kk question -> en context and en question -> kk context
            nq = df.filter(pl.col("config").str.starts_with("nq-"))
            kk = nq.filter(pl.col("lang") == "kk").select(
                "example_id", "split", q_kk="question", d_kk="doc_id"
            )
            en = nq.filter(pl.col("lang") == "en").select("example_id", q_en="question", d_en="doc_id")
            j = kk.join(en, on="example_id")
            for ql, dl, q, d in (("kk", "en", "q_kk", "d_en"), ("en", "kk", "q_en", "d_kk")):
                frames.append(finish_pairs(j.select(
                    query_id=pl.concat_str(pl.lit("kazqad/"), pl.lit(ql), "example_id", separator="/"), query=q, doc_id=d,
                    q_lang=pl.lit(ql), d_lang=pl.lit(dl), split="split", neg_ids=pl.lit([], dtype=pl.List(pl.String)),
                ), "kazqad", "mt_published"))  # fmt: skip
        if mt is not None:  # our translations of the questions: query in another language, passage unchanged
            src = df.filter(~pl.col("config").str.starts_with("nq-")).select(
                "lang", "example_id", "split", "question", "doc_id"
            )
            t = src.join(mt, left_on=["lang", "question"], right_on=["src_lang", "text"], how="inner")
            frames.append(finish_pairs(t.select(
                query_id=pl.concat_str(pl.lit(f"{s}/mt"), "tgt_lang", "example_id", separator="/"), query="translation",
                doc_id="doc_id", q_lang="tgt_lang", d_lang="lang", split="split", neg_ids=pl.lit([], dtype=pl.List(pl.String)),
            ), s, "mt_query"))  # fmt: skip
    return pl.concat(frames), pl.concat(passages, how="diagonal_relaxed")


def kazparc(root: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    frames = []
    for table, origin in (("pairs", "parallel"), ("sync_pairs", "synthetic_mt")):
        df = (
            scan_raw(root, "kazparc", table)
            .filter(
                pl.col("pair").is_in(["kk_en", "kk_ru", "en_ru"])
                & pl.col("_split").is_in(["train", "validation"])
            )
            .select("id", "pair", "_split", a="source_lang", b="target_lang")
            .collect(engine="streaming")
        )
        df = df.with_columns(la=pl.col("pair").str.slice(0, 2), lb=pl.col("pair").str.slice(3, 2))
        for qa, qb, ql, dl in (("a", "b", "la", "lb"), ("b", "a", "lb", "la")):
            frames.append(df.select(
                query_id=pl.concat_str(pl.lit(f"kazparc/{table}"), "id", pl.col(ql), separator="/"), query=qa,
                doc_id=pl.concat_str(pl.lit("kazparc/"), pl.col(dl), pl.col(qb).hash(seed=0).cast(pl.String), separator="/"),
                q_lang=ql, d_lang=dl, split="_split", neg_ids=pl.lit([], dtype=pl.List(pl.String)), text=qb, origin=pl.lit(origin),
            ))  # fmt: skip
    allp = pl.concat(frames)
    passages = (
        allp.select("doc_id", "text", lang="d_lang")
        .unique("doc_id")
        .with_columns(source=pl.lit("kazparc"), title=pl.lit(None, dtype=pl.String))
    )
    pairs = finish_pairs(allp, "kazparc", pl.col("origin"))
    return pairs, passages


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=["pilot", "full"])
    ap.add_argument("--synthetic", type=Path, help="filtered teacher queries (task/synthetic/queries)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    params = load_params()
    profile = args.profile or params["scale"]["profile"]
    root = (ROOT / params["lake"]["root"]).resolve()
    out = root / "task" / "retrieval"
    k = int(params["retrieval"]["hard_negatives_per_query"])
    val, seed = float(params["split"]["val"]), int(params["seed"])
    mt_dir = root / "task/mt/qa"
    stage_params = {"hard_negatives_per_query": k, "val": val, "profile": profile, "cap": params["scale"][profile]["retrieval_pairs"],
                    "mt_queries": mt_dir.is_dir(), "synthetic": bool(args.synthetic)}  # fmt: skip
    inputs_m = [root / f"raw/{s}/source.yaml" for s in SOURCES] + [root / "task/splits/dataset.yaml"]
    if mt_dir.is_dir():
        inputs_m.append(mt_dir / "dataset.yaml")
    if args.synthetic:
        inputs_m.append(args.synthetic / "dataset.yaml")
    fp = fingerprint(
        "build_pairs", stage_params, [Path(__file__), ROOT / "src/trifact/data/bm25.py"], inputs_m
    )
    t0 = time.perf_counter()
    mt = None
    if mt_dir.is_dir():
        mt = (
            pl.scan_parquet(str(mt_dir / "*/translations-*.parquet"))
            .filter(qc_ok(params["nli"]["kk_filters"]["min_chrf_back"]))
            .select("src_lang", "tgt_lang", "text", "translation")
            .collect()
        )
    with ResourceMonitor() as mon:
        if out.exists():
            shutil.rmtree(out)
        w = DatasetWriter(out, LakeConfig.from_params(params["lake"], root))
        stats: list[dict[str, Any]] = []
        builders = [
            ("ms_marco", lambda: ms_marco(root, k, val, seed)), ("mmarco_ru", lambda: mmarco(root, k, val, seed)),
            ("miracl", lambda: miracl(root, k)), ("ru_hnp", lambda: ru_hnp(root, val, seed)),
            ("qa", lambda: qa_pairs(root, k, mt)), ("kazparc", lambda: kazparc(root)),
        ]  # fmt: skip
        if args.synthetic:
            from trifact.data.synthetic import synthetic_pairs

            builders.append(("synthetic", lambda: synthetic_pairs(args.synthetic)))
        cap = stage_params["cap"]
        for name, fn in builders:
            pairs, passages = fn()
            if profile == "pilot":
                pairs = pairs.filter(
                    pl.col("pair_id").hash(seed=seed) % 1000 < max(1, int(1000 * cap / 20_000_000))
                )
                used = (
                    pl.concat([pairs.select("doc_id"), pairs.select(doc_id="neg_ids").explode("doc_id")])
                    .drop_nulls()
                    .unique()
                )
                passages = passages.join(used, on="doc_id", how="semi")
            w.write("pairs", pairs, lang_col="q_lang")
            w.write("passages", passages.select("doc_id", "lang", "source", "title", "text"), lang_col=None)
            stats.append({"builder": name, "pairs": pairs.height, "passages": passages.height})
            stats += (
                pairs.group_by("source", "origin", "direction", "split")
                .agg(n=pl.len(), with_negatives=(pl.col("neg_ids").list.len() > 0).sum())
                .sort("source", "origin", "direction", "split")
                .to_dicts()
            )
            log.info("%s: %d pairs, %d passages", name, pairs.height, passages.height)
            del pairs, passages
        shards = w.close()
    total = sum(s.rows for s in shards if s.table == "pairs")
    by_dir: dict[str, int] = {}
    for s in stats:
        if "direction" in s:
            by_dir[s["direction"]] = by_dir.get(s["direction"], 0) + s["n"]
    metrics = {"pairs_total": total, "by_direction": dict(sorted(by_dir.items())), "detail": stats}
    lic = inherited(root, SOURCES) + ([MT_LICENSE] if mt is not None else [])
    inputs = [raw_input(root, s) for s in SOURCES] + [dataset_input_of(root, root / "task/splits")]
    if mt is not None:
        inputs.append(dataset_input_of(root, mt_dir))
    finish(
        out, name="task/retrieval", layer="task", stage="build_pairs", source="+".join(SOURCES),
        description="Retrieval training pairs (query, positive passage, hard negatives) over kk/ru/en and their "
        "cross-lingual directions; passages stored once in table `passages`.",
        params=stage_params, inputs=inputs, licenses=lic, shards=shards, rows_in={"multi": total},
        metrics=metrics, stats=mon.stats(total, 0, 1), fingerprint=fp, partial=profile != "full",
    )  # fmt: skip
    write_report(ROOT / "reports" / "retrieval_data.json", metrics)
    log_mlflow("task", "build_pairs", {**stage_params, "fingerprint": fp},
               {"pairs": total, "seconds": round(time.perf_counter() - t0, 1), **{f"dir_{d}": n for d, n in by_dir.items()}})  # fmt: skip
    print(metrics["by_direction"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
