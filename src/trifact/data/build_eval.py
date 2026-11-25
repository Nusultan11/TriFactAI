"""Stage build_eval: frozen, leakage-checked evaluation sets over the 9 language directions (ADR-0007, ADR-0015).

python -m trifact.data.build_eval build      # (re)build sets that are not frozen yet, write review sheets
python -m trifact.data.build_eval status     # human review progress per sheet
python -m trifact.data.build_eval freeze     # apply finished reviews, freeze every set that may be frozen
python -m trifact.data.build_eval leakage    # overlap of all eval sets with the current train data

Sets in `data/eval/<set>/` (each a dataset with manifest, data card and `frozen.json`):
* `ner` — gold test sentences of KazNERD, WikiANN, WikiNEuRal, NEREL (task/ner `gold_test`);
* `qa` — extractive QA test: SQuAD v2 (official dev), SberQuAD (re-split test), KazQAD test incl. its nq part;
* `retrieval` — tables `queries` (query_id, query, q_lang, d_lang, direction, origin, needs_review), `qrels`
  (query_id, doc_id, relevance) and `docs` (closed pool per document language: every judged passage of the test
  topics and every test context). Directions: kk-kk, ru-ru, en-en from human queries; kk-en / en-kk from the
  parallel KazQAD nq test part (published translation); the remaining cross-lingual directions from NLLB
  translations of test queries (`origin = mt_query`, `needs_review = true`);
* `nli` — XNLI test en/ru (human) and kk (NLLB, `needs_review = true`).

Freezing: a set whose rows need no review is frozen at build time; a set with `needs_review` rows stays a
candidate until the full review packet in `data/eval/review/` is filled by human reviewers (verdict ok / fix / drop,
fixed text), with at least `evaluation.human_verified_min` items verified (ROADMAP 3.5). `freeze`
applies the verdicts (drop rows, replace fixed text) and freezes. A frozen set is never rebuilt: its
fingerprint and shard hashes are recorded in `frozen.json` and checked by `status`. AI review never counts.
Leakage: normalized questions, queries, contexts and NER sentences of every set are checked against all
train/validation task data; the overlap is reported in `metrics.leakage` and must be zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from trifact.core.params import ROOT, load_params
from trifact.data.curation import fingerprint, write_report
from trifact.data.qa_common import QA_SOURCES, load_qa
from trifact.data.task_common import (
    DatasetWriter,
    by_lang,
    dataset_input_of,
    finish,
    inherited,
    key_expr,
    raw_files_by_lang,
    raw_input,
    rows_with_keys,
    scan_raw,
    stable_hash,
)
from trifact.data.translations import MT_LICENSE, qc_ok
from trifact.lake.layout import LakeConfig, sha256_file
from trifact.lake.manifest import DatasetManifest, verify_dataset
from trifact.lake.stats import ResourceMonitor, log_mlflow

log = logging.getLogger("trifact.build_eval")
REVIEW_COLS = ["item_id", "set", "kind", "direction", "source_text", "candidate_text", "context", "label",
               "verdict", "fixed_text", "reviewer", "note"]  # fmt: skip


def frozen(d: Path) -> dict[str, Any] | None:
    f = d / "frozen.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None


def mt_table(root: Path) -> pl.DataFrame | None:
    d = root / "task/mt/qa"
    if not d.is_dir():
        return None
    return (
        pl.scan_parquet(str(d / "*/translations-*.parquet"))
        .filter(qc_ok(30.0))  # eval candidates are human-reviewed afterwards: a looser round-trip floor
        .select("src_lang", "tgt_lang", "text", "translation", "chrf_back")
        .collect()
    )


# ----------------------------------------------------------------------------- sets


def ner_set(root: Path) -> pl.DataFrame:
    return (
        pl.scan_parquet(str(root / "task/ner/*/gold_test-*.parquet"))
        .collect()
        .with_columns(needs_review=pl.lit(False))
    )


def qa_set(root: Path) -> pl.DataFrame:
    out = []
    for s in QA_SOURCES:
        df = load_qa(root, s, ("test",))
        out.append(df.select(
            example_id=pl.concat_str(pl.lit(f"{s}/"), "lang", "example_id", separator="/"), source=pl.lit(s), lang="lang",
            config="config", title="title", question="question", context="context", answers="answers", answerable="answerable",
            needs_review=pl.lit(False),
        ))  # fmt: skip
    return pl.concat(out)


def retrieval_set(root: Path, mt: pl.DataFrame | None) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    asg = (
        pl.scan_parquet(str(root / "task/splits/*/assignments-*.parquet"))
        .filter(pl.col("split") == "test")
        .select("source", "lang", "example_id")
        .collect()
    )
    queries, qrels, docs = [], [], []
    # MIRACL dev (= our test): human queries, judged passages
    mt_ids = asg.filter(pl.col("source") == "miracl").select("lang", qid="example_id")
    topics = (
        scan_raw(root, "miracl", "topics")
        .select("lang", "qid", "query")
        .collect()
        .join(mt_ids, on=["lang", "qid"])
    )
    qr = (
        scan_raw(root, "miracl", "qrels")
        .select("lang", "qid", "docid", relevance=pl.col("relevance").cast(pl.Int32))
        .collect()
    )
    qr = qr.join(mt_ids, on=["lang", "qid"])
    qrels.append(qr.select(query_id=pl.concat_str(pl.lit("miracl"), "lang", "qid", separator="/"),
                           doc_id=pl.concat_str(pl.lit("miracl"), "lang", "docid", separator="/"), relevance="relevance", lang="lang"))  # fmt: skip
    queries.append(topics.select(query_id=pl.concat_str(pl.lit("miracl"), "lang", "qid", separator="/"), query="query",
                                 q_lang="lang", d_lang="lang", origin=pl.lit("gold"), source=pl.lit("miracl")))  # fmt: skip
    # docids are Wikipedia page ids shared by the ru and en corpora: the id carries the language
    keys = {str(lg): g["docid"].to_list() for (lg,), g in qr.group_by("lang")}
    docs.append(
        rows_with_keys(raw_files_by_lang(root, "miracl_corpus"), "docid", keys, ["docid", "title", "text"])
        .select(
            "lang", "title", "text", doc_id=pl.concat_str(pl.lit("miracl"), "lang", "docid", separator="/")
        )
        .unique("doc_id")
    )
    # KazQAD test (kazqad config + nq part): question -> its context; nq kk <-> en parallel by id
    kq = load_qa(root, "kazqad", ("test",))
    queries.append(kq.select(query_id=pl.concat_str(pl.lit("kazqad"), "lang", "example_id", separator="/"), query="question",
                             q_lang="lang", d_lang="lang",
                             origin=pl.when(pl.col("config") == "nq-translate-kk").then(pl.lit("mt_published")).otherwise(pl.lit("gold")),
                             source=pl.lit("kazqad")))  # fmt: skip
    qrels.append(kq.select(query_id=pl.concat_str(pl.lit("kazqad"), "lang", "example_id", separator="/"), doc_id="doc_id",
                           relevance=pl.lit(1, dtype=pl.Int32), lang="lang"))  # fmt: skip
    docs.append(kq.unique("doc_id").select("lang", "title", text="context", doc_id="doc_id"))
    nq = kq.filter(pl.col("config").str.starts_with("nq-"))
    kk = nq.filter(pl.col("lang") == "kk").select("example_id", q_kk="question", d_kk="doc_id")
    en = nq.filter(pl.col("lang") == "en").select("example_id", q_en="question", d_en="doc_id")
    j = kk.join(en, on="example_id")
    for ql, dl, q, d in (("kk", "en", "q_kk", "d_en"), ("en", "kk", "q_en", "d_kk")):
        qid = pl.concat_str(pl.lit(f"kazqad/{ql}-{dl}"), "example_id", separator="/")
        queries.append(
            j.select(
                query_id=qid,
                query=q,
                q_lang=pl.lit(ql),
                d_lang=pl.lit(dl),
                origin=pl.lit("mt_published"),
                source=pl.lit("kazqad"),
            )
        )
        qrels.append(j.select(query_id=qid, doc_id=d, relevance=pl.lit(1, dtype=pl.Int32), lang=pl.lit(dl)))
    # our NLLB translations of test queries: the remaining directions
    if mt is not None:
        base = (
            pl.concat(queries)
            .filter(pl.col("origin") == "gold")
            .select("query_id", "query", "q_lang", "source")
        )
        t = base.join(mt, left_on=["q_lang", "query"], right_on=["src_lang", "text"])
        mq = t.select(query_id=pl.concat_str("query_id", pl.lit("mt"), "tgt_lang", separator="/"), query="translation",
                      q_lang="tgt_lang", d_lang="q_lang", origin=pl.lit("mt_query"), source="source", orig_id="query_id",
                      source_text="query")  # fmt: skip
        allq = pl.concat(qrels)
        qrels.append(
            mq.join(allq.rename({"query_id": "orig_id"}), on="orig_id").select(
                "query_id", "doc_id", "relevance", "lang"
            )
        )
        queries.append(mq.drop("orig_id"))
    q = (
        pl.concat(queries, how="diagonal_relaxed")
        .with_columns(
            direction=pl.concat_str("q_lang", "d_lang", separator="-"),
            needs_review=pl.col("origin") == "mt_query",
        )
        .unique("query_id")
    )
    return (
        q,
        pl.concat(qrels).unique(["query_id", "doc_id"]),
        pl.concat(docs, how="diagonal_relaxed").unique("doc_id"),
    )


def nli_set(root: Path) -> pl.DataFrame:
    return pl.scan_parquet(str(root / "task/nli/*/test-*.parquet")).collect()


# ----------------------------------------------------------------------------- leakage


def train_keys(root: Path) -> dict[str, pl.Series]:
    """64-bit hashes of the normalized texts of all train/validation task data, by kind (texts of
    non-empty keys only). Tens of millions of queries, passages and sentences do not fit in memory as
    Python strings; a hash collision can only add a spurious overlap, never hide one."""
    parts: dict[str, list[pl.Series]] = {}

    def add(kind: str, lf: pl.LazyFrame, expr: pl.Expr) -> None:
        h = (
            lf.select(k=key_expr(expr))
            .filter(pl.col("k") != "")
            .select(pl.col("k").hash(seed=0))
            .unique()
            .collect(engine="streaming")["k"]
        )
        parts.setdefault(kind, []).append(h)

    t = root / "task"
    if (t / "ner").is_dir():
        for tab in ("gold_train", "gold_validation", "silver_train"):
            files = [str(p) for p in (t / "ner").glob(f"*/{tab}-*.parquet")]
            if files:
                add("sentence", pl.scan_parquet(files), pl.col("tokens").list.join(" "))
    if (t / "retrieval").is_dir():
        add("query", pl.scan_parquet(str(t / "retrieval/*/pairs-*.parquet")), pl.col("query"))
        docs = pl.scan_parquet(str(t / "retrieval/multi/passages-*.parquet"))
        shared = (pl.col("source") == "miracl_corpus") & pl.col("doc_id").str.starts_with("miracl/")
        add("passage", docs, pl.col("text"))
        add("miracl_passage", docs.filter(~shared.fill_null(False)), pl.col("text"))
        add("miracl_shared", docs.filter(shared), pl.col("text"))
    if (t / "sft").is_dir():
        sft = pl.scan_parquet(str(t / "sft/*/*.parquet"))
        add("query", sft, pl.col("question"))
        evidence = sft.select("evidence").explode("evidence").unnest("evidence")
        for kind in ("passage", "miracl_passage"):
            add(kind, evidence, pl.col("text"))
    if (t / "nli").is_dir():
        for tab in ("train", "validation"):
            add(
                "nli",
                pl.scan_parquet(str(t / f"nli/*/{tab}-*.parquet")),
                pl.concat_str("premise", "hypothesis", separator=" || "),
            )
    return {k: pl.concat(v).unique() for k, v in parts.items()}


def leakage(eval_texts: dict[str, pl.Series], keys: dict[str, pl.Series]) -> dict[str, Any]:
    out = {}
    for name, s in eval_texts.items():
        kind = name.split(":")[1]
        k = pl.DataFrame({"t": s}).select(k=key_expr("t"))["k"]
        train = keys.get(kind, pl.Series([], dtype=pl.UInt64))
        h = k.hash(seed=0)
        hit = h.is_in(train.implode()) & (k != "")
        out[name] = {"items": len(k), "overlap": int(hit.sum()), "examples": k.filter(hit).head(5).to_list()}
        if kind == "miracl_passage":
            shared = keys.get("miracl_shared", pl.Series([], dtype=pl.UInt64))
            out[name]["allowed_shared_miracl"] = int((h.is_in(shared.implode()) & (k != "") & ~hit).sum())
    return out


def retrieval_passage_texts(docs: pl.DataFrame) -> dict[str, pl.Series]:
    shared = docs["doc_id"].str.starts_with("miracl/")
    return {
        "retrieval:miracl_passage": docs.filter(shared)["text"],
        "retrieval:passage": docs.filter(~shared)["text"],
    }


# ----------------------------------------------------------------------------- review sheets


def review_items(q: pl.DataFrame, nli: pl.DataFrame, per_kind: int, seed: int) -> pl.DataFrame:
    """MT-derived eval rows to verify (all of them when few) + a stratified sample of gold rows as a control."""
    mtq = q.filter(pl.col("needs_review")).with_columns(r=pl.col("query_id").hash(seed=seed)).sort("r")
    rows = [mtq.group_by("direction").head(per_kind).select(
        item_id=pl.concat_str(pl.lit("retrieval/"), "query_id"), set=pl.lit("retrieval"), kind=pl.lit("query_translation"),
        direction="direction", source_text=pl.col("source_text").fill_null(""), candidate_text="query", context=pl.lit(""), label=pl.lit(""),
    )]  # fmt: skip
    kk = (
        nli.filter(pl.col("needs_review"))
        .with_columns(r=pl.col("pair_id").hash(seed=seed))
        .sort("r")
        .head(per_kind)
    )
    rows.append(kk.select(
        item_id=pl.concat_str(pl.lit("nli/"), "pair_id"), set=pl.lit("nli"), kind=pl.lit("nli_translation"),
        direction=pl.lit("en-kk"), source_text=pl.col("premise"), candidate_text="hypothesis", context=pl.lit(""),
        label=pl.col("label").replace_strict({0: "entailment", 1: "neutral", 2: "contradiction"}, return_dtype=pl.String),
    ))  # fmt: skip
    df = pl.concat(rows)
    return df.with_columns(verdict=pl.lit(""), fixed_text=pl.lit(""), reviewer=pl.lit(""), note=pl.lit(""))


def write_sheet(path: Path, df: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():  # never overwrite reviewer work
        log.info("review sheet %s exists, kept", path)
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLS)
        w.writeheader()
        for r in df.select(REVIEW_COLS).iter_rows(named=True):
            w.writerow(r)


def read_sheets(review: Path) -> pl.DataFrame:
    rows = []
    for p in sorted(review.glob("*.csv")):
        with open(p, encoding="utf-8-sig", newline="") as f:
            rows += [dict(r, sheet=p.name) for r in csv.DictReader(f)]
    if not rows:
        return pl.DataFrame(schema={c: pl.String for c in [*REVIEW_COLS, "sheet"]})
    return pl.DataFrame(rows)


# ----------------------------------------------------------------------------- driver


def write_set(root: Path, name: str, tables: dict[str, pl.DataFrame], params: dict[str, Any], inputs: list[Any],
              licenses: list[Any], metrics: dict[str, Any], fp: str, lang_col: str | None, stats: Any) -> DatasetManifest:  # fmt: skip
    out = root / "eval" / name
    if out.exists():
        shutil.rmtree(out)
    w = DatasetWriter(out, LakeConfig.from_params(params["lake"], root))
    for t, df in tables.items():
        w.write(t, df, lang_col=lang_col if lang_col and lang_col in df.columns else None)
    shards = w.close()
    review = sum(int(df["needs_review"].sum()) for df in tables.values() if "needs_review" in df.columns)
    rows = {
        k: v
        for df in tables.values()
        if lang_col and lang_col in df.columns
        for k, v in by_lang(df, lang_col).items()
    }
    m = finish(
        out, name=f"eval/{name}", layer="eval", stage="build_eval", source=name,
        description=f"Frozen evaluation set `{name}` (ADR-0007); rows with `needs_review` are machine translated and "
        "count only after human review.", params={"seed": params["seed"]}, inputs=inputs, licenses=licenses, shards=shards,
        rows_in=rows or {"multi": sum(df.height for df in tables.values())}, metrics={**metrics, "needs_review_rows": review},
        stats=stats, fingerprint=fp,
    )  # fmt: skip
    status = "candidate" if review else "frozen"
    (out / "frozen.json").write_text(json.dumps({"status": status, "fingerprint": fp, "at": datetime.now(UTC).isoformat(timespec="seconds"),
                                                 "shards": {s.path: s.sha256 for s in shards}, "needs_review_rows": review,
                                                 "human_verified": 0}, indent=1), encoding="utf-8")  # fmt: skip
    return m


def build(params: dict[str, Any], root: Path) -> int:
    t0 = time.perf_counter()
    mt = mt_table(root)
    with ResourceMonitor() as mon:
        q, qrels, docs = retrieval_set(root, mt)
        sets = {
            "ner": ({"test": ner_set(root)}, "lang", ["kaznerd", "wikiann", "wikineural", "nerel"], [root / "task/ner"]),
            "qa": ({"test": qa_set(root)}, "lang", list(QA_SOURCES), [root / "task/splits"]),
            "retrieval": ({"queries": q, "qrels": qrels, "docs": docs}, "q_lang", ["miracl", "miracl_corpus", "kazqad"], [root / "task/splits"]),
            "nli": ({"test": nli_set(root)}, "lang", ["xnli"], [root / "task/nli"]),
        }  # fmt: skip
        keys = train_keys(root)
        report: dict[str, Any] = {}
        for name, (tables, lang_col, sources, deps) in sets.items():
            d = root / "eval" / name
            fz = frozen(d)
            if fz and fz["status"] == "frozen":
                problems = verify_dataset(d, deep=True)
                report[name] = {"status": "frozen (kept)", "problems": problems}
                continue
            texts: dict[str, pl.Series] = {}
            if name == "ner":
                texts["ner:sentence"] = tables["test"]["tokens"].list.join(" ")
            elif name == "qa":
                texts["qa:query"] = tables["test"]["question"]
                texts["qa:passage"] = tables["test"]["context"].unique()
            elif name == "retrieval":
                texts["retrieval:query"] = tables["queries"]["query"]
                texts.update(retrieval_passage_texts(tables["docs"]))
            else:
                texts["nli:nli"] = tables["test"].select(
                    pl.concat_str("premise", "hypothesis", separator=" || ")
                )[:, 0]
            leak = leakage(texts, keys)
            inputs = [raw_input(root, s) for s in sources] + [
                dataset_input_of(root, x) for x in deps if (x / "dataset.yaml").is_file()
            ]
            uses_mt = name in ("retrieval",) and mt is not None or name == "nli"
            lic = inherited(root, sources) + ([MT_LICENSE] if uses_mt else [])
            fp = fingerprint(f"build_eval/{name}", {"seed": params["seed"]}, [Path(__file__)],
                             [i for i in [root / f"raw/{s}/source.yaml" for s in sources]])  # fmt: skip
            metrics: dict[str, Any] = {"leakage": leak}
            for t, df in tables.items():
                metrics[f"rows_{t}"] = df.height
            if name == "retrieval":
                metrics["queries_by_direction"] = (
                    q.group_by("direction", "origin").len().sort("direction", "origin").to_dicts()
                )
                metrics["docs_by_lang"] = by_lang(docs)
            write_set(root, name, tables, params, inputs, lic, metrics, fp, lang_col, mon.stats(1, 0, 1))
            report[name] = {
                "status": "built",
                "rows": {t: df.height for t, df in tables.items()},
                "leakage": leak,
            }
        sheet = review_items(
            q, sets["nli"][0]["test"], int(params["evaluation"]["review_per_kind"]), params["seed"]
        )
        gold_needed = max(0, int(params["evaluation"]["human_verified_min"]) - sheet.height)
        if gold_needed:  # top up with gold control items so that the total meets the human-verified minimum
            gold = (
                q.filter(~pl.col("needs_review"))
                .with_columns(r=pl.col("query_id").hash(seed=params["seed"]))
                .sort("r")
            )
            gold = gold.group_by("direction").head(gold_needed // 3 + 1).head(gold_needed)
            sheet = pl.concat([sheet, gold.select(
                item_id=pl.concat_str(pl.lit("retrieval/"), "query_id"), set=pl.lit("retrieval"), kind=pl.lit("gold_query_relevance"),
                direction="direction", source_text=pl.lit(""), candidate_text="query", context=pl.lit(""), label=pl.lit("relevant"),
                verdict=pl.lit(""), fixed_text=pl.lit(""), reviewer=pl.lit(""), note=pl.lit(""),
            )])  # fmt: skip
        write_sheet(root / "eval" / "review" / "sheet-01.csv", sheet)
    report["review_items"] = sheet.height
    report["seconds"] = round(time.perf_counter() - t0, 1)
    write_report(ROOT / "reports" / "eval.json", report)
    log_mlflow("eval", "build_eval", {"seed": params["seed"]}, {"review_items": sheet.height, "seconds": report["seconds"],
               **{f"rows_{k}": sum(v.get("rows", {}).values()) for k, v in report.items() if isinstance(v, dict) and "rows" in v}})  # fmt: skip
    print(json.dumps({k: (v.get("status") if isinstance(v, dict) else v) for k, v in report.items()}))
    return 0


MODEL_REVIEWER = r"(?i)llm|claude|gpt|qwen|machine|auto"


def human_done(sheets: pl.DataFrame) -> pl.DataFrame:
    """Rows with a verdict and a person's name (a reviewer naming a model is not a human review, ADR-0022)."""
    who = pl.col("reviewer").fill_null("").str.strip_chars()
    return sheets.filter(
        pl.col("verdict").is_in(["ok", "fix", "drop"]) & (who != "") & ~who.str.contains(MODEL_REVIEWER)
    )


def query_cluster(qid: pl.Expr) -> pl.Expr:
    """ADR-0022 cluster of a retrieval query: the source query with all its translations (our NLLB `/mt/<lang>`
    suffix; the KazQAD nq part shares one example id across `kk`, `en`, `kk-en`, `en-kk`)."""
    return qid.str.replace(r"/mt/[a-z]+$", "").str.replace(r"^kazqad/[a-z-]+/nq/", "kazqad/nq/")


def nli_cluster(df: pl.DataFrame, en: pl.DataFrame) -> pl.Series:
    """ADR-0022 cluster of an XNLI pair: its English premise (three hypotheses share it, translations too)."""
    ex = pl.col("pair_id").str.replace(r"^xnli/(kk|ru|en)/", "")
    prem = en.select(ex=ex, p=pl.col("premise"))
    keys = {p: f"xnli/{stable_hash(p):016x}" for p in prem["p"].unique().to_list()}
    m = dict(zip(prem["ex"].to_list(), [keys[p] for p in prem["p"].to_list()], strict=True))
    return df.select(ex.replace_strict(m, default=None, return_dtype=pl.String)).to_series()


def eval_texts(root: Path) -> dict[str, pl.Series]:
    """Normalizable texts of every built eval set (frozen or not), by `<set>:<kind>` as in `leakage`."""

    def read(name: str, table: str, cols: list[str]) -> pl.DataFrame:
        files = sorted((root / "eval" / name).glob(f"*/{table}-*.parquet"))
        return pl.read_parquet([str(f) for f in files], columns=cols) if files else pl.DataFrame()

    out: dict[str, pl.Series] = {}
    if (ner := read("ner", "test", ["tokens"])).height:
        out["ner:sentence"] = ner["tokens"].list.join(" ")
    if (qa := read("qa", "test", ["question", "context"])).height:
        out["qa:query"], out["qa:passage"] = qa["question"], qa["context"].unique()
    if (q := read("retrieval", "queries", ["query"])).height:
        out["retrieval:query"] = q["query"]
        out.update(retrieval_passage_texts(read("retrieval", "docs", ["doc_id", "text"])))
    if (n := read("nli", "test", ["premise", "hypothesis"])).height:
        out["nli:nli"] = n.select(pl.concat_str("premise", "hypothesis", separator=" || "))[:, 0]
    return out


def leakage_check(params: dict[str, Any], root: Path) -> int:
    """Overlap of every eval set (frozen ones included) with the current train/validation task data
    (`reports/eval_leakage.json`, DVC stage `eval_leakage`); queries and NLI pairs must not overlap. Retrieval
    passages may: MIRACL dev and train share one corpus, and judged passages of eval topics serve as negatives
    of train topics (PROGRESS, stage 3)."""
    t0 = time.perf_counter()
    with ResourceMonitor() as mon:
        res = leakage(eval_texts(root), train_keys(root))
    res = {
        "sets": res,
        "seconds": round(time.perf_counter() - t0, 1),
        "peak_rss_mb": round(mon.peak_rss / 2**20),
    }
    write_report(ROOT / "reports" / "eval_leakage.json", res)
    log_mlflow(
        "eval",
        "eval_leakage",
        {"seed": params["seed"]},
        {k.replace(":", "_"): v["overlap"] for k, v in res["sets"].items()},
    )
    print(json.dumps({k: v["overlap"] for k, v in res["sets"].items()}))
    return int(any(v["overlap"] for v in res["sets"].values()))


def status(params: dict[str, Any], root: Path) -> int:
    sheets = read_sheets(root / "eval" / "review")
    done = human_done(sheets)
    need = int(params["evaluation"]["human_verified_min"])
    out: dict[str, Any] = {"items": sheets.height, "human_verified": done.height, "required": need,
                           "by_verdict": dict(done.group_by("verdict").len().iter_rows()) if done.height else {}, "sets": {}}  # fmt: skip
    for d in sorted((root / "eval").iterdir()):
        fz = frozen(d)
        if fz:
            out["sets"][d.name] = {
                "status": fz["status"],
                "intact": not verify_dataset(d, deep=True),
                "needs_review_rows": fz["needs_review_rows"],
            }
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


def rewrite(d: Path, m: DatasetManifest, table: str, fn: Any) -> int:
    """Apply `fn` to every shard of `table` in place (shards keep their language partition); rows written."""
    rows, shards = 0, []
    for s in m.shards:
        if s.table != table:
            shards.append(s)
            continue
        f = d / s.path
        df = fn(pl.read_parquet(f))
        df.write_parquet(f)
        rows += df.height
        shards.append(
            s.model_copy(update={"rows": df.height, "bytes": f.stat().st_size, "sha256": sha256_file(f)})
        )
    m.shards = shards
    return rows


def freeze(
    params: dict[str, Any],
    root: Path,
    *,
    calibrated: bool = False,
    machine_validated: bool = False,
    report_approved_at: str | None = None,
) -> int:
    """Apply verdicts and freeze `retrieval` and `nli`: drop `drop` items and MT rows outside the validated
    packet, replace fixed texts (`fixed_text` -> query / hypothesis, `fixed_source` -> NLI premise), add
    `cluster_id` (ADR-0022) and `verification_tier`, drop qrels of removed queries, record shard hashes in
    `frozen.json`.

    Modes:
    * default: requires full human review of every item in the review sheet.
    * `--calibrated`: requires human confirmation of the stratified calibration sample + machine assessments.
    * `--machine-validated` (ADR-0026): applies 100% dual-signal cross-validated machine assessments
      (`machine_labels.csv`) under report-level human governance; keeps `human_verified: 0` for MT rows and
      tags MT rows `verification_tier = "machine_cross_validated"` (gold rows stay `"human_verified"`)."""
    sheets = read_sheets(root / "eval" / "review")
    done = human_done(sheets)
    if done["item_id"].is_duplicated().any():
        print("Duplicate human verdicts for eval items; nothing frozen")
        return 1
    if "fixed_text" not in done.columns:
        done = done.with_columns(fixed_text=pl.lit(""))
    if "fixed_source" not in done.columns:
        done = done.with_columns(fixed_source=pl.lit(""))

    calibration_meta: dict[str, Any] = {}
    machine_reviewed: set[str] = set()
    human_reviewed = set(done["item_id"].to_list())
    drop: set[str] = set(done.filter(pl.col("verdict") == "drop")["item_id"].to_list())
    fx = done.filter(pl.col("verdict") == "fix")
    fixes: dict[str, str] = {
        i: t for i, t in fx.select("item_id", "fixed_text").iter_rows() if (t or "").strip()
    }
    fixes_src: dict[str, str] = {
        i: t for i, t in fx.select("item_id", "fixed_source").iter_rows() if (t or "").strip()
    }

    if machine_validated:
        mfile = root / "review" / "eval_mt" / "machine_labels.csv"
        if not mfile.is_file():
            print(f"Missing {mfile}; run `prelabel import` first")
            return 1
        mach = pl.read_csv(mfile, infer_schema_length=10000)
        if mach["judge_verdict"].is_null().any():
            print("Incomplete machine_labels.csv; nothing frozen")
            return 1
        for r in mach.iter_rows(named=True):
            iid = str(r["item_id"])
            if iid in human_reviewed:
                continue
            machine_reviewed.add(iid)
            v = str(r["judge_verdict"])
            if v == "drop":
                drop.add(iid)
            elif v == "fix":
                ft = str(r.get("judge_fixed_text") or "").strip()
                fs = str(r.get("judge_fixed_source") or "").strip()
                if ft:
                    fixes[iid] = ft
                if fs:
                    fixes_src[iid] = fs
        reviewed = human_reviewed | machine_reviewed
    elif calibrated:
        from trifact.evaluation.review_quality import cohen_kappa

        cal_path = root / "review" / "calibration_sample.json"
        if not cal_path.is_file():
            print(f"Missing {cal_path}; run `prelabel calibrate` first")
            return 1
        cal_ids = set(json.loads(cal_path.read_text(encoding="utf-8"))["ids"]["eval"])
        missing_cal = cal_ids - human_reviewed
        if missing_cal:
            print(
                f"ADR-0026 calibration human-verified items: {len(cal_ids) - len(missing_cal)} / {len(cal_ids)} required; nothing frozen"
            )
            return 1
        mfile = root / "review" / "eval_mt" / "machine_labels.csv"
        if not mfile.is_file():
            print(f"Missing {mfile}; run `prelabel import` first")
            return 1
        mach = pl.read_csv(mfile, infer_schema_length=10000)
        if mach["judge_verdict"].is_null().any():
            print("Incomplete machine_labels.csv; nothing frozen")
            return 1
        jcal = done.select("item_id", "verdict").join(
            mach.select("item_id", "kind", "direction", "judge_verdict"), on="item_id"
        )
        hl, jl = jcal["verdict"].to_list(), jcal["judge_verdict"].to_list()
        agree = (
            round(sum(1 for x, y in zip(hl, jl, strict=True) if x == y) / jcal.height, 4)
            if jcal.height
            else 0.0
        )
        calibration_meta = {
            "protocol": "ADR-0026",
            "human_audit_n": jcal.height,
            "agreement": agree,
            "cohen_kappa": cohen_kappa(hl, jl),
            "by_slice": {
                f"{k}/{dr}": {
                    "n": g.height,
                    "agreement": round(float((g["verdict"] == g["judge_verdict"]).mean() or 0), 4),
                    "cohen_kappa": cohen_kappa(g["verdict"].to_list(), g["judge_verdict"].to_list()),
                }
                for (k, dr), g in sorted(
                    jcal.group_by("kind", "direction"), key=lambda x: (str(x[0][0]), str(x[0][1]))
                )
            },
        }
        # Apply calibrated machine verdicts ONLY to non-human-audited items (human always wins)
        for r in mach.iter_rows(named=True):
            iid = str(r["item_id"])
            if iid in human_reviewed:
                continue
            machine_reviewed.add(iid)
            v = str(r["judge_verdict"])
            if v == "drop":
                drop.add(iid)
            elif v == "fix":
                ft = str(r.get("judge_fixed_text") or "").strip()
                fs = str(r.get("judge_fixed_source") or "").strip()
                if ft:
                    fixes[iid] = ft
                if fs:
                    fixes_src[iid] = fs
        reviewed = human_reviewed | machine_reviewed
    else:
        need = max(int(params["evaluation"]["human_verified_min"]), sheets["item_id"].n_unique())
        if done.height < need:
            print(f"human-verified items: {done.height} / {need} required; nothing frozen")
            return 1
        reviewed = human_reviewed

    def keep(df: pl.DataFrame, id_col: str, prefix: str) -> pl.DataFrame:
        item = pl.lit(prefix) + pl.col(id_col)
        return df.filter(~item.is_in(list(drop)) & (~pl.col("needs_review") | item.is_in(list(reviewed))))

    mt_tier = "machine_cross_validated" if machine_validated else "machine_calibrated"

    def tag_tier(df: pl.DataFrame, id_col: str, prefix: str) -> pl.DataFrame:
        item = pl.lit(prefix) + pl.col(id_col)
        return df.with_columns(
            verification_tier=pl.when(~pl.col("needs_review") | item.is_in(list(human_reviewed)))
            .then(pl.lit("human_verified"))
            .otherwise(pl.lit(mt_tier))
        )

    def fixed(df: pl.DataFrame, id_col: str, prefix: str, col: str, table: dict[str, str]) -> pl.DataFrame:
        item = (pl.lit(prefix) + pl.col(id_col)).alias("_item")
        return df.with_columns(
            pl.struct(item, col)
            .map_elements(lambda r: table.get(r["_item"], r[col]), return_dtype=pl.String)
            .alias(col)
        )

    for name in ("retrieval", "nli"):
        d = root / "eval" / name
        fz = frozen(d)
        if not fz or fz["status"] == "frozen":
            continue
        m = DatasetManifest.read(d)
        if name == "retrieval":
            rows = rewrite(
                d,
                m,
                "queries",
                lambda df: tag_tier(
                    fixed(keep(df, "query_id", "retrieval/"), "query_id", "retrieval/", "query", fixes),
                    "query_id",
                    "retrieval/",
                ).with_columns(cluster_id=query_cluster(pl.col("query_id")), needs_review=pl.lit(False)),
            )
            qids = pl.concat(
                [pl.read_parquet(d / s.path, columns=["query_id"]) for s in m.shards if s.table == "queries"]
            )
            rewrite(
                d, m, "qrels", lambda df, q=qids: df.filter(pl.col("query_id").is_in(q["query_id"].implode()))
            )
        else:
            en = pl.concat(
                [pl.read_parquet(d / s.path) for s in m.shards if s.table == "test" and s.lang == "en"]
            )

            def nli_fn(df: pl.DataFrame, en: pl.DataFrame = en) -> pl.DataFrame:
                df = tag_tier(keep(df, "pair_id", "nli/"), "pair_id", "nli/")
                df = fixed(
                    fixed(df, "pair_id", "nli/", "hypothesis", fixes), "pair_id", "nli/", "premise", fixes_src
                )
                return df.with_columns(cluster_id=nli_cluster(df, en), needs_review=pl.lit(False))

            rows = rewrite(d, m, "test", nli_fn)
        tier_name = (
            "machine_cross_validated"
            if machine_validated
            else ("human_calibrated_machine" if calibrated else "human_verified")
        )
        proto = "ADR-0026" if (machine_validated or calibrated) else "full_human"
        m.metrics["needs_review_rows"] = 0
        m.metrics["verification_tier"] = tier_name
        m.metrics["human_review"] = {
            "human_verified": len(human_reviewed),
            "machine_validated": len(machine_reviewed),
            "dropped": len(drop),
            "fixed": len(fixes) + len(fixes_src),
            "assist": dict(done.group_by("assist").len().iter_rows()) if "assist" in done.columns else {},
            **(
                {"report_governance_approved": True, "report_approved_at": report_approved_at}
                if report_approved_at
                else {}
            ),
            **({"calibration": calibration_meta} if calibration_meta else {}),
        }
        m.write(d)
        (d / "frozen.json").write_text(
            json.dumps(
                {
                    **fz,
                    "status": "frozen",
                    "protocol": proto,
                    "verification_tier": tier_name,
                    "at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "human_verified": len(human_reviewed),
                    "machine_validated": len(machine_reviewed),
                    "needs_review_rows": 0,
                    **(
                        {"report_governance_approved": True, "report_approved_at": report_approved_at}
                        if report_approved_at
                        else {}
                    ),
                    **({"calibration": calibration_meta} if calibration_meta else {}),
                    "shards": {s.path: s.sha256 for s in m.shards},
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        log.info("%s frozen (%s): %d rows", name, tier_name, rows)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "status", "freeze", "leakage"], nargs="?", default="build")
    ap.add_argument("--calibrated", action="store_true", help="ADR-0026 calibrated human-audit freeze mode")
    ap.add_argument(
        "--machine-validated",
        action="store_true",
        help="ADR-0026 autonomous machine cross-validation freeze mode (requires report-level governance approval)",
    )
    ap.add_argument(
        "--report-approved-at",
        default=None,
        help="ISO timestamp of report-level human governance approval (never represents item-level human labeling)",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    params = load_params()
    root = (ROOT / params["lake"]["root"]).resolve()
    if args.command == "freeze":
        return freeze(
            params,
            root,
            calibrated=args.calibrated,
            machine_validated=args.machine_validated,
            report_approved_at=args.report_approved_at,
        )
    return {"build": build, "status": status, "leakage": leakage_check}[args.command](params, root)


if __name__ == "__main__":
    raise SystemExit(main())
