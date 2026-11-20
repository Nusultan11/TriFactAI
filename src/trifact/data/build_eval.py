"""Stage build_eval: frozen, leakage-checked evaluation sets over the 9 language directions (ADR-0007, ADR-0015).

python -m trifact.data.build_eval build      # (re)build sets that are not frozen yet, write review sheets
python -m trifact.data.build_eval status     # human review progress per sheet
python -m trifact.data.build_eval freeze     # apply finished reviews, freeze every set that may be frozen

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
candidate until the review sheets in `data/eval/review/` are filled by human reviewers (verdict ok / fix / drop,
fixed text) and at least `evaluation.human_verified_min` items are verified in total (ROADMAP 3.5). `freeze`
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


def train_keys(root: Path) -> dict[str, set[str]]:
    """Normalized texts of all train/validation task data, by kind."""
    keys: dict[str, set[str]] = {}

    def add(kind: str, lf: pl.LazyFrame, expr: pl.Expr) -> None:
        keys.setdefault(kind, set()).update(
            lf.select(k=key_expr(expr)).collect(engine="streaming")["k"].to_list()
        )

    t = root / "task"
    if (t / "ner").is_dir():
        for tab in ("gold_train", "gold_validation", "silver_train"):
            files = [str(p) for p in (t / "ner").glob(f"*/{tab}-*.parquet")]
            if files:
                add("sentence", pl.scan_parquet(files), pl.col("tokens").list.join(" "))
    if (t / "retrieval").is_dir():
        add("query", pl.scan_parquet(str(t / "retrieval/*/pairs-*.parquet")), pl.col("query"))
        add("passage", pl.scan_parquet(str(t / "retrieval/multi/passages-*.parquet")), pl.col("text"))
    if (t / "sft").is_dir():
        add("query", pl.scan_parquet(str(t / "sft/*/*.parquet")), pl.col("question"))
    if (t / "nli").is_dir():
        for tab in ("train", "validation"):
            add(
                "nli",
                pl.scan_parquet(str(t / f"nli/*/{tab}-*.parquet")),
                pl.concat_str("premise", "hypothesis", separator=" || "),
            )
    keys.pop("", None)
    for v in keys.values():
        v.discard("")
    return keys


def leakage(eval_texts: dict[str, pl.Series], keys: dict[str, set[str]]) -> dict[str, Any]:
    out = {}
    for name, s in eval_texts.items():
        kind = name.split(":")[1]
        k = pl.DataFrame({"t": s}).select(k=key_expr("t"))["k"]
        hit = k.is_in(list(keys.get(kind, set()))) & (k.str.len_chars() >= 20)
        out[name] = {"items": len(k), "overlap": int(hit.sum()), "examples": k.filter(hit).head(5).to_list()}
    return out


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
                texts["retrieval:passage"] = tables["docs"]["text"]
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


def status(params: dict[str, Any], root: Path) -> int:
    sheets = read_sheets(root / "eval" / "review")
    done = sheets.filter(
        pl.col("verdict").is_in(["ok", "fix", "drop"]) & (pl.col("reviewer").str.strip_chars() != "")
    )
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


def freeze(params: dict[str, Any], root: Path) -> int:
    sheets = read_sheets(root / "eval" / "review")
    done = sheets.filter(
        pl.col("verdict").is_in(["ok", "fix", "drop"]) & (pl.col("reviewer").str.strip_chars() != "")
    )
    need = int(params["evaluation"]["human_verified_min"])
    if done.height < need:
        print(f"human-verified items: {done.height} < evaluation.human_verified_min = {need}; nothing frozen")
        return 1
    reviewed = set(done["item_id"].to_list())
    drop = set(done.filter(pl.col("verdict") == "drop")["item_id"].to_list())
    fixes = dict(done.filter(pl.col("verdict") == "fix").select("item_id", "fixed_text").iter_rows())
    for name, id_col, table, text_col, prefix in (
        ("retrieval", "query_id", "queries", "query", "retrieval/"),
        ("nli", "pair_id", "test", "hypothesis", "nli/"),
    ):
        d = root / "eval" / name
        fz = frozen(d)
        if not fz or fz["status"] == "frozen":
            continue
        m = DatasetManifest.read(d)
        files = [d / s.path for s in m.shards if s.table == table]
        df = pl.concat([pl.read_parquet(f) for f in files])
        item = pl.lit(prefix) + pl.col(id_col)
        df = df.with_columns(_item=item).filter(
            ~pl.col("_item").is_in(list(drop))
            & (~pl.col("needs_review") | pl.col("_item").is_in(list(reviewed)))
        )
        df = df.with_columns(
            pl.struct("_item", text_col)
            .map_elements(lambda r, c=text_col: fixes.get(r["_item"], r[c]), return_dtype=pl.String)
            .alias(text_col)
        )
        df = df.drop("_item").with_columns(needs_review=pl.lit(False))
        for f in files:
            f.unlink()
        df.write_parquet(files[0])
        m.shards = [s for s in m.shards if s.table != table] + [
            s.model_copy(update={"rows": df.height, "bytes": files[0].stat().st_size, "sha256": sha256_file(files[0])})
            for s in m.shards if s.table == table and (d / s.path) == files[0]
        ]  # fmt: skip
        m.metrics["needs_review_rows"] = 0
        m.metrics["human_review"] = {"verified": len(reviewed), "dropped": len(drop), "fixed": len(fixes)}
        m.write(d)
        (d / "frozen.json").write_text(json.dumps({**fz, "status": "frozen", "at": datetime.now(UTC).isoformat(timespec="seconds"),
                                                   "human_verified": done.height, "needs_review_rows": 0,
                                                   "shards": {s.path: s.sha256 for s in m.shards}}, indent=1), encoding="utf-8")  # fmt: skip
        log.info("%s frozen after human review", name)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "status", "freeze"], nargs="?", default="build")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    params = load_params()
    root = (ROOT / params["lake"]["root"]).resolve()
    return {"build": build, "status": status, "freeze": freeze}[args.command](params, root)


if __name__ == "__main__":
    raise SystemExit(main())
