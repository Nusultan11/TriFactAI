"""EDA of the task datasets (ROADMAP 3.7): NER, QA/SFT, retrieval, NLI, synthetic, train vs eval.

python -m trifact.data.task_eda      # reports/eda/task.json (+ MLflow); rendered by trifact.data.eda_report

All statistics are computed over every row of `data/task/*` and `data/eval/*`, by language, source / origin and
split: NER label distribution, sentence length, entity density and span length, deviations from the core schema,
silver vs gold type distribution; QA question / context / answer lengths, answer position, unanswerable share,
question types, question-context lexical overlap; retrieval positives per query, lengths, BM25 rank of the
positive among its hard negatives (ease) and negative/positive BM25 score ratio (hardness), directions; NLI class
balance, lengths, a hypothesis-only Naive Bayes baseline and negation cues; synthetic distinct-n and query openings;
train vs eval length shift (Jensen-Shannon) and entity / question / document overlap.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from trifact.core.params import ROOT, load_params
from trifact.data.bm25 import BM25
from trifact.data.curation import write_report
from trifact.data.ner_common import entities
from trifact.data.task_common import key_expr
from trifact.lake.stats import ResourceMonitor, log_mlflow

log = logging.getLogger("trifact.task_eda")
OUT = ROOT / "reports" / "eda" / "task.json"
CORE = {"PER", "ORG", "LOC"}
QWORDS = {
    "kk": ["қандай", "қай", "кім", "не", "қашан", "қанша", "неше", "қайда", "неге", "қалай"],
    "ru": [
        "что",
        "кто",
        "когда",
        "где",
        "сколько",
        "какой",
        "какая",
        "какие",
        "почему",
        "как",
        "чем",
        "в каком",
    ],
    "en": [
        "what",
        "who",
        "when",
        "where",
        "how many",
        "how much",
        "which",
        "why",
        "how",
        "is",
        "did",
        "does",
    ],
}
NEGATION = {"kk": ["емес", "жоқ", "ешқашан", "ешкім", "ештеңе"], "ru": ["не", "нет", "никогда", "никто", "ничего"],
            "en": ["not", "no", "never", "nobody", "nothing", "n't"]}  # fmt: skip
WORD = re.compile(r"\w+", re.UNICODE)
BINS = [0, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 10**9]


def hist(v: np.ndarray, bins: list[int] = BINS) -> list[int]:
    return np.histogram(v, bins=bins)[0].tolist()


def js(p: list[int], q: list[int]) -> float:
    a, b = np.asarray(p, float), np.asarray(q, float)
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    a, b = a / a.sum(), b / b.sum()
    m = (a + b) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 0.5 * np.nansum(np.where(a > 0, a * np.log2(a / m), 0)) + 0.5 * np.nansum(
            np.where(b > 0, b * np.log2(b / m), 0)
        )
    return round(float(d), 4)


def scan(path: Path, pattern: str) -> pl.DataFrame | None:
    files = sorted(str(p) for p in path.glob(pattern))
    return pl.read_parquet(files) if files else None


def distinct_n(texts: list[str], n: int) -> float:
    grams: Counter[tuple[str, ...]] = Counter()
    total = 0
    for t in texts:
        w = WORD.findall(t.lower())
        g = list(zip(*[w[i:] for i in range(n)], strict=False))
        grams.update(g)
        total += len(g)
    return round(len(grams) / max(total, 1), 4)


# ----------------------------------------------------------------------------- NER


def ner(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    import pyarrow.parquet as pq

    base = root / "task/ner"
    counts: dict[tuple[str, str, str], Counter[int]] = {}
    stats: dict[tuple[str, str, str], Counter[str]] = {}
    labels: Counter[tuple[str, str, str]] = Counter()
    test_surfaces: set[tuple[str, str]] = set()
    seen: dict[str, set[tuple[str, str]]] = {"gold_train": set(), "silver_train": set()}
    for tab in ("gold_test", "gold_train", "gold_validation", "silver_train"):
        for path in sorted(base.glob(f"*/{tab}-*.parquet")):
            with pq.ParquetFile(path) as parquet:
                for batch in parquet.iter_batches(
                    batch_size=32_768, columns=["source", "lang", "tokens", "tags"]
                ):
                    for source, lang, tokens, tags in pl.DataFrame(batch).iter_rows():
                        key = (tab, source, lang)
                        n = len(tokens)
                        counts.setdefault(key, Counter())[n] += 1
                        acc = stats.setdefault(key, Counter())
                        spans = entities(tags)
                        acc["tokens"] += n
                        acc["entities"] += len(spans)
                        acc["absent"] += not spans
                        acc["off_schema"] += sum(tag != "O" and tag[2:] not in CORE for tag in tags)
                        for start, end, label in spans:
                            acc["tagged_tokens"] += end - start
                            labels[tab, lang, label] += 1
                            if tab == "gold_test":
                                test_surfaces.add((lang, " ".join(tokens[start:end]).lower()))
                            elif tab in seen:
                                surface = (lang, " ".join(tokens[start:end]).lower())
                                if surface in test_surfaces:
                                    seen[tab].add(surface)
    if not stats:
        return {}
    rows = []
    for (tab, source, lang), acc in sorted(stats.items()):
        lengths = sorted(counts[tab, source, lang].items())
        sentences = sum(n for _, n in lengths)
        p95 = np.searchsorted(
            np.cumsum([n for _, n in lengths]), math.floor(0.95 * (sentences - 1) + 0.5) + 1
        )
        tokens, n_ent, ent_tokens = acc["tokens"], acc["entities"], acc["tagged_tokens"]
        rows.append({"table": tab, "source": source, "lang": lang, "sentences": sentences,
                     "tokens": tokens, "mean_len": round(tokens / sentences, 2), "p95_len": lengths[p95][0],
                     "entities": n_ent, "ent_per_sent": round(n_ent / sentences, 3),
                     "ent_token_share": round(ent_tokens / tokens, 4) if tokens else None,
                     "no_entity_share": round(acc["absent"] / sentences, 4), "off_schema_tags": acc["off_schema"],
                     "mean_span": round(ent_tokens / n_ent, 3) if n_ent else None})  # fmt: skip
    out["by_table_source_lang"] = rows
    t = pl.DataFrame(
        [(tab, lang, label, n) for (tab, lang, label), n in labels.items()],
        schema={"table": pl.String, "lang": pl.String, "types": pl.String, "len": pl.UInt32},
        orient="row",
    )
    out["labels"] = t.sort("table", "lang", "types").to_dicts()
    # silver vs gold type distribution per language
    shift = {}
    for lang in ("kk", "ru", "en"):
        gold = (
            t.filter(pl.col("table").str.starts_with("gold") & (pl.col("lang") == lang))
            .group_by("types")
            .agg(pl.col("len").sum())
        )
        sil = (
            t.filter((pl.col("table") == "silver_train") & (pl.col("lang") == lang))
            .group_by("types")
            .agg(pl.col("len").sum())
        )
        dg = dict(gold.iter_rows())
        ds = dict(sil.iter_rows())
        shift[lang] = {
            "gold": dg,
            "silver": ds,
            "js": js([dg.get(k, 0) for k in sorted(CORE)], [ds.get(k, 0) for k in sorted(CORE)]),
        }
    out["silver_vs_gold_types"] = shift
    totals = Counter(lang for lang, _ in test_surfaces)
    overlap = {}
    for tab, matches in seen.items():
        by_lang = Counter(lang for lang, _ in matches)
        overlap.update({f"{tab}/{lang}": round(by_lang[lang] / n, 4) for lang, n in sorted(totals.items())})
    out["entity_overlap_train_test"] = overlap
    return out


# ----------------------------------------------------------------------------- QA / SFT


def qtype(q: str, lang: str) -> str:
    low = (q or "").lower()
    for w in QWORDS.get(lang, []):
        if re.search(rf"(^|\W){re.escape(w)}(\W|$)", low):
            return w
    return "other"


def overlap(q: str, c: str) -> float:
    qw = set(WORD.findall((q or "").lower()))
    cw = set(WORD.findall((c or "").lower()))
    return len(qw & cw) / max(len(qw), 1)


def qa(root: Path) -> dict[str, Any]:
    df = scan(root / "eval/qa", "*/test-*.parquet")
    out: dict[str, Any] = {}
    if df is not None:
        df = df.with_columns(
            q_len=pl.col("question").str.count_matches(r"\w+"), c_len=pl.col("context").str.count_matches(r"\w+"),
            a_len=pl.col("answers").list.first().str.count_matches(r"\w+"),
            a_pos=(pl.col("context").str.find(pl.col("answers").list.first().fill_null(""), literal=True) / pl.col("context").str.len_chars()),
        )  # fmt: skip
        types = [qtype(q, lg) for q, lg in df.select("question", "lang").iter_rows()]
        ov = [overlap(q, c) for q, c in df.select("question", "context").iter_rows()]
        df = df.with_columns(qtype=pl.Series(types), q_overlap=pl.Series(ov))
        out["eval_qa"] = df.group_by("source", "lang").agg(
            n=pl.len(), unanswerable=(~pl.col("answerable")).mean().round(4), q_len=pl.col("q_len").mean().round(2),
            c_len=pl.col("c_len").mean().round(1), a_len=pl.col("a_len").mean().round(2),
            a_pos_p50=pl.col("a_pos").median().round(3), q_context_overlap=pl.col("q_overlap").mean().round(3),
        ).sort("source", "lang").to_dicts()  # fmt: skip
        out["eval_qa_answer_position_hist"] = {lg: np.histogram(g["a_pos"].drop_nulls().clip(0, 1).to_numpy(), bins=10, range=(0, 1))[0].tolist()
                                               for (lg,), g in df.group_by("lang")}  # fmt: skip
        out["eval_question_types"] = (
            df.group_by("lang", "qtype").len().sort("lang", "len", descending=[False, True]).to_dicts()
        )
    sft = scan(root / "task/sft", "*/*.parquet")
    if sft is not None:
        sft = sft.with_columns(
            q_len=pl.col("question").str.count_matches(r"\w+"), a_len=pl.col("answer").str.count_matches(r"\w+"),
            n_evidence=pl.col("evidence").list.len(), n_cites=pl.col("answer").str.count_matches(r"\[c\d+\]"),
            ev_tokens=pl.col("evidence").list.eval(pl.element().struct.field("text").str.count_matches(r"\w+")).list.sum(),
        )  # fmt: skip
        out["sft"] = sft.group_by("type", "lang", "origin").agg(
            n=pl.len(), q_len=pl.col("q_len").mean().round(2), a_len=pl.col("a_len").mean().round(2),
            evidence=pl.col("n_evidence").mean().round(2), cites=pl.col("n_cites").mean().round(2),
            evidence_words=pl.col("ev_tokens").mean().round(1),
        ).sort("type", "lang", "origin").to_dicts()  # fmt: skip
        out["sft_question_types"] = (
            sft.with_columns(qtype=pl.Series([qtype(q, lg) for q, lg in sft.select("question", "lang").iter_rows()]))
            .group_by("lang", "qtype").len().sort("lang", "len", descending=[False, True]).to_dicts()
        )  # fmt: skip
        out["sft_distinct"] = {lg: {f"distinct_{n}": distinct_n(g["question"].head(50_000).to_list(), n) for n in (1, 2, 3)}
                               for (lg,), g in sft.group_by("lang")}  # fmt: skip
    return out


# ----------------------------------------------------------------------------- retrieval


def retrieval(root: Path, sample: int, seed: int) -> dict[str, Any]:
    files = sorted((root / "task/retrieval").glob("*/pairs-*.parquet"))
    out: dict[str, Any] = {}
    if files:
        pairs = pl.scan_parquet(files).with_columns(
            q_len=pl.col("query").str.count_matches(r"\w+"), n_neg=pl.col("neg_ids").list.len()
        )
        out["train"] = pairs.group_by("source", "origin", "direction", "split").agg(
            n=pl.len(), queries=pl.col("query_id").n_unique(), q_len=pl.col("q_len").mean().round(2),
            with_negatives=(pl.col("n_neg") > 0).mean().round(4), neg_per_query=pl.col("n_neg").mean().round(2),
        ).sort("source", "origin", "direction", "split").collect(engine="streaming").to_dicts()  # fmt: skip
        out["positives_per_query"] = pairs.group_by("source", "query_id").len().group_by("source").agg(
            mean=pl.col("len").mean().round(3), p99=pl.col("len").quantile(0.99), max=pl.col("len").max()
        ).sort("source").collect(engine="streaming").to_dicts()  # fmt: skip
        # BM25 ease and hardness on a sample of monolingual pairs with negatives
        pas = pl.scan_parquet(str(root / "task/retrieval/multi/passages-*.parquet")).select("doc_id", "text")
        mono = pairs.filter(
            (pl.col("n_neg") > 0)
            & (pl.col("direction").str.slice(0, 2) == pl.col("direction").str.slice(3, 2))
        )
        smp = (
            mono.with_columns(r=pl.col("pair_id").hash(seed=seed))
            .group_by("source")
            .agg(pl.struct("query", "doc_id", "neg_ids").top_k_by("r", sample, reverse=True).alias("pairs"))
            .explode("pairs")
            .unnest("pairs")
            .collect(engine="streaming")
        )
        ids = pl.concat([smp.select("doc_id"), smp.select(doc_id="neg_ids").explode("doc_id")]).unique()
        texts = dict(
            pas.filter(pl.col("doc_id").is_in(ids["doc_id"].implode()))
            .collect(engine="streaming")
            .iter_rows()
        )
        res: dict[str, list[tuple[int, float]]] = {}
        for src, q, d, negs in smp.select("source", "query", "doc_id", "neg_ids").iter_rows():
            pool = [d, *[n for n in negs if n in texts]]
            if d not in texts or len(pool) < 2:
                continue
            bm = BM25([texts[x] for x in pool])
            idx, sc = bm.search([q], len(pool))
            order = [int(i) for i in idx[0] if i >= 0]
            rank = order.index(0) + 1 if 0 in order else len(pool)
            spos = float(sc[0][order.index(0)]) if 0 in order else 0.0
            sneg = max((float(s) for i, s in zip(idx[0], sc[0], strict=True) if i > 0), default=0.0)
            res.setdefault(src, []).append((rank, sneg / spos if spos > 0 else float("inf")))
        out["bm25_ease"] = {
            s: {"pairs": len(v), "positive_rank1_share": round(sum(1 for r, _ in v if r == 1) / len(v), 4),
                "mean_rank": round(float(np.mean([r for r, _ in v])), 3),
                "hard_negative_ratio_p50": round(float(np.median([x for _, x in v if math.isfinite(x)] or [0])), 3)}
            for s, v in res.items() if v
        }  # fmt: skip
    q = scan(root / "eval/retrieval", "*/queries-*.parquet")
    if q is not None:
        qr = scan(root / "eval/retrieval", "*/qrels-*.parquet")
        out["eval_queries"] = q.group_by("direction", "origin").agg(
            n=pl.len(), q_len=pl.col("query").str.count_matches(r"\w+").mean().round(2), needs_review=pl.col("needs_review").sum()
        ).sort("direction", "origin").to_dicts()  # fmt: skip
        if qr is not None:
            rel = qr.filter(pl.col("relevance") > 0).group_by("query_id").len()
            out["eval_relevant_per_query"] = {
                "mean": round(float(rel["len"].mean() or 0), 3),
                "max": int(rel["len"].max() or 0),
            }
    return out


# ----------------------------------------------------------------------------- NLI


def nli(root: Path) -> dict[str, Any]:
    tr = scan(root / "task/nli", "*/train-*.parquet")
    te = scan(root / "task/nli", "*/test-*.parquet")
    if tr is None or te is None:
        return {}
    out: dict[str, Any] = {}
    allp = pl.concat(
        [tr.with_columns(split=pl.lit("train")), te.with_columns(split=pl.lit("test"))],
        how="diagonal_relaxed",
    )
    out["balance"] = allp.group_by("lang", "split", "label").len().sort("lang", "split", "label").to_dicts()
    out["lengths"] = allp.group_by("lang", "split").agg(
        premise=pl.col("premise").str.count_matches(r"\w+").mean().round(2),
        hypothesis=pl.col("hypothesis").str.count_matches(r"\w+").mean().round(2),
    ).sort("lang", "split").to_dicts()  # fmt: skip
    hb = {}
    neg = {}
    for lang in ("kk", "ru", "en"):
        a, b = tr.filter(pl.col("lang") == lang), te.filter(pl.col("lang") == lang)
        if a.height and b.height:
            hb[lang] = hypothesis_only(
                a["hypothesis"].to_list(),
                a["label"].to_list(),
                b["hypothesis"].to_list(),
                b["label"].to_list(),
            )
            cue = a.with_columns(
                has=pl.col("hypothesis")
                .str.to_lowercase()
                .str.contains(r"(^|\W)(" + "|".join(map(re.escape, NEGATION[lang])) + r")(\W|$)")
            )
            neg[lang] = cue.group_by("has", "label").len().sort("has", "label").to_dicts()
    out["hypothesis_only_baseline"] = hb
    out["negation_cue_by_label"] = neg
    return out


def hypothesis_only(tr_x: list[str], tr_y: list[int], te_x: list[str], te_y: list[int]) -> dict[str, float]:
    """Multinomial Naive Bayes on hypothesis unigrams + bigrams: accuracy above 1/3 = annotation artifacts."""
    counts = [Counter() for _ in range(3)]
    prior = Counter(tr_y)

    def feats(t: str) -> list[str]:
        w = WORD.findall(t.lower())
        return w + [f"{a} {b}" for a, b in itertools.pairwise(w)]

    for x, y in zip(tr_x, tr_y, strict=True):
        counts[y].update(feats(x))
    vocab = set().union(*counts)
    tot = [sum(c.values()) for c in counts]
    v = len(vocab)
    correct = 0
    for x, y in zip(te_x, te_y, strict=True):
        f = feats(x)
        scores = [
            math.log(prior[k] / len(tr_y)) + sum(math.log((counts[k][w] + 1) / (tot[k] + v)) for w in f)
            for k in range(3)
        ]
        correct += int(int(np.argmax(scores)) == y)
    return {
        "accuracy": round(correct / max(len(te_y), 1), 4),
        "chance": round(max(Counter(te_y).values()) / max(len(te_y), 1), 4),
        "test": len(te_y),
    }


# ----------------------------------------------------------------------------- synthetic, train vs eval


def synthetic(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("queries", "qa"):
        d = root / "task/synthetic" / name
        if not (d / "dataset.yaml").is_file():
            continue
        from trifact.lake.manifest import DatasetManifest

        m = DatasetManifest.read(d)
        df = scan(d, "*/kept-*.parquet")
        res: dict[str, Any] = {"drops": m.drops, "rows_in": m.rows_in, "kept": m.rows_of("kept")}
        if df is not None:
            col = "query" if name == "queries" else "question"
            res["distinct"] = {lg: {f"distinct_{n}": distinct_n(g[col].to_list()[:50_000], n) for n in (1, 2, 3)}
                               for (lg,), g in df.group_by("lang")}  # fmt: skip
            res["openings"] = (df.with_columns(o=pl.col(col).str.to_lowercase().str.extract(r"^(\w+(?:\s+\w+)?)"))
                               .group_by("lang", "o").len().sort("lang", "len", descending=[False, True]).group_by("lang").head(10).to_dicts())  # fmt: skip
        out[name] = res
    return out


def train_vs_eval(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    tr = scan(root / "task/ner", "*/gold_train-*.parquet")
    te = scan(root / "eval/ner", "*/test-*.parquet")
    if tr is not None and te is not None:
        out["ner_sentence_length_js"] = {
            lg: js(hist(tr.filter(pl.col("lang") == lg)["tokens"].list.len().to_numpy()), hist(te.filter(pl.col("lang") == lg)["tokens"].list.len().to_numpy()))
            for lg in ("kk", "ru", "en")
        }  # fmt: skip
    sft = scan(root / "task/sft", "*/train-*.parquet")
    qa_te = scan(root / "eval/qa", "*/test-*.parquet")
    if sft is not None and qa_te is not None:
        out["question_length_js"] = {
            lg: js(hist(sft.filter(pl.col("lang") == lg)["question"].str.count_matches(r"\w+").to_numpy()),
                   hist(qa_te.filter(pl.col("lang") == lg)["question"].str.count_matches(r"\w+").to_numpy()))
            for lg in ("kk", "ru", "en")
        }  # fmt: skip
        tq = set(sft.select(k=key_expr("question"))["k"])
        eq = qa_te.select(k=key_expr("question"))["k"]
        out["eval_questions_in_sft_train"] = int(eq.is_in(list(tq)).sum())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--bm25-sample", type=int, default=2000, help="pairs per source for the BM25 ease analysis"
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    params = load_params()
    root = (ROOT / params["lake"]["root"]).resolve()
    t0 = time.perf_counter()
    with ResourceMonitor() as monitor:
        rep = {"ner": ner(root), "qa": qa(root), "retrieval": retrieval(root, args.bm25_sample, params["seed"]),
               "nli": nli(root), "synthetic": synthetic(root), "train_vs_eval": train_vs_eval(root)}  # fmt: skip
    meta = {"seconds": round(time.perf_counter() - t0, 1), "peak_rss_mb": round(monitor.peak_rss_mb, 1)}
    write_report(OUT, {"meta": meta, "profiles": rep})
    log_mlflow("eda", "eda-task", {"bm25_sample": args.bm25_sample}, meta)
    log.info("task EDA in %.0f s", meta["seconds"])
    return 0


# ----------------------------------------------------------------------------- report section


def task_section(rep: dict[str, Any], fig: Path) -> tuple[list[str], list[str]]:
    from trifact.data.eda_report import fig_bars, table

    p = rep["profiles"]
    md = ["## Task datasets (ROADMAP 3.7)", "", f"Profiled in {rep['meta']['seconds']} s.", ""]
    findings: list[str] = []
    n = p.get("ner") or {}
    if n:
        rows = [[r["table"], r["source"], r["lang"], r["sentences"], r["mean_len"], r["entities"], r["ent_per_sent"],
                 r["ent_token_share"], r["no_entity_share"], r["mean_span"], r["off_schema_tags"]] for r in n["by_table_source_lang"]]  # fmt: skip
        md += ["### NER", "", table(rows, ["table", "source", "lang", "sentences", "mean tokens", "entities", "ent/sent", "entity token share",
                                          "no-entity share", "mean span", "off-schema tags"]), ""]  # fmt: skip
        for lang, v in n.get("silver_vs_gold_types", {}).items():
            md.append(f"- {lang}: gold types {v['gold']}, silver types {v['silver']}, JS {v['js']}")
            if v["js"] == v["js"] and v["js"] > 0.05:
                findings.append(f"NER {lang}: silver type distribution differs from gold (JS {v['js']})")
        md += ["", f"Gold test entity surfaces seen in train: {n.get('entity_overlap_train_test')}", ""]
    q = p.get("qa") or {}
    if q.get("eval_qa"):
        rows = [
            [
                r["source"],
                r["lang"],
                r["n"],
                r["unanswerable"],
                r["q_len"],
                r["c_len"],
                r["a_len"],
                r["a_pos_p50"],
                r["q_context_overlap"],
            ]
            for r in q["eval_qa"]
        ]
        md += ["### QA (eval)", "", table(rows, ["source", "lang", "n", "unanswerable", "question words", "context words", "answer words",
                                                "answer position p50", "question-context overlap"]), ""]  # fmt: skip
        for r in q["eval_qa"]:
            if r["q_context_overlap"] > 0.7:
                findings.append(
                    f"QA {r['source']}/{r['lang']}: question-context word overlap {r['q_context_overlap']} — lexical shortcut risk"
                )
    if q.get("sft"):
        rows = [
            [
                r["type"],
                r["lang"],
                r["origin"],
                r["n"],
                r["q_len"],
                r["a_len"],
                r["evidence"],
                r["cites"],
                r["evidence_words"],
            ]
            for r in q["sft"]
        ]
        md += ["### SFT", "", table(rows, ["type", "lang", "origin", "n", "question words", "answer words", "evidence", "citations", "evidence words"]), "",
               f"Distinct n-grams of questions: {q.get('sft_distinct')}", ""]  # fmt: skip
    r = p.get("retrieval") or {}
    if r.get("train"):
        dirs: dict[str, int] = {}
        for x in r["train"]:
            dirs[x["direction"]] = dirs.get(x["direction"], 0) + x["n"]
        md += ["### Retrieval", "", f"![dirs]({fig_bars('task_retrieval_directions', list(dirs), list(dirs.values()), 'pairs', log=True)})", "",
               table([[s, v["pairs"], v["positive_rank1_share"], v["mean_rank"], v["hard_negative_ratio_p50"]] for s, v in r.get("bm25_ease", {}).items()],
                     ["source", "sampled pairs", "BM25 ranks positive first", "mean rank", "neg/pos BM25 p50"]), ""]  # fmt: skip
        for s, v in r.get("bm25_ease", {}).items():
            if v["positive_rank1_share"] > 0.9:
                findings.append(
                    f"Retrieval {s}: BM25 ranks the positive first in {100 * v['positive_rank1_share']:.0f}% — negatives are easy"
                )
    if r.get("eval_queries"):
        md += [table([[x["direction"], x["origin"], x["n"], x["q_len"], x["needs_review"]] for x in r["eval_queries"]],
                     ["direction", "origin", "eval queries", "query words", "needs review"]), ""]  # fmt: skip
        have = {x["direction"] for x in r["eval_queries"]}
        miss = [f"{a}-{b}" for a in ("kk", "ru", "en") for b in ("kk", "ru", "en") if f"{a}-{b}" not in have]
        if miss:
            findings.append(f"Retrieval eval: directions without queries: {', '.join(miss)}")
    nl = p.get("nli") or {}
    if nl:
        md += ["### NLI", "", f"Hypothesis-only Naive Bayes accuracy: {nl.get('hypothesis_only_baseline')}", "",
               f"Negation cues by label (train): {nl.get('negation_cue_by_label')}", ""]  # fmt: skip
        for lang, v in nl.get("hypothesis_only_baseline", {}).items():
            if v["accuracy"] - v["chance"] > 0.15:
                findings.append(
                    f"NLI {lang}: hypothesis-only accuracy {v['accuracy']} vs chance {v['chance']} — annotation artifacts"
                )
    s = p.get("synthetic") or {}
    if s:
        md += (
            ["### Synthetic", ""]
            + [
                f"- {k}: kept {v.get('kept')}, drops {v.get('drops')}, distinct {v.get('distinct')}"
                for k, v in s.items()
            ]
            + [""]
        )
    tv = p.get("train_vs_eval") or {}
    if tv:
        md += ["### Train vs eval", "", f"{tv}", ""]
        if tv.get("eval_questions_in_sft_train"):
            findings.append(f"{tv['eval_questions_in_sft_train']} eval questions also in SFT train")
    _ = fig
    return md, findings


if __name__ == "__main__":
    raise SystemExit(main())
