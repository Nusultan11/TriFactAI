"""Teacher synthetic data (ADR-0015): prompts from corpus chunks, parsing and filters of the teacher answers.

python -m trifact.data.synthetic prepare  --kind queries|qa --n N --out DIR      # Kaggle input: prompts + meta
python -m trifact.data.synthetic import   --kind queries|qa --kaggle-out DIR     # filters -> task/synthetic/<kind>
python -m trifact.data.synthetic finalize --kind qa --scores DIR                  # NLI filter of answers

Queries (`configs/prompts/query_gen.md`): one chunk -> a query in kk, ru and en (3 language directions per chunk, so
9 directions over the three chunk languages). Filters, in order (drop reason in parentheses): JSON object with the
three keys (`parse`); 3..20 words (`length`); lid.176 language equals the requested one (`language`); the query is not a
copied passage sentence (`copy`, >= 0.8 of the query words in one sentence in order); round-trip (`round_trip`): the
query in the passage language must retrieve its chunk in the BM25 top `teacher.filters.round_trip_top_k` of the pool of
all prepared chunks of that language; the queries in the other languages are the teacher's translations of it and share
its verdict (the cross-lingual dense round-trip needs the stage 6 embedder); exact duplicates (`duplicate`).

Grounded QA (`configs/prompts/grounded_qa_gen.md`): a target chunk + 2 chunks of the same document or language as
evidence, question language and mode (answerable / unanswerable) drawn to `sft.mix` and `scale.min_kk_share`. Filters:
JSON (`parse`); question and answer language (`language`); answerable: at least one citation and only evidence ids
(`citation`); unanswerable: the exact refusal phrase (`refusal`); NLI support of the answer by the cited chunks
(`finalize`, Kaggle job `nli_score`, entailment >= `teacher.filters.min_nli_entail`, `nli`).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl

from trifact.core.params import ROOT, load_params
from trifact.data.bm25 import BM25
from trifact.data.curation import fingerprint, write_report
from trifact.data.task_common import (
    DatasetWriter,
    by_lang,
    dataset_input_of,
    finish,
    inherited,
    key_expr,
    stable_hash,
)
from trifact.ingestion.langdetect import LangId, model_path
from trifact.ingestion.segment import sentences
from trifact.lake.layout import LakeConfig
from trifact.lake.manifest import DatasetManifest, InheritedLicense, LicenseUse
from trifact.lake.stats import ResourceMonitor, log_mlflow

log = logging.getLogger("trifact.synthetic")
NAMES = {"kk": "Kazakh", "ru": "Russian", "en": "English"}
REFUSAL = {
    "kk": "Берілген құжаттарда бұл сұраққа жауап жоқ.",
    "ru": "В предоставленных документах нет ответа на этот вопрос.",
    "en": "The provided documents do not contain an answer to this question.",
}
TEACHER_LICENSE = InheritedLicense(
    source="qwen3.5-35b-a3b", license="Apache-2.0", use=LicenseUse.allowed, release=True
)
CITE = re.compile(r"\[([^\[\]\s]+)\]")
WORD = re.compile(r"\w+", re.UNICODE)


def lid() -> LangId:
    spec = load_params()["curation"]["langid"]
    return LangId(model_path("lid176", spec["models"]["lid176"], ROOT / spec["cache"]), "lid176")


def parse_json(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) else None


def lang_key(d: dict[str, Any], lang: str) -> str | None:
    """Value for `lang`; tolerates near-miss keys the teacher writes ("k", "kka", "Kazakh")."""
    if isinstance(d.get(lang), str):
        return d[lang]
    for k, v in d.items():
        kk = str(k).lower()
        if isinstance(v, str) and (
            kk.startswith(lang[0]) and (len(kk) <= 3 or kk.startswith(NAMES[lang].lower()[:3]))
        ):
            return v
    return None


def copied(query: str, passage: str) -> bool:
    qw = WORD.findall(query.lower())
    if len(qw) < 4:
        return False
    for a, b in sentences(passage):
        sw = WORD.findall(passage[a:b].lower())
        it = iter(sw)
        if sum(1 for w in qw if w in it) / len(qw) >= 0.8:  # in-order subsequence share
            return True
    return False


# ----------------------------------------------------------------------------- prepare


def sample_chunks(root: Path, n: int, seed: int, min_kk: float) -> pl.DataFrame:
    """`n` chunks: kk >= `min_kk`, the rest split between ru and en; within a language spread over sources/domains."""
    lf = (
        pl.scan_parquet(str(root / "task/chunks/*/*/part-*.parquet"))
        .select("chunk_id", "doc_id", "source", "lang", "domain", "title", "text", "n_tokens")
        .filter(pl.col("n_tokens") >= 64)
    )
    quota = {
        "kk": int(n * max(min_kk, 0.34)),
        "ru": int(n * 0.33),
        "en": n - int(n * max(min_kk, 0.34)) - int(n * 0.33),
    }
    out = []
    for lang, q in quota.items():
        part = (
            lf.filter(pl.col("lang") == lang)
            .with_columns(r=pl.col("chunk_id").hash(seed=seed))
            .sort("r")
            .head(q * 3)
            .collect(engine="streaming")
        )
        # balance domains: round-robin by domain rank
        part = part.with_columns(k=pl.int_range(pl.len()).over("domain")).sort("k", "r").head(q)
        out.append(part.drop("r", "k"))
    return pl.concat(out)


def prepare(kind: str, n: int, out: Path) -> int:
    params = load_params()
    root = (ROOT / params["lake"]["root"]).resolve()
    seed = params["seed"]
    chunks = sample_chunks(root, n, seed, params["scale"]["min_kk_share"])
    out.mkdir(parents=True, exist_ok=True)
    rows, meta = [], []
    if kind == "queries":
        tpl = (ROOT / "configs/prompts/query_gen.md").read_text(encoding="utf-8")
        for r in chunks.iter_rows(named=True):
            pid = f"q/{r['chunk_id']}"
            rows.append(
                {
                    "id": pid,
                    "system": "",
                    "user": tpl.replace("{passage_lang_name}", NAMES[r["lang"]]).replace(
                        "{passage}", r["text"]
                    ),
                    "max_tokens": 200,
                }
            )
            meta.append(
                {
                    "id": pid,
                    "chunk_id": r["chunk_id"],
                    "lang": r["lang"],
                    "source": r["source"],
                    "domain": r["domain"],
                    "text": r["text"],
                }
            )
    else:
        tpl = (ROOT / "configs/prompts/grounded_qa_gen.md").read_text(encoding="utf-8")
        by_doc = chunks.group_by("doc_id").agg("chunk_id", "text")
        docs = {d: list(zip(c, t, strict=True)) for d, c, t in by_doc.iter_rows()}
        pool = {lg: g.select("chunk_id", "text").rows() for (lg,), g in chunks.group_by("lang")}
        mix = params["sft"]["mix"]
        for r in chunks.iter_rows(named=True):
            rnd = __import__("random").Random(stable_hash(r["chunk_id"], seed))
            same = [x for x in docs[r["doc_id"]] if x[0] != r["chunk_id"]]
            others = [
                x for x in rnd.sample(pool[r["lang"]], min(4, len(pool[r["lang"]]))) if x[0] != r["chunk_id"]
            ]
            ev = [(r["chunk_id"], r["text"]), *(same + others)[:2]]
            rnd.shuffle(ev)
            u = rnd.random()
            mode = "unanswerable" if u < mix["unanswerable"] else "answerable"
            # question language: cross-lingual share of sft.mix, otherwise the chunk language; kk favoured to min_kk
            if rnd.random() < mix["cross_lingual"] / (mix["answerable"] + mix["cross_lingual"]):
                ql = rnd.choice([x for x in ("kk", "ru", "en") if x != r["lang"]])
            else:
                ql = r["lang"]
            evid = "\n\n".join(f"[{c}] {t}" for c, t in ev)
            pid = f"qa/{r['chunk_id']}"
            user = (
                tpl.replace("{evidence}", evid)
                .replace("{mode}", mode)
                .replace("{question_lang_name}", NAMES[ql])
                .replace("{refusal}", REFUSAL[ql])
            )
            rows.append({"id": pid, "system": "", "user": user, "max_tokens": 400})
            meta.append({"id": pid, "chunk_id": r["chunk_id"], "lang": ql, "chunk_lang": r["lang"], "mode": mode, "source": r["source"],
                         "evidence_ids": [c for c, _ in ev], "evidence_texts": [t for _, t in ev]})  # fmt: skip
    (out / "in").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(out / "in" / "prompts.parquet")
    pl.DataFrame(meta).write_parquet(out / "meta.parquet")
    t = params["teacher"]
    (out / "in" / "config.json").write_text(
        json.dumps({**t["serve"], "time_budget_h": t["time_budget_h"]}), encoding="utf-8"
    )
    print(f"{len(rows)} prompts -> {out}")
    return 0


# ----------------------------------------------------------------------------- import


def import_queries(kaggle_out: Path, meta: pl.DataFrame, top_k: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    gen = pl.read_parquet(sorted(str(p) for p in (kaggle_out / "out").glob("*.parquet")))
    df = meta.rename({"text": "passage"}).join(gen.select("id", raw="text"), on="id", how="left")
    rows = []
    for r in df.iter_rows(named=True):
        d = parse_json(r["raw"] or "")
        for ql in ("kk", "ru", "en"):
            q = lang_key(d, ql) if d else None
            rows.append({"id": r["id"], "chunk_id": r["chunk_id"], "d_lang": r["lang"], "q_lang": ql, "source": r["source"],
                         "query": (q or "").strip(), "passage": r["passage"], "reason": None if q else "parse"})  # fmt: skip
    q = pl.DataFrame(rows)
    n_words = q["query"].str.count_matches(r"\w+")
    q = q.with_columns(
        reason=pl.when(pl.col("reason").is_null() & ~n_words.is_between(3, 20))
        .then(pl.lit("length"))
        .otherwise(pl.col("reason"))
    )
    langs, _ = lid().predict([x.replace("\n", " ") for x in q["query"].to_list()])
    q = q.with_columns(pred=pl.Series(langs)).with_columns(
        reason=pl.when(pl.col("reason").is_null() & (pl.col("pred") != pl.col("q_lang")))
        .then(pl.lit("language"))
        .otherwise(pl.col("reason"))
    )
    cp = [copied(a, b) for a, b in q.select("query", "passage").iter_rows()]
    q = q.with_columns(
        reason=pl.when(pl.col("reason").is_null() & pl.Series(cp))
        .then(pl.lit("copy"))
        .otherwise(pl.col("reason"))
    )
    # round-trip with the same-language query over the pool of all prepared chunks of that language
    verdict: dict[str, bool] = {}
    for (lang,), g in meta.group_by("lang"):
        pool = g.select("chunk_id", "text")
        bm = BM25(pool["text"].to_list())
        ids = pool["chunk_id"].to_list()
        mono = q.filter((pl.col("q_lang") == lang) & (pl.col("d_lang") == lang) & pl.col("reason").is_null())
        idx, _ = bm.search(mono["query"].to_list(), top_k)
        for cid, row in zip(mono["chunk_id"].to_list(), idx, strict=True):
            verdict[cid] = cid in {ids[j] for j in row if j >= 0}
    q = q.with_columns(
        rt=pl.col("chunk_id").replace_strict(verdict, default=False, return_dtype=pl.Boolean)
    ).with_columns(
        reason=pl.when(pl.col("reason").is_null() & ~pl.col("rt"))
        .then(pl.lit("round_trip"))
        .otherwise(pl.col("reason"))
    )
    q = q.with_columns(k=key_expr("query"))
    dup = q.filter(pl.col("reason").is_null()).with_columns(first=pl.int_range(pl.len()).over("k") == 0)
    dup_ids = set(dup.filter(~pl.col("first")).select(pl.concat_str("id", "q_lang"))[:, 0].to_list())
    q = q.with_columns(
        reason=pl.when(pl.col("reason").is_null() & pl.concat_str("id", "q_lang").is_in(list(dup_ids)))
        .then(pl.lit("duplicate"))
        .otherwise(pl.col("reason"))
    )
    kept = q.filter(pl.col("reason").is_null()).select(
        "id", "chunk_id", "q_lang", "d_lang", "source", "query", "passage", lang="q_lang"
    )
    return kept, q.filter(pl.col("reason").is_not_null()).select(
        "id", "q_lang", "query", "reason", lang="q_lang"
    )


def import_qa(kaggle_out: Path, meta: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    gen = pl.read_parquet(sorted(str(p) for p in (kaggle_out / "out").glob("*.parquet")))
    df = meta.join(gen.select("id", raw="text"), on="id", how="left")
    rows = []
    for r in df.iter_rows(named=True):
        d = parse_json(r["raw"] or "") or {}
        qq, aa = d.get("question"), d.get("answer")
        rows.append({**{k: r[k] for k in ("id", "chunk_id", "lang", "chunk_lang", "mode", "source", "evidence_ids", "evidence_texts")},
                     "question": qq if isinstance(qq, str) else "", "answer": aa if isinstance(aa, str) else "",
                     "reason": None if isinstance(qq, str) and isinstance(aa, str) and qq.strip() and aa.strip() else "parse"})  # fmt: skip
    q = pl.DataFrame(rows)
    plain = q["answer"].str.replace_all(r"\[[^\[\]]*\]", "").str.strip_chars()
    lq, _ = lid().predict([x.replace("\n", " ") for x in q["question"].to_list()])
    la, _ = lid().predict([x.replace("\n", " ") for x in plain.to_list()])
    q = q.with_columns(lq=pl.Series(lq), la=pl.Series(la), answer_plain=plain)
    bad_lang = (pl.col("lq") != pl.col("lang")) | (
        (pl.col("mode") == "answerable") & (pl.col("la") != pl.col("lang"))
    )
    q = q.with_columns(
        reason=pl.when(pl.col("reason").is_null() & bad_lang)
        .then(pl.lit("language"))
        .otherwise(pl.col("reason"))
    )
    cites = [CITE.findall(a) for a in q["answer"].to_list()]
    ok_cite = [bool(c) and set(c) <= set(e) for c, e in zip(cites, q["evidence_ids"].to_list(), strict=True)]
    q = q.with_columns(cited=pl.Series(cites, dtype=pl.List(pl.String)), ok_cite=pl.Series(ok_cite))
    q = q.with_columns(reason=pl.when(pl.col("reason").is_null() & (pl.col("mode") == "answerable") & ~pl.col("ok_cite")).then(pl.lit("citation"))
                       .when(pl.col("reason").is_null() & (pl.col("mode") == "unanswerable") & (pl.col("answer").str.strip_chars() != pl.col("lang").replace_strict(REFUSAL))).then(pl.lit("refusal"))
                       .otherwise(pl.col("reason")))  # fmt: skip
    kept = q.filter(pl.col("reason").is_null())
    return kept, q.filter(pl.col("reason").is_not_null()).select(
        "id", "lang", "mode", "question", "answer", "reason"
    )


def write_kind(
    kind: str,
    kept: pl.DataFrame,
    dropped: pl.DataFrame,
    params: dict[str, Any],
    extra: dict[str, Any],
    final: bool,
) -> DatasetManifest:
    root = (ROOT / params["lake"]["root"]).resolve()
    out = root / "task" / "synthetic" / kind
    if out.exists():
        shutil.rmtree(out)
    w = DatasetWriter(out, LakeConfig.from_params(params["lake"], root))
    w.write("kept", kept)
    w.write("dropped", dropped)
    shards = w.close()
    drops: dict[str, dict[str, int]] = {}
    for reason, lang, n in dropped.group_by("reason", "lang").len().iter_rows():
        drops.setdefault(str(reason), {})[str(lang)] = int(n)
    rows_in = {k: by_lang(kept).get(k, 0) + by_lang(dropped).get(k, 0) for k in ("kk", "ru", "en")}
    lic = inherited(root, [], [root / "task/chunks"]) + [TEACHER_LICENSE]
    return finish(
        out, name=f"task/synthetic/{kind}", layer="task", stage="synthetic", source="teacher",
        description=f"Teacher ({params['teacher']['model']}) synthetic {kind} over corpus chunks with the ADR-0015 filters; "
        "table `kept` passes every filter, `dropped` keeps the reason." + ("" if final else " NLI filter pending."),
        params={"teacher": params["teacher"], "kind": kind}, inputs=[dataset_input_of(root, root / "task/chunks")],
        licenses=lic, shards=shards, rows_in=rows_in, drops=drops, metrics=extra, stats=None,
        fingerprint=fingerprint(f"synthetic/{kind}", params["teacher"], [Path(__file__)], []),
    )  # fmt: skip


def synthetic_pairs(d: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Kept synthetic queries as retrieval pairs + their passages (for build_pairs)."""
    k = pl.read_parquet(sorted(str(p) for p in d.glob("*/kept-*.parquet")))
    pairs = k.select(
        pair_id=pl.concat_str(pl.lit("synthetic"), "id", "q_lang", separator="|"), query_id=pl.concat_str("id", "q_lang", separator="/"),
        query="query", q_lang="q_lang", doc_id=pl.concat_str(pl.lit("chunk/"), "chunk_id"), d_lang="d_lang",
        direction=pl.concat_str("q_lang", "d_lang", separator="-"), origin=pl.lit("synthetic"), source=pl.lit("teacher"),
        split=pl.when(pl.col("chunk_id").hash(seed=7) % 20 == 0).then(pl.lit("validation")).otherwise(pl.lit("train")),
        neg_ids=pl.lit([], dtype=pl.List(pl.String)),
    )  # fmt: skip
    passages = k.unique("chunk_id").select(doc_id=pl.concat_str(pl.lit("chunk/"), "chunk_id"), lang="d_lang", source="source",
                                            title=pl.lit(None, dtype=pl.String), text="passage")  # fmt: skip
    return pairs, passages


def synthetic_sft(d: Path, template: str) -> pl.DataFrame:
    """Kept synthetic grounded QA as SFT rows (for build_sft)."""
    from trifact.data.build_sft import render

    k = pl.read_parquet(sorted(str(p) for p in d.glob("*/kept-*.parquet")))
    rows = []
    for r in k.iter_rows(named=True):
        ev = [
            {"chunk_id": c, "lang": r["chunk_lang"], "text": t}
            for c, t in zip(r["evidence_ids"], r["evidence_texts"], strict=True)
        ]
        typ = (
            "unanswerable"
            if r["mode"] == "unanswerable"
            else ("cross_lingual" if r["lang"] != r["chunk_lang"] else "answerable")
        )
        rows.append({"example_id": f"synthetic/{r['id']}", "lang": r["lang"], "type": typ, "origin": "synthetic", "source": "teacher",
                     "split": "validation" if stable_hash(r["id"]) % 20 == 0 else "train", "question": r["question"], "evidence": ev,
                     "gold_chunk_ids": r["cited"] or [], "answer": r["answer"], "prompt": render(template, r["lang"], r["question"], ev)})  # fmt: skip
    return pl.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "import", "finalize"])
    ap.add_argument("--kind", choices=["queries", "qa"], required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", type=Path, help="prepare: work directory (prompts in <out>/in, meta.parquet)")
    ap.add_argument("--kaggle-out", type=Path)
    ap.add_argument("--scores", type=Path)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    params = load_params()
    if args.command == "prepare":
        return prepare(args.kind, args.n, args.out)
    work = args.kaggle_out.parent if args.kaggle_out else args.scores.parent
    meta = pl.read_parquet(work / "meta.parquet")
    t0 = time.perf_counter()
    with ResourceMonitor():
        if args.command == "import":
            run = json.loads((args.kaggle_out / "run.json").read_text())
            summary = json.loads((args.kaggle_out / "summary.json").read_text())
            if args.kind == "queries":
                kept, dropped = import_queries(
                    args.kaggle_out, meta, int(params["teacher"]["filters"]["round_trip_top_k"])
                )
            else:
                kept, dropped = import_qa(args.kaggle_out, meta)
                # NLI pairs for the Kaggle job: cited evidence -> answer without citations
                ev = kept.select(
                    "id", "answer_plain", ids="evidence_ids", texts="evidence_texts", cited="cited"
                )
                pairs = [{"id": i, "premise": " ".join(t for c, t in zip(ids, texts, strict=True) if c in set(cited)), "hypothesis": a}
                         for i, a, ids, texts, cited in ev.iter_rows() if cited]  # fmt: skip
                (work / "nli_in").mkdir(exist_ok=True)
                pl.DataFrame(pairs).write_parquet(work / "nli_in" / "pairs.parquet")
                (work / "nli_in" / "config.json").write_text(
                    json.dumps(params["verification"]["nli_job"]), encoding="utf-8"
                )
                kept.write_parquet(work / "qa_candidates.parquet")
            extra = {"kaggle": run, "teacher_summary": summary, "seconds": round(time.perf_counter() - t0, 1)}
            m = write_kind(args.kind, kept.drop([c for c in ("answer_plain", "ok_cite", "lq", "la", "rt", "k", "pred") if c in kept.columns]),
                           dropped, params, extra, final=args.kind == "queries")  # fmt: skip
        else:
            kept = pl.read_parquet(work / "qa_candidates.parquet")
            s = pl.concat([pl.read_parquet(f) for f in sorted(args.scores.glob("out/*.parquet"))])
            kept = kept.join(s.select("id", "p_entailment"), on="id", how="left")
            thr = float(params["teacher"]["filters"]["min_nli_entail"])
            bad = (pl.col("mode") == "answerable") & (pl.col("p_entailment").fill_null(0) < thr)
            old = DatasetManifest.read((ROOT / params["lake"]["root"]).resolve() / "task/synthetic/qa")
            dropped = kept.filter(bad).select(
                "id", "lang", "mode", "question", "answer", reason=pl.lit("nli")
            )
            prev = pl.scan_parquet(
                str((ROOT / params["lake"]["root"]).resolve() / "task/synthetic/qa/*/dropped-*.parquet")
            ).collect()
            dropped = pl.concat([prev, dropped], how="diagonal_relaxed")
            kept = kept.filter(~bad)
            m = write_kind("qa", kept.drop([c for c in ("answer_plain", "ok_cite", "lq", "la") if c in kept.columns]), dropped, params,
                           {**old.metrics, "nli_threshold": thr}, final=True)  # fmt: skip
    write_report(
        ROOT / "reports" / f"synthetic_{args.kind}.json", {"rows": m.rows_of("kept"), "drops": m.drops}
    )
    log_mlflow("task", f"synthetic-{args.kind}-{args.command}", {"kind": args.kind}, {
        "kept": sum(m.rows_of("kept").values()), **{f"drop_{r}": sum(v.values()) for r, v in m.drops.items()}})  # fmt: skip
    print("kept", m.rows_of("kept"), "drops", {r: sum(v.values()) for r, v in m.drops.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
