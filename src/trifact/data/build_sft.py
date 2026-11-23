"""Stage build_sft: grounded QA SFT examples with citations, cross-lingual and unanswerable (ROADMAP 3.3, ADR-0015).

python -m trifact.data.build_sft [--profile full|pilot] [--synthetic DIR]

Every example is the prompt of `configs/prompts/grounded_answer.md` (numbered evidence, question, answer language)
and the target answer citing the evidence as `[chunk_id]`. Rows of `data/task/sft/<answer lang>/{train,validation}`:
`example_id, lang, type (answerable | cross_lingual | unanswerable), origin (gold | mt_query | synthetic), source,
question, evidence (list of {chunk_id, lang, text}), gold_chunk_ids, answer, prompt`.

Built from gold extractive QA (train/validation splits only; test is the frozen eval, ADR-0007):
* answerable — the gold context plus `sft.distractors` BM25 contexts (same language, different title, not containing
  the answer), shuffled; answer = the gold answer span with the citation of the gold context;
* cross_lingual — the question and the answer translated by NLLB (`task/mt/qa`, round-trip filtered) into another
  language, evidence unchanged: the model must answer in the question language from foreign evidence;
* unanswerable — SQuAD v2 questions without an answer (their own context as evidence), and answerable questions whose
  gold context is removed so that only distractors remain (`unanswerable_from_distractors`); answer = the refusal
  phrase of the question language.
Teacher examples (`--synthetic`, ADR-0015 filters applied in `trifact.data.synthetic`) are added with origin `synthetic`.
The mix follows `sft.mix` and `scale.min_kk_share` by sampling each pool down to the largest feasible total.
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl

from trifact.core.params import ROOT, load_params
from trifact.data.curation import fingerprint, write_report
from trifact.data.qa_common import QA_SOURCES, bm25_negatives, load_qa
from trifact.data.task_common import (
    DatasetWriter,
    dataset_input_of,
    eval_query_hashes,
    finish,
    inherited,
    key_expr,
    raw_input,
    stable_hash,
)
from trifact.data.translations import MT_LICENSE, qc_ok
from trifact.lake.layout import LakeConfig
from trifact.lake.stats import ResourceMonitor, log_mlflow

log = logging.getLogger("trifact.build_sft")
PROMPT = ROOT / "configs" / "prompts" / "grounded_answer.md"
LANG_NAME = {"kk": "Kazakh", "ru": "Russian", "en": "English"}
REFUSAL = {
    "kk": "Берілген құжаттарда бұл сұраққа жауап жоқ.",
    "ru": "В предоставленных документах нет ответа на этот вопрос.",
    "en": "The provided documents do not contain an answer to this question.",
}
EVIDENCE = pl.List(pl.Struct({"chunk_id": pl.String, "lang": pl.String, "text": pl.String}))


def chunk_id(doc_id: str) -> str:
    return f"c{stable_hash(doc_id) % 10**8:08d}"


def render(template: str, lang: str, question: str, evidence: list[dict[str, str]]) -> str:
    ev = "\n\n".join(f"[{e['chunk_id']}] {e['text']}" for e in evidence)
    return (
        template.replace("{answer_lang_name}", LANG_NAME[lang])
        .replace("{evidence}", ev)
        .replace("{question}", question)
    )


def examples(
    root: Path, k: int, mt: pl.DataFrame | None, seed: int, template: str
) -> tuple[pl.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    for s in QA_SOURCES:
        df = load_qa(root, s, ("train", "validation"))
        df = df.with_columns(neg_ids=bm25_negatives(df, k, exclude_answer=True))
        ctx = dict(df.unique("doc_id").select("doc_id", "context").iter_rows())
        mt_q = mt_a = {}
        if mt is not None:
            m = mt.filter(pl.col("src_lang").is_in(df["lang"].unique().to_list()))
            mt_q = {
                (r[0], r[1], r[2]): r[3]
                for r in m.select("src_lang", "tgt_lang", "text", "translation").iter_rows()
            }
            mt_a = mt_q
        n0 = len(rows)
        for r in df.iter_rows(named=True):
            rnd = random.Random(stable_hash(r["example_id"], seed))
            lang = r["lang"]
            negs = [{"chunk_id": chunk_id(d), "lang": lang, "text": ctx[d]} for d in r["neg_ids"]]
            gold = {"chunk_id": chunk_id(r["doc_id"]), "lang": lang, "text": r["context"]}
            ev = [gold, *negs]
            rnd.shuffle(ev)
            base = {"source": s, "split": r["split"], "question": r["question"]}
            eid = f"{s}/{lang}/{r['example_id']}"
            if r["answerable"]:
                ans = r["answers"][0].strip().rstrip(".")
                rows.append({**base, "example_id": f"{eid}/a", "lang": lang, "type": "answerable", "origin": "gold",
                             "evidence": ev, "gold_chunk_ids": [gold["chunk_id"]], "answer": f"{ans} [{gold['chunk_id']}]."})  # fmt: skip
                if (
                    negs and r["config"] != "nq-original-en"
                ):  # one refusal variant per question, from distractors only
                    rows.append({**base, "example_id": f"{eid}/u", "lang": lang, "type": "unanswerable",
                                 "origin": "gold", "evidence": negs, "gold_chunk_ids": [], "answer": REFUSAL[lang]})  # fmt: skip
                for tgt in ("kk", "ru", "en"):
                    q_t, a_t = mt_q.get((lang, tgt, r["question"])), mt_a.get((lang, tgt, r["answers"][0]))
                    if tgt == lang or not q_t or not a_t:
                        continue
                    rows.append({**base, "question": q_t, "example_id": f"{eid}/x{tgt}", "lang": tgt, "type": "cross_lingual",
                                 "origin": "mt_query", "evidence": ev, "gold_chunk_ids": [gold["chunk_id"]],
                                 "answer": f"{a_t.strip().rstrip('.')} [{gold['chunk_id']}]."})  # fmt: skip
            else:  # SQuAD v2 unanswerable: the plausible context itself does not answer
                rows.append({**base, "example_id": f"{eid}/n", "lang": lang, "type": "unanswerable", "origin": "gold",
                             "evidence": ev, "gold_chunk_ids": [], "answer": REFUSAL[lang]})  # fmt: skip
        stats[s] = len(rows) - n0
        log.info("%s: %d examples", s, len(rows) - n0)
    df = pl.DataFrame(rows, schema_overrides={"evidence": EVIDENCE, "gold_chunk_ids": pl.List(pl.String)})
    df = df.with_columns(
        prompt=pl.struct("lang", "question", "evidence").map_elements(
            lambda x: render(template, x["lang"], x["question"], x["evidence"]), return_dtype=pl.String
        )
    )
    return df, stats


def mix(
    df: pl.DataFrame, shares: dict[str, float], min_kk: float, cap: int | None, seed: int
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Largest sample with the type shares of `sft.mix` and at least `min_kk` Kazakh examples per type."""
    pools = {t: df.filter(pl.col("type") == t) for t in shares}
    n = min(pools[t].height / shares[t] for t in shares)
    for t, sh in shares.items():  # kk examples of a type must cover min_kk of its quota
        kk = pools[t].filter(pl.col("lang") == "kk").height
        n = min(n, kk / (sh * min_kk))
    if cap:
        n = min(n, cap)
    out = []
    for t, sh in shares.items():
        quota = int(n * sh)
        p = pools[t].with_columns(r=pl.col("example_id").hash(seed=seed)).sort("r")
        kk = p.filter(pl.col("lang") == "kk").head(max(int(quota * min_kk), 0))
        rest = p.filter(pl.col("lang") != "kk")
        others = rest.head(quota - kk.height)
        extra_kk = p.filter(pl.col("lang") == "kk").slice(kk.height, quota - kk.height - others.height)
        out.append(pl.concat([kk, others, extra_kk]).drop("r"))
    res = pl.concat(out)
    return res, {
        "pool": df.group_by("type", "lang").len().sort("type", "lang").to_dicts(),
        "total_feasible": int(n),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=["pilot", "full"])
    ap.add_argument("--synthetic", type=Path)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    params = load_params()
    profile = args.profile or params["scale"]["profile"]
    root = (ROOT / params["lake"]["root"]).resolve()
    out = root / "task" / "sft"
    cfg = params["sft"]
    mt_dir = root / "task/mt/qa"
    stage_params = {"sft": cfg, "min_kk_share": params["scale"]["min_kk_share"], "profile": profile,
                    "cap": params["scale"][profile]["sft_examples"], "mt": mt_dir.is_dir(), "synthetic": bool(args.synthetic)}  # fmt: skip
    inputs_m = [root / f"raw/{s}/source.yaml" for s in QA_SOURCES] + [root / "task/splits/dataset.yaml"]
    inputs_m += [mt_dir / "dataset.yaml"] if mt_dir.is_dir() else []
    inputs_m += [args.synthetic / "dataset.yaml"] if args.synthetic else []
    code = [Path(__file__), PROMPT, ROOT / "src/trifact/data/qa_common.py", ROOT / "src/trifact/data/bm25.py"]
    fp = fingerprint("build_sft", stage_params, code, inputs_m)
    template = PROMPT.read_text(encoding="utf-8")
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
        df, per_source = examples(root, int(cfg["distractors"]), mt, params["seed"], template)
        if args.synthetic:
            from trifact.data.synthetic import synthetic_sft

            df = pl.concat([df, synthetic_sft(args.synthetic, template)], how="diagonal_relaxed")
        # a question of the frozen QA eval never appears in SFT, even from another source (an MT SQuAD question
        # repeated a SberQuAD test question word for word, task EDA), nor a query of the retrieval eval
        eq_h = eval_query_hashes(root)
        hit = key_expr(pl.col("question")).hash(seed=0).is_in(eq_h.implode())
        n_eval_q = df.filter(hit).height
        df = df.filter(~hit)
        cap = stage_params["cap"] if profile == "pilot" else None
        final, mix_stats = mix(df, cfg["mix"], params["scale"]["min_kk_share"], cap, params["seed"])
        if out.exists():
            shutil.rmtree(out)
        w = DatasetWriter(out, LakeConfig.from_params(params["lake"], root))
        for split in ("train", "validation"):
            w.write(split, final.filter(pl.col("split") == split).drop("split").sort("lang", "example_id"))
        shards = w.close()
    dist = (
        final.group_by("split", "type", "lang", "origin")
        .len()
        .sort("split", "type", "lang", "origin")
        .to_dicts()
    )
    kk_share = final.filter(pl.col("lang") == "kk").height / max(final.height, 1)
    metrics = {"per_source_pool": per_source, **mix_stats, "final": dist, "kk_share": round(kk_share, 4),
               "eval_questions_dropped": n_eval_q,
               "type_share": {t: round(final.filter(pl.col("type") == t).height / max(final.height, 1), 4) for t in cfg["mix"]}}  # fmt: skip
    extra_datasets = [args.synthetic] if args.synthetic else []
    lic = inherited(root, list(QA_SOURCES), extra_datasets) + ([MT_LICENSE] if mt is not None else [])
    inputs = [raw_input(root, s) for s in QA_SOURCES] + [dataset_input_of(root, root / "task/splits")]
    inputs += [dataset_input_of(root, mt_dir)] if mt is not None else []
    inputs += [dataset_input_of(root, args.synthetic)] if args.synthetic else []
    finish(
        out, name="task/sft", layer="task", stage="build_sft", source="+".join(QA_SOURCES),
        description="Grounded QA SFT examples: numbered evidence with [chunk_id] citations, answerable / cross-lingual / "
        "unanswerable, in the prompt format of configs/prompts/grounded_answer.md.",
        params=stage_params, inputs=inputs, licenses=lic, shards=shards, rows_in={"multi": df.height},
        metrics=metrics, stats=mon.stats(final.height, 0, 1), fingerprint=fp, partial=profile != "full",
    )  # fmt: skip
    write_report(ROOT / "reports" / "sft_data.json", metrics)
    log_mlflow("task", "build_sft", {**stage_params, "fingerprint": fp},
               {"examples": final.height, "pool": df.height, "kk_share": kk_share, "seconds": round(time.perf_counter() - t0, 1)})  # fmt: skip
    print(metrics["type_share"], "kk", round(kk_share, 3), "examples", final.height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
