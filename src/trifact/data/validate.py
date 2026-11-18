"""Stage validate: integrity of the gold sources the task datasets are built from (ROADMAP 3, ADR-0007).

python -m trifact.data.validate   # reports/data_quality.json (+ MLflow); exit 1 on structural errors

Checks, per source, language and split (counts and examples of every violation):
* NER (KazNERD, WikiANN, WikiNEuRal): tokens and tags of equal length, tag ids in the tag set,
  BIO well-formedness (`I-X` not after `B-X`/`I-X`), empty / whitespace tokens, empty sentences,
  label distribution; NEREL: brat spans inside the text and equal to the text they cover.
* QA (SQuAD v2, SberQuAD, KazQAD): answers equal `context[start:start+len]`, empty questions,
  unanswerable share, duplicate (question, context) pairs.
* NLI (XNLI): labels in {0,1,2}, empty fields, en/ru rows aligned (same label at the same position).
* Retrieval (MIRACL): topics without judgements, positive passages missing from the corpus.

Structural errors (length mismatch, unknown tag id, spans outside the text, misaligned XNLI) fail
the stage; content issues (bad BIO, offset drift, duplicates) are reported and handled downstream.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from trifact.core.params import ROOT, load_params
from trifact.data.curation import write_report
from trifact.data.task_common import scan_raw
from trifact.lake.stats import log_mlflow

log = logging.getLogger("trifact.validate")
LABELS = ROOT / "configs" / "labels.yaml"
EXAMPLES = 5


def tagsets() -> dict[str, list[str]]:
    return {k: v for k, v in yaml.safe_load(LABELS.read_text(encoding="utf-8"))["tagsets"].items()}


def _bio(tags: list[str]) -> int:
    bad, prev = 0, "O"
    for t in tags:
        if t.startswith("I-") and not (prev != "O" and prev[2:] == t[2:]):
            bad += 1
        prev = t
    return bad


def bio_errors(tags: pl.Expr) -> pl.Expr:
    """Number of `I-X` tags not preceded by `B-X` or `I-X` in a list of string tags."""
    return tags.map_elements(_bio, return_dtype=pl.Int64)


def check_ner(root: Path, source: str, names: list[str]) -> dict[str, Any]:
    df = scan_raw(root, source).select("lang", "_split", "tokens", "ner_tags").collect()
    n = len(names)
    df = df.with_columns(
        n_tok=pl.col("tokens").list.len(),
        n_tag=pl.col("ner_tags").list.len(),
        bad_id=pl.col("ner_tags").list.eval((pl.element() < 0) | (pl.element() >= n)).list.any(),
        empty_tok=pl.col("tokens").list.eval(pl.element().str.strip_chars() == "").list.sum(),
        space_tok=pl.col("tokens").list.eval(pl.element().str.contains(r"\s")).list.sum(),
    )
    ok = df.filter((pl.col("n_tok") == pl.col("n_tag")) & ~pl.col("bad_id"))
    tags = ok.with_columns(
        tags=pl.col("ner_tags").list.eval(pl.element().replace_strict(dict(enumerate(names))))
    )
    tags = tags.with_columns(bio=bio_errors(pl.col("tags")))
    labels = (
        tags.select("lang", tag=pl.col("tags"))
        .explode("tag")
        .filter(pl.col("tag").str.starts_with("B-"))
        .group_by("lang", "tag")
        .len()
        .sort("lang", "len", descending=[False, True])
    )
    per = df.group_by("lang", "_split").agg(
        sentences=pl.len(),
        tokens=pl.col("n_tok").sum(),
        length_mismatch=(pl.col("n_tok") != pl.col("n_tag")).sum(),
        unknown_tag_id=pl.col("bad_id").sum(),
        empty_sentences=(pl.col("n_tok") == 0).sum(),
        empty_tokens=pl.col("empty_tok").sum(),
        whitespace_tokens=pl.col("space_tok").sum(),
    )
    bio = tags.group_by("lang", "_split").agg(
        bio_violations=pl.col("bio").sum(), bio_sentences=(pl.col("bio") > 0).sum()
    )
    per = per.join(bio, on=["lang", "_split"], how="left").sort("lang", "_split")
    examples = (
        tags.filter(pl.col("bio") > 0)
        .head(EXAMPLES)
        .select(tokens=pl.col("tokens").list.head(30), tags=pl.col("tags").list.head(30))
    )
    return {
        "by_split": per.to_dicts(),
        "entities_by_type": labels.to_dicts(),
        "bio_examples": examples.to_dicts(),
        "structural_errors": int(per["length_mismatch"].sum() + per["unknown_tag_id"].sum()),
    }


def check_nerel(root: Path) -> dict[str, Any]:
    df = scan_raw(root, "nerel").select("_split", "id", "text", "entities").collect()
    ent = df.explode("entities").filter(pl.col("entities").is_not_null())
    ent = ent.with_columns(
        p=pl.col("entities").str.extract_groups(r"^(T\d+)\t(\S+) (\d+) (\d+)(?:;\d+ \d+)*\t(.*)$")
    ).unnest("p")
    ent = ent.rename({"1": "tid", "2": "type", "3": "start", "4": "end", "5": "surface"}).with_columns(
        pl.col("start").cast(pl.Int64), pl.col("end").cast(pl.Int64)
    )
    ent = ent.with_columns(
        unparsed=pl.col("tid").is_null(),
        outside=pl.col("end") > pl.col("text").str.len_chars(),
        covered=pl.col("text").str.slice(pl.col("start"), pl.col("end") - pl.col("start")),
    )
    ent = ent.with_columns(mismatch=~pl.col("outside") & (pl.col("covered") != pl.col("surface")))
    per = ent.group_by("_split").agg(
        documents=pl.col("id").n_unique(), entities=pl.len(), unparsed=pl.col("unparsed").sum(),
        outside_text=pl.col("outside").sum(), surface_mismatch=pl.col("mismatch").sum(),
    ).sort("_split")  # fmt: skip
    types = ent.group_by("type").len().sort("len", descending=True)
    return {
        "by_split": per.to_dicts(),
        "entities_by_type": types.to_dicts(),
        "mismatch_examples": ent.filter(pl.col("mismatch"))
        .head(EXAMPLES)
        .select("surface", "covered")
        .to_dicts(),
        "structural_errors": int(per["outside_text"].sum()),
    }


def check_qa(root: Path, source: str) -> dict[str, Any]:
    df = (
        scan_raw(root, source).select("lang", "_config", "_split", "context", "question", "answers").collect()
    )
    df = df.with_columns(
        a_text=pl.col("answers").struct.field("text"), a_start=pl.col("answers").struct.field("answer_start")
    )
    ans = (
        df.with_row_index("row")
        .explode("a_text", "a_start")
        .filter(pl.col("a_text").is_not_null() & (pl.col("a_text") != ""))
    )
    ans = ans.with_columns(
        covered=pl.col("context").str.slice(pl.col("a_start"), pl.col("a_text").str.len_chars()),
        found=pl.col("context").str.contains(pl.col("a_text"), literal=True),
    ).with_columns(exact=pl.col("covered") == pl.col("a_text"))
    per_ans = ans.group_by("lang", "_config", "_split").agg(
        answers=pl.len(), offset_exact=pl.col("exact").sum(), text_found=pl.col("found").sum(),
    )  # fmt: skip
    per = df.group_by("lang", "_config", "_split").agg(
        questions=pl.len(),
        unanswerable=(
            pl.col("a_text").list.eval(pl.element().filter(pl.element() != "")).list.len() == 0
        ).sum(),
        empty_question=(pl.col("question").str.strip_chars() == "").sum(),
        duplicate_pairs=pl.len() - pl.struct("question", "context").n_unique(),
        contexts=pl.col("context").n_unique(),
    )
    per = per.join(per_ans, on=["lang", "_config", "_split"], how="left").sort("lang", "_config", "_split")
    bad = ans.filter(~pl.col("exact")).head(EXAMPLES).select("a_text", "covered")
    return {"by_split": per.to_dicts(), "offset_examples": bad.to_dicts(), "structural_errors": 0}


def check_xnli(root: Path) -> dict[str, Any]:
    df = scan_raw(root, "xnli").select("lang", "_split", "premise", "hypothesis", "label").collect()
    df = df.with_columns(row=pl.int_range(pl.len()).over("lang", "_split"))
    per = (
        df.group_by("lang", "_split")
        .agg(
            pairs=pl.len(),
            bad_label=(~pl.col("label").is_in([0, 1, 2])).sum(),
            empty=(
                (pl.col("premise").str.strip_chars() == "") | (pl.col("hypothesis").str.strip_chars() == "")
            ).sum(),
            **{f"label_{i}": (pl.col("label") == i).sum() for i in range(3)},
        )
        .sort("lang", "_split")
    )
    en = df.filter(pl.col("lang") == "en").select("_split", "row", "label")
    ru = df.filter(pl.col("lang") == "ru").select("_split", "row", label_ru="label")
    j = en.join(ru, on=["_split", "row"], how="full", coalesce=True)
    misaligned = int(((j["label"] != j["label_ru"]) | j["label"].is_null() | j["label_ru"].is_null()).sum())
    return {"by_split": per.to_dicts(), "en_ru_misaligned": misaligned,
            "structural_errors": misaligned + int(per["bad_label"].sum())}  # fmt: skip


def check_miracl(root: Path) -> dict[str, Any]:
    topics = scan_raw(root, "miracl", "topics").select("lang", "_split", "qid").collect()
    qrels = scan_raw(root, "miracl", "qrels").select("lang", "_split", "qid", "docid", "relevance").collect()
    pos = qrels.filter(pl.col("relevance").cast(pl.Int64) > 0)
    corpus_ids = scan_raw(root, "miracl_corpus").select("lang", "docid")
    found = pos.lazy().join(corpus_ids, on=["lang", "docid"], how="semi").collect()
    judged = topics.join(qrels.select("lang", "qid").unique(), on=["lang", "qid"], how="semi")
    per = (
        topics.group_by("lang", "_split")
        .agg(topics=pl.len())
        .join(judged.group_by("lang", "_split").agg(judged=pl.len()), on=["lang", "_split"], how="left")
    )
    pos_per = (
        pos.group_by("lang", "_split")
        .agg(positives=pl.len())
        .join(
            found.group_by("lang", "_split").agg(positives_in_corpus=pl.len()),
            on=["lang", "_split"],
            how="left",
        )
    )
    per = per.join(pos_per, on=["lang", "_split"], how="left").sort("lang", "_split")
    return {"by_split": per.to_dicts(), "structural_errors": 0}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "data_quality.json")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    params = load_params()
    root = (ROOT / params["lake"]["root"]).resolve()
    t0 = time.perf_counter()
    ts = tagsets()
    rep: dict[str, Any] = {}
    for s in ("kaznerd", "wikiann", "wikineural"):
        rep[s] = check_ner(root, s, ts[s])
    rep["nerel"] = check_nerel(root)
    for s in ("squad_v2", "sberquad", "kazqad"):
        rep[s] = check_qa(root, s)
    rep["xnli"] = check_xnli(root)
    rep["miracl"] = check_miracl(root)
    errors = {k: v["structural_errors"] for k, v in rep.items()}
    rep["summary"] = {"structural_errors": errors, "seconds": round(time.perf_counter() - t0, 1)}
    write_report(args.report, rep)
    log_mlflow(
        "task", "validate", {"sources": ",".join(errors)}, {f"errors_{k}": v for k, v in errors.items()}
    )
    print(errors)
    return 1 if any(errors.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
