"""Machine pre-labels for the human review sheets: stage 2 quality, stage 3 silver NER and eval MT items.

python -m trifact.evaluation.prelabel prompts --out DIR          # teacher judge prompts (Kaggle `teacher` job input)
python -m trifact.evaluation.prelabel signals [--sheet S]        # independent local signals per item
python -m trifact.evaluation.prelabel import --teacher-out DIR   # judge verdicts + signals -> machine_labels.csv
python -m trifact.evaluation.prelabel report                     # machine-only estimates -> reports/prelabel.json

Sheets (`SHEETS`): `quality` (data/review/quality_human/to_label.csv, label good/bad), `ner` (data/review/ner_silver
/to_label.csv, error counts per sentence) and `eval` (data/eval/review/sheet-01.csv, verdict ok/fix/drop).

Two independent machine opinions per item, never written into the human columns of a sheet:
* the teacher judge (`teacher.model`, llama.cpp on Kaggle, temperature 0, ADR-0022) with a rubric equal to the
  human instructions and a short reason. The judged text never shows the filter decision (kept/dropped) or the
  automatic label it checks, only the item;
* signals that do not depend on the judge: GlotLID language (stage 2 filters use lid.176), the share of Kazakh
  letters; bge-m3 cosine of source and translation, the round-trip chrF of the NLLB translation, digits kept,
  translation language; mDeBERTa-xnli on the Kazakh and on the English pair; GLiNER spans against the silver spans.

`machine_labels.csv` (in `MACHINE_DIRS`) keeps both opinions, a `machine_agree` flag (judge and signals agree)
and a `priority` (1 = disagreement or a doubtful signal: review first). `labeler` names the models. Nothing here
is a human label: the scoring commands (`review_quality score`, `review_ner score`, `build_eval freeze`) read only
the human columns, and `reports/prelabel.json` carries `"human": false`. ADR-0022: a judge counts only after it is
calibrated on human labels; `report` therefore also prints its agreement with any human labels present.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl

from trifact.core.params import ROOT, load_params
from trifact.data.curation import write_report
from trifact.lake.stats import ResourceMonitor, git_sha, log_mlflow

log = logging.getLogger("trifact.prelabel")
NAMES = {"kk": "Kazakh", "ru": "Russian", "en": "English"}
KK_LETTERS = "әғқңөұүһіӘҒҚҢӨҰҮҺІ"
SHEETS: dict[str, tuple[str, str]] = {
    "quality": ("data/review/quality_human/to_label.csv", "doc_id"),
    "ner": ("data/review/ner_silver/to_label.csv", "sent_id"),
    "eval": ("data/eval/review/sheet-01.csv", "item_id"),
}
MACHINE_DIRS = {
    "quality": "data/review/quality_human",
    "ner": "data/review/ner_silver",
    "eval": "data/review/eval_mt",
}
SIGNALS = "machine_signals.parquet"
LABELS = "machine_labels.csv"

QUALITY_PROMPT = """You review documents for a Kazakh/Russian/English corpus used to train and evaluate search and question answering models.
Expected language: {lang}. Below are the first {chars} characters of the document (the end may be cut mid-word).

Document:
\"\"\"
{text}
\"\"\"

Label it:
- good: natural text in {lang} that is worth training and retrieval on (prose, news, encyclopedia, books, posts), also when short;
- bad: site boilerplate or navigation, lists or tables without prose, broken encoding or OCR garbage, spam or SEO text, machine-generated nonsense, a meaningless fragment, or mostly another language.
Category: prose | stub | list_table | boilerplate | spam | broken_encoding | ocr_noise | wrong_language | fragment | other.
Output only one JSON object: {{"category": "...", "label": "good" or "bad", "language": "<ISO 639-1 code of the main language>", "reason": "<one short English sentence>"}}"""

NER_PROMPT = """You check automatic named-entity annotations of a {lang} Wikipedia sentence. Entities are marked as [surface|TYPE]:
- PER: a person (real or fictional human);
- ORG: an organization (company, institution, party, government body, university, sports club, association, band);
- LOC: a location (country, city, region, administrative unit, geographic feature, river, mountain, building, facility).
In Kazakh and Russian a span is the inflected word form: case suffixes belong to the span (e.g. [Қазақстанда|LOC]). Dates, nationalities, languages, events, titles of works and products are not entities.

Sentence: {text}

List the errors:
- wrong_type: marked spans with a wrong type (give the correct type);
- wrong_span: marked spans with wrong boundaries, i.e. missing or extra words (give the correct surface);
- not_entity: marked spans that are not a PER/ORG/LOC name at all;
- missed: PER/ORG/LOC names in the sentence that are not marked (give the type).
Output only one JSON object: {{"wrong_type": [{{"text": "...", "type": "..."}}], "wrong_span": [{{"text": "...", "correct": "..."}}], "not_entity": ["..."], "missed": [{{"text": "...", "type": "..."}}], "reason": "<short English>"}}"""

QUERY_PROMPT = """You check a machine translation of a search query for a cross-lingual retrieval evaluation set.
Source query ({src}): {source}
Translation ({tgt}): {candidate}

Verdict:
- ok: fluent {tgt} that keeps the full meaning, names, numbers and dates of the source (minor style issues are fine);
- fix: meaning, a name or a number is wrong, something is missing or added, or it is not fluent {tgt}; give the corrected translation;
- drop: the source query is itself broken or meaningless.
Output only one JSON object: {{"verdict": "ok" | "fix" | "drop", "fixed_text": "<corrected {tgt} translation when fix, else empty>", "reason": "<short English>"}}"""

NLI_PROMPT = """You check a Kazakh machine translation of an English natural language inference pair (XNLI test).
English premise: {en_premise}
English hypothesis: {en_hypothesis}
Gold label of the English pair: {label}
Kazakh premise: {premise}
Kazakh hypothesis: {hypothesis}

Verdict:
- ok: both Kazakh sentences are faithful, fluent Kazakh and the gold label holds for the Kazakh pair;
- fix: a Kazakh sentence must be corrected for the label to hold or for fluency; give the corrected sentence(s), empty for a sentence that is fine;
- drop: the label cannot hold for any faithful translation, or the English pair is itself broken.
Output only one JSON object: {{"verdict": "ok" | "fix" | "drop", "fixed_premise": "", "fixed_hypothesis": "", "reason": "<short English>"}}"""


def machine_dir(name: str) -> Path:
    """Where the machine signals, labels and the review page of a sheet live (`data/review`, DVC `review.dvc`);
    the eval sheet's directory is a `build_eval` output whose every CSV is read as a review sheet."""
    d = ROOT / MACHINE_DIRS[name]
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_sheet(name: str) -> pl.DataFrame:
    path, _ = SHEETS[name]
    return pl.read_csv(ROOT / path, infer_schema_length=0, encoding="utf8-lossy")


def parse_json(text: str | None) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) else None


def query_langs(direction: str) -> tuple[str, str]:
    """A query translation in direction `q-d` translates the original query (document language d) into q."""
    q, d = direction.split("-")
    return d, q


def en_nli(root: Path) -> dict[str, tuple[str, str]]:
    """English XNLI test pair by example id (`<split>/<row>`), the source of the Kazakh translation."""
    en = pl.read_parquet(sorted(str(p) for p in (root / "eval/nli/en").glob("test-*.parquet")))
    return {
        pid.removeprefix("xnli/en/"): (p, h)
        for pid, p, h in en.select("pair_id", "premise", "hypothesis").iter_rows()
    }


# ----------------------------------------------------------------------------- prompts


def build_prompts(root: Path) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    q = read_sheet("quality")
    for r in q.iter_rows(named=True):
        user = QUALITY_PROMPT.format(
            lang=NAMES[r["review_lang"]], chars=len(r["text"] or ""), text=r["text"] or ""
        )
        rows.append({"id": f"judge/quality/{r['doc_id']}", "system": "", "user": user, "max_tokens": 160})
    n = read_sheet("ner")
    for r in n.iter_rows(named=True):
        user = NER_PROMPT.format(lang=NAMES[r["lang"]], text=r["text"])
        rows.append({"id": f"judge/ner/{r['sent_id']}", "system": "", "user": user, "max_tokens": 300})
    e = read_sheet("eval")
    en = en_nli(root)
    labels = {"entailment": "entailment", "neutral": "neutral", "contradiction": "contradiction"}
    for r in e.iter_rows(named=True):
        if r["kind"] == "query_translation":
            src, tgt = query_langs(r["direction"])
            user = QUERY_PROMPT.format(
                src=NAMES[src], tgt=NAMES[tgt], source=r["source_text"], candidate=r["candidate_text"]
            )
        else:
            ep, eh = en.get(r["item_id"].removeprefix("nli/xnli/kk/"), ("", ""))
            user = NLI_PROMPT.format(en_premise=ep, en_hypothesis=eh, label=labels[r["label"]],
                                     premise=r["source_text"], hypothesis=r["candidate_text"])  # fmt: skip
        rows.append({"id": f"judge/eval/{r['item_id']}", "system": "", "user": user, "max_tokens": 300})
    return pl.DataFrame(rows).with_columns(temperature=pl.lit(0.0))


# ----------------------------------------------------------------------------- signals


def quality_signals(params: dict[str, Any]) -> pl.DataFrame:
    from trifact.ingestion.langdetect import LangId, model_path

    spec = params["curation"]["langid"]
    df = read_sheet("quality")
    texts = [(t or "").replace("\n", " ") for t in df["text"].to_list()]
    out = df.select("doc_id")
    for name in ("glotlid", "lid176"):
        m = LangId(model_path(name, spec["models"][name], ROOT / spec["cache"]), name)
        langs, scores = m.predict(texts)
        out = out.with_columns(pl.Series(f"{name}_lang", langs), pl.Series(f"{name}_score", scores).round(3))
    letters = df["text"].fill_null("").str.count_matches(r"\p{L}")
    kk = df["text"].fill_null("").str.count_matches(f"[{KK_LETTERS}]")
    return out.with_columns(kk_letter_share=(kk / letters.clip(1)).round(4))


_EMB: list[Any] = []


def embed_cos(a: list[str], b: list[str]) -> list[float]:
    """bge-m3 dense cosine: a translation-independent cross-lingual similarity (NLLB did the translation)."""
    from sentence_transformers import SentenceTransformer

    if not _EMB:
        _EMB.append(SentenceTransformer("BAAI/bge-m3", device="cpu"))
    m = _EMB[0]
    ea = m.encode(a, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
    eb = m.encode(b, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
    return [round(float(x), 4) for x in (ea * eb).sum(1)]


def nli_probs(premises: list[str], hyps: list[str], model: str, revision: str) -> list[dict[str, float]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    mdl = AutoModelForSequenceClassification.from_pretrained(model, revision=revision).eval()
    names = [mdl.config.id2label[i].lower() for i in range(mdl.config.num_labels)]
    out: list[dict[str, float]] = []
    with torch.inference_mode():
        for i in range(0, len(premises), 32):
            enc = tok(
                premises[i : i + 32],
                hyps[i : i + 32],
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors="pt",
            )
            p = mdl(**enc).logits.softmax(-1).tolist()
            out += [dict(zip(names, row, strict=True)) for row in p]
    return out


def digits(s: str) -> list[str]:
    return sorted(re.findall(r"\d+", s or ""))


def eval_signals(params: dict[str, Any], root: Path) -> pl.DataFrame:
    from trifact.ingestion.langdetect import LangId, model_path

    df = read_sheet("eval")
    spec = params["curation"]["langid"]
    lid = LangId(model_path("lid176", spec["models"]["lid176"], ROOT / spec["cache"]), "lid176")
    cand_lang, _ = lid.predict([(t or "").replace("\n", " ") for t in df["candidate_text"].to_list()])
    src, cand = df["source_text"].fill_null("").to_list(), df["candidate_text"].fill_null("").to_list()
    is_q = df["kind"] == "query_translation"
    # translation similarity: query items compare source and translation, NLI items the Kazakh hypothesis with
    # the English one (the premise is checked by the NLI signals)
    en = en_nli(root)
    en_pairs = [en.get(i.removeprefix("nli/xnli/kk/"), ("", "")) for i in df["item_id"].to_list()]
    a = [s if q else eh for s, q, (_, eh) in zip(src, is_q.to_list(), en_pairs, strict=True)]
    cos_cand = embed_cos(a, cand)  # query: source vs translation; NLI: English vs Kazakh hypothesis
    nli_idx = [i for i, q in enumerate(is_q.to_list()) if not q]
    cos_prem = dict(
        zip(nli_idx, embed_cos([en_pairs[i][0] for i in nli_idx], [src[i] for i in nli_idx]), strict=True)
    )
    out = df.select("item_id", "kind").with_columns(
        cand_lang=pl.Series(cand_lang),
        cos_source=pl.Series(cos_cand),
        cos_premise=pl.Series([cos_prem.get(i) for i in range(df.height)], dtype=pl.Float64),
        # query: source vs translation; NLI: each English sentence vs its Kazakh translation
        digits_kept=pl.Series(
            [
                digits(s) == digits(c) if q else digits(ep) == digits(s) and digits(eh) == digits(c)
                for s, c, q, (ep, eh) in zip(src, cand, is_q.to_list(), en_pairs, strict=True)
            ]
        ),
        len_ratio=pl.Series([round(len(c) / max(len(s), 1), 3) for s, c in zip(src, cand, strict=True)]),
    )
    mt = pl.scan_parquet(str(root / "task/mt/qa/*/translations-*.parquet")).select(
        source_text="text", candidate_text="translation", chrf_back="chrf_back"
    )
    out = out.join(
        df.select("item_id", "source_text", "candidate_text")
        .join(
            mt.collect().unique(["source_text", "candidate_text"]),
            on=["source_text", "candidate_text"],
            how="left",
        )
        .select("item_id", "chrf_back"),
        on="item_id",
        how="left",
    )
    nli = df.filter(~is_q)
    m, rev = params["verification"]["nli_job"]["model"], params["verification"]["nli_job"]["revision"]
    pk = nli_probs(nli["source_text"].to_list(), nli["candidate_text"].to_list(), m, rev)
    ep = [en.get(i.removeprefix("nli/xnli/kk/"), ("", "")) for i in nli["item_id"].to_list()]
    pe = nli_probs([p for p, _ in ep], [h for _, h in ep], m, rev)
    top = lambda d: max(d, key=d.get)
    sig = nli.select("item_id").with_columns(
        nli_kk=pl.Series([top(d) for d in pk]), nli_kk_p=pl.Series([round(d[r], 3) for d, r in zip(pk, nli["label"].to_list(), strict=True)]),
        nli_en=pl.Series([top(d) for d in pe]),
    )  # fmt: skip
    return out.join(sig, on="item_id", how="left")


SPAN = re.compile(r"\[([^\[\]|]+)\|(PER|ORG|LOC)\]")


def plain_spans(marked: str) -> tuple[str, list[tuple[int, int, str]]]:
    """Text without the markup and the character spans of the marked entities."""
    out, spans, pos = [], [], 0
    for m in SPAN.finditer(marked):
        out.append(marked[pos : m.start()])
        a = sum(len(x) for x in out)
        out.append(m[1])
        spans.append((a, a + len(m[1]), m[2]))
        pos = m.end()
    out.append(marked[pos:])
    return "".join(out), spans


def ner_signals() -> pl.DataFrame:
    """GLiNER multi v2.1 (zero-shot, not trained on our silver) against the silver spans of each sentence."""
    from gliner import GLiNER

    df = read_sheet("ner")
    model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
    types = {"person": "PER", "organization": "ORG", "location": "LOC"}
    rows = []
    for sid, marked in df.select("sent_id", "text").iter_rows():
        text, silver = plain_spans(marked)
        pred = [
            (e["start"], e["end"], types[e["label"]])
            for e in model.predict_entities(text, list(types), threshold=0.5)
        ]
        overlap = lambda a, b: a[0] < b[1] and b[0] < a[1]
        exact = sum(1 for s in silver if any(s == p for p in pred))
        type_diff = [text[s[0] : s[1]] for s in silver if any(overlap(s, p) and s[2] != p[2] for p in pred)]
        bound = [
            text[s[0] : s[1]]
            for s in silver
            if any(overlap(s, p) and s[2] == p[2] and s[:2] != p[:2] for p in pred)
        ]
        silver_only = [text[s[0] : s[1]] for s in silver if not any(overlap(s, p) for p in pred)]
        gliner_only = [f"{text[p[0] : p[1]]}|{p[2]}" for p in pred if not any(overlap(s, p) for s in silver)]
        rows.append({"sent_id": sid, "gliner_exact": exact, "gliner_type_diff": "; ".join(type_diff),
                     "gliner_boundary_diff": "; ".join(bound), "silver_only": "; ".join(silver_only),
                     "gliner_only": "; ".join(gliner_only)})  # fmt: skip
    return pl.DataFrame(rows)


# ----------------------------------------------------------------------------- import


def teacher_answers(out: Path) -> dict[str, str]:
    gen = pl.read_parquet(sorted(str(p) for p in (out / "out").glob("*.parquet")))
    return dict(gen.filter(pl.col("id").str.starts_with("judge/")).select("id", "text").iter_rows())


def as_list(x: Any) -> list[Any]:
    return x if isinstance(x, list) else []


def import_quality(ans: dict[str, str], sig: pl.DataFrame) -> pl.DataFrame:
    df = read_sheet("quality").select("doc_id", "review_lang", "status", "stratum")
    rows = []
    for doc_id in df["doc_id"].to_list():
        d = parse_json(ans.get(f"judge/quality/{doc_id}")) or {}
        rows.append({"doc_id": doc_id, "judge_label": d.get("label") if d.get("label") in ("good", "bad") else None,
                     "judge_category": str(d.get("category") or ""), "judge_language": str(d.get("language") or ""),
                     "judge_reason": str(d.get("reason") or "")})  # fmt: skip
    j = df.join(pl.DataFrame(rows), on="doc_id").join(sig, on="doc_id", how="left")
    # signal opinion: the language is right by GlotLID (independent of the lid.176 filters)
    sig_ok = (pl.col("glotlid_lang") == pl.col("review_lang")) & (pl.col("glotlid_score") >= 0.5)
    kk_doubt = (pl.col("review_lang") == "kk") & (pl.col("kk_letter_share") < 0.05)
    return (
        j.with_columns(
            machine_label=pl.col("judge_label"),
            signal_language_ok=sig_ok & ~kk_doubt,
        )
        .with_columns(
            # GlotLID only speaks about the language: a `good` needs it to agree, a `wrong_language` needs it to object
            machine_agree=pl.when(pl.col("judge_label").is_null())
            .then(pl.lit(False))
            .when(pl.col("judge_label") == "good")
            .then(pl.col("signal_language_ok"))
            .when(pl.col("judge_category") == "wrong_language")
            .then(~pl.col("signal_language_ok"))
            .otherwise(pl.lit(True)),
            # the filters decided too: disagreement of the judge with the filter decision is worth a human look
            filter_disagree=((pl.col("status") == "kept") & (pl.col("judge_label") == "bad"))
            | ((pl.col("status") == "dropped") & (pl.col("judge_label") == "good")),
        )
        .with_columns(
            priority=pl.when(
                pl.col("judge_label").is_null() | ~pl.col("machine_agree") | pl.col("filter_disagree")
            )
            .then(1)
            .otherwise(2)
        )
    )


def import_ner(ans: dict[str, str], sig: pl.DataFrame) -> pl.DataFrame:
    df = read_sheet("ner").select("sent_id", "lang", "entities")
    rows = []
    for sid in df["sent_id"].to_list():
        d = parse_json(ans.get(f"judge/ner/{sid}"))
        if d is None:
            rows.append({"sent_id": sid, "judge_ok": False})
            continue
        wt, ws, ne, mi = (as_list(d.get(k)) for k in ("wrong_type", "wrong_span", "not_entity", "missed"))
        fmt = lambda xs, k: "; ".join(
            f"{x.get('text', '')}->{x.get(k, '')}" if isinstance(x, dict) else str(x) for x in xs
        )
        rows.append({"sent_id": sid, "judge_ok": True, "judge_wrong_type": len(wt), "judge_wrong_span": len(ws),
                     "judge_not_entity": len(ne), "judge_missed": len(mi), "judge_wrong_type_items": fmt(wt, "type"),
                     "judge_wrong_span_items": fmt(ws, "correct"), "judge_not_entity_items": fmt(ne, ""),
                     "judge_missed_items": fmt(mi, "type"), "judge_reason": str(d.get("reason") or "")})  # fmt: skip
    j = df.join(pl.DataFrame(rows, infer_schema_length=None), on="sent_id").join(
        sig, on="sent_id", how="left"
    )
    judge_err = pl.sum_horizontal(
        pl.col("judge_wrong_type", "judge_wrong_span", "judge_not_entity", "judge_missed").fill_null(0)
    )
    gliner_err = (
        (pl.col("gliner_type_diff").fill_null("") != "")
        | (pl.col("gliner_boundary_diff").fill_null("") != "")
        | (pl.col("gliner_only").fill_null("") != "")
    )
    return j.with_columns(
        machine_errors=judge_err,
        machine_agree=pl.col("judge_ok") & ((judge_err > 0) == gliner_err),
    ).with_columns(
        priority=pl.when(~pl.col("machine_agree") | (pl.col("machine_errors") > 0)).then(1).otherwise(2)
    )


def import_eval(ans: dict[str, str], sig: pl.DataFrame) -> pl.DataFrame:
    df = read_sheet("eval").select("item_id", "kind", "direction", "label")
    rows = []
    for iid in df["item_id"].to_list():
        d = parse_json(ans.get(f"judge/eval/{iid}")) or {}
        v = d.get("verdict") if d.get("verdict") in ("ok", "fix", "drop") else None
        rows.append({"item_id": iid, "judge_verdict": v, "judge_fixed_text": str(d.get("fixed_text") or d.get("fixed_hypothesis") or ""),
                     "judge_fixed_source": str(d.get("fixed_premise") or ""), "judge_reason": str(d.get("reason") or "")})  # fmt: skip
    j = df.join(pl.DataFrame(rows), on="item_id").join(sig.drop("kind"), on="item_id", how="left")
    is_q = pl.col("kind") == "query_translation"
    tgt = pl.col("direction").str.split("-").list.first()
    # signal opinion: the translation is plausible by similarity, language and digits; NLI items also need the
    # label to be predicted on the Kazakh pair whenever it is predicted on the English pair
    sig_ok = (
        (pl.col("cos_source") >= 0.75)
        & pl.col("digits_kept")
        & pl.when(is_q).then(pl.col("cand_lang") == tgt).otherwise(pl.lit(True))
        & pl.when(is_q).then(pl.lit(True)).otherwise((pl.col("nli_en") != pl.col("label")) | (pl.col("nli_kk") == pl.col("label")))
    )  # fmt: skip
    return (
        j.with_columns(signal_ok=sig_ok)
        .with_columns(
            machine_agree=pl.col("judge_verdict").is_not_null()
            & ((pl.col("judge_verdict") == "ok") == pl.col("signal_ok")),
        )
        .with_columns(
            priority=pl.when(~pl.col("machine_agree") | (pl.col("judge_verdict") != "ok"))
            .then(1)
            .otherwise(2)
        )
    )


def import_all(teacher_out: Path, params: dict[str, Any]) -> dict[str, Any]:
    ans = teacher_answers(teacher_out)
    labeler = f"machine: judge {params['teacher']['model']} (llama.cpp {params['teacher']['serve']['file']}); signals"
    summary: dict[str, Any] = {}
    for name, fn in (("quality", import_quality), ("ner", import_ner), ("eval", import_eval)):
        d = machine_dir(name)
        sig = pl.read_parquet(d / SIGNALS)
        out = fn(ans, sig).with_columns(labeler=pl.lit(labeler))
        out.write_csv(d / LABELS)
        summary[name] = {"items": out.height, "judged": int(out["machine_agree"].is_not_null().sum()),
                         "priority_1": int((out["priority"] == 1).sum())}  # fmt: skip
        log.info("%s: %s -> %s", name, summary[name], d / LABELS)
    return summary


# ----------------------------------------------------------------------------- report


def wilson(k: int, n: int) -> list[float]:
    from trifact.evaluation.review_quality import wilson as w

    return w(k, n)


def report(params: dict[str, Any]) -> dict[str, Any]:
    """Machine-only estimates (`human: false`): judge opinion per stratum and agreement of the two opinions.
    Where a human label exists for the same item, the judge's agreement with it is reported (ADR-0022)."""
    res: dict[str, Any] = {
        "human": False,
        "note": "machine pre-labels, not a human review; uncalibrated judge",
    }
    d = machine_dir("quality")
    q = pl.read_csv(d / LABELS, infer_schema_length=10000)
    qs: dict[str, Any] = {}
    for (lang, status), g in q.group_by("review_lang", "status"):
        k = int((g["judge_label"] == ("good" if status == "kept" else "bad")).sum())
        n = int(g["judge_label"].is_not_null().sum())
        qs.setdefault(lang, {})[status] = {"n": n, "judge_agrees_with_filter": round(k / n, 4) if n else None,
                                           "ci95": wilson(k, n),
                                           "categories": dict(g.group_by("judge_category").len().sort("len", descending=True).iter_rows())}  # fmt: skip
        qs[lang][f"{status}_by_stratum"] = {
            s: round(float((gg["judge_label"] == ("good" if status == "kept" else "bad")).mean() or 0), 3)
            for (s,), gg in g.group_by("stratum")
        }
    res["quality"] = qs
    human = d / "labels.csv"
    if human.exists():
        h = pl.read_csv(human, infer_schema_length=0).select("doc_id", human_label="label")
        j = q.join(h, on="doc_id")
        res["quality_judge_vs_human"] = {
            "n": j.height,
            "agreement": round(float((j["judge_label"] == j["human_label"]).mean() or 0), 4),
        }
    d = machine_dir("ner")
    n = pl.read_csv(d / LABELS, infer_schema_length=10000)
    ns: dict[str, Any] = {}
    for (lang,), g in n.group_by("lang"):
        ents = int(g["entities"].cast(pl.Int64).sum())
        wrong = int(
            g.select(
                pl.sum_horizontal(
                    pl.col("judge_wrong_type", "judge_wrong_span", "judge_not_entity").fill_null(0)
                )
            )
            .sum()
            .item()
        )
        missed = int(g["judge_missed"].fill_null(0).sum())
        ns[lang] = {"sentences": g.height, "entities": ents,
                    "judge_precision": round(1 - wrong / ents, 4) if ents else None,
                    "judge_missed_per_sentence": round(missed / g.height, 3),
                    "judge_recall": round((ents - wrong) / (ents - wrong + missed), 4) if ents else None,
                    "by_error": {k: int(g[f"judge_{k}"].fill_null(0).sum()) for k in ("wrong_type", "wrong_span", "not_entity", "missed")},
                    "gliner_exact_share": round(int(g["gliner_exact"].sum()) / ents, 4) if ents else None,
                    "sentences_with_gliner_only_span": int((g["gliner_only"].fill_null("") != "").sum()),
                    "machine_agree": round(float(g["machine_agree"].mean() or 0), 4)}  # fmt: skip
    res["ner"] = ns
    d = machine_dir("eval")
    e = pl.read_csv(d / LABELS, infer_schema_length=10000)
    es: dict[str, Any] = {}
    for (kind, direction), g in e.group_by("kind", "direction"):
        v = g["judge_verdict"].value_counts()
        es[f"{kind}/{direction}"] = {"n": g.height, "verdicts": dict(v.iter_rows()),
                                    "signal_ok": round(float(g["signal_ok"].mean() or 0), 4),
                                    "machine_agree": round(float(g["machine_agree"].mean() or 0), 4),
                                    "cos_source_p10": round(float(g["cos_source"].quantile(0.1) or 0), 3)}  # fmt: skip
    res["eval"] = es
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prompts", "signals", "import", "report"])
    ap.add_argument("--out", type=Path, help="prompts: directory for prompts-judge.parquet")
    ap.add_argument("--teacher-out", type=Path, help="import: downloaded Kaggle teacher output")
    ap.add_argument("--sheet", choices=list(SHEETS), action="append", help="signals: only these sheets")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    params = load_params()
    root = (ROOT / params["lake"]["root"]).resolve()
    t0 = time.perf_counter()
    if args.command == "prompts":
        p = build_prompts(root)
        args.out.mkdir(parents=True, exist_ok=True)
        p.write_parquet(args.out / "prompts-judge.parquet")
        print(f"{p.height} judge prompts -> {args.out}")
        return 0
    if args.command == "signals":
        with ResourceMonitor() as mon:
            for name in args.sheet or list(SHEETS):
                t = time.perf_counter()
                sig = {"quality": lambda: quality_signals(params), "ner": ner_signals,
                       "eval": lambda: eval_signals(params, root)}[name]()  # fmt: skip
                sig.write_parquet(machine_dir(name) / SIGNALS)
                log.info("%s signals: %d items, %.1f s", name, sig.height, time.perf_counter() - t)
        log.info("peak RSS %.0f MB", mon.peak_rss / 2**20)
        return 0
    if args.command == "import":
        s = import_all(args.teacher_out, params)
        print(json.dumps(s))
        return 0
    res = report(params)
    res["seconds"] = round(time.perf_counter() - t0, 1)
    write_report(ROOT / "reports" / "prelabel.json", res)
    metrics: dict[str, float] = {}
    for lang, r in res["quality"].items():
        for st in ("kept", "dropped"):
            if r.get(st, {}).get("judge_agrees_with_filter") is not None:
                metrics[f"quality_{lang}_{st}_judge_agree"] = r[st]["judge_agrees_with_filter"]
    for lang, r in res["ner"].items():
        metrics[f"ner_{lang}_judge_precision"] = r["judge_precision"] or 0.0
        metrics[f"ner_{lang}_judge_recall"] = r["judge_recall"] or 0.0
    log_mlflow("review", "prelabel", {"human": False, "code_version": git_sha(ROOT)}, metrics)
    print(json.dumps(res, ensure_ascii=False, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
