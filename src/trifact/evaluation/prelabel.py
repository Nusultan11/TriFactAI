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


def _extract_array_items(s: str, key: str) -> list[Any]:
    """Extract complete JSON elements from a possibly truncated top-level array field."""
    m = re.search(rf'"{re.escape(key)}"\s*:\s*\[', s)
    if not m:
        return []
    pos = m.end()
    dec = json.JSONDecoder()
    items: list[Any] = []
    n = len(s)
    while pos < n:
        while pos < n and s[pos] in " \t\r\n,":
            pos += 1
        if pos >= n or s[pos] == "]":
            break
        try:
            obj, end = dec.raw_decode(s, pos)
            items.append(obj)
            pos = end
        except json.JSONDecodeError:
            break
    return items


TOP_KEYS = {"label", "verdict", "wrong_type", "wrong_span", "not_entity", "missed"}


def parse_json(text: str | None) -> dict[str, Any] | None:
    """Parse judge JSON, preferring the last complete top-level object and recovering truncated outputs."""
    if not text:
        return None
    # If the model emitted an explicit "Revised JSON" block, parse from that block first.
    rev = text.rfind("Revised JSON")
    if rev != -1:
        revised = parse_json(text[rev + len("Revised JSON") :])
        if revised is not None:
            return revised
    # Prefer the last complete TOP-LEVEL JSON object (must contain a known judge key, not an inner entity dict).
    starts = [m.start() for m in re.finditer(r"\{", text)]
    dec = json.JSONDecoder()
    for pos in reversed(starts):
        try:
            obj, _ = dec.raw_decode(text, pos)
            if isinstance(obj, dict) and (set(obj.keys()) & TOP_KEYS):
                return obj
        except json.JSONDecodeError:
            continue
    if not starts:
        return None
    s = text[starts[0] :]
    idx = s.rfind('"reason"')
    if idx != -1:
        prefix = s[:idx].rstrip()
        prefix = prefix.removesuffix(",")
        rm = re.search(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)', s[idx:])
        reason = rm.group(1) if rm else ""
        try:
            obj = json.loads(prefix + "}")
            if isinstance(obj, dict) and (set(obj.keys()) & TOP_KEYS):
                obj["reason"] = reason
                obj["_truncated"] = True
                return obj
        except json.JSONDecodeError:
            pass
    # Fallback 1: eval verdict object truncated mid-field
    vm = re.search(r'"verdict"\s*:\s*"(ok|fix|drop)"', s)
    if vm:
        out: dict[str, Any] = {"verdict": vm.group(1), "_truncated": True}
        for k in ("fixed_text", "fixed_hypothesis", "fixed_premise", "reason"):
            km = re.search(rf'"{k}"\s*:\s*"((?:[^"\\]|\\.)*)', s)
            if km:
                out[k] = km.group(1)
        return out
    # Fallback 2: NER object truncated mid-array
    if any(f'"{k}"' in s for k in ("wrong_type", "wrong_span", "not_entity", "missed")):
        out = {k: _extract_array_items(s, k) for k in ("wrong_type", "wrong_span", "not_entity", "missed")}
        rm = re.search(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)', s)
        out["reason"] = rm.group(1) if rm else ""
        out["_truncated"] = True
        return out
    return None


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


CORE_TYPES = {"PER", "ORG", "LOC"}
FULL_DOC_DROP_REASONS = {"dup_lines", "dup_ngrams", "bad_chars", "few_stopwords", "bullets", "ellipsis"}
SIGNAL_COLS = {
    "quality": ["doc_id", "glotlid_lang", "glotlid_score", "lid176_lang", "lid176_score", "kk_letter_share"],
    "ner": [
        "sent_id",
        "gliner_exact",
        "gliner_type_diff",
        "gliner_boundary_diff",
        "silver_only",
        "gliner_only",
    ],
    "eval": [
        "item_id",
        "kind",
        "cand_lang",
        "cos_source",
        "cos_premise",
        "digits_kept",
        "len_ratio",
        "chrf_back",
        "nli_kk",
        "nli_kk_p",
        "nli_en",
    ],
}


def teacher_answers(out: Path) -> dict[str, str]:
    if out.is_file():
        files = [str(out)]
    elif (out / "out").is_dir():
        files = sorted(str(p) for p in (out / "out").glob("*.parquet"))
    else:
        files = sorted(str(p) for p in out.glob("*.parquet"))
    gen = pl.read_parquet(files)
    return dict(gen.filter(pl.col("id").str.starts_with("judge/")).select("id", "text").iter_rows())


def as_list(x: Any) -> list[Any]:
    return x if isinstance(x, list) else []


def _strip_span_markup(s: str) -> str:
    m = SPAN.fullmatch(s.strip())
    return m.group(1).strip() if m else s.strip()


def _sanitize_ner_judge(
    marked_text: str, d: dict[str, Any]
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str], list[dict[str, str]], int]:
    """Validate NER judge errors against marked spans and ADR-0006 core types (PER/ORG/LOC).

    Prevents non-core types (EVENT/WORK/DATE), already-marked spans in `missed`, no-op fixes,
    and double-counting the same marked span across multiple error buckets.
    """
    plain, spans = plain_spans(marked_text)
    plain_low = plain.lower()
    # Track remaining occurrences of each marked (surface_lower, type) so a span cannot be counted twice
    avail: list[tuple[str, str]] = [(plain[a:b].lower(), ty) for a, b, ty in spans]
    marked_surfaces = {s for s, _ in avail}

    def consume_marked(surface: str) -> tuple[bool, str | None]:
        low = surface.lower()
        for idx, (s, ty) in enumerate(avail):
            if s == low:
                avail.pop(idx)
                return True, ty
        return False, None

    sanitized = 0
    wt_clean: list[dict[str, str]] = []
    ws_clean: list[dict[str, str]] = []
    ne_clean: list[str] = []
    mi_clean: list[dict[str, str]] = []

    # 1. not_entity first (highest precedence for non-entities)
    for x in as_list(d.get("not_entity")):
        raw_txt = x.get("text", "") if isinstance(x, dict) else str(x)
        tx = _strip_span_markup(str(raw_txt))
        if not tx:
            sanitized += 1
            continue
        ok, _ = consume_marked(tx)
        if ok:
            ne_clean.append(tx)
        else:
            sanitized += 1

    # 2. wrong_type (reclassify non-core target types like O/DATE/EVENT into not_entity)
    for x in as_list(d.get("wrong_type")):
        if not isinstance(x, dict):
            sanitized += 1
            continue
        tx = _strip_span_markup(str(x.get("text") or ""))
        ty = str(x.get("type") or "").strip().upper()
        if not tx:
            sanitized += 1
            continue
        ok, cur_ty = consume_marked(tx)
        if not ok:
            sanitized += 1
            continue
        if ty not in CORE_TYPES:
            ne_clean.append(tx)
            sanitized += 1
        elif ty == cur_ty:
            # Put back if judge suggested the same type it already had
            avail.append((tx.lower(), cur_ty or ""))
            sanitized += 1
        else:
            wt_clean.append({"text": tx, "type": ty})

    # 3. wrong_span
    for x in as_list(d.get("wrong_span")):
        if not isinstance(x, dict):
            sanitized += 1
            continue
        tx = _strip_span_markup(str(x.get("text") or ""))
        co = _strip_span_markup(str(x.get("correct") or ""))
        if not tx or not co or tx.lower() == co.lower():
            sanitized += 1
            continue
        ok, _ = consume_marked(tx)
        if ok:
            ws_clean.append({"text": tx, "correct": co})
        else:
            sanitized += 1

    # 4. missed (only valid PER/ORG/LOC present in plain text and not already marked)
    seen_missed: set[str] = set()
    for x in as_list(d.get("missed")):
        if not isinstance(x, dict):
            sanitized += 1
            continue
        tx = _strip_span_markup(str(x.get("text") or ""))
        ty = str(x.get("type") or "").strip().upper()
        low = tx.lower()
        if (
            not tx
            or ty not in CORE_TYPES
            or low in marked_surfaces
            or low in seen_missed
            or low not in plain_low
        ):
            sanitized += 1
            continue
        seen_missed.add(low)
        mi_clean.append({"text": tx, "type": ty})

    return wt_clean, ws_clean, ne_clean, mi_clean, sanitized


def import_quality(ans: dict[str, str], sig: pl.DataFrame) -> pl.DataFrame:
    df = read_sheet("quality").select("doc_id", "review_lang", "status", "stratum", "text")
    rows = []
    for r in df.iter_rows(named=True):
        doc_id = r["doc_id"]
        d = parse_json(ans.get(f"judge/quality/{doc_id}")) or {}
        text = r["text"] or ""
        rows.append(
            {
                "doc_id": doc_id,
                "judge_label": d.get("label") if d.get("label") in ("good", "bad") else None,
                "judge_category": str(d.get("category") or ""),
                "judge_language": str(d.get("language") or ""),
                "judge_reason": str(d.get("reason") or ""),
                "judge_truncated": bool(d.get("_truncated", False)),
                "text_chars": len(text),
            }
        )
    j = df.drop("text").join(pl.DataFrame(rows), on="doc_id").join(sig, on="doc_id", how="left")
    sig_ok = (pl.col("glotlid_lang") == pl.col("review_lang")) & (pl.col("glotlid_score") >= 0.5)
    kk_doubt = (pl.col("review_lang") == "kk") & (pl.col("kk_letter_share") < 0.05)
    j = j.with_columns(
        machine_label=pl.col("judge_label"),
        signal_language_ok=sig_ok & ~kk_doubt,
    ).with_columns(
        machine_agree=pl.when(pl.col("judge_label").is_null())
        .then(pl.lit(False))
        .when(pl.col("judge_label") == "good")
        .then(pl.col("signal_language_ok"))
        .when(pl.col("judge_category") == "wrong_language")
        .then(~pl.col("signal_language_ok"))
        .otherwise(pl.lit(True)),
        filter_disagree=((pl.col("status") == "kept") & (pl.col("judge_label") == "bad"))
        | ((pl.col("status") == "dropped") & (pl.col("judge_label") == "good")),
    )
    unc_levels: list[str] = []
    unc_reasons: list[str] = []
    for r in j.iter_rows(named=True):
        reasons: list[str] = []
        if r["judge_label"] is None:
            reasons.append("judge_missing")
        if r["judge_truncated"]:
            reasons.append("judge_truncated")
        if not r["machine_agree"]:
            reasons.append("signal_disagrees")
        if r["filter_disagree"]:
            reasons.append("filter_disagrees")
        if r["review_lang"] == "kk" and (r["kk_letter_share"] or 0.0) < 0.05:
            reasons.append("kk_low_letter_share")
        if (
            r["status"] == "dropped"
            and r["stratum"] in FULL_DOC_DROP_REASONS
            and (r["text_chars"] or 0) >= 600
        ):
            reasons.append("preview_truncated_600c")
        if not r["machine_agree"] or r["judge_label"] is None:
            lvl = "high"
        elif r["filter_disagree"] or "preview_truncated_600c" in reasons or "kk_low_letter_share" in reasons:
            lvl = "medium"
        else:
            lvl = "low"
        unc_levels.append(lvl)
        unc_reasons.append("; ".join(reasons))
    return j.with_columns(
        uncertainty=pl.Series(unc_levels),
        uncertainty_reasons=pl.Series(unc_reasons),
    ).with_columns(
        priority=pl.when(
            pl.col("judge_label").is_null() | ~pl.col("machine_agree") | pl.col("filter_disagree")
        )
        .then(1)
        .otherwise(2)
    )


def import_ner(ans: dict[str, str], sig: pl.DataFrame) -> pl.DataFrame:
    df = read_sheet("ner").select("sent_id", "lang", "text", "entities")
    rows = []
    for r in df.iter_rows(named=True):
        sid = r["sent_id"]
        d = parse_json(ans.get(f"judge/ner/{sid}"))
        if d is None:
            rows.append(
                {
                    "sent_id": sid,
                    "judge_ok": False,
                    "judge_truncated": False,
                    "judge_sanitized": 0,
                    "judge_wrong_type": 0,
                    "judge_wrong_span": 0,
                    "judge_not_entity": 0,
                    "judge_missed": 0,
                    "judge_wrong_type_items": "",
                    "judge_wrong_span_items": "",
                    "judge_not_entity_items": "",
                    "judge_missed_items": "",
                    "judge_reason": "",
                }
            )
            continue
        wt, ws, ne, mi, san = _sanitize_ner_judge(r["text"], d)
        fmt = lambda xs, k: "; ".join(
            f"{x.get('text', '')}->{x.get(k, '')}" if isinstance(x, dict) else str(x) for x in xs
        )
        rows.append(
            {
                "sent_id": sid,
                "judge_ok": True,
                "judge_truncated": bool(d.get("_truncated", False)),
                "judge_sanitized": san,
                "judge_wrong_type": len(wt),
                "judge_wrong_span": len(ws),
                "judge_not_entity": len(ne),
                "judge_missed": len(mi),
                "judge_wrong_type_items": fmt(wt, "type"),
                "judge_wrong_span_items": fmt(ws, "correct"),
                "judge_not_entity_items": fmt(ne, ""),
                "judge_missed_items": fmt(mi, "type"),
                "judge_reason": str(d.get("reason") or ""),
            }
        )
    j = (
        df.drop("text")
        .join(pl.DataFrame(rows, infer_schema_length=None), on="sent_id")
        .join(sig, on="sent_id", how="left")
    )
    judge_err = pl.sum_horizontal(
        pl.col("judge_wrong_type", "judge_wrong_span", "judge_not_entity", "judge_missed").fill_null(0)
    )
    gliner_err = (
        (pl.col("gliner_type_diff").fill_null("") != "")
        | (pl.col("gliner_boundary_diff").fill_null("") != "")
        | (pl.col("gliner_only").fill_null("") != "")
    )
    j = j.with_columns(
        machine_errors=judge_err,
        machine_agree=pl.col("judge_ok") & ((judge_err > 0) == gliner_err),
    )
    unc_levels: list[str] = []
    unc_reasons: list[str] = []
    for r in j.iter_rows(named=True):
        reasons: list[str] = []
        if not r["judge_ok"]:
            reasons.append("judge_missing")
        if r["judge_truncated"]:
            reasons.append("judge_truncated")
        if (r["judge_sanitized"] or 0) > 0:
            reasons.append(f"judge_sanitized({r['judge_sanitized']})")
        if not r["machine_agree"]:
            reasons.append("gliner_disagrees")
        if (r["machine_errors"] or 0) > 0:
            reasons.append("errors_found")
        if (r["gliner_only"] or "").strip():
            reasons.append("gliner_only_candidates")
        if not r["judge_ok"] or not r["machine_agree"] or r["judge_truncated"]:
            lvl = "high"
        elif (r["machine_errors"] or 0) > 0 or (r["judge_sanitized"] or 0) > 0:
            lvl = "medium"
        else:
            lvl = "low"
        unc_levels.append(lvl)
        unc_reasons.append("; ".join(reasons))
    return j.with_columns(
        uncertainty=pl.Series(unc_levels),
        uncertainty_reasons=pl.Series(unc_reasons),
    ).with_columns(
        priority=pl.when(
            ~pl.col("machine_agree") | (pl.col("machine_errors") > 0) | pl.col("judge_truncated")
        )
        .then(1)
        .otherwise(2)
    )


def import_eval(ans: dict[str, str], sig: pl.DataFrame) -> pl.DataFrame:
    df = read_sheet("eval").select("item_id", "kind", "direction", "label", "source_text", "candidate_text")
    rows = []
    for r in df.iter_rows(named=True):
        iid = r["item_id"]
        d = parse_json(ans.get(f"judge/eval/{iid}")) or {}
        v = d.get("verdict") if d.get("verdict") in ("ok", "fix", "drop") else None
        ft = str(d.get("fixed_text") or d.get("fixed_hypothesis") or "").strip()
        fs = str(d.get("fixed_premise") or "").strip()
        san = False
        if v == "fix":
            if r["kind"] == "query_translation":
                if not ft or ft == (r["candidate_text"] or "").strip():
                    v, ft, san = "ok", "", True
            else:
                ft_same = not ft or ft == (r["candidate_text"] or "").strip()
                fs_same = not fs or fs == (r["source_text"] or "").strip()
                if ft_same and fs_same:
                    v, ft, fs, san = "ok", "", "", True
        rows.append(
            {
                "item_id": iid,
                "judge_verdict": v,
                "judge_fixed_text": ft,
                "judge_fixed_source": fs,
                "judge_reason": str(d.get("reason") or ""),
                "judge_truncated": bool(d.get("_truncated", False)),
                "judge_sanitized": san,
            }
        )
    sig_clean = sig.drop("kind") if "kind" in sig.columns else sig
    j = (
        df.drop("source_text", "candidate_text")
        .join(pl.DataFrame(rows), on="item_id")
        .join(sig_clean, on="item_id", how="left")
    )
    is_q = pl.col("kind") == "query_translation"
    tgt = pl.col("direction").str.split("-").list.first()
    sig_ok = (
        (pl.col("cos_source") >= 0.75)
        & pl.col("digits_kept")
        & pl.when(is_q).then(pl.col("cand_lang") == tgt).otherwise(pl.lit(True))
        & pl.when(is_q)
        .then(pl.lit(True))
        .otherwise((pl.col("nli_en") != pl.col("label")) | (pl.col("nli_kk") == pl.col("label")))
    )
    j = j.with_columns(signal_ok=sig_ok).with_columns(
        machine_agree=pl.col("judge_verdict").is_not_null()
        & ((pl.col("judge_verdict") == "ok") == pl.col("signal_ok")),
    )
    unc_levels: list[str] = []
    unc_reasons: list[str] = []
    for r in j.iter_rows(named=True):
        reasons: list[str] = []
        if r["judge_verdict"] is None:
            reasons.append("judge_missing")
        if r["judge_truncated"]:
            reasons.append("judge_truncated")
        if r["judge_sanitized"]:
            reasons.append("noop_fix_normalized")
        if not r["machine_agree"]:
            reasons.append("signal_disagrees")
        if r["cos_source"] is not None and float(r["cos_source"]) < 0.75:
            reasons.append("low_cos_source")
        if r["chrf_back"] is not None and float(r["chrf_back"]) < 45.0:
            reasons.append("low_chrf_back")
        if r["digits_kept"] is False:
            reasons.append("digits_mismatch")
        if r["kind"] == "nli_translation" and r["nli_en"] == r["label"] and r["nli_kk"] != r["label"]:
            reasons.append("nli_label_mismatch")
        if r["judge_verdict"] in ("fix", "drop"):
            reasons.append(f"verdict_{r['judge_verdict']}")
        if r["judge_verdict"] is None or not r["machine_agree"] or r["judge_truncated"]:
            lvl = "high"
        elif r["judge_verdict"] in ("fix", "drop") or r["judge_sanitized"] or "low_chrf_back" in reasons:
            lvl = "medium"
        else:
            lvl = "low"
        unc_levels.append(lvl)
        unc_reasons.append("; ".join(reasons))
    return j.with_columns(
        uncertainty=pl.Series(unc_levels),
        uncertainty_reasons=pl.Series(unc_reasons),
    ).with_columns(
        priority=pl.when(
            ~pl.col("machine_agree")
            | (pl.col("judge_verdict") != "ok")
            | pl.col("judge_truncated")
            | pl.col("judge_sanitized")
        )
        .then(1)
        .otherwise(2)
    )


def _load_signals(name: str, d: Path) -> pl.DataFrame:
    if (d / SIGNALS).is_file():
        return pl.read_parquet(d / SIGNALS)
    if (d / LABELS).is_file():
        prev = pl.read_csv(d / LABELS, infer_schema_length=10000)
        cols = [c for c in SIGNAL_COLS[name] if c in prev.columns]
        return prev.select(cols)
    raise FileNotFoundError(f"missing {d / SIGNALS} (and no fallback {d / LABELS})")


def import_all(teacher_out: Path, params: dict[str, Any]) -> dict[str, Any]:
    ans = teacher_answers(teacher_out)
    labeler = (
        f"machine: judge {params['teacher']['model']} (llama.cpp {params['teacher']['serve']['file']}); "
        "sanitized + independent signals"
    )
    summary: dict[str, Any] = {}
    for name, fn in (("quality", import_quality), ("ner", import_ner), ("eval", import_eval)):
        d = machine_dir(name)
        sig = _load_signals(name, d)
        out = fn(ans, sig).with_columns(labeler=pl.lit(labeler))
        out.write_csv(d / LABELS)
        valid_col = {"quality": "judge_label", "ner": "judge_ok", "eval": "judge_verdict"}[name]
        judged = int(out[valid_col].sum()) if name == "ner" else int(out[valid_col].is_not_null().sum())
        summary[name] = {
            "items": out.height,
            "judged": judged,
            "truncated_recovered": int(out["judge_truncated"].sum()),
            "priority_1": int((out["priority"] == 1).sum()),
            "uncertainty": dict(out.group_by("uncertainty").len().iter_rows()),
        }
        log.info("%s: %s -> %s", name, summary[name], d / LABELS)
    return summary


# ----------------------------------------------------------------------------- report


def wilson(k: int, n: int) -> list[float]:
    from trifact.evaluation.review_quality import wilson as w

    return w(k, n)


def report(params: dict[str, Any]) -> dict[str, Any]:
    """Machine-only estimates (`human: false`): judge opinion per stratum and agreement of the two opinions.
    Where a human label exists for the same item, the judge's agreement with it is reported (ADR-0022)."""
    import numpy as np

    res: dict[str, Any] = {
        "human": False,
        "note": "machine pre-labels, not a human review; uncalibrated judge",
        "coverage": {},
    }
    d = machine_dir("quality")
    q = pl.read_csv(d / LABELS, infer_schema_length=10000)
    res["coverage"]["quality"] = {
        "items": q.height,
        "judged": int(q["judge_label"].is_not_null().sum()),
        "priority_1": int((q["priority"] == 1).sum()),
        "machine_agree": int(q["machine_agree"].sum()),
        "uncertainty": dict(q.group_by("uncertainty").len().iter_rows())
        if "uncertainty" in q.columns
        else {},
    }
    sample_path = d / "sample.parquet"
    stratum_sizes: dict[tuple[str, str, str], int] = {}
    if sample_path.is_file():
        sp = pl.read_parquet(sample_path, columns=["review_lang", "status", "stratum", "stratum_size"])
        for r in sp.unique(["review_lang", "status", "stratum"]).iter_rows(named=True):
            stratum_sizes[(r["review_lang"], r["status"], r["stratum"])] = int(r["stratum_size"])
    qs: dict[str, Any] = {}
    for (lang, status), g in sorted(
        q.group_by("review_lang", "status"), key=lambda x: (str(x[0][0]), str(x[0][1]))
    ):
        positive = "good" if status == "kept" else "bad"
        k = int((g["judge_label"] == positive).sum())
        n = int(g["judge_label"].is_not_null().sum())
        by_stratum: dict[str, Any] = {}
        pop_total = 0
        weighted_num = 0.0
        for (s,), gg in sorted(g.group_by("stratum"), key=lambda x: str(x[0][0])):
            sk = int((gg["judge_label"] == positive).sum())
            sn = int(gg["judge_label"].is_not_null().sum())
            share = round(sk / sn, 4) if sn else 0.0
            pop = stratum_sizes.get((str(lang), str(status), str(s)), 0)
            pop_total += pop
            weighted_num += share * pop
            by_stratum[str(s)] = {
                "n": sn,
                positive: sk,
                "share": share,
                "ci95": wilson(sk, sn),
                "population": pop,
            }
        weighted = round(weighted_num / pop_total, 4) if pop_total else None
        qs.setdefault(str(lang), {})[str(status)] = {
            "n": n,
            "judge_agrees_with_filter": round(k / n, 4) if n else None,
            "ci95": wilson(k, n),
            "weighted_share": weighted,
            "population": pop_total,
            "categories": dict(g.group_by("judge_category").len().sort("len", descending=True).iter_rows()),
            "by_stratum": by_stratum,
        }
        qs[str(lang)][f"{status}_by_stratum"] = {s: v["share"] for s, v in by_stratum.items()}
    for lang, lr in qs.items():
        if "kept" in lr and "dropped" in lr and lr["kept"]["weighted_share"] is not None:
            kept_pop = lr["kept"]["population"]
            kept_good = lr["kept"]["weighted_share"]
            lost = sum(v["population"] * (1.0 - v["share"]) for v in lr["dropped"]["by_stratum"].values())
            good_total = kept_pop * kept_good + lost
            lr["estimated_good_lost"] = round(lost / good_total, 4) if good_total else None
            lr["estimated_false_accept"] = round(1.0 - kept_good, 4)
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
    res["coverage"]["ner"] = {
        "items": n.height,
        "judged": int(n["judge_ok"].sum()),
        "truncated_recovered": int(n["judge_truncated"].sum()) if "judge_truncated" in n.columns else 0,
        "sanitized_entries": int(n["judge_sanitized"].sum()) if "judge_sanitized" in n.columns else 0,
        "priority_1": int((n["priority"] == 1).sum()),
        "machine_agree": int(n["machine_agree"].sum()),
        "uncertainty": dict(n.group_by("uncertainty").len().iter_rows())
        if "uncertainty" in n.columns
        else {},
    }
    rng = np.random.default_rng(int(params["seed"]))
    boot_n = int(params["evaluation"]["bootstrap_samples"])
    ns: dict[str, Any] = {}
    for (lang,), g in sorted(n.group_by("lang"), key=lambda x: str(x[0][0])):
        ents = int(g["entities"].cast(pl.Int64).sum())
        wt = int(g["judge_wrong_type"].fill_null(0).sum())
        ws = int(g["judge_wrong_span"].fill_null(0).sum())
        ne = int(g["judge_not_entity"].fill_null(0).sum())
        mi = int(g["judge_missed"].fill_null(0).sum())
        wrong = wt + ws + ne
        correct = max(0, ents - wrong)
        gold = correct + mi + wt + ws
        # Sentence-level bootstrap CI matching review_ner.py
        arr_e = g["entities"].cast(pl.Int64).to_numpy()
        arr_wt = g["judge_wrong_type"].fill_null(0).to_numpy()
        arr_ws = g["judge_wrong_span"].fill_null(0).to_numpy()
        arr_ne = g["judge_not_entity"].fill_null(0).to_numpy()
        arr_mi = g["judge_missed"].fill_null(0).to_numpy()
        bp, br = [], []
        for _ in range(boot_n):
            idx = rng.integers(0, g.height, g.height)
            be = int(arr_e[idx].sum())
            bwt, bws, bne, bmi = (
                int(arr_wt[idx].sum()),
                int(arr_ws[idx].sum()),
                int(arr_ne[idx].sum()),
                int(arr_mi[idx].sum()),
            )
            bc = max(0, be - bwt - bws - bne)
            bg = bc + bmi + bwt + bws
            bp.append(bc / be if be else 0.0)
            br.append(bc / bg if bg else 0.0)
        ns[str(lang)] = {
            "sentences": g.height,
            "entities": ents,
            "judge_precision": round(correct / ents, 4) if ents else None,
            "judge_precision_ci95_bootstrap": [
                round(float(np.quantile(bp, 0.025)), 4),
                round(float(np.quantile(bp, 0.975)), 4),
            ],
            "judge_missed_per_sentence": round(mi / g.height, 3),
            "judge_recall": round(correct / gold, 4) if gold else None,
            "judge_recall_ci95_bootstrap": [
                round(float(np.quantile(br, 0.025)), 4),
                round(float(np.quantile(br, 0.975)), 4),
            ],
            "by_error": {"wrong_type": wt, "wrong_span": ws, "not_entity": ne, "missed": mi},
            "sentences_without_errors": round(float((g["machine_errors"] == 0).mean() or 0), 4),
            "gliner_exact_share": round(int(g["gliner_exact"].sum()) / ents, 4) if ents else None,
            "sentences_with_gliner_only_span": int((g["gliner_only"].fill_null("") != "").sum()),
            "machine_agree": round(float(g["machine_agree"].mean() or 0), 4),
        }
    res["ner"] = ns
    d = machine_dir("eval")
    e = pl.read_csv(d / LABELS, infer_schema_length=10000)
    res["coverage"]["eval"] = {
        "items": e.height,
        "judged": int(e["judge_verdict"].is_not_null().sum()),
        "truncated_recovered": int(e["judge_truncated"].sum()) if "judge_truncated" in e.columns else 0,
        "sanitized_noop_fixes": int(e["judge_sanitized"].sum()) if "judge_sanitized" in e.columns else 0,
        "priority_1": int((e["priority"] == 1).sum()),
        "machine_agree": int(e["machine_agree"].sum()),
        "uncertainty": dict(e.group_by("uncertainty").len().iter_rows())
        if "uncertainty" in e.columns
        else {},
    }
    es: dict[str, Any] = {}
    for (kind, direction), g in sorted(
        e.group_by("kind", "direction"), key=lambda x: (str(x[0][0]), str(x[0][1]))
    ):
        v = g["judge_verdict"].value_counts()
        chrf_vals = g["chrf_back"].drop_nulls()
        entry: dict[str, Any] = {
            "n": g.height,
            "verdicts": dict(v.iter_rows()),
            "signal_ok": round(float(g["signal_ok"].mean() or 0), 4),
            "machine_agree": round(float(g["machine_agree"].mean() or 0), 4),
            "digits_kept_rate": round(float(g["digits_kept"].mean() or 0), 4),
            "cos_source_p10": round(float(g["cos_source"].quantile(0.1) or 0), 3),
            "cos_source_p50": round(float(g["cos_source"].quantile(0.5) or 0), 3),
        }
        if chrf_vals.len() > 0:
            entry["chrf_back_p10"] = round(float(chrf_vals.quantile(0.1) or 0), 2)
            entry["chrf_back_p50"] = round(float(chrf_vals.quantile(0.5) or 0), 2)
        if kind == "nli_translation":
            en_ok = g.filter(pl.col("nli_en") == pl.col("label"))
            entry["nli_kk_label_match"] = round(float((g["nli_kk"] == g["label"]).mean() or 0), 4)
            entry["nli_kk_given_en_match"] = (
                round(float((en_ok["nli_kk"] == en_ok["label"]).mean() or 0), 4) if en_ok.height else None
            )
        es[f"{kind}/{direction}"] = entry
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
