"""Multilingual rule-based DATE extractor for kk/ru/en (ADR-0006, ROADMAP 5.3).

Extracts calendar dates, years, ranges, centuries, and (optionally) relative date expressions
with exact character offsets, conforms to ``EntityExtractor``, and provides evaluation against
KazNERD ``DATE`` annotations plus curated RU/EN reference cases.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from trifact.core.params import ROOT
from trifact.core.schemas import Chunk, EntityMention, EntityType, Lang
from trifact.evaluation.eval_ner import extract_iob2_entities, span_error_taxonomy, strict_entity_metrics

log = logging.getLogger("trifact.extraction.dates")

# --------------------------------------------------------------------------- month lexicons

KK_MONTHS = r"қаңтар|ақпан|наурыз|сәуір|мамыр|маусым|шілде|тамыз|қыркүйек|қазан|қараша|желтоқсан"
KK_MONTH_SFX = (
    r"(?:да|де|та|те|дан|ден|тан|тен|ға|ге|қа|ке|дың|дің|тың|тің|ның|нің|ында|інде|ына|іне|ынан|інен|ы|і)?"
)

RU_MONTHS = (
    r"январ[ьяею]|феврал[ьяею]|март[аеу]?|апрел[ьяею]|ма[йяею]|июн[ьяею]|июл[ьяею]|"
    r"август[аеу]?|сентябр[ьяею]|октябр[ьяею]|ноябр[ьяею]|декабр[ьяею]"
)

EN_MONTHS = (
    r"January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?"
)

# --------------------------------------------------------------------------- compiled patterns

_COMMON_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # ISO YYYY-MM-DD
    (
        "iso_date",
        re.compile(
            r"(?<!\d)(?:1[0-9]{3}|20[0-9]{2}|21[0-9]{2})-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])(?!\d)"
        ),
    ),
    # DD.MM.YYYY or DD/MM/YYYY
    (
        "dmy_numeric",
        re.compile(
            r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])[./](?:0?[1-9]|1[0-2])[./](?:1[0-9]{3}|20[0-9]{2}|21[0-9]{2})(?!\d)"
        ),
    ),
    # Quarter + year: Q1 2024
    ("quarter_year", re.compile(r"\bQ[1-4]\s+(?:1[0-9]{3}|20[0-9]{2})\b", re.IGNORECASE)),
]

_KK_ABSOLUTE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # 2021 жылғы 16 желтоқсанда / 1991 жылдың 16 желтоқсанында
    (
        "kk_ymd",
        re.compile(
            rf"(?<!\d)(?:1[0-9]{{3}}|20[0-9]{{2}})(?:-(?:ші|шы|інші|ыншы|сінші|сыншы))?\s+"
            rf"жыл(?:ғы|дың|ы|да)?\s+(?:[1-2]?\d|3[01])(?:-(?:ші|шы|інші|ыншы))?\s+"
            rf"(?:{KK_MONTHS}){KK_MONTH_SFX}\b",
            re.IGNORECASE,
        ),
    ),
    # 16 желтоқсан 1991 жылы / 16 желтоқсанда
    (
        "kk_dmy",
        re.compile(
            rf"(?<!\d)(?:[1-2]?\d|3[01])(?:-(?:ші|шы|інші|ыншы))?\s+"
            rf"(?:{KK_MONTHS}){KK_MONTH_SFX}(?:\s+(?:1[0-9]{{3}}|20[0-9]{{2}})(?:\s*ж(?:ыл(?:ы|ғы|да|дан|ға)?|\.))?)?",
            re.IGNORECASE,
        ),
    ),
    # 2020-2025 жылдары / 2020–2025 жж.
    (
        "kk_year_range",
        re.compile(
            r"(?<!\d)(?:1[0-9]{3}|20[0-9]{2})\s*[–—-]\s*(?:1[0-9]{3}|20[0-9]{2})\s*"
            r"(?:жылдар(?:ы|да|дан|ға|дың)?|жж\.)",
            re.IGNORECASE,
        ),
    ),
    # 2020-сыншы жылы / 2020 жылы / 1991 ж. / 2020 жылдың басынан
    (
        "kk_year",
        re.compile(
            r"(?<!\d)(?:1[0-9]{3}|20[0-9]{2})(?:-(?:ші|шы|інші|ыншы|сінші|сыншы|жылдар(?:ы|да)?))?\s+"
            r"(?:жыл(?:ы|ғы|да|дан|ға|дың|дар(?:ы|да)?|мен|дан бері|ға дейін)?|ж\.)",
            re.IGNORECASE,
        ),
    ),
    # 1990-жылдары
    ("kk_decade", re.compile(r"(?<!\d)(?:1[0-9]{3}|20[0-9]{2})-жылдар(?:ы|да|дан|ға|дың)?\b", re.IGNORECASE)),
    # XXI ғасыр(да|дың)
    ("kk_century", re.compile(r"\b[IVXLCDM]+\s+ғасыр(?:да|дың|ға|дан|ында)?\b", re.IGNORECASE)),
    # қаңтар айында / қаңтарда / қаңтар
    (
        "kk_month",
        re.compile(
            rf"\b(?:{KK_MONTHS})(?:{KK_MONTH_SFX}|\s+айы(?:нда|ның|на|нан)?)\b",
            re.IGNORECASE,
        ),
    ),
]

_KK_RELATIVE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "kk_relative_phrase",
        re.compile(
            r"\b(?:бүгінге\s+дейін|бүгінгі\s+күні|бүгінгі\s+таңда|"
            r"жыл\s+басынан\s+бері|жыл\s+соңына\s+дейін|жыл\s+сайын|күн\s+сайын|ай\s+сайын|апта\s+сайын|"
            r"өткен\s+жыл(?:ы|ғы|дың|мен|дан)?|келесі\s+жыл(?:ы|ғы|дан)?|келер\s+жыл(?:ы|ғы)?|"
            r"былтырдан\s+бері|тәулік\s+бойы|бірнеше\s+жылдан\s+бері)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "kk_relative_word",
        re.compile(
            r"\b(?:биыл(?:ғы)?|бүгін(?:де|гі)?|былтыр(?:ғы)?|кеше(?:гі)?|ертең(?:гі)?|күнделікті)\b",
            re.IGNORECASE,
        ),
    ),
]

_RU_ABSOLUTE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # 16 декабря 1991 года / 16 декабря 1991 г.
    (
        "ru_dmy",
        re.compile(
            rf"(?<!\d)(?:0?[1-9]|[12]\d|3[01])\s+(?:{RU_MONTHS})"
            rf"(?:\s+(?:1[0-9]{{3}}|20[0-9]{{2}})(?:\s*(?:года|году|год|годом|г\.))?)?",
            re.IGNORECASE,
        ),
    ),
    # в декабре 1991 года / декабрь 1991 г.
    (
        "ru_my",
        re.compile(
            rf"\b(?:{RU_MONTHS})\s+(?:1[0-9]{{3}}|20[0-9]{{2}})(?:\s*(?:года|году|год|г\.))?",
            re.IGNORECASE,
        ),
    ),
    # 2020–2025 гг. / 2020-2025 годов
    (
        "ru_year_range",
        re.compile(
            r"(?<!\d)(?:1[0-9]{3}|20[0-9]{2})\s*[–—-]\s*(?:1[0-9]{3}|20[0-9]{2})\s*(?:гг\.|годов|годах|годы)",
            re.IGNORECASE,
        ),
    ),
    # 1991 года / 1991 году / 1991 г.
    (
        "ru_year",
        re.compile(
            r"(?<!\d)(?:1[0-9]{3}|20[0-9]{2})(?:-х|-е|-м)?\s*(?:года|году|год|годом|годов|годах|годы|г\.)",
            re.IGNORECASE,
        ),
    ),
    # XXI века / XX веке
    ("ru_century", re.compile(r"\b[IVXLCDM]+\s+век[аеу]?\b", re.IGNORECASE)),
    # I квартале 2024 года
    (
        "ru_quarter",
        re.compile(
            r"\b(?:[IV]{1,3}|[1-4](?:-м|-й)?)\s+квартал[аеу]?\s+(?:1[0-9]{3}|20[0-9]{2})(?:\s*(?:года|г\.))?",
            re.IGNORECASE,
        ),
    ),
]

_RU_RELATIVE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "ru_relative",
        re.compile(
            r"\b(?:в\s+прошлом\s+году|в\s+этом\s+году|в\s+следующем\s+году|"
            r"сегодня|вчера|завтра|ежегодно|ежедневно)\b",
            re.IGNORECASE,
        ),
    ),
]

_EN_ABSOLUTE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # December 16, 1991 / Dec 16 1991 / December 16th, 1991
    (
        "en_mdy",
        re.compile(
            rf"\b(?:{EN_MONTHS})\s+(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?(?:,\s*|\s+)(?:1[0-9]{{3}}|20[0-9]{{2}})\b",
            re.IGNORECASE,
        ),
    ),
    # 16 December 1991
    (
        "en_dmy",
        re.compile(
            rf"(?<!\d)(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+(?:{EN_MONTHS})(?:,?\s+(?:1[0-9]{{3}}|20[0-9]{{2}}))?\b",
            re.IGNORECASE,
        ),
    ),
    # December 1991 / December 16
    (
        "en_my_or_md",
        re.compile(
            rf"\b(?:{EN_MONTHS})\s+(?:(?:1[0-9]{{3}}|20[0-9]{{2}})|(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?)\b",
            re.IGNORECASE,
        ),
    ),
    # 1990s / 21st century
    (
        "en_decade_century",
        re.compile(
            r"\b(?:(?:1[0-9]{2}0|20[0-9]0)s|(?:1?\d|2[01])(?:st|nd|rd|th)\s+century)\b", re.IGNORECASE
        ),
    ),
    # Year range 2020–2025
    (
        "en_year_range",
        re.compile(r"(?<!\d)(?:1[0-9]{3}|20[0-9]{2})\s*[–—-]\s*(?:1[0-9]{3}|20[0-9]{2})(?!\d)"),
    ),
    # Contextual year: in 1991 / since 1991 / by 2025 / from 2020
    (
        "en_year_ctx",
        re.compile(
            r"\b(?:in|since|by|from|until|before|after|during)\s+((?:1[0-9]{3}|20[0-9]{2}))\b", re.IGNORECASE
        ),
    ),
]

_EN_RELATIVE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "en_relative",
        re.compile(
            r"\b(?:last\s+year|this\s+year|next\s+year|today|yesterday|tomorrow|annually)\b",
            re.IGNORECASE,
        ),
    ),
]


@dataclass(frozen=True)
class DateSpan:
    start: int
    end: int
    surface: str
    rule: str
    is_relative: bool
    normalized: str | None = None


def parse_date_iso(surface: str, lang: str) -> str | None:
    """Best-effort ISO date normalization via ``dateparser`` (optional enrichment)."""
    try:
        import dateparser
    except ImportError:
        return None
    # Fast path for explicit YYYY-MM-DD
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", surface.strip())
    if m:
        return surface.strip()
    langs = [lang] if lang in {"kk", "ru", "en"} else ["en", "ru", "kk"]
    dt = dateparser.parse(surface, languages=langs, settings={"STRICT_PARSING": False})
    if dt is None:
        return None
    return dt.date().isoformat()


class RuleBasedDateExtractor:
    """Multilingual rule-based DATE extractor conforming to ``EntityExtractor``."""

    version: str = "date-rules-v1"

    def __init__(self, include_relative: bool = True, normalize_iso: bool = False) -> None:
        self.include_relative = include_relative
        self.normalize_iso = normalize_iso

    def _patterns_for(self, lang: str) -> list[tuple[str, re.Pattern[str], bool]]:
        pats: list[tuple[str, re.Pattern[str], bool]] = [(n, p, False) for n, p in _COMMON_PATTERNS]
        if lang == "kk":
            pats.extend((n, p, False) for n, p in _KK_ABSOLUTE_PATTERNS)
            if self.include_relative:
                pats.extend((n, p, True) for n, p in _KK_RELATIVE_PATTERNS)
        elif lang == "ru":
            pats.extend((n, p, False) for n, p in _RU_ABSOLUTE_PATTERNS)
            if self.include_relative:
                pats.extend((n, p, True) for n, p in _RU_RELATIVE_PATTERNS)
        elif lang == "en":
            pats.extend((n, p, False) for n, p in _EN_ABSOLUTE_PATTERNS)
            if self.include_relative:
                pats.extend((n, p, True) for n, p in _EN_RELATIVE_PATTERNS)
        else:
            for grp in (_KK_ABSOLUTE_PATTERNS, _RU_ABSOLUTE_PATTERNS, _EN_ABSOLUTE_PATTERNS):
                pats.extend((n, p, False) for n, p in grp)
            if self.include_relative:
                for grp in (_KK_RELATIVE_PATTERNS, _RU_RELATIVE_PATTERNS, _EN_RELATIVE_PATTERNS):
                    pats.extend((n, p, True) for n, p in grp)
        return pats

    def extract_spans(self, text: str, lang: str) -> list[DateSpan]:
        """Extract non-overlapping DATE spans (longest match wins on overlap) with exact char offsets."""
        raw: list[DateSpan] = []
        for rule_name, pattern, is_rel in self._patterns_for(lang):
            for m in pattern.finditer(text):
                if rule_name == "en_year_ctx" and m.lastindex:
                    s, e = m.start(1), m.end(1)
                else:
                    s, e = m.start(), m.end()
                surf = text[s:e]
                if not surf.strip():
                    continue
                norm = parse_date_iso(surf, lang) if (self.normalize_iso and not is_rel) else None
                raw.append(
                    DateSpan(
                        start=s, end=e, surface=surf, rule=rule_name, is_relative=is_rel, normalized=norm
                    )
                )

        # Resolve overlaps: prefer longer span, then earlier start, then absolute over relative
        raw.sort(key=lambda sp: (-(sp.end - sp.start), sp.start, sp.is_relative))
        kept: list[DateSpan] = []
        for cand in raw:
            if any(cand.start < k.end and k.start < cand.end for k in kept):
                continue
            kept.append(cand)
        kept.sort(key=lambda sp: sp.start)
        return kept

    def extract(self, chunks: Sequence[Chunk]) -> list[EntityMention]:
        mentions: list[EntityMention] = []
        for ch in chunks:
            lg = ch.language.value if isinstance(ch.language, Lang) else str(ch.language)
            for sp in self.extract_spans(ch.text, lg):
                mentions.append(
                    EntityMention(
                        chunk_id=ch.chunk_id,
                        surface=sp.surface,
                        type=EntityType.DATE,
                        char_start=sp.start,
                        char_end=sp.end,
                        score=1.0,
                        wikidata_qid=None,
                        model_version=self.version,
                    )
                )
        return mentions

    def extract_token_bio(self, tokens: Sequence[str], lang: str) -> list[str]:
        """Project extracted character spans onto whitespace-joined tokens as IOB2 DATE tags."""
        offsets: list[tuple[int, int]] = []
        pos = 0
        for tok in tokens:
            s = pos
            e = pos + len(tok)
            offsets.append((s, e))
            pos = e + 1
        text = " ".join(tokens)
        spans = self.extract_spans(text, lang)
        tags = ["O"] * len(tokens)
        for sp in spans:
            idx = [i for i, (ts, te) in enumerate(offsets) if ts < sp.end and te > sp.start]
            if not idx or any(tags[i] != "O" for i in idx):
                continue
            tags[idx[0]] = "B-DATE"
            for i in idx[1:]:
                tags[i] = "I-DATE"
        return tags


# --------------------------------------------------------------------------- RU/EN curated reference cases

RU_EN_REFERENCE_CASES: list[dict[str, Any]] = [
    {
        "lang": "ru",
        "text": "Конституция Республики Казахстан была принята 30 августа 1995 года на референдуме.",
        "expected": ["30 августа 1995 года"],
    },
    {
        "lang": "ru",
        "text": "В 2020–2025 гг. объём инвестиций вырос, особенно в I квартале 2024 года.",
        "expected": ["2020–2025 гг.", "I квартале 2024 года"],
    },
    {
        "lang": "ru",
        "text": "Договор подписан 15.03.2023 и вступил в силу в декабре 2023 года.",
        "expected": ["15.03.2023", "декабре 2023 года"],
    },
    {
        "lang": "ru",
        "text": "Памятник относится к XXI века и сегодня открыт для посетителей.",
        "expected": ["XXI века", "сегодня"],
    },
    {
        "lang": "ru",
        "text": "По данным отчёта, опубликованного в прошлом году, рост составил 5%.",
        "expected": ["в прошлом году"],
    },
    {
        "lang": "en",
        "text": "Kazakhstan declared independence on December 16, 1991.",
        "expected": ["December 16, 1991"],
    },
    {
        "lang": "en",
        "text": "The audit covers 2020-2024 and Q3 2024 financial results published on 2024-11-01.",
        "expected": ["2020-2024", "Q3 2024", "2024-11-01"],
    },
    {
        "lang": "en",
        "text": "Founded in 1997, the institute expanded rapidly in the 2010s and the 21st century.",
        "expected": ["1997", "2010s", "21st century"],
    },
    {
        "lang": "en",
        "text": "The summit on 12 March 2025 will review policies adopted last year.",
        "expected": ["12 March 2025", "last year"],
    },
    {
        "lang": "en",
        "text": "No dates appear in this sentence about Almaty and Astana.",
        "expected": [],
    },
]


def evaluate_kaznerd_dates(
    eval_dir: Path,
    include_relative: bool = True,
) -> dict[str, Any]:
    """Evaluate rule-based DATE extractor on frozen KazNERD ``tags_full`` DATE spans."""
    ext = RuleBasedDateExtractor(include_relative=include_relative)
    df = pl.read_parquet(eval_dir / "kk" / "test-00000.parquet").filter(pl.col("source") == "kaznerd")
    tokens_list = df["tokens"].to_list()
    tags_full_list = df["tags_full"].to_list()

    y_true: list[list[str]] = []
    y_pred: list[list[str]] = []
    missed_examples: Counter[str] = Counter()
    spurious_examples: Counter[str] = Counter()

    for toks, full in zip(tokens_list, tags_full_list, strict=True):
        gt = [t if t in {"B-DATE", "I-DATE"} else "O" for t in full]
        pr = ext.extract_token_bio(toks, "kk")
        y_true.append(gt)
        y_pred.append(pr)
        gt_s = set(extract_iob2_entities(gt))
        pr_s = set(extract_iob2_entities(pr))
        for s, e, _ in gt_s - pr_s:
            missed_examples[" ".join(toks[s:e])] += 1
        for s, e, _ in pr_s - gt_s:
            spurious_examples[" ".join(toks[s:e])] += 1

    strict = strict_entity_metrics(y_true, y_pred, labels=("DATE",))
    tax = span_error_taxonomy(y_true, y_pred, tokens_list=tokens_list, langs=["kk"] * len(tokens_list))
    return {
        "benchmark": "kaznerd_frozen_test_kk",
        "include_relative": include_relative,
        "n_sentences": len(tokens_list),
        "strict": strict["per_type"]["DATE"],
        "partial_and_boundary": tax["scores"],
        "error_counts": tax["counts"],
        "top_missed": missed_examples.most_common(15),
        "top_spurious": spurious_examples.most_common(15),
    }


def evaluate_ru_en_reference(include_relative: bool = True) -> dict[str, Any]:
    """Evaluate on curated RU/EN synthetic reference cases (clearly labeled as non-benchmark)."""
    ext = RuleBasedDateExtractor(include_relative=include_relative)
    by_lang: dict[str, dict[str, int]] = {
        "ru": {"tp": 0, "fp": 0, "fn": 0},
        "en": {"tp": 0, "fp": 0, "fn": 0},
    }
    failures: list[dict[str, Any]] = []
    for case in RU_EN_REFERENCE_CASES:
        lg = case["lang"]
        got = [sp.surface for sp in ext.extract_spans(case["text"], lg)]
        exp = list(case["expected"])
        got_set, exp_set = set(got), set(exp)
        by_lang[lg]["tp"] += len(got_set & exp_set)
        by_lang[lg]["fp"] += len(got_set - exp_set)
        by_lang[lg]["fn"] += len(exp_set - got_set)
        if got_set != exp_set:
            failures.append({"lang": lg, "text": case["text"], "expected": exp, "got": got})
    summary: dict[str, Any] = {}
    for lg, c in by_lang.items():
        p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 1.0
        r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 1.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        summary[lg] = {**c, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}
    return {
        "benchmark": "curated_synthetic_reference_cases_not_full_benchmark",
        "n_cases": len(RU_EN_REFERENCE_CASES),
        "by_lang": summary,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, default=ROOT / "data" / "eval" / "ner")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "dates.json")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    rep = {
        "version": RuleBasedDateExtractor.version,
        "kaznerd_with_relative": evaluate_kaznerd_dates(args.eval_dir, include_relative=True),
        "kaznerd_absolute_only": evaluate_kaznerd_dates(args.eval_dir, include_relative=False),
        "ru_en_reference": evaluate_ru_en_reference(include_relative=True),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    k = rep["kaznerd_with_relative"]
    log.info(
        "DATE eval KazNERD (kk): strict F1=%.4f (P=%.4f R=%.4f), partial F1=%.4f",
        k["strict"]["f1"],
        k["strict"]["precision"],
        k["strict"]["recall"],
        k["partial_and_boundary"]["partial_f1"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
