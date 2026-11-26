"""Multilingual Wikidata Entity Linking with Kazakh suffix normalization (ROADMAP 5.4, architecture §7.2).

Supports:
- Exact normalized alias lookup + Kazakh morphological suffix stripping + fuzzy fallback.
- Multilingual label/description fallback (kk -> ru -> en -> mul) addressing the 0.38% kk label gap.
- Candidate ranking by prior (sitelinks + alias match) x NER/KB type compatibility x context-description similarity.
- Tunable NIL threshold and evaluation metrics (candidate recall@k, accuracy@1, NIL rate, tail accuracy).
- Compact SQLite B-tree KB index built streamingly from ``data/task/kb`` Parquet shards, or in-memory index for tests.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sqlite3
import time
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from rapidfuzz import fuzz

from trifact.core.params import ROOT, load_params
from trifact.core.schemas import Chunk, EntityMention, EntityType, Lang

log = logging.getLogger("trifact.extraction.linking")

_WS_RE = re.compile(r"\s+")
_OUTER_PUNCT_RE = re.compile(r'^[\s"«»„“”\'()\[\]{},.;:!?—–-]+|[\s"«»„“”\'()\[\]{},.;:!?—–-]+$')
_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁәғқңөұүһіӘҒҚҢӨҰҮҺІ0-9]+")

# Kazakh case, possessive, and plural suffixes ordered longest-first for greedy stripping.
_KK_CASE_SUFFIXES: tuple[str, ...] = (
    # Compound possessive + case
    "ында",
    "інде",
    "ынан",
    "інен",
    "ына",
    "іне",
    "ының",
    "інің",
    "ын",
    "ін",
    "сында",
    "сінде",
    "сынан",
    "сінен",
    "сына",
    "сіне",
    "сының",
    "сінің",
    "сын",
    "сін",
    "мызда",
    "мізде",
    "мыздан",
    "мізден",
    "мызға",
    "мізге",
    "мыздың",
    "міздің",
    # Ablative / Locative / Genitive / Dative / Accusative / Instrumental
    "дан",
    "ден",
    "тан",
    "тен",
    "нан",
    "нен",
    "дың",
    "дің",
    "тың",
    "тің",
    "ның",
    "нің",
    "мен",
    "бен",
    "пен",
    "нда",
    "нде",
    "да",
    "де",
    "та",
    "те",
    "ға",
    "ге",
    "қа",
    "ке",
    "на",
    "не",
    "ды",
    "ді",
    "ты",
    "ті",
    "ны",
    "ні",
)
_KK_PLURAL_SUFFIXES: tuple[str, ...] = ("лар", "лер", "дар", "дер", "тар", "тер")
_KK_POSS_SUFFIXES: tuple[str, ...] = ("сы", "сі", "ы", "і")
_KK_HYPHEN_SFX_RE = re.compile(
    r"-(?:дың|дің|тың|тің|ның|нің|дан|ден|тан|тен|нан|нен|да|де|та|те|ға|ге|қа|ке|ны|ні|ды|ді|ты|ті|мен|бен|пен|қа|ке)$",
    re.IGNORECASE,
)

# Lightweight Russian oblique-case endings for proper nouns
_RU_CASE_ENDINGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ой", ("а", "ая", "")),
    ("ей", ("я", "е", "")),
    ("ом", ("", "о", "а")),
    ("ем", ("ь", "й", "е", "")),
    ("ам", ("а", "")),
    ("ям", ("я", "ь", "")),
    ("ах", ("а", "")),
    ("ях", ("я", "ь", "")),
    ("ою", ("а", "")),
    ("ею", ("я", "")),
    ("ы", ("а", "")),
    ("и", ("я", "ь", "ия", "")),
    ("е", ("а", "я", "ь", "")),
    ("у", ("а", "")),
    ("ю", ("я", "ь", "")),
    ("а", ("", "о")),
    ("я", ("ь", "й", "")),
)


def normalize_mention(text: str) -> str:
    """Canonical lowercase normalized surface form for alias matching."""
    s = unicodedata.normalize("NFC", text).strip()
    s = _OUTER_PUNCT_RE.sub("", s)
    s = _WS_RE.sub(" ", s).strip().casefold()
    return s


def strip_kazakh_suffixes(surface: str, min_stem_len: int = 2) -> list[str]:
    """Generate normalized candidate stems by stripping Kazakh case, possessive, and plural suffixes.

    Handles both single-word mentions (``Астанада`` -> ``астана``) and multi-word phrases
    (``Қазақстан Республикасында`` -> ``қазақстан республикасы``, ``қазақстан республика``)
    as well as hyphenated acronyms (``АҚШ-тың`` -> ``ақш``).
    """
    norm = normalize_mention(surface)
    if not norm:
        return []
    candidates: list[str] = [norm]
    seen: set[str] = {norm}

    def _add(v: str) -> None:
        v = v.strip(" -")
        if len(v) >= min_stem_len and v not in seen:
            seen.add(v)
            candidates.append(v)

    # 1. Hyphenated acronym suffixes (АҚШ-тың, БҰҰ-да)
    if "-" in norm:
        stripped = _KK_HYPHEN_SFX_RE.sub("", norm)
        _add(stripped)

    parts = norm.split(" ")
    prefix = " ".join(parts[:-1])
    last = parts[-1]

    def _join(stem: str) -> str:
        return f"{prefix} {stem}".strip() if prefix else stem

    # 2. Strip case suffix from last word, then optionally possessive or plural
    stems_stage1: list[str] = [last]
    for sfx in _KK_CASE_SUFFIXES:
        if last.endswith(sfx) and len(last) - len(sfx) >= min_stem_len:
            st = last[: -len(sfx)]
            _add(_join(st))
            stems_stage1.append(st)
            # Voicing alternation repair when vowel-initial suffix stripped (ғ->қ, г->к, б->п)
            if sfx[0] in "ыіае":
                for voiced, voiceless in (("ғ", "қ"), ("г", "к"), ("б", "п")):
                    if st.endswith(voiced):
                        repaired = st[:-1] + voiceless
                        _add(_join(repaired))
                        stems_stage1.append(repaired)
            break

    # 3. Strip plural or possessive from stage1 stems
    for st in stems_stage1:
        for sfx in _KK_PLURAL_SUFFIXES:
            if st.endswith(sfx) and len(st) - len(sfx) >= min_stem_len:
                _add(_join(st[: -len(sfx)]))
        for sfx in _KK_POSS_SUFFIXES:
            if st.endswith(sfx) and len(st) - len(sfx) >= min_stem_len + 1:
                _add(_join(st[: -len(sfx)]))

    return candidates


def strip_russian_suffixes(surface: str, min_stem_len: int = 3) -> list[str]:
    """Generate lightweight Russian nominative stem candidates for inflected proper nouns."""
    norm = normalize_mention(surface)
    if not norm:
        return []
    out: list[str] = [norm]
    seen: set[str] = {norm}
    parts = norm.split(" ")
    prefix = " ".join(parts[:-1])
    last = parts[-1]
    for ending, repls in _RU_CASE_ENDINGS:
        if last.endswith(ending) and len(last) - len(ending) >= min_stem_len:
            base = last[: -len(ending)]
            for r in repls:
                cand = f"{prefix} {base}{r}".strip() if prefix else f"{base}{r}"
                if len(cand) >= min_stem_len and cand not in seen:
                    seen.add(cand)
                    out.append(cand)
            break
    return out


# --------------------------------------------------------------------------- KB records & storage


@dataclass(frozen=True)
class KBEntity:
    qid: str
    ner_type: str
    n_sitelinks: int
    label_kk: str = ""
    label_ru: str = ""
    label_en: str = ""
    desc_kk: str = ""
    desc_ru: str = ""
    desc_en: str = ""

    def best_label(self, lang: str) -> str:
        order = {
            "kk": (self.label_kk, self.label_ru, self.label_en),
            "ru": (self.label_ru, self.label_en, self.label_kk),
            "en": (self.label_en, self.label_ru, self.label_kk),
        }.get(lang, (self.label_en, self.label_ru, self.label_kk))
        for x in order:
            if x:
                return x
        return self.qid

    def all_descriptions(self) -> str:
        return " ".join(x for x in (self.desc_kk, self.desc_ru, self.desc_en) if x)


@dataclass(frozen=True)
class LinkCandidate:
    entity: KBEntity
    match_kind: str  # "exact" | "kk_stem" | "ru_stem" | "fuzzy"
    alias_weight: float
    matched_alias: str
    score: float = 0.0


class InMemoryKBIndex:
    """Lightweight in-memory KB index for unit tests, pilots, and bounded benchmarks."""

    def __init__(self) -> None:
        self.entities: dict[str, KBEntity] = {}
        self.aliases: dict[str, list[tuple[str, float]]] = defaultdict(list)

    def add_entity(self, ent: KBEntity, aliases: Iterable[tuple[str, float]] = ()) -> None:
        self.entities[ent.qid] = ent
        for label, w in (
            (ent.label_kk, 1.0),
            (ent.label_ru, 1.0),
            (ent.label_en, 1.0),
            *aliases,
        ):
            norm = normalize_mention(label)
            if norm:
                self.aliases[norm].append((ent.qid, w))

    def lookup_exact(self, norm_alias: str, limit: int = 50) -> list[tuple[KBEntity, float]]:
        hits = self.aliases.get(norm_alias, [])
        best_w: dict[str, float] = {}
        for qid, w in hits:
            if qid not in best_w or w > best_w[qid]:
                best_w[qid] = w
        out = [(self.entities[q], w) for q, w in best_w.items() if q in self.entities]
        out.sort(key=lambda x: (x[0].n_sitelinks, x[1]), reverse=True)
        return out[:limit]

    def lookup_fuzzy(
        self, norm_alias: str, limit: int = 20, min_ratio: float = 78.0
    ) -> list[tuple[KBEntity, float, str]]:
        if len(norm_alias) < 3:
            return []
        prefix = norm_alias[:3]
        scored: list[tuple[float, KBEntity, str]] = []
        seen_qids: set[str] = set()
        for al, hits in self.aliases.items():
            if not al.startswith(prefix) and prefix not in al:
                continue
            ratio = fuzz.ratio(norm_alias, al)
            if ratio < min_ratio:
                continue
            for qid, w in hits:
                if qid in seen_qids or qid not in self.entities:
                    continue
                seen_qids.add(qid)
                scored.append(((ratio / 100.0) * w, self.entities[qid], al))
        scored.sort(key=lambda x: (x[0], x[1].n_sitelinks), reverse=True)
        return [(ent, sc, al) for sc, ent, al in scored[:limit]]


class SQLiteKBIndex:
    """Disk-backed SQLite B-tree KB index for large-scale Wikidata linking (low RAM, <1 ms lookup)."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA query_only = ON;")

    def close(self) -> None:
        self._conn.close()

    def lookup_exact(self, norm_alias: str, limit: int = 50) -> list[tuple[KBEntity, float]]:
        cur = self._conn.execute(
            """
            SELECT e.qid, e.ner_type, e.n_sitelinks, e.label_kk, e.label_ru, e.label_en,
                   e.desc_kk, e.desc_ru, e.desc_en, MAX(a.weight) AS w
            FROM aliases a
            JOIN entities e ON e.qid = a.qid
            WHERE a.alias_norm = ?
            GROUP BY e.qid
            ORDER BY e.n_sitelinks DESC, w DESC
            LIMIT ?
            """,
            (norm_alias, limit),
        )
        return [
            (
                KBEntity(
                    qid=r[0],
                    ner_type=r[1] or "O",
                    n_sitelinks=int(r[2] or 0),
                    label_kk=r[3] or "",
                    label_ru=r[4] or "",
                    label_en=r[5] or "",
                    desc_kk=r[6] or "",
                    desc_ru=r[7] or "",
                    desc_en=r[8] or "",
                ),
                float(r[9] or 1.0),
            )
            for r in cur.fetchall()
        ]

    def lookup_fuzzy(
        self, norm_alias: str, limit: int = 20, min_ratio: float = 80.0
    ) -> list[tuple[KBEntity, float, str]]:
        if len(norm_alias) < 3:
            return []
        prefix = norm_alias[: min(5, max(3, len(norm_alias) - 2))]
        cur = self._conn.execute(
            """
            SELECT a.alias_norm, e.qid, e.ner_type, e.n_sitelinks, e.label_kk, e.label_ru, e.label_en,
                   e.desc_kk, e.desc_ru, e.desc_en, a.weight
            FROM aliases a
            JOIN entities e ON e.qid = a.qid
            WHERE a.alias_norm >= ? AND a.alias_norm < ?
            LIMIT 250
            """,
            (prefix, prefix[:-1] + chr(ord(prefix[-1]) + 1)),
        )
        scored: list[tuple[float, KBEntity, str]] = []
        seen: set[str] = set()
        for r in cur.fetchall():
            al = str(r[0])
            qid = str(r[1])
            if qid in seen:
                continue
            ratio = fuzz.ratio(norm_alias, al)
            if ratio < min_ratio:
                continue
            seen.add(qid)
            ent = KBEntity(
                qid=qid,
                ner_type=r[2] or "O",
                n_sitelinks=int(r[3] or 0),
                label_kk=r[4] or "",
                label_ru=r[5] or "",
                label_en=r[6] or "",
                desc_kk=r[7] or "",
                desc_ru=r[8] or "",
                desc_en=r[9] or "",
            )
            scored.append(((ratio / 100.0) * float(r[10] or 1.0), ent, al))
        scored.sort(key=lambda x: (x[0], x[1].n_sitelinks), reverse=True)
        return [(ent, sc, al) for sc, ent, al in scored[:limit]]


def build_sqlite_kb_index(
    kb_dir: Path,
    out_db: Path,
    min_sitelinks: int = 1,
    core_only: bool = True,
    max_shards: int | None = None,
) -> dict[str, Any]:
    """Stream ``data/task/kb/multi/{entities,titles}-*.parquet`` into a compact SQLite B-tree index."""
    t0 = time.perf_counter()
    out_db.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_db.with_suffix(".sqlite.tmp")
    if tmp.exists():
        tmp.unlink()

    conn = sqlite3.connect(str(tmp))
    conn.execute("PRAGMA journal_mode = OFF;")
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.executescript(
        """
        CREATE TABLE entities (
            qid TEXT PRIMARY KEY,
            ner_type TEXT,
            n_sitelinks INTEGER,
            label_kk TEXT,
            label_ru TEXT,
            label_en TEXT,
            desc_kk TEXT,
            desc_ru TEXT,
            desc_en TEXT
        );
        CREATE TABLE aliases (
            alias_norm TEXT NOT NULL,
            qid TEXT NOT NULL,
            weight REAL NOT NULL
        );
        """
    )

    multi_dir = kb_dir / "multi"
    ent_files = sorted(multi_dir.glob("entities-*.parquet"))
    title_files = sorted(multi_dir.glob("titles-*.parquet"))
    if max_shards is not None:
        ent_files = ent_files[:max_shards]
        title_files = title_files[:max_shards]

    n_entities = 0
    n_aliases = 0

    if ent_files:
        for fp in ent_files:
            lf = pl.scan_parquet(fp)
            if core_only:
                lf = lf.filter(pl.col("ner_type").is_in(["PER", "ORG", "LOC"]))
            if min_sitelinks > 0:
                lf = lf.filter(pl.col("n_sitelinks") >= min_sitelinks)
            cols = [
                c
                for c in (
                    "qid",
                    "ner_type",
                    "n_sitelinks",
                    "label_kk",
                    "label_ru",
                    "label_en",
                    "label_mul",
                    "description_kk",
                    "description_ru",
                    "description_en",
                    "aliases_kk",
                    "aliases_ru",
                    "aliases_en",
                    "aliases_mul",
                )
                if c in lf.collect_schema().names()
            ]
            df = lf.select(cols).collect()
            if df.is_empty():
                continue
            ent_rows = []
            al_rows = []
            for r in df.iter_rows(named=True):
                qid = str(r["qid"])
                ent_rows.append(
                    (
                        qid,
                        r.get("ner_type") or "O",
                        int(r.get("n_sitelinks") or 0),
                        r.get("label_kk") or "",
                        r.get("label_ru") or "",
                        r.get("label_en") or "",
                        r.get("description_kk") or "",
                        r.get("description_ru") or "",
                        r.get("description_en") or "",
                    )
                )
                seen_al: set[str] = set()
                for col, w in (
                    ("label_kk", 1.0),
                    ("label_ru", 1.0),
                    ("label_en", 1.0),
                    ("label_mul", 0.95),
                ):
                    v = r.get(col)
                    if v:
                        nm = normalize_mention(str(v))
                        if nm and nm not in seen_al:
                            seen_al.add(nm)
                            al_rows.append((nm, qid, w))
                for col, w in (
                    ("aliases_kk", 0.9),
                    ("aliases_ru", 0.9),
                    ("aliases_en", 0.9),
                    ("aliases_mul", 0.85),
                ):
                    for a in r.get(col) or ():
                        nm = normalize_mention(str(a))
                        if nm and nm not in seen_al:
                            seen_al.add(nm)
                            al_rows.append((nm, qid, w))
            conn.executemany("INSERT OR IGNORE INTO entities VALUES (?,?,?,?,?,?,?,?,?)", ent_rows)
            conn.executemany("INSERT INTO aliases VALUES (?,?,?)", al_rows)
            n_entities += len(ent_rows)
            n_aliases += len(al_rows)
            conn.commit()
    elif title_files:
        # Fallback when only titles shards are locally present (e.g. lightweight pilot)
        for fp in title_files:
            lf = pl.scan_parquet(fp)
            if core_only:
                lf = lf.filter(pl.col("ner_type").is_in(["PER", "ORG", "LOC"]))
            df = lf.select("lang", "title", "qid", "ner_type").collect()
            if df.is_empty():
                continue
            by_qid: dict[str, dict[str, Any]] = {}
            al_rows = []
            for r in df.iter_rows(named=True):
                qid = str(r["qid"])
                lg = str(r["lang"])
                title = str(r["title"])
                nt = str(r["ner_type"] or "O")
                rec = by_qid.setdefault(qid, {"ner_type": nt, "sitelinks": 0, "kk": "", "ru": "", "en": ""})
                rec["sitelinks"] += 1
                if lg in {"kk", "ru", "en"} and not rec[lg]:
                    rec[lg] = title
                nm = normalize_mention(title)
                if nm:
                    al_rows.append((nm, qid, 1.0))
                # Also index title without parenthetical disambiguator, e.g. "Астана (қала)" -> "астана"
                base = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
                nm_base = normalize_mention(base)
                if nm_base and nm_base != nm:
                    al_rows.append((nm_base, qid, 0.92))
            ent_rows = [
                (q, v["ner_type"], v["sitelinks"], v["kk"], v["ru"], v["en"], "", "", "")
                for q, v in by_qid.items()
            ]
            conn.executemany("INSERT OR IGNORE INTO entities VALUES (?,?,?,?,?,?,?,?,?)", ent_rows)
            conn.executemany("INSERT INTO aliases VALUES (?,?,?)", al_rows)
            n_entities += len(ent_rows)
            n_aliases += len(al_rows)
            conn.commit()

    conn.execute("CREATE INDEX idx_aliases_norm ON aliases(alias_norm);")
    conn.commit()
    conn.close()
    tmp.replace(out_db)
    return {
        "db_path": str(out_db),
        "n_entities": n_entities,
        "n_aliases": n_aliases,
        "seconds": round(time.perf_counter() - t0, 2),
        "bytes": out_db.stat().st_size,
    }


# --------------------------------------------------------------------------- Linker


def _type_compatibility(mention_type: str, kb_type: str) -> float:
    m = mention_type.upper()
    k = kb_type.upper()
    if m == k:
        return 1.0
    if k in {"", "O"}:
        return 0.45
    if {m, k} == {"ORG", "LOC"}:
        return 0.30  # metonymy tolerance (e.g. government vs country)
    return 0.08


def _context_similarity(context: str, ent: KBEntity) -> float:
    """Lightweight multilingual lexical + character-trigram similarity between context and KB metadata."""
    if not context:
        return 0.0
    meta = f"{ent.label_kk} {ent.label_ru} {ent.label_en} {ent.all_descriptions()}".strip()
    if not meta:
        return 0.0
    ctx_toks = {t.casefold() for t in _TOKEN_RE.findall(context) if len(t) >= 3}
    meta_toks = {t.casefold() for t in _TOKEN_RE.findall(meta) if len(t) >= 3}
    if not ctx_toks or not meta_toks:
        return 0.0
    inter = len(ctx_toks & meta_toks)
    if inter == 0:
        return 0.0
    return min(1.0, inter / math.sqrt(len(meta_toks) + 1.0))


class WikidataEntityLinker:
    """EntityLinker implementation over Wikidata KB (ADR-0006, ROADMAP 5.4)."""

    def __init__(
        self,
        index: InMemoryKBIndex | SQLiteKBIndex,
        candidates_k: int = 20,
        nil_threshold: float = 0.35,
        use_kk_suffix_strip: bool = True,
        use_fuzzy_fallback: bool = True,
    ) -> None:
        self.index = index
        self.candidates_k = candidates_k
        self.nil_threshold = nil_threshold
        self.use_kk_suffix_strip = use_kk_suffix_strip
        self.use_fuzzy_fallback = use_fuzzy_fallback

    def generate_candidates(self, surface: str, lang: str, k: int | None = None) -> list[LinkCandidate]:
        limit = k or self.candidates_k
        norm = normalize_mention(surface)
        if not norm:
            return []

        best: dict[str, LinkCandidate] = {}

        # 1. Exact normalized alias lookup
        for ent, w in self.index.lookup_exact(norm, limit=limit * 2):
            best[ent.qid] = LinkCandidate(entity=ent, match_kind="exact", alias_weight=w, matched_alias=norm)

        # 2. Kazakh morphological suffix stripping
        if self.use_kk_suffix_strip and (lang == "kk" or len(best) < limit):
            for stem in strip_kazakh_suffixes(surface)[1:]:
                for ent, w in self.index.lookup_exact(stem, limit=limit):
                    if ent.qid not in best:
                        best[ent.qid] = LinkCandidate(
                            entity=ent, match_kind="kk_stem", alias_weight=w * 0.94, matched_alias=stem
                        )

        # 3. Russian oblique-case stemming
        if lang == "ru" and len(best) < limit:
            for stem in strip_russian_suffixes(surface)[1:]:
                for ent, w in self.index.lookup_exact(stem, limit=limit):
                    if ent.qid not in best:
                        best[ent.qid] = LinkCandidate(
                            entity=ent, match_kind="ru_stem", alias_weight=w * 0.92, matched_alias=stem
                        )

        # 4. Fuzzy fallback
        if self.use_fuzzy_fallback and len(best) < max(5, limit // 2):
            for ent, sc, al in self.index.lookup_fuzzy(norm, limit=limit):
                if ent.qid not in best:
                    best[ent.qid] = LinkCandidate(
                        entity=ent, match_kind="fuzzy", alias_weight=sc * 0.80, matched_alias=al
                    )

        cands = list(best.values())
        cands.sort(key=lambda c: (c.alias_weight, c.entity.n_sitelinks), reverse=True)
        return cands[:limit]

    def rank_candidates(
        self,
        surface: str,
        context: str,
        lang: str,
        ner_type: str,
        candidates: Sequence[LinkCandidate],
    ) -> list[LinkCandidate]:
        ranked: list[LinkCandidate] = []
        for c in candidates:
            prior = math.log1p(max(0, c.entity.n_sitelinks)) / math.log1p(250.0)
            prior = min(1.0, max(0.05, prior))
            type_compat = _type_compatibility(ner_type, c.entity.ner_type)
            ctx_sim = _context_similarity(context, c.entity)
            # Weighted multiplicative + additive blend so high-prior exact matches with matching type win,
            # while context breaks ambiguity among homonyms (e.g. "Абай" person vs region).
            raw_score = c.alias_weight * type_compat * (0.55 + 0.30 * prior + 0.15 * ctx_sim)
            # Language label availability bonus (small, preserves ru/en fallback for kk)
            if lang == "kk" and c.entity.label_kk:
                raw_score += 0.02
            score = round(min(1.0, max(0.0, raw_score)), 6)
            ranked.append(
                LinkCandidate(
                    entity=c.entity,
                    match_kind=c.match_kind,
                    alias_weight=c.alias_weight,
                    matched_alias=c.matched_alias,
                    score=score,
                )
            )
        ranked.sort(key=lambda c: (c.score, c.entity.n_sitelinks), reverse=True)
        return ranked

    def link_one(
        self,
        surface: str,
        context: str,
        lang: str,
        ner_type: str,
    ) -> tuple[str | None, float, list[LinkCandidate]]:
        if ner_type.upper() == EntityType.DATE.value:
            return None, 0.0, []
        cands = self.generate_candidates(surface, lang)
        if not cands:
            return None, 0.0, []
        ranked = self.rank_candidates(surface, context, lang, ner_type, cands)
        top = ranked[0]
        if top.score < self.nil_threshold:
            return None, top.score, ranked
        return top.entity.qid, top.score, ranked

    def link(self, mentions: Sequence[EntityMention], chunks: Sequence[Chunk]) -> list[EntityMention]:
        chunk_by_id = {c.chunk_id: c for c in chunks}
        out: list[EntityMention] = []
        for m in mentions:
            ch = chunk_by_id.get(m.chunk_id)
            ctx = ch.text if ch else ""
            lg = (ch.language.value if isinstance(ch.language, Lang) else str(ch.language)) if ch else "en"
            mt = m.type.value if isinstance(m.type, EntityType) else str(m.type)
            qid, _, _ = self.link_one(m.surface, ctx, lg, mt)
            out.append(m.model_copy(update={"wikidata_qid": qid}))
        return out


# --------------------------------------------------------------------------- Evaluation


def evaluate_entity_linking(
    examples: Sequence[dict[str, Any]],
    index: InMemoryKBIndex | SQLiteKBIndex,
    candidates_k: int = 20,
    nil_threshold: float = 0.35,
    reference_provenance: str = "wikipedia_anchor_derived",
) -> dict[str, Any]:
    """Evaluate entity linking per language, quantifying Kazakh suffix-stripping candidate recall lift.

    Each example dict requires: ``surface``, ``context``, ``lang``, ``ner_type``, ``gold_qid`` (or None for NIL),
    and optional ``gold_n_sitelinks`` (for tail-entity slicing where ``<= 5``).
    """
    linker_with = WikidataEntityLinker(
        index, candidates_k=candidates_k, nil_threshold=nil_threshold, use_kk_suffix_strip=True
    )
    linker_exact_only = WikidataEntityLinker(
        index,
        candidates_k=candidates_k,
        nil_threshold=nil_threshold,
        use_kk_suffix_strip=False,
        use_fuzzy_fallback=False,
    )
    linker_exact_plus_stem = WikidataEntityLinker(
        index,
        candidates_k=candidates_k,
        nil_threshold=nil_threshold,
        use_kk_suffix_strip=True,
        use_fuzzy_fallback=False,
    )

    by_lang: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        by_lang[str(ex["lang"])].append(ex)

    results_by_lang: dict[str, Any] = {}
    for lg, rows in sorted(by_lang.items()):
        total = len(rows)
        linkable = 0
        cand_hit_with = 0
        cand_hit_exact = 0
        cand_hit_stem = 0
        acc1_correct = 0
        nil_preds = 0
        nil_gold = 0
        nil_tp = 0
        tail_total = 0
        tail_correct = 0
        ambig_total = 0
        ambig_correct = 0
        infl_total = 0
        infl_correct = 0
        err_cand_miss = 0
        err_disambig = 0
        err_false_nil = 0
        err_false_link = 0
        lat_ms: list[float] = []

        for r in rows:
            surf = str(r["surface"])
            ctx = str(r.get("context", ""))
            nt = str(r.get("ner_type", "PER"))
            gold = r.get("gold_qid")
            gold_sitelinks = r.get("gold_n_sitelinks")
            is_ambig = bool(r.get("is_ambiguous", False))
            is_infl = bool(r.get("is_inflected", False))

            t_m0 = time.perf_counter()
            pred_qid, _, ranked_with = linker_with.link_one(surf, ctx, lg, nt)
            lat_ms.append((time.perf_counter() - t_m0) * 1000.0)

            cands_exact = linker_exact_only.generate_candidates(surf, lg, k=candidates_k)
            cands_stem = linker_exact_plus_stem.generate_candidates(surf, lg, k=candidates_k)

            if pred_qid is None:
                nil_preds += 1

            if gold is None:
                nil_gold += 1
                if pred_qid is None:
                    acc1_correct += 1
                    nil_tp += 1
                else:
                    err_false_link += 1
            else:
                linkable += 1
                in_cands = any(c.entity.qid == gold for c in ranked_with[:candidates_k])
                if in_cands:
                    cand_hit_with += 1
                if any(c.entity.qid == gold for c in cands_exact[:candidates_k]):
                    cand_hit_exact += 1
                if any(c.entity.qid == gold for c in cands_stem[:candidates_k]):
                    cand_hit_stem += 1
                if pred_qid == gold:
                    acc1_correct += 1
                elif not in_cands:
                    err_cand_miss += 1
                elif pred_qid is None:
                    err_false_nil += 1
                else:
                    err_disambig += 1

                if gold_sitelinks is not None and int(gold_sitelinks) <= 5:
                    tail_total += 1
                    if pred_qid == gold:
                        tail_correct += 1
                if is_ambig or len(ranked_with) >= 2:
                    ambig_total += 1
                    if pred_qid == gold:
                        ambig_correct += 1
                if is_infl:
                    infl_total += 1
                    if pred_qid == gold:
                        infl_correct += 1

        rec_with = cand_hit_with / linkable if linkable else 0.0
        rec_exact = cand_hit_exact / linkable if linkable else 0.0
        rec_stem = cand_hit_stem / linkable if linkable else 0.0
        lat_sorted = sorted(lat_ms)
        tot_s = sum(lat_ms) / 1000.0
        p50 = lat_sorted[len(lat_sorted) // 2] if lat_sorted else 0.0
        p95 = lat_sorted[min(int(len(lat_sorted) * 0.95), len(lat_sorted) - 1)] if lat_sorted else 0.0
        nil_prec = (nil_tp / nil_preds) if nil_preds > 0 else 1.0
        nil_rec = (nil_tp / nil_gold) if nil_gold > 0 else None
        results_by_lang[lg] = {
            "n_examples": total,
            "n_linkable": linkable,
            "n_nil_gold": nil_gold,
            f"candidate_recall@{candidates_k}": round(rec_with, 4),
            f"exact_recall@{candidates_k}": round(rec_exact, 4),
            f"exact_plus_kk_stem_recall@{candidates_k}": round(rec_stem, 4),
            "kk_suffix_strip_recall_gain": round(rec_stem - rec_exact, 4),
            "accuracy@1": round(acc1_correct / total, 4) if total else 0.0,
            "nil_rate": round(nil_preds / total, 4) if total else 0.0,
            "nil_accuracy": round(nil_tp / nil_gold, 4) if nil_gold else None,
            "nil_precision": round(nil_prec, 4),
            "nil_recall": round(nil_rec, 4) if nil_rec is not None else None,
            "ambiguous_n": ambig_total,
            "ambiguous_accuracy@1": round(ambig_correct / ambig_total, 4) if ambig_total else None,
            "inflected_n": infl_total,
            "inflected_accuracy@1": round(infl_correct / infl_total, 4) if infl_total else None,
            "tail_n": tail_total,
            "tail_accuracy@1": round(tail_correct / tail_total, 4) if tail_total else None,
            "errors": {
                "candidate_miss": err_cand_miss,
                "disambiguation_error": err_disambig,
                "false_nil": err_false_nil,
                "false_link": err_false_link,
            },
            "latency_p50_ms": round(p50, 3),
            "latency_p95_ms": round(p95, 3),
            "mentions_per_s": round(total / max(tot_s, 1e-6), 1),
        }

    return {
        "reference_provenance": reference_provenance,
        "human_verified": reference_provenance == "human_verified",
        "candidates_k": candidates_k,
        "nil_threshold": nil_threshold,
        "by_lang": results_by_lang,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-index", help="Build compact SQLite B-tree KB index from data/task/kb")
    b.add_argument("--kb-dir", type=Path, default=ROOT / "data" / "task" / "kb")
    b.add_argument("--out", type=Path, default=ROOT / "data" / "task" / "kb" / "kb_index.sqlite")
    b.add_argument("--min-sitelinks", type=int, default=1)
    b.add_argument("--max-shards", type=int, default=None)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_params()

    if args.cmd == "build-index":
        info = build_sqlite_kb_index(
            args.kb_dir, args.out, min_sitelinks=args.min_sitelinks, max_shards=args.max_shards
        )
        log.info("Built KB index: %s", json.dumps(info))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
