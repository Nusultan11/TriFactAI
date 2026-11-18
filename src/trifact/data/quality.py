"""Quality signals and filter rules of the clean stage (architecture §5.3, ROADMAP 2.3).

Signals (one row per document, computed on normalized text after boilerplate-line removal):
`n_chars`, `n_words`, `n_lines`, `alpha_ratio` (letters / non-space chars), `dup_line_ratio`
(1 - unique/total non-empty lines), `dup_ngram_ratio` (share of word 5-gram positions repeating an
earlier 5-gram), `mean_word_len`, `bullet_ratio` / `ellipsis_ratio` (lines starting with a bullet /
ending with an ellipsis), `stopword_ratio` (words from the document language's function-word list),
`replacement_ratio` (U+FFFD per char), plus `lang`, `lang_score`, `lang_conf` from language id.

Bullets exclude dialogue dashes (`–`, `—` open direct speech in kk/ru prose); `bullet_ratio` and
`ellipsis_ratio` are 0 for documents with fewer than `MIN_LINES_RATIO` lines, where one list item
or a trailing "..." would otherwise make the ratio 1.

Rules are applied in `RULES` order; a document is dropped for the first rule it fails
(`drop_reason`), and every failed rule is counted separately (`fails`), so the report shows both the
exclusive and the overlapping picture. Thresholds resolve as `params.yaml: curation.quality`
`default` -> `languages[lang]` -> `domains[domain]` -> `domains[domain].languages[lang]`: e.g. Kazakh
is agglutinative and has few free function words; sentence-level corpora (cc100, leipzig) are
allowed to be short. `few_stopwords` applies only to documents of >= `stopword_min_words` words.
"""

from __future__ import annotations

from typing import Any

import polars as pl

TARGET_LANGS = ("kk", "ru", "en")

STOPWORDS = {
    "kk": "және мен бен пен да де та те бұл осы ол бір үшін деп еді болып болды бар жоқ ғана туралы сол "
    "оның кейін арқылы бойынша сияқты дейін ретінде ал не немесе біз сіз мұнда онда қазір",
    "ru": "и в не на что с по как а то это из к у за от для о же так но все она он они его ее бы был "
    "была были или уже только мы вы при до если когда где также",
    "en": "the of and to in a is that for it as was with by on are be this from at or an which his "
    "have has not were but their they he she its been also",
}

# name, signal column, comparison, threshold key
RULES: list[tuple[str, str, str, str]] = [
    ("lang_other", "lang", "not_target", ""),
    ("lang_low_score", "lang_score", "lt", "min_lang_score"),
    ("lang_low_conf", "lang_conf", "lt", "min_lang_conf"),
    ("too_short", "n_chars", "lt", "min_chars"),
    ("too_few_words", "n_words", "lt", "min_words"),
    ("low_alpha", "alpha_ratio", "lt", "min_alpha_ratio"),
    ("bad_chars", "replacement_ratio", "gt", "max_replacement_ratio"),
    ("dup_lines", "dup_line_ratio", "gt", "max_dup_line_ratio"),
    ("dup_ngrams", "dup_ngram_ratio", "gt", "max_dup_ngram_ratio"),
    ("short_words", "mean_word_len", "lt", "min_mean_word_len"),
    ("long_words", "mean_word_len", "gt", "max_mean_word_len"),
    ("bullets", "bullet_ratio", "gt", "max_bullet_ratio"),
    ("ellipsis", "ellipsis_ratio", "gt", "max_ellipsis_ratio"),
    ("few_stopwords", "stopword_ratio", "lt", "min_stopword_ratio"),
]
REASONS = [r[0] for r in RULES]
MIN_LINES_RATIO = 3
SIGNALS = [
    "n_chars", "n_words", "n_lines", "alpha_ratio", "dup_line_ratio", "dup_ngram_ratio", "mean_word_len",
    "bullet_ratio", "ellipsis_ratio", "stopword_ratio", "replacement_ratio",
]  # fmt: skip


def _stop_regex(lang: str) -> str:
    return r"(?i)\b(?:" + "|".join(STOPWORDS[lang].split()) + r")\b"


def dup_ngram_ratio(texts: list[str], n: int = 5) -> list[float]:
    out = []
    for t in texts:
        w = t.lower().split()
        total = len(w) - n + 1
        if total <= 1:
            out.append(0.0)
            continue
        grams = set(zip(*(w[i:] for i in range(n)), strict=False))
        out.append(1.0 - len(grams) / total)
    return out


def signals(df: pl.DataFrame) -> pl.DataFrame:
    """Add signal columns to a frame with `text` and `lang`."""
    t = pl.col("text")
    lines = t.str.split("\n").list.eval(pl.element().filter(pl.element().str.len_bytes() > 0))
    out = df.with_columns(
        n_chars=t.str.len_chars().cast(pl.Int32),
        n_words=t.str.count_matches(r"[\p{L}\p{N}]+").cast(pl.Int32),
        _letters=t.str.count_matches(r"\p{L}"),
        _spaces=t.str.count_matches(r"\s"),
        n_lines=lines.list.len().cast(pl.Int32),
        _uniq=lines.list.n_unique(),
        _bul=t.str.count_matches("(?m)^[•\\-*·►▪●◦]"),
        _ell=t.str.count_matches("(?m)(?:\\.\\.\\.|…)$"),
        _fffd=t.str.count_matches("�", literal=True),
        _stop=pl.when(pl.col("lang") == "kk")
        .then(t.str.count_matches(_stop_regex("kk")))
        .when(pl.col("lang") == "ru")
        .then(t.str.count_matches(_stop_regex("ru")))
        .when(pl.col("lang") == "en")
        .then(t.str.count_matches(_stop_regex("en")))
        .otherwise(0),
    )
    out = out.with_columns(
        alpha_ratio=(pl.col("_letters") / (pl.col("n_chars") - pl.col("_spaces")).clip(1)).cast(pl.Float32),
        dup_line_ratio=(1 - pl.col("_uniq") / pl.col("n_lines").clip(1)).cast(pl.Float32),
        mean_word_len=(pl.col("_letters") / pl.col("n_words").clip(1)).cast(pl.Float32),
        bullet_ratio=pl.when(pl.col("n_lines") >= MIN_LINES_RATIO)
        .then(pl.col("_bul") / pl.col("n_lines"))
        .otherwise(0.0)
        .cast(pl.Float32),
        ellipsis_ratio=pl.when(pl.col("n_lines") >= MIN_LINES_RATIO)
        .then(pl.col("_ell") / pl.col("n_lines"))
        .otherwise(0.0)
        .cast(pl.Float32),
        stopword_ratio=(pl.col("_stop") / pl.col("n_words").clip(1)).cast(pl.Float32),
        replacement_ratio=(pl.col("_fffd") / pl.col("n_chars").clip(1)).cast(pl.Float32),
        dup_ngram_ratio=pl.Series(dup_ngram_ratio(df["text"].to_list()), dtype=pl.Float32),
    )
    return out.drop("_letters", "_spaces", "_uniq", "_bul", "_ell", "_fffd", "_stop")


def thresholds(cfg: dict[str, Any], domain: str, lang: str) -> dict[str, float]:
    th = dict(cfg["default"])
    th.update((cfg.get("languages") or {}).get(lang, {}))
    dom = dict((cfg.get("domains") or {}).get(domain, {}))
    by_lang = dom.pop("languages", None) or {}
    th.update(dom)
    th.update(by_lang.get(lang, {}))
    return th


def apply_rules(df: pl.DataFrame, cfg: dict[str, Any]) -> pl.DataFrame:
    """Add `drop_reason` (null = kept) and boolean `fail_<rule>` columns; thresholds depend on (domain, lang)."""
    key = pl.concat_str("domain", "lang", separator="|")
    keys = df.select(key.alias("k"))["k"].unique().to_list()
    per = {k: thresholds(cfg, *k.split("|", 1)) for k in keys}
    fails = []
    for name, col, op, th_key in RULES:
        if op == "not_target":
            cond = ~pl.col(col).is_in(TARGET_LANGS)
        else:
            thr = key.replace_strict({k: float(per[k][th_key]) for k in keys}, return_dtype=pl.Float64)
            cond = pl.col(col) < thr if op == "lt" else pl.col(col) > thr
            if name == "few_stopwords":  # short texts legitimately have no function words
                min_w = key.replace_strict({k: int(per[k]["stopword_min_words"]) for k in keys})
                cond = cond & (pl.col("n_words") >= min_w)
        fails.append(cond.fill_null(True).alias(f"fail_{name}"))
    out = df.with_columns(fails)
    reason = pl.lit(None, dtype=pl.String)
    for name in reversed(REASONS):
        reason = pl.when(pl.col(f"fail_{name}")).then(pl.lit(name)).otherwise(reason)
    return out.with_columns(drop_reason=reason)
