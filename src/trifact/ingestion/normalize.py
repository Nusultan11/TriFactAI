"""Text normalization shared by ingestion and curation (architecture §5.3).

Steps, in order:
1. Line endings CRLF/CR -> LF, tab -> space, NUL removed.
2. Mojibake repair: runs of UTF-8 Cyrillic shown as cp1252/latin-1 (`Ð¼Ð¸Ñ€`) are decoded back.
   It runs before step 3 because latin-1 mojibake contains C1 controls.
3. Invisible format characters (soft hyphen, zero-width, bidi marks, BOM), C0/C1 controls and
   private-use code points (Symbol-font bullets) are removed.
4. Unicode NFC; stress marks (U+0300/U+0301) after a Cyrillic letter are removed.
5. Unicode spaces -> ASCII space, runs of spaces collapsed, lines stripped, >= 3 newlines -> 2.
6. Homoglyphs inside mixed-script words: the minority script is rewritten into the majority one
   when every minority letter has a look-alike (`Cілтемелер` -> `Сілтемелер`, Latin `i`/`ə` in
   Kazakh words -> `і`/`ә`, `Photо` -> `Photo`). Ties follow the text's dominant script. Words
   whose minority letters have no look-alike (`Кf3`, `бөлісіңізTwitter`) are left as they are.

`normalize_text` handles one string (online path). `normalize_series` is the vectorized form for
Polars batches: steps 1, 3-5 are Polars expressions over the whole column; steps 2 and 6 run in
Python only on rows a vectorized regex flags (a few percent), so both forms give identical output.
Character classes are written with literal characters, so one pattern serves Python `re` and
Polars (Rust regex).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

import polars as pl

# soft hyphen, zero-width space/joiners, LRM/RLM, bidi embeddings/isolates, word joiner, BOM
_INVISIBLE = "\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff"
_CONTROL = "\x01-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f"
_PRIVATE_USE = "\ue000-\uf8ff"
REMOVE = f"[{_INVISIBLE}{_CONTROL}{_PRIVATE_USE}]"
SPACES = "[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]"
CYR = "Ѐ-ԯ"
LAT = "A-Za-zÀ-ÖØ-öø-ɏə"
STRESS = f"([{CYR}])[\u0300\u0301]+"
MIXED = f"[{CYR}][{LAT}]|[{LAT}][{CYR}]"
# how cp1252 shows UTF-8 continuation bytes 0x80-0xBF (undefined bytes stay C1 controls)
_CONT = "\u0080-¿ŒœŠšŸŽžƒˆ˜–—‘-„†-•…‰‹›€™"
MOJIBAKE = f"[ÐÑ][{_CONT}]"

# Latin -> Cyrillic look-alikes (incl. Kazakh letters typed with Latin keys), and the reverse
LAT2CYR = dict(zip("aABcCeEHiIKMoOpPTxXyYhəƏ", "аАВсСеЕНіІКМоОрРТхХуУһәӘ", strict=True))
CYR2LAT = {c: lat for lat, c in LAT2CYR.items()}

_RE_REMOVE = re.compile(REMOVE)
_RE_SPACES = re.compile(SPACES)
_RE_STRESS = re.compile(STRESS)
_RE_MIXED = re.compile(MIXED)
_RE_MOJIBAKE = re.compile(f"(?:[ÐÑ][{_CONT}])+")
_RE_CYR = re.compile(f"[{CYR}]")
_RE_LAT = re.compile(f"[{LAT}]")
_RE_MULTISPACE = re.compile(" {2,}")
_RE_LINE_EDGES = re.compile("(?m)^ +| +$")
_RE_BLANKS = re.compile("\n{3,}")


# --------------------------------------------------------------------------- python parts


def _to_bytes(s: str) -> bytes | None:
    out = bytearray()
    for ch in s:
        try:
            out += ch.encode("cp1252")
        except UnicodeEncodeError:
            if ord(ch) > 0xFF:
                return None
            out.append(ord(ch))  # byte undefined in cp1252, passed through as a C1 control
    return bytes(out)


def fix_mojibake(text: str) -> str:
    """Decode runs of UTF-8 Cyrillic shown as cp1252/latin-1; runs that do not decode stay as they are."""

    def repl(m: re.Match[str]) -> str:
        raw = _to_bytes(m.group(0))
        if raw is None:
            return m.group(0)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return m.group(0)

    return _RE_MOJIBAKE.sub(repl, text)


def dominant_script(text: str) -> str:
    return "cyr" if len(_RE_CYR.findall(text)) >= len(_RE_LAT.findall(text)) else "lat"


def _fix_word(w: str, dom: Callable[[], str]) -> str:
    cyr = len(_RE_CYR.findall(w))
    lat = len(_RE_LAT.findall(w))
    target = "cyr" if cyr > lat else "lat" if lat > cyr else dom()
    table, minority = (LAT2CYR, _RE_LAT) if target == "cyr" else (CYR2LAT, _RE_CYR)
    if all(ch in table for ch in w if minority.match(ch)):
        return "".join(table[ch] if minority.match(ch) else ch for ch in w)
    return w


def fix_homoglyphs(text: str, dominant: str | None = None) -> str:
    """Rewrite mixed-script words into their majority script where every minority letter has a look-alike.

    A word is a maximal run of `str.isalpha` characters around a Cyrillic/Latin letter pair.
    """
    cache: list[str] = [dominant] if dominant else []

    def dom() -> str:
        if not cache:
            cache.append(dominant_script(text))
        return cache[0]

    out: list[str] = []
    last = 0
    n = len(text)
    for m in _RE_MIXED.finditer(text):
        if m.start() < last:
            continue
        a, b = m.start(), m.end()
        while a > last and text[a - 1].isalpha():
            a -= 1
        while b < n and text[b].isalpha():
            b += 1
        out.append(text[last:a])
        out.append(_fix_word(text[a:b], dom))
        last = b
    out.append(text[last:])
    return "".join(out)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ").replace("\x00", "")
    text = fix_mojibake(text)
    text = _RE_REMOVE.sub("", text)
    if not unicodedata.is_normalized("NFC", text):
        text = unicodedata.normalize("NFC", text)
    text = _RE_STRESS.sub(r"\1", text)
    text = _RE_SPACES.sub(" ", text)
    text = _RE_MULTISPACE.sub(" ", text)
    text = _RE_LINE_EDGES.sub("", text)
    text = _RE_BLANKS.sub("\n\n", text).strip()
    if _RE_MIXED.search(text):
        text = fix_homoglyphs(text)
    return text


# --------------------------------------------------------------------------- vectorized form


def _apply_where(s: pl.Series, mask: pl.Series, fn: Callable[[str], str]) -> pl.Series:
    if not mask.any():
        return s
    idx = mask.arg_true()
    fixed = pl.Series([fn(t) for t in s.gather(idx).to_list()], dtype=pl.String)
    return s.scatter(idx, fixed)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _not_nfc(s: pl.Series) -> pl.Series:
    return pl.Series([t is not None and not unicodedata.is_normalized("NFC", t) for t in s.to_list()])


def normalize_series(s: pl.Series) -> pl.Series:
    """Vectorized `normalize_text` over a String series (nulls stay null)."""
    name = s.name
    s = (
        s.str.replace_all("\r\n", "\n", literal=True)
        .str.replace_all("\r", "\n", literal=True)
        .str.replace_all("\t", " ", literal=True)
        .str.replace_all("\x00", "", literal=True)
    )
    s = _apply_where(s, s.str.contains(MOJIBAKE).fill_null(False), fix_mojibake)
    s = s.str.replace_all(REMOVE, "")
    s = _apply_where(s, _not_nfc(s), _nfc)
    s = (
        s.str.replace_all(STRESS, "${1}")
        .str.replace_all(SPACES, " ")
        .str.replace_all(" {2,}", " ")
        .str.replace_all("(?m)^ +| +$", "")
        .str.replace_all("\n{3,}", "\n\n")
        .str.strip_chars()
    )
    s = _apply_where(s, s.str.contains(MIXED).fill_null(False), fix_homoglyphs)
    return s.alias(name)
