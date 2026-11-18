from __future__ import annotations

import polars as pl
import pytest

from trifact.ingestion.normalize import (
    dominant_script,
    fix_homoglyphs,
    fix_mojibake,
    normalize_series,
    normalize_text,
)


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        # Latin i / ə used for Kazakh і / ә inside Cyrillic words
        ("үшiн бiр жəне", "үшін бір және"),
        ("Cілтемелер", "Сілтемелер"),  # Latin C at the start of a Cyrillic word
        ("Photо", "Photo"),  # Cyrillic о in a Latin word
        ("Қазақстан", "Қазақстан"),  # already clean
        ("Кf3", "Кf3"),  # chess notation: f has no Cyrillic look-alike, word kept
        ("бөлісіңізTwitter", "бөлісіңізTwitter"),  # concatenated words of two scripts, kept
    ],
)
def test_homoglyphs(raw: str, want: str) -> None:
    assert normalize_text(raw) == want


def test_homoglyph_tie_uses_dominant_script() -> None:
    # "Мo" -> 1 Cyrillic + 1 Latin, every letter has a look-alike: the document script decides
    assert fix_homoglyphs("Мo", dominant="cyr") == "Мо"
    assert fix_homoglyphs("Мo", dominant="lat") == "Mo"
    assert dominant_script("Бұл қазақ мәтіні, a few words") == "cyr"


def test_mojibake_repaired() -> None:
    assert fix_mojibake("Ð¼Ð¸Ñ\x80 Ñ\x81") == "мир с"
    assert fix_mojibake("нормальный текст") == "нормальный текст"
    assert fix_mojibake("café") == "café"


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("a\r\nb\rc", "a\nb\nc"),
        ("\ufeffТекст", "Текст"),  # BOM
        ("сло\u00adво", "слово"),  # soft hyphen
        ("a\u200bb\u200dc", "abc"),  # zero-width
        ("x\ue001y", "xy"),  # private use area (Symbol font)
        ("a\x00b\x07c", "abc"),  # NUL / control
        ("Москва\u0301", "Москва"),  # stress mark
        ("a\u00a0 \u2003b", "a b"),  # exotic spaces collapse
        ("  a  \n\n\n\n  b  ", "a\n\nb"),  # line edges, blank runs
        ("е\u0308", "ё"),  # NFC
        ("", ""),
        ("\n\n \t", ""),
    ],
)
def test_unicode_cleanup(raw: str, want: str) -> None:
    assert normalize_text(raw) == want


def test_idempotent() -> None:
    texts = ["үшiн Photо  \u00ad x", "Ð¼Ð¸Ñ\x80", "е\u0308 \ufeff", "Кf3 Nf6"]
    for t in texts:
        once = normalize_text(t)
        assert normalize_text(once) == once


def test_series_equals_scalar() -> None:
    texts = [
        "үшiн бiр жəне",
        "Photо\r\nCілтемелер",
        "Ð¼Ð¸Ñ\x80 Ñ\x81",
        "сло\u00adво\u200b",
        "Москва\u0301 е\u0308",
        "plain English text",
        "Кf3 бөлісіңізTwitter",
        "",
        None,
    ]
    got = normalize_series(pl.Series("text", texts, dtype=pl.String)).to_list()
    want = [None if t is None else normalize_text(t) for t in texts]
    assert got == want
