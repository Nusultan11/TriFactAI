from __future__ import annotations

import numpy as np
import polars as pl

from trifact.data.eda import hist, iin_valid, luhn, merge, quantiles_from_hist, text_profile


def test_merge_and_quantiles() -> None:
    a = {"n": 1, "h": [1, 2], "d": {"x": 1}, "parts": [["a", 1]]}
    b = {"n": 2, "h": [3, 4], "d": {"x": 2, "y": 1}, "parts": [["b", 2]]}
    assert merge(a, b) == {"n": 3, "h": [4, 6], "d": {"x": 3, "y": 1}, "parts": [["a", 1], ["b", 2]]}
    q = quantiles_from_hist(hist(np.arange(1, 1001)))
    assert 400 <= q["p50"] <= 520 and q["p99"] >= 850  # log bins: ~6% resolution


def test_text_profile_scripts_encoding_pii() -> None:
    s = pl.Series(
        ["Қазақ тілі әдемі", "Plain english text", "bad � char", "mail me a@b.kz +7 701 123 45 67", ""]
    )
    p = text_profile(s)
    assert p["docs"] == 5 and p["empty"] == 1
    assert p["script"]["docs_with_kk_letters"] == 1 and p["script"]["docs_mostly_latin"] >= 2
    assert p["encoding"]["docs_replacement_char"] == 1
    assert p["pii"]["docs_email"] == 1 and p["pii"]["docs_phone"] == 1


def test_iin_and_luhn() -> None:
    assert luhn("4111 1111 1111 1111") and not luhn("4111 1111 1111 1112")
    assert not iin_valid("000000000000")
