from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from trifact.data.hashing import MinHasher, exact_hash, jaccard, ngram_hashes, word_hashes

BASE = (
    "the quick brown fox jumps over the lazy dog and runs away into the dark forest "
    "near the old river bank today while the farmer sleeps in his small wooden house"
)


def test_word_and_ngram_hashes() -> None:
    words, counts = word_hashes(pl.Series(["A b, c!", "", None, "Қазақ тілі"]))
    assert counts.tolist() == [3, 0, 0, 2]
    grams, per_doc = ngram_hashes(words, counts, 2)
    assert per_doc.tolist() == [2, 0, 0, 1]
    # case and punctuation do not matter; the same n-gram has the same hash in any document
    w2, c2 = word_hashes(pl.Series(["x a B", "a b"]))
    g2, _ = ngram_hashes(w2, c2, 2)
    assert g2[1] == grams[0] == g2[2]


@pytest.mark.parametrize("texts", [["a b"], ["a"], ["a b", "c"], [""]])
def test_ngram_fewer_words_than_n(texts: list[str]) -> None:
    words, counts = word_hashes(pl.Series(texts))
    grams, per_doc = ngram_hashes(words, counts, 13)
    assert per_doc.tolist() == [1 if c else 0 for c in counts]  # one shingle of all words
    assert len(grams) == per_doc.sum()


def test_windows_do_not_cross_documents() -> None:
    w, c = word_hashes(pl.Series(["a b c", "d e f"]))
    g, per = ngram_hashes(w, c, 3)
    w1, c1 = word_hashes(pl.Series(["b c d"]))
    g1, _ = ngram_hashes(w1, c1, 3)
    assert per.tolist() == [1, 1] and g1[0] not in g


def test_minhash_estimates_jaccard() -> None:
    mh = MinHasher(126, 14, 5)
    near = BASE.replace("today", "yesterday")
    sig, _ = mh.signatures(
        pl.Series([BASE, near, "completely different words about steppe horses and sun", ""])
    )
    true = jaccard(BASE, near, 5)
    assert abs((sig[0] == sig[1]).mean() - true) < 0.15
    assert (sig[0] == sig[2]).mean() < 0.1
    bands = mh.band_hashes(sig)
    assert bands.shape == (4, 14) and (bands[0] == bands[1]).any() and not (bands[0] == bands[2]).any()
    assert 0.7 < mh.threshold < 0.8


def test_exact_hash_ignores_case_and_spacing() -> None:
    h = exact_hash(pl.Series(["Hello  World\n", "hello world", "hello world!"]))
    assert h[0] == h[1] != h[2]
    assert h.dtype == np.uint64


def test_minhash_requires_divisible_bands() -> None:
    with pytest.raises(ValueError):
        MinHasher(128, 14, 5)
