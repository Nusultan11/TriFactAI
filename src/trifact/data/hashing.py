"""Vectorized text hashing for dedup and decontamination (architecture §5.3).

Words = maximal runs of letters/digits of the lower-cased text, hashed by Polars (seeded, stable for
a pinned Polars version). A word n-gram hash combines the n word hashes with fixed odd multipliers
and a splitmix64 finalizer; documents shorter than n words get one shingle of all their words.
Everything works on flat numpy arrays plus per-document word counts, without Python loops over
words.

MinHash: `num_perm` random affine permutations `a*x + b mod 2^32` (a odd) of the high 32 bits of
the shingle hashes; the shingle hashes are already well mixed, and on random sets the estimate
matches theory (bias < 0.003, sd = sqrt(J(1-J)/num_perm), checked in stage 2) at 2x the speed
of 64-bit multiply-shift. Band hash = finalized weighted sum of the band's rows.
"""

from __future__ import annotations

import numpy as np
import polars as pl

SEED = 42
WORD = r"[\p{L}\p{N}]+"
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def mix64(x: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer on a uint64 array (wraparound arithmetic)."""
    x = x.astype(np.uint64, copy=True)
    x ^= x >> np.uint64(30)
    x *= _M1
    x ^= x >> np.uint64(27)
    x *= _M2
    x ^= x >> np.uint64(31)
    return x


def _odd_constants(n: int, salt: int) -> np.ndarray:
    rng = np.random.default_rng(SEED + salt)
    return rng.integers(1, 2**63, size=n, dtype=np.uint64) * np.uint64(2) + np.uint64(1)


def word_hashes(texts: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    """Flat uint64 hashes of all words and the word count of every document."""
    words = texts.fill_null("").str.to_lowercase().str.extract_all(WORD)
    counts = words.list.len().fill_null(0).to_numpy().astype(np.int64)
    flat = words.explode().drop_nulls()
    return flat.hash(seed=SEED).to_numpy().astype(np.uint64), counts


def ngram_hashes(words: np.ndarray, counts: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Word n-gram hashes (flat) and the n-gram count of every document (0 for empty documents)."""
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.uint64), np.zeros(len(counts), dtype=np.int64)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    pos = np.arange(total) - starts  # position of each word inside its document
    length = np.repeat(counts, counts)
    k = _odd_constants(n, salt=n)
    acc = np.zeros(total, dtype=np.uint64)
    for j in range(min(n, total)):
        shifted = np.zeros(total, dtype=np.uint64)
        shifted[: total - j] = words[j:]
        shifted[pos + j >= length] = 0  # window ran past the end of its document
        acc += shifted * k[j]
    valid = pos <= np.maximum(length - n, 0)
    grams = np.where(counts > 0, np.maximum(counts - n + 1, 1), 0)
    return mix64(acc[valid]), grams


class MinHasher:
    def __init__(self, num_perm: int, bands: int, ngram: int):
        if num_perm % bands:
            raise ValueError(f"num_perm {num_perm} is not divisible by bands {bands}")
        self.num_perm, self.bands, self.rows, self.ngram = num_perm, bands, num_perm // bands, ngram
        rng = np.random.default_rng(SEED)
        self.a = (rng.integers(0, 2**31, size=num_perm, dtype=np.uint64) * 2 + 1).astype(np.uint32)
        self.b = rng.integers(0, 2**32, size=num_perm, dtype=np.uint64).astype(np.uint32)
        self.band_k = _odd_constants(self.rows, salt=1000)

    @property
    def threshold(self) -> float:
        """Jaccard at which the LSH S-curve 1-(1-s^r)^b crosses 1/2 (approximately (1/b)^(1/r))."""
        return float((1 / self.bands) ** (1 / self.rows))

    def signatures(self, texts: pl.Series) -> tuple[np.ndarray, np.ndarray]:
        """(n_docs, num_perm) uint32 signatures and the word count of every document."""
        words, counts = word_hashes(texts)
        grams, per_doc = ngram_hashes(words, counts, self.ngram)
        sig = np.full((len(counts), self.num_perm), np.iinfo(np.uint32).max, dtype=np.uint32)
        has = per_doc > 0
        if grams.size:
            offsets = (np.cumsum(per_doc) - per_doc)[has]
            g32 = (grams >> np.uint64(32)).astype(np.uint32)
            buf = np.empty_like(g32)
            for p in range(self.num_perm):
                np.multiply(g32, self.a[p], out=buf)
                np.add(buf, self.b[p], out=buf)
                sig[has, p] = np.minimum.reduceat(buf, offsets)
        return sig, counts

    def band_hashes(self, sig: np.ndarray) -> np.ndarray:
        """(n_docs, bands) uint64 band hashes."""
        s = sig.astype(np.uint64).reshape(len(sig), self.bands, self.rows)
        return mix64((s * self.band_k).sum(axis=2, dtype=np.uint64))


def exact_hash(texts: pl.Series) -> np.ndarray:
    """Hash of the lower-cased text with whitespace runs collapsed."""
    norm = texts.fill_null("").str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars()
    return norm.hash(seed=SEED).to_numpy().astype(np.uint64)


def jaccard(a: str, b: str, n: int) -> float:
    """Exact word n-gram Jaccard of two texts (for checking LSH candidates on a sample)."""

    def grams(t: str) -> set[tuple[str, ...]]:
        w = pl.Series([t]).str.to_lowercase().str.extract_all(WORD)[0].to_list()
        if len(w) < n:
            return {tuple(w)} if w else set()
        return set(zip(*(w[i:] for i in range(n)), strict=False))

    ga, gb = grams(a), grams(b)
    return len(ga & gb) / len(ga | gb) if ga or gb else 1.0
