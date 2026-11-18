"""Sparse BM25 (Okapi, k1 = 0.9, b = 0.4 as in Anserini) for hard negatives and BM25-ease analysis.

Documents and queries are lower-cased word tokens (Unicode letters/digits); the index is a SciPy CSR
term-document matrix of BM25 weights, queries are binary term vectors, so a batch of queries is
one sparse matrix product. Suited to the pools of stage 3 (up to a few million passages).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from scipy import sparse

WORD = re.compile(r"\w+", re.UNICODE)


def _tok(text: str) -> list[str]:
    return WORD.findall((text or "").lower())


class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 0.9, b: float = 0.4):
        vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        lengths = np.zeros(len(docs), dtype=np.float32)
        for i, d in enumerate(docs):
            toks = _tok(d)
            lengths[i] = len(toks)
            tf: dict[int, int] = {}
            for t in toks:
                j = vocab.setdefault(t, len(vocab))
                tf[j] = tf.get(j, 0) + 1
            rows.extend([i] * len(tf))
            cols.extend(tf.keys())
            vals.extend(tf.values())
        self.vocab = vocab
        tfm = sparse.csr_matrix((np.asarray(vals, np.float32), (rows, cols)), shape=(len(docs), len(vocab)))
        df = np.bincount(tfm.indices, minlength=len(vocab)).astype(np.float32)
        idf = np.log(1 + (len(docs) - df + 0.5) / (df + 0.5)).astype(np.float32)
        avg = float(lengths.mean()) if len(docs) else 1.0
        norm = k1 * (1 - b + b * lengths / max(avg, 1e-9))
        coo = tfm.tocoo()
        w = coo.data * (k1 + 1) / (coo.data + norm[coo.row]) * idf[coo.col]
        self.matrix = sparse.csr_matrix((w, (coo.row, coo.col)), shape=tfm.shape).T.tocsr()  # terms x docs

    def search(self, queries: Sequence[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k (doc indices, scores) per query, -1 where fewer than k documents match."""
        n_docs = self.matrix.shape[1]
        batch = max(16, min(2048, int(2e7 / max(n_docs, 1))))  # dense score block <= ~80 MB
        idx = np.full((len(queries), k), -1, dtype=np.int64)
        sc = np.zeros((len(queries), k), dtype=np.float32)
        for s in range(0, len(queries), batch):
            qs = queries[s : s + batch]
            rows, cols = [], []
            for i, q in enumerate(qs):
                for t in set(_tok(q)):
                    j = self.vocab.get(t)
                    if j is not None:
                        rows.append(i)
                        cols.append(j)
            qm = sparse.csr_matrix(
                (np.ones(len(rows), np.float32), (rows, cols)), shape=(len(qs), len(self.vocab))
            )
            scores = (qm @ self.matrix).toarray()
            for i in range(len(qs)):
                row = scores[i]
                kk = min(k, n_docs)
                top = np.argpartition(-row, kk - 1)[:kk]
                top = top[np.argsort(-row[top])]
                top = top[row[top] > 0]
                idx[s + i, : len(top)] = top
                sc[s + i, : len(top)] = row[top]
        return idx, sc
