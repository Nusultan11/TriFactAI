"""Multilingual dense + sparse embedders (BGE-M3, mE5, Qwen3-Embedding, BM25 sparse) (ADR-0003).

Implements ``trifact.core.interfaces.Embedder`` with:
- Strict model-card query/passage prefixes (mE5, Qwen3-Embedding).
- L2-normalized dense embeddings with optional Matryoshka (MRL) truncation.
- Native BGE-M3 lexical sparse weights or deterministic hashed BM25 sparse vectors for Qdrant IDF.
- Truncation-rate telemetry so ``max_length`` clipping is always observable.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("trifact.retrieval.embedder")

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_DEFAULT_INSTRUCT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
)

# Common Kazakh agglutinative suffixes (plural, possessive, case) for optional BM25 stemming ablation.
_KK_SUFFIXES = (
    "ларындағы",
    "леріндегі",
    "дарындағы",
    "деріндегі",
    "тарындағы",
    "теріндегі",
    "ларынан",
    "лерінен",
    "дарынан",
    "дерінен",
    "тарынан",
    "терінен",
    "ларында",
    "лерінде",
    "дарында",
    "дерінде",
    "тарында",
    "терінде",
    "лардың",
    "лердің",
    "дардың",
    "дердің",
    "тардың",
    "тердің",
    "лардан",
    "лерден",
    "дардан",
    "дерден",
    "тардан",
    "терден",
    "лармен",
    "лермен",
    "дармен",
    "дермен",
    "тармен",
    "термен",
    "ларға",
    "лерге",
    "дарға",
    "дерге",
    "тарға",
    "терге",
    "ларда",
    "лерде",
    "дарда",
    "дерде",
    "тарда",
    "терде",
    "ларды",
    "лерді",
    "дарды",
    "дерді",
    "тарды",
    "терді",
    "ның",
    "нің",
    "дың",
    "дің",
    "тың",
    "тің",
    "дан",
    "ден",
    "тан",
    "тен",
    "нан",
    "нен",
    "мен",
    "бен",
    "пен",
    "лар",
    "лер",
    "дар",
    "дер",
    "тар",
    "тер",
    "ға",
    "ге",
    "қа",
    "ке",
    "да",
    "де",
    "та",
    "те",
    "ны",
    "ні",
    "ды",
    "ді",
    "ты",
    "ті",
)


def detect_model_family(model_name_or_path: str) -> str:
    """Infer embedder family (``bge_m3``, ``me5_instruct``, ``me5``, ``qwen3``, ``generic``)."""
    low = model_name_or_path.lower()
    if "bge-m3" in low or "bge_m3" in low:
        return "bge_m3"
    if "e5" in low and "instruct" in low:
        return "me5_instruct"
    if "e5" in low:
        return "me5"
    if "qwen3" in low and "embed" in low:
        return "qwen3"
    return "generic"


def format_query_text(text: str, family: str, instruction: str | None = None) -> str:
    """Apply model-card query prefix strictly per family."""
    t = (text or "").strip()
    if family == "me5":
        return f"query: {t}"
    if family in {"me5_instruct", "qwen3"}:
        prefix = instruction if instruction is not None else _DEFAULT_INSTRUCT
        return f"{prefix}{t}"
    return t


def format_passage_text(text: str, family: str, title: str | None = None) -> str:
    """Apply model-card passage prefix (and optional contextual header)."""
    body = (text or "").strip()
    if title:
        body = f"{title.strip()}\n{body}" if body else title.strip()
    if family == "me5":
        return f"passage: {body}"
    return body


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization (returns float32 array)."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 1e-12 else arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return arr / norms


def apply_mrl(vectors: np.ndarray, mrl_dim: int | None) -> np.ndarray:
    """Truncate dense vectors to ``mrl_dim`` and re-normalize L2."""
    arr = np.asarray(vectors, dtype=np.float32)
    if mrl_dim is not None:
        if mrl_dim <= 0 or mrl_dim > arr.shape[-1]:
            raise ValueError(f"Invalid mrl_dim={mrl_dim} for embedding dim={arr.shape[-1]}")
        arr = arr[..., :mrl_dim]
    return l2_normalize(arr)


def _stem_kk(token: str) -> str:
    if len(token) < 5:
        return token
    for suf in _KK_SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[: -len(suf)]
    return token


def tokenize_sparse(text: str, *, stem_kk: bool = False) -> list[str]:
    """Unicode word tokenization with optional Kazakh suffix stripping."""
    toks = _WORD_RE.findall((text or "").lower())
    if stem_kk:
        return [_stem_kk(t) for t in toks]
    return toks


def token_to_sparse_id(token: str, buckets: int = 2_000_003) -> int:
    """Deterministic uint31 sparse index for Qdrant IDF sparse vectors."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") % buckets


class BM25SparseEncoder:
    """Stateless TF sparse encoder compatible with Qdrant ``Modifier.IDF`` sparse vectors."""

    def __init__(self, *, k1: float = 0.9, b: float = 0.4, avgdl: float = 120.0, stem_kk: bool = False):
        self.k1 = k1
        self.b = b
        self.avgdl = avgdl
        self.stem_kk = stem_kk

    def encode_documents(self, texts: Sequence[str]) -> list[dict[int, float]]:
        out: list[dict[int, float]] = []
        for t in texts:
            toks = tokenize_sparse(t, stem_kk=self.stem_kk)
            dl = len(toks)
            if dl == 0:
                out.append({})
                continue
            counts = Counter(toks)
            norm = self.k1 * (1.0 - self.b + self.b * (dl / max(self.avgdl, 1.0)))
            vec: dict[int, float] = {}
            for tok, tf in counts.items():
                idx = token_to_sparse_id(tok)
                w = float(tf * (self.k1 + 1.0) / (tf + norm))
                vec[idx] = vec.get(idx, 0.0) + w
            out.append(vec)
        return out

    def encode_queries(self, texts: Sequence[str]) -> list[dict[int, float]]:
        out: list[dict[int, float]] = []
        for t in texts:
            toks = tokenize_sparse(t, stem_kk=self.stem_kk)
            if not toks:
                out.append({})
                continue
            vec: dict[int, float] = {}
            for tok in set(toks):
                idx = token_to_sparse_id(tok)
                vec[idx] = 1.0
            out.append(vec)
        return out


@dataclass
class TruncationStats:
    total_texts: int = 0
    truncated_texts: int = 0
    max_length: int = 512

    @property
    def truncation_rate(self) -> float:
        return self.truncated_texts / self.total_texts if self.total_texts else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_texts": self.total_texts,
            "truncated_texts": self.truncated_texts,
            "max_length": self.max_length,
            "truncation_rate": round(self.truncation_rate, 6),
        }


@dataclass
class HashEmbedder:
    """Deterministic lightweight embedder for fast unit/integration tests (no model download)."""

    dim: int = 64
    version: str = "hash-embedder@test"
    mrl_dim: int | None = None
    sparse_encoder: BM25SparseEncoder = field(default_factory=BM25SparseEncoder)

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        toks = tokenize_sparse(text)
        if not toks:
            vec[0] = 1.0
            return vec
        for tok in toks:
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if (h[4] & 1) == 0 else -1.0
            vec[idx] += sign
        return vec

    def embed_dense(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        mat = np.stack([self._encode_one(t) for t in texts], axis=0)
        mat = apply_mrl(mat, self.mrl_dim)
        return mat.tolist()

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_dense(texts)

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_dense(texts)

    def embed_sparse(self, texts: Sequence[str]) -> list[dict[int, float]]:
        return self.sparse_encoder.encode_documents(texts)

    def embed_sparse_queries(self, texts: Sequence[str]) -> list[dict[int, float]]:
        return self.sparse_encoder.encode_queries(texts)


class MultilingualEmbedder:
    """Production multilingual embedder supporting BGE-M3, mE5, and Qwen3-Embedding.

    Satisfies ``trifact.core.interfaces.Embedder``.
    """

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-m3",
        *,
        version: str | None = None,
        family: str | None = None,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 64,
        use_fp16: bool = True,
        mrl_dim: int | None = None,
        use_bge_sparse: bool = True,
        stem_kk_bm25: bool = False,
        instruction: str | None = None,
    ):
        self.model_name_or_path = str(model_name_or_path)
        self.family = family or detect_model_family(self.model_name_or_path)
        self.version = version or f"{Path(self.model_name_or_path).name}@{self.family}"
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.use_fp16 = bool(use_fp16)
        self.mrl_dim = mrl_dim
        self.use_bge_sparse = bool(use_bge_sparse and self.family == "bge_m3")
        self.instruction = instruction
        self.sparse_encoder = BM25SparseEncoder(stem_kk=stem_kk_bm25)
        self.truncation_stats = TruncationStats(max_length=self.max_length)

        import torch

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self._dtype = torch.float16 if (self.use_fp16 and self.device.startswith("cuda")) else torch.float32
        self._tokenizer: Any = None
        self._model: Any = None
        self._sparse_linear: Any = None
        self._dim: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        log.info(
            "Loading embedder %s (family=%s, device=%s, dtype=%s)",
            self.model_name_or_path,
            self.family,
            self.device,
            self._dtype,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if self.family == "qwen3" and self._tokenizer.padding_side != "left":
            self._tokenizer.padding_side = "left"
        self._model = AutoModel.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self._dtype if self.device.startswith("cuda") else torch.float32,
        )
        self._model.to(self.device)
        self._model.eval()
        hidden_size = getattr(self._model.config, "hidden_size", 1024)
        self._dim = self.mrl_dim or int(hidden_size)

        if self.use_bge_sparse:
            self._try_load_bge_sparse_head(hidden_size)

    def _try_load_bge_sparse_head(self, hidden_size: int) -> None:
        import torch

        sparse_path: Path | None = None
        local = Path(self.model_name_or_path) / "sparse_linear.pt"
        if local.is_file():
            sparse_path = local
        else:
            try:
                from huggingface_hub import hf_hub_download

                sparse_path = Path(hf_hub_download(self.model_name_or_path, "sparse_linear.pt"))
            except (OSError, RuntimeError, ValueError, ImportError) as exc:
                log.warning("Could not load BGE-M3 sparse_linear.pt (%s); falling back to BM25 sparse", exc)
                self.use_bge_sparse = False
                return
        if sparse_path is not None and sparse_path.is_file():
            state = torch.load(sparse_path, map_location="cpu", weights_only=True)
            lin = torch.nn.Linear(hidden_size, 1)
            lin.load_state_dict(state)
            lin.to(device=self.device, dtype=self._dtype if self.device.startswith("cuda") else torch.float32)
            lin.eval()
            self._sparse_linear = lin

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        assert self._dim is not None
        return self._dim

    def _pool(self, last_hidden_state: Any, attention_mask: Any) -> Any:
        import torch

        if self.family == "bge_m3":
            return last_hidden_state[:, 0]
        if self.family == "qwen3":
            # Last-token pooling (works for both left and right padding)
            left_padding = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_state[:, -1]
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
            return last_hidden_state[batch_idx, seq_lengths]
        # Mean pooling for mE5 / generic
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

    def _encode_batch(
        self, texts: Sequence[str], *, return_sparse: bool = False
    ) -> tuple[np.ndarray, list[dict[int, float]]]:
        import torch

        self._ensure_loaded()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32), []

        # Length-sort for minimal padding overhead
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        sorted_texts = [texts[i] for i in order]

        dense_parts: list[np.ndarray] = []
        sparse_out: list[dict[int, float]] = [{} for _ in range(len(texts))]
        unused_ids = set(getattr(self._tokenizer, "all_special_ids", []) or [])

        with torch.inference_mode():
            for start in range(0, len(sorted_texts), self.batch_size):
                batch = sorted_texts[start : start + self.batch_size]
                enc = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                # Track truncation rate
                lens = enc["attention_mask"].sum(dim=1).tolist()
                self.truncation_stats.total_texts += len(lens)
                self.truncation_stats.truncated_texts += sum(1 for ln in lens if ln >= self.max_length)

                enc = {k: v.to(self.device) for k, v in enc.items()}
                outputs = self._model(**enc)
                hidden = outputs.last_hidden_state
                pooled = self._pool(hidden, enc["attention_mask"])
                vec_np = pooled.detach().float().cpu().numpy()
                dense_parts.append(vec_np)

                if return_sparse and self._sparse_linear is not None:
                    weights = (
                        torch.relu(self._sparse_linear(hidden)).squeeze(-1).detach().float().cpu().numpy()
                    )
                    input_ids = enc["input_ids"].detach().cpu().numpy()
                    attn = enc["attention_mask"].detach().cpu().numpy()
                    for b_i in range(len(batch)):
                        orig_idx = order[start + b_i]
                        tok_weights: dict[int, float] = {}
                        for pos in range(input_ids.shape[1]):
                            if attn[b_i, pos] == 0:
                                continue
                            tid = int(input_ids[b_i, pos])
                            if tid in unused_ids:
                                continue
                            w = float(weights[b_i, pos])
                            if w > 0.0:
                                prev = tok_weights.get(tid, 0.0)
                                if w > prev:
                                    tok_weights[tid] = w
                        sparse_out[orig_idx] = tok_weights

        all_dense_sorted = np.concatenate(dense_parts, axis=0)
        all_dense = np.empty_like(all_dense_sorted)
        for sorted_pos, orig_idx in enumerate(order):
            all_dense[orig_idx] = all_dense_sorted[sorted_pos]
        all_dense = apply_mrl(all_dense, self.mrl_dim)

        if return_sparse and self._sparse_linear is None:
            sparse_out = self.sparse_encoder.encode_documents(texts)

        return all_dense, sparse_out

    def embed_both(
        self, texts: Sequence[str], *, is_query: bool = False
    ) -> tuple[np.ndarray, list[dict[int, float]]]:
        """Single-pass dense + sparse encoding."""
        formatted = [
            format_query_text(t, self.family, self.instruction)
            if is_query
            else format_passage_text(t, self.family)
            for t in texts
        ]
        dense, sparse = self._encode_batch(formatted, return_sparse=True)
        if not self.use_bge_sparse or self._sparse_linear is None:
            sparse = (
                self.sparse_encoder.encode_queries(texts)
                if is_query
                else self.sparse_encoder.encode_documents(texts)
            )
        return dense, sparse

    def embed_queries_np(self, texts: Sequence[str]) -> np.ndarray:
        formatted = [format_query_text(t, self.family, self.instruction) for t in texts]
        dense, _ = self._encode_batch(formatted, return_sparse=False)
        return dense

    def embed_passages_np(self, texts: Sequence[str]) -> np.ndarray:
        formatted = [format_passage_text(t, self.family) for t in texts]
        dense, _ = self._encode_batch(formatted, return_sparse=False)
        return dense

    def embed_dense(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_passages_np(texts).tolist()

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_queries_np(texts).tolist()

    def embed_sparse(self, texts: Sequence[str]) -> list[dict[int, float]]:
        if self.use_bge_sparse:
            _, sparse = self.embed_both(texts, is_query=False)
            return sparse
        return self.sparse_encoder.encode_documents(texts)

    def embed_sparse_queries(self, texts: Sequence[str]) -> list[dict[int, float]]:
        if self.use_bge_sparse:
            _, sparse = self.embed_both(texts, is_query=True)
            return sparse
        return self.sparse_encoder.encode_queries(texts)


def sparse_dot_search(
    query_vecs: Sequence[dict[int, float]],
    doc_vecs: Sequence[dict[int, float]],
    doc_ids: Sequence[str],
    top_k: int = 100,
) -> list[list[tuple[str, float]]]:
    """Exact inverted-index dot-product search over sparse dicts (for RET-2 BGE-M3 sparse eval)."""
    from scipy import sparse as sp

    if not query_vecs or not doc_vecs:
        return [[] for _ in query_vecs]
    vocab: dict[int, int] = {}
    d_rows, d_cols, d_vals = [], [], []
    for i, dv in enumerate(doc_vecs):
        for tid, w in dv.items():
            if w <= 0.0:
                continue
            j = vocab.setdefault(tid, len(vocab))
            d_rows.append(i)
            d_cols.append(j)
            d_vals.append(w)
    if not vocab:
        return [[] for _ in query_vecs]
    d_mat = sp.csr_matrix(
        (np.asarray(d_vals, dtype=np.float32), (d_rows, d_cols)),
        shape=(len(doc_vecs), len(vocab)),
    ).T.tocsr()  # terms x docs

    n_docs = len(doc_vecs)
    batch = max(16, min(1024, int(2e7 / max(n_docs, 1))))
    results: list[list[tuple[str, float]]] = []
    for start in range(0, len(query_vecs), batch):
        q_slice = query_vecs[start : start + batch]
        q_rows, q_cols, q_vals = [], [], []
        for i, qv in enumerate(q_slice):
            for tid, w in qv.items():
                j = vocab.get(tid)
                if j is not None and w > 0.0:
                    q_rows.append(i)
                    q_cols.append(j)
                    q_vals.append(w)
        q_mat = sp.csr_matrix(
            (np.asarray(q_vals, dtype=np.float32), (q_rows, q_cols)),
            shape=(len(q_slice), len(vocab)),
        )
        scores = (q_mat @ d_mat).toarray()
        kk = min(top_k, n_docs)
        for i in range(len(q_slice)):
            row = scores[i]
            top_idx = np.argpartition(-row, kk - 1)[:kk]
            top_idx = top_idx[np.argsort(-row[top_idx])]
            res = [(doc_ids[int(idx)], float(row[idx])) for idx in top_idx if row[idx] > 0.0]
            results.append(res)
    return results


def dense_exact_search(
    query_vecs: np.ndarray,
    doc_vecs: np.ndarray,
    doc_ids: Sequence[str],
    top_k: int = 100,
    batch_size: int = 512,
) -> list[list[tuple[str, float]]]:
    """Exact cosine (dot-product on L2-normalized vectors) top-k search in batched float32."""
    n_q = int(query_vecs.shape[0])
    n_d = int(doc_vecs.shape[0])
    if n_q == 0 or n_d == 0:
        return [[] for _ in range(n_q)]
    q_mat = np.asarray(query_vecs, dtype=np.float32)
    d_mat = np.asarray(doc_vecs, dtype=np.float32)
    kk = min(top_k, n_d)
    results: list[list[tuple[str, float]]] = []
    for start in range(0, n_q, batch_size):
        qb = q_mat[start : start + batch_size]
        scores = qb @ d_mat.T
        for i in range(scores.shape[0]):
            row = scores[i]
            top_idx = np.argpartition(-row, kk - 1)[:kk]
            top_idx = top_idx[np.argsort(-row[top_idx])]
            results.append([(doc_ids[int(idx)], float(row[idx])) for idx in top_idx])
    return results


def estimate_vector_bytes(n_vectors: int, dim: int, quantization: str = "binary") -> dict[str, float]:
    """Analytical memory & disk estimate (in GB) for Qdrant corpus collection (ADR-0013)."""
    fp16_disk_gb = (n_vectors * dim * 2) / (1024**3)
    if quantization == "none":
        ram_vec_gb = fp16_disk_gb
    elif quantization == "scalar":
        ram_vec_gb = (n_vectors * dim * 1) / (1024**3)
    elif quantization == "binary":
        ram_vec_gb = (n_vectors * math.ceil(dim / 8)) / (1024**3)
    elif quantization == "turbo":
        # 2-bit asymmetric/turbo quantization (~2 bits/dim)
        ram_vec_gb = (n_vectors * math.ceil(dim / 4)) / (1024**3)
    else:
        raise ValueError(f"Unknown quantization: {quantization}")
    return {
        "fp16_disk_gb": round(fp16_disk_gb, 3),
        "quant_ram_gb": round(ram_vec_gb, 3),
    }
