"""Cross-encoder rerankers (BAAI/bge-reranker-v2-m3, Qwen3-Reranker-0.6B/4B) with Platt calibration.

Implements ``trifact.core.interfaces.Reranker``.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trifact.core.schemas import ScoredChunk

log = logging.getLogger("trifact.retrieval.reranker")

_QWEN_RERANK_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
    "<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n"
)
_QWEN_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def detect_reranker_family(model_name_or_path: str) -> str:
    low = model_name_or_path.lower()
    if "qwen3" in low and "rerank" in low:
        return "qwen3_reranker"
    return "cross_encoder"


def platt_sigmoid(logit: float, a: float = 1.0, b: float = 0.0) -> float:
    z = max(-60.0, min(60.0, a * logit + b))
    return 1.0 / (1.0 + math.exp(-z))


def fit_platt_scaling(
    logits: Sequence[float], labels: Sequence[int], lr: float = 0.05, steps: int = 300
) -> dict[str, float]:
    """Fit 1D logistic (Platt) calibration ``p = sigmoid(a * logit + b)`` on validation pairs."""
    if not logits or len(logits) != len(labels):
        return {"a": 1.0, "b": 0.0}
    a, b = 1.0, 0.0
    n = float(len(logits))
    for _ in range(steps):
        ga = 0.0
        gb = 0.0
        for x, y in zip(logits, labels, strict=True):
            p = platt_sigmoid(x, a, b)
            err = p - (1.0 if y else 0.0)
            ga += err * x
            gb += err
        a -= lr * (ga / n)
        b -= lr * (gb / n)
    return {"a": round(float(a), 6), "b": round(float(b), 6)}


@dataclass
class LexicalOverlapReranker:
    """Fast deterministic reranker for unit/integration tests without model downloads."""

    version: str = "lexical-reranker@test"

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for q, d in pairs:
            q_toks = set(q.lower().split())
            d_toks = set(d.lower().split())
            if not q_toks or not d_toks:
                scores.append(0.0)
            else:
                scores.append(len(q_toks & d_toks) / len(q_toks))
        return scores

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates or top_k <= 0:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self.score_pairs(pairs)
        scored = sorted(
            zip(candidates, scores, strict=True),
            key=lambda kv: (-kv[1], kv[0].chunk.chunk_id),
        )[:top_k]
        out: list[ScoredChunk] = []
        for rank_1, (cand, sc) in enumerate(scored, start=1):
            ranks = dict(cand.ranks)
            ranks["rerank"] = rank_1
            out.append(ScoredChunk(chunk=cand.chunk, score=float(sc), ranks=ranks))
        return out


@dataclass
class CrossEncoderReranker:
    """Production cross-encoder reranker supporting ``bge-reranker-v2-m3`` and ``Qwen3-Reranker``.

    Satisfies ``trifact.core.interfaces.Reranker``.
    """

    model_name_or_path: str = "BAAI/bge-reranker-v2-m3"
    version: str = ""
    family: str = ""
    device: str | None = None
    max_length: int = 512
    batch_size: int = 32
    use_fp16: bool = True
    calibration: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.family:
            self.family = detect_reranker_family(self.model_name_or_path)
        if not self.version:
            self.version = f"{Path(self.model_name_or_path).name}@{self.family}"
        self._tokenizer: Any = None
        self._model: Any = None
        self._yes_id: int | None = None
        self._no_id: int | None = None
        # Auto-load Platt calibration if present alongside local checkpoint
        cal_path = Path(self.model_name_or_path) / "calibration.json"
        if not self.calibration and cal_path.is_file():
            try:
                raw = json.loads(cal_path.read_text(encoding="utf-8"))
                self.calibration = raw.get("platt_by_lang", raw)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                log.warning("Failed loading reranker calibration %s: %s", cal_path, exc)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if (self.use_fp16 and self.device.startswith("cuda")) else torch.float32
        log.info(
            "Loading reranker %s (family=%s, device=%s, dtype=%s)",
            self.model_name_or_path,
            self.family,
            self.device,
            dtype,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        if self.family == "qwen3_reranker":
            self._tokenizer.padding_side = "left"
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=dtype if self.device.startswith("cuda") else torch.float32,
            )
            self._yes_id = self._tokenizer.convert_tokens_to_ids("yes")
            self._no_id = self._tokenizer.convert_tokens_to_ids("no")
        else:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path,
                torch_dtype=dtype if self.device.startswith("cuda") else torch.float32,
            )
        self._model.to(self.device)
        self._model.eval()

    def score_pairs(
        self, pairs: Sequence[tuple[str, str]], *, lang: str | None = None, calibrated: bool = False
    ) -> list[float]:
        """Score ``(query, passage)`` pairs with length-sorted batching."""
        import torch

        if not pairs:
            return []
        self._ensure_loaded()

        # Length-sort to minimize padding waste
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        sorted_pairs = [pairs[i] for i in order]
        raw_scores_sorted: list[float] = []

        with torch.inference_mode():
            for start in range(0, len(sorted_pairs), self.batch_size):
                batch = sorted_pairs[start : start + self.batch_size]
                if self.family == "qwen3_reranker":
                    prompts = [
                        f"{_QWEN_RERANK_PREFIX}<Query>: {q}\n<Document>: {d}{_QWEN_RERANK_SUFFIX}"
                        for q, d in batch
                    ]
                    enc = self._tokenizer(
                        prompts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    enc = {k: v.to(self.device) for k, v in enc.items()}
                    logits = self._model(**enc).logits[:, -1, :]
                    yes_l = logits[:, self._yes_id]
                    no_l = logits[:, self._no_id]
                    scores_b = (yes_l - no_l).detach().float().cpu().tolist()
                else:
                    qs = [q for q, _ in batch]
                    ds = [d for _, d in batch]
                    enc = self._tokenizer(
                        qs,
                        ds,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    enc = {k: v.to(self.device) for k, v in enc.items()}
                    logits = self._model(**enc).logits
                    if logits.ndim == 2 and logits.shape[1] == 1:
                        scores_b = logits.squeeze(-1).detach().float().cpu().tolist()
                    elif logits.ndim == 2 and logits.shape[1] >= 2:
                        scores_b = (logits[:, 1] - logits[:, 0]).detach().float().cpu().tolist()
                    else:
                        scores_b = logits.view(-1).detach().float().cpu().tolist()
                raw_scores_sorted.extend(float(s) for s in scores_b)

        raw_scores = [0.0] * len(pairs)
        for pos, orig_idx in enumerate(order):
            raw_scores[orig_idx] = raw_scores_sorted[pos]

        if calibrated and lang and lang in self.calibration:
            cal = self.calibration[lang]
            a, b = float(cal.get("a", 1.0)), float(cal.get("b", 0.0))
            return [platt_sigmoid(s, a, b) for s in raw_scores]
        return raw_scores

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates or top_k <= 0:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self.score_pairs(pairs)
        scored = sorted(
            zip(candidates, scores, strict=True),
            key=lambda kv: (-kv[1], kv[0].chunk.chunk_id),
        )[:top_k]
        out: list[ScoredChunk] = []
        for rank_1, (cand, sc) in enumerate(scored, start=1):
            ranks = dict(cand.ranks)
            ranks["rerank"] = rank_1
            out.append(ScoredChunk(chunk=cand.chunk, score=float(sc), ranks=ranks))
        return out
