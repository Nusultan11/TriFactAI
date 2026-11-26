"""Evidence verification via NLI over 3-sentence passage windows (VER-0, ADR-0022).

Implements ``trifact.core.interfaces.EvidenceVerifier`` for Stage 7 preliminary faithfulness
evaluation using ``MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`` (VER-0) with
identical 3-sentence sliding windows, plus a deterministic lexical verifier for offline tests.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from trifact.core.schemas import Citation, Claim, ScoredChunk, Verdict
from trifact.verification.claims import make_sentence_windows, split_claims

log = logging.getLogger("trifact.verification.nli")

DEFAULT_VER0_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


def _resolve_nli_label_indices(id2label: Mapping[int, str] | Mapping[str, str]) -> tuple[int, int, int]:
    """Return ``(entailment_idx, neutral_idx, contradiction_idx)`` from model config ``id2label``."""
    ent = neu = con = -1
    for k, v in id2label.items():
        idx = int(k)
        low = str(v).lower()
        if "entail" in low:
            ent = idx
        elif "contra" in low:
            con = idx
        elif "neutral" in low:
            neu = idx
    if ent < 0 or con < 0:
        # Standard MNLI / mDeBERTa-xnli default: 0=entailment, 1=neutral, 2=contradiction
        return 0, 1, 2
    if neu < 0:
        neu = ({0, 1, 2} - {ent, con}).pop()
    return ent, neu, con


@dataclass
class LexicalOverlapVerifier:
    """Fast deterministic ``EvidenceVerifier`` for offline unit/integration tests."""

    window_size: int = 3
    support_threshold: float = 0.45

    def verify(
        self,
        raw_answer: str,
        context: Sequence[ScoredChunk],
        *,
        id_to_chunk_id: Mapping[str, str] | None = None,
    ) -> list[Claim]:
        chunk_map = {sc.chunk.chunk_id: sc for sc in context}
        parsed = split_claims(raw_answer, context, id_to_chunk_id=id_to_chunk_id)
        claims: list[Claim] = []
        for item in parsed:
            if not item.valid_chunk_ids:
                claims.append(
                    Claim(
                        text=item.text,
                        citations=[],
                        verdict=Verdict.NOT_ENOUGH_INFO,
                        nli_score=0.0,
                    )
                )
                continue
            h_toks = {w for w in re.findall(r"\w+", item.text.lower()) if len(w) > 2}
            best_score = 0.0
            best_cid = item.valid_chunk_ids[0]
            best_quote = chunk_map[best_cid].chunk.text[:240]
            for cid in item.valid_chunk_ids:
                text = chunk_map[cid].chunk.text
                for win in make_sentence_windows(text, window_size=self.window_size):
                    p_toks = {w for w in re.findall(r"\w+", win.lower()) if len(w) > 2}
                    score = (len(h_toks & p_toks) / len(h_toks)) if h_toks else 0.0
                    if score > best_score:
                        best_score = score
                        best_cid = cid
                        best_quote = win

            verdict = Verdict.SUPPORTED if best_score >= self.support_threshold else Verdict.NOT_ENOUGH_INFO
            citations = [
                Citation(
                    chunk_id=cid,
                    document_id=chunk_map[cid].chunk.document_id,
                    page=chunk_map[cid].chunk.page,
                    quote=best_quote if cid == best_cid else chunk_map[cid].chunk.text[:240],
                )
                for cid in item.valid_chunk_ids
            ]
            claims.append(
                Claim(
                    text=item.text,
                    citations=citations,
                    verdict=verdict,
                    nli_score=round(float(best_score), 4),
                )
            )
        return claims


@dataclass
class NLIEvidenceVerifier:
    """Stage 7 ``VER-0`` claim verifier using 3-sentence passage windows (ADR-0022).

    Scores ``NLI(premise=3-sentence window, hypothesis=claim)`` over all windows of cited
    passages, takes the maximum entailment probability across windows/citations, and stores
    the winning window as ``Citation.quote``.
    """

    model_name_or_path: str = DEFAULT_VER0_MODEL
    revision: str | None = "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"
    device: str | None = None
    batch_size: int = 32
    max_length: int = 512
    window_size: int = 3
    entail_threshold: float = 0.5
    contradict_threshold: float = 0.5

    def __post_init__(self) -> None:
        self._tokenizer: Any = None
        self._model: Any = None
        self._label_idx: tuple[int, int, int] = (0, 1, 2)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs: dict[str, Any] = {}
        if self.revision and "/" in self.model_name_or_path and not self.model_name_or_path.startswith("."):
            kwargs["revision"] = self.revision
        log.info("Loading NLIEvidenceVerifier %s (device=%s)", self.model_name_or_path, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, **kwargs)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path, **kwargs)
        if self.device.startswith("cuda"):
            self._model.half()
        self._model.to(self.device)
        self._model.eval()
        self._label_idx = _resolve_nli_label_indices(getattr(self._model.config, "id2label", {}))

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[tuple[float, float, float]]:
        """Score ``(premise, hypothesis)`` pairs -> ``[(p_entail, p_neutral, p_contradict), ...]``."""
        if not pairs:
            return []
        import torch

        self._ensure_loaded()
        ent_i, neu_i, con_i = self._label_idx
        out: list[tuple[float, float, float]] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            premises = [p for p, _ in batch]
            hyps = [h for _, h in batch]
            enc = self._tokenizer(
                premises,
                hyps,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self._model(**enc).logits
                probs = torch.softmax(logits.float(), dim=-1).cpu().tolist()
            for row in probs:
                out.append((float(row[ent_i]), float(row[neu_i]), float(row[con_i])))
        return out

    def verify(
        self,
        raw_answer: str,
        context: Sequence[ScoredChunk],
        *,
        id_to_chunk_id: Mapping[str, str] | None = None,
    ) -> list[Claim]:
        """Verify every atomic claim in ``raw_answer`` against its cited 3-sentence windows."""
        chunk_map = {sc.chunk.chunk_id: sc for sc in context}
        parsed = split_claims(raw_answer, context, id_to_chunk_id=id_to_chunk_id)
        if not parsed:
            return []

        # Collect all (claim_idx, chunk_id, window_text) pairs for batched NLI scoring
        pair_meta: list[tuple[int, str, str]] = []
        pairs: list[tuple[str, str]] = []
        for idx, item in enumerate(parsed):
            for cid in item.valid_chunk_ids:
                sc = chunk_map.get(cid)
                if sc is None:
                    continue
                for win in make_sentence_windows(sc.chunk.text, window_size=self.window_size):
                    pair_meta.append((idx, cid, win))
                    pairs.append((win, item.text))

        scored = self.score_pairs(pairs)
        per_claim: dict[int, list[tuple[str, str, float, float, float]]] = {}
        for (c_idx, cid, win), (p_ent, p_neu, p_con) in zip(pair_meta, scored, strict=True):
            per_claim.setdefault(c_idx, []).append((cid, win, p_ent, p_neu, p_con))

        claims: list[Claim] = []
        for idx, item in enumerate(parsed):
            cands = per_claim.get(idx, [])
            if not cands:
                claims.append(
                    Claim(
                        text=item.text,
                        citations=[],
                        verdict=Verdict.NOT_ENOUGH_INFO,
                        nli_score=0.0,
                    )
                )
                continue

            # Winning window maximizes entailment probability (ADR-0015 / ADR-0022)
            best_cid, best_win, best_ent, _, _ = max(cands, key=lambda x: x[2])
            max_con = max(x[4] for x in cands)

            if best_ent >= self.entail_threshold:
                verdict = Verdict.SUPPORTED
                score = best_ent
            elif max_con >= self.contradict_threshold:
                verdict = Verdict.CONTRADICTED
                score = max_con
            else:
                verdict = Verdict.NOT_ENOUGH_INFO
                score = best_ent

            citations = [
                Citation(
                    chunk_id=cid,
                    document_id=chunk_map[cid].chunk.document_id,
                    page=chunk_map[cid].chunk.page,
                    quote=best_win if cid == best_cid else chunk_map[cid].chunk.text[:240],
                )
                for cid in item.valid_chunk_ids
                if cid in chunk_map
            ]
            claims.append(
                Claim(
                    text=item.text,
                    citations=citations,
                    verdict=verdict,
                    nli_score=round(float(score), 4),
                )
            )
        return claims
