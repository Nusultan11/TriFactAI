"""Multilingual NER inference, subword alignment, temperature calibration, and NER-5 optimization.

Implements ``EntityExtractor`` backends for:
- ``NER-0``: ``GLiNERExtractor`` (zero-shot ``urchade/gliner_multi-v2.1``).
- ``NER-1..4``: ``TokenClassificationExtractor`` (PyTorch HF XLM-R base/large & KazNERD-25).
- ``NER-5``: ``OnnxNERExtractor`` (vocabulary-trimmed + ONNX Runtime dynamic int8 CPU / fp16 GPU).
- ``CombinedExtractor``: merges neural PER/ORG/LOC mentions with rule-based DATE mentions.

All tokenizers use first-subword label alignment (``-100`` on continuation subwords) and
overlapping sliding windows for sequences exceeding ``max_length`` (never silent truncation).
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from trifact.core.schemas import Chunk, EntityMention, EntityType, Lang
from trifact.evaluation.eval_ner import extract_iob2_entities
from trifact.evaluation.stats import equal_mass_ece, precision_coverage_curve, select_annotation_threshold
from trifact.extraction.dates import RuleBasedDateExtractor
from trifact.ingestion.segment import tokenize as word_tokenize

log = logging.getLogger("trifact.extraction.ner")

CORE_BIO_LABELS: tuple[str, ...] = ("O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC")
CORE_LABEL2ID: dict[str, int] = {lab: i for i, lab in enumerate(CORE_BIO_LABELS)}
CORE_ID2LABEL: dict[int, str] = dict(enumerate(CORE_BIO_LABELS))

_CAP_RE = re.compile(r"^[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ]")
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$")


# --------------------------------------------------------------------------- silver cleaning & alignment


def clean_silver_tags(
    tokens: Sequence[str],
    tags: Sequence[str],
    mask_unlabeled_capitals: bool = True,
) -> list[str | None]:
    """Sanitize silver NER tags based on Stage 3 audit findings:
    1. Drop punctuation-only silver spans (e.g. ``[.|LOC]`` artifact).
    2. Optionally mask non-initial unlabeled capitalized tokens as ``None`` (mapped to ``-100``)
       so partial-annotation false-``O`` noise does not penalize true entities.
    """
    cleaned = list(tags)
    # 1. Remove punctuation-only spans
    for s, e, _ in extract_iob2_entities(cleaned):
        surf = " ".join(tokens[s:e]).strip()
        if _PUNCT_ONLY_RE.match(surf):
            for i in range(s, e):
                cleaned[i] = "O"
    # 2. Partial-annotation masking on non-initial capitalized O tokens
    out: list[str | None] = []
    for i, (tok, tag) in enumerate(zip(tokens, cleaned, strict=True)):
        if mask_unlabeled_capitals and i > 0 and tag == "O" and _CAP_RE.match(tok):
            out.append(None)
        else:
            out.append(tag)
    return out


def align_word_labels_to_subwords(
    word_ids: Sequence[int | None],
    word_labels: Sequence[str | None],
    label2id: dict[str, int],
) -> list[int]:
    """Assign entity label ID to the FIRST subword of each word; ``-100`` to continuations,
    special tokens, and masked (``None``) tokens."""
    aligned: list[int] = []
    prev_word: int | None = None
    for wid in word_ids:
        if wid is None or wid == prev_word:
            aligned.append(-100)
        else:
            lab = word_labels[wid] if wid < len(word_labels) else None
            aligned.append(label2id[lab] if (lab is not None and lab in label2id) else -100)
        prev_word = wid
    return aligned


@dataclass
class TokenWindow:
    input_ids: list[int]
    attention_mask: list[int]
    word_ids: list[int | None]
    labels: list[int] | None = None


def build_sliding_windows(
    tokens: Sequence[str],
    tokenizer: Any,
    max_length: int = 256,
    stride: int = 128,
    word_labels: Sequence[str | None] | None = None,
    label2id: dict[str, int] | None = None,
) -> list[TokenWindow]:
    """Tokenize word-split ``tokens`` into one or more overlapping windows of at most ``max_length``."""
    enc = tokenizer(
        list(tokens),
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        stride=min(stride, max(0, max_length - 16)),
        return_overflowing_tokens=True,
        add_special_tokens=True,
    )
    n_win = len(enc["input_ids"])
    windows: list[TokenWindow] = []
    for w in range(n_win):
        wids = enc.word_ids(batch_index=w)
        labs: list[int] | None = None
        if word_labels is not None and label2id is not None:
            labs = align_word_labels_to_subwords(wids, word_labels, label2id)
        windows.append(
            TokenWindow(
                input_ids=list(enc["input_ids"][w]),
                attention_mask=list(enc["attention_mask"][w]),
                word_ids=list(wids),
                labels=labs,
            )
        )
    return windows


def merge_window_word_logits(
    n_words: int,
    windows_word_ids: Sequence[Sequence[int | None]],
    windows_logits: Sequence[np.ndarray],
) -> np.ndarray:
    """Merge first-subword logits across overlapping windows by selecting the most central window per word."""
    num_labels = int(windows_logits[0].shape[-1])
    merged = np.zeros((n_words, num_labels), dtype=np.float32)
    best_dist = np.full(n_words, 1e9, dtype=np.float32)

    for wids, logits in zip(windows_word_ids, windows_logits, strict=True):
        seq_len = len(wids)
        center = seq_len / 2.0
        seen_in_win: set[int] = set()
        for pos, wid in enumerate(wids):
            if wid is None or wid in seen_in_win or wid >= n_words:
                continue
            seen_in_win.add(wid)
            dist = abs(pos - center)
            if dist < best_dist[wid]:
                best_dist[wid] = dist
                merged[wid] = logits[pos]
    return merged


def decode_word_logits_to_bio(
    word_logits: np.ndarray,
    id2label: dict[int, str],
    temperature: float = 1.0,
    score_agg: str = "mean",
) -> tuple[list[str], list[tuple[int, int, str, float]]]:
    """Decode per-word logits ``(n_words, n_labels)`` into repaired IOB2 tags and scored entity spans."""
    if word_logits.shape[0] == 0:
        return [], []
    scaled = word_logits.astype(np.float64) / max(float(temperature), 1e-4)
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=-1, keepdims=True)
    pred_ids = np.argmax(probs, axis=-1)
    max_probs = np.max(probs, axis=-1)

    raw_tags = [id2label[int(i)] for i in pred_ids]
    # Repair illegal I-X after O or different type
    tags: list[str] = []
    prev = "O"
    for t in raw_tags:
        if t.startswith("I-") and (prev == "O" or prev[2:] != t[2:]):
            t = "B-" + t[2:]
        tags.append(t)
        prev = t

    scored_spans: list[tuple[int, int, str, float]] = []
    for s, e, lab in extract_iob2_entities(tags):
        span_probs = [float(max_probs[i]) for i in range(s, e)]
        sc = min(span_probs) if score_agg == "min" else (sum(span_probs) / len(span_probs))
        scored_spans.append((s, e, lab, round(sc, 6)))
    return tags, scored_spans


# --------------------------------------------------------------------------- Calibration


def fit_temperature_scaling(
    word_logits_list: Sequence[np.ndarray],
    gold_tag_ids_list: Sequence[Sequence[int]],
    t_grid: Sequence[float] | None = None,
) -> dict[str, float]:
    """Fit scalar temperature ``T > 0`` minimizing token-level negative log-likelihood on validation data.

    Uses a fine grid search followed by golden-section refinement (zero external optimizer dependency).
    """
    logits_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    for lg, y in zip(word_logits_list, gold_tag_ids_list, strict=True):
        if len(y) == 0 or lg.shape[0] == 0:
            continue
        arr_y = np.asarray(y[: lg.shape[0]], dtype=np.int64)
        mask = arr_y >= 0
        if np.any(mask):
            logits_parts.append(lg[: len(arr_y)][mask].astype(np.float64))
            labels_parts.append(arr_y[mask])

    if not logits_parts:
        return {"temperature": 1.0, "nll_before": 0.0, "nll_after": 0.0}

    logits = np.concatenate(logits_parts, axis=0)
    labels = np.concatenate(labels_parts, axis=0)
    idx = np.arange(len(labels))

    def _nll(temp: float) -> float:
        scaled = logits / max(temp, 1e-4)
        m = np.max(scaled, axis=-1, keepdims=True)
        lse = m.squeeze(-1) + np.log(np.sum(np.exp(scaled - m), axis=-1))
        return float(np.mean(lse - scaled[idx, labels]))

    nll_before = _nll(1.0)
    grid = list(t_grid) if t_grid is not None else [round(0.3 + 0.05 * i, 3) for i in range(55)]
    best_t = min(grid, key=_nll)

    # Golden-section refinement in [best_t - 0.08, best_t + 0.08]
    a = max(0.05, best_t - 0.08)
    b = best_t + 0.08
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(25):
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        if _nll(c) < _nll(d):
            b = d
        else:
            a = c
    opt_t = round((a + b) / 2.0, 4)
    nll_after = _nll(opt_t)
    return {
        "temperature": opt_t,
        "nll_before": round(nll_before, 6),
        "nll_after": round(nll_after, 6),
    }


def evaluate_entity_calibration(
    word_logits_list: Sequence[np.ndarray],
    gold_tags_list: Sequence[Sequence[str]],
    id2label: dict[int, str],
    temperature: float = 1.0,
    n_bins: int = 10,
    min_precision: float = 0.85,
) -> dict[str, Any]:
    """Compute entity-level equal-mass ECE, Brier score, reliability curve, and precision-coverage threshold."""
    probs: list[float] = []
    correct: list[int] = []
    for lg, gt_tags in zip(word_logits_list, gold_tags_list, strict=True):
        _, pred_spans = decode_word_logits_to_bio(lg, id2label, temperature=temperature)
        gt_set = set(extract_iob2_entities(gt_tags))
        for s, e, lab, sc in pred_spans:
            probs.append(sc)
            correct.append(1 if (s, e, lab) in gt_set else 0)
    cal = equal_mass_ece(probs, correct, n_bins=n_bins)
    curve = precision_coverage_curve(probs, correct)
    chosen = select_annotation_threshold(curve, min_precision=min_precision)
    return {
        "temperature": temperature,
        "calibration": cal,
        "precision_coverage": curve,
        "recommended_threshold": chosen,
    }


# --------------------------------------------------------------------------- Extractors


class TokenClassificationExtractor:
    """PyTorch Hugging Face token-classification extractor (NER-1..4)."""

    def __init__(
        self,
        model_path: str | Path,
        version: str = "ner-xlmr",
        device: str | None = None,
        max_length: int = 256,
        batch_size: int = 32,
        temperatures: dict[str, float] | None = None,
        confidence_thresholds: dict[str, float] | None = None,
        fp16: bool = False,
        require_calibrated: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.model_path = Path(model_path)
        self.version = version
        self.max_length = max_length
        self.batch_size = batch_size
        self.temperatures = temperatures or {}
        self.confidence_thresholds = confidence_thresholds or {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16 and self.device.startswith("cuda")

        meta_file = self.model_path / "model_meta.json"
        self.model_meta: dict[str, Any] = (
            json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else {}
        )
        if self.model_meta.get("eligible") is False:
            raise RuntimeError(f"Refusing to load ineligible model checkpoint at {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(str(model_path))
        self.model.to(self.device)
        if self.fp16:
            self.model.half()
        self.model.eval()
        self.id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}

        # Load calibration.json if present alongside model checkpoint
        cal_file = self.model_path / "calibration.json"
        self.calibrated = False
        if cal_file.is_file():
            cal = json.loads(cal_file.read_text(encoding="utf-8"))
            if not temperatures:
                self.temperatures = {str(k): float(v) for k, v in cal.get("temperatures", {}).items()}
            if not confidence_thresholds:
                self.confidence_thresholds = {str(k): float(v) for k, v in cal.get("thresholds", {}).items()}
            self.calibrated = True
        elif temperatures is not None and confidence_thresholds is not None:
            self.calibrated = True
        elif require_calibrated:
            raise FileNotFoundError(
                f"Required calibration.json missing at {cal_file}; run calibration or pass require_calibrated=False"
            )

    def predict_word_logits(self, sentences_tokens: Sequence[Sequence[str]]) -> list[np.ndarray]:
        """Predict per-word logits for a list of word-tokenized sentences using length-sorted batching."""
        import torch

        all_windows: list[tuple[int, TokenWindow]] = []
        for s_idx, toks in enumerate(sentences_tokens):
            if not toks:
                continue
            for win in build_sliding_windows(toks, self.tokenizer, max_length=self.max_length):
                all_windows.append((s_idx, win))

        if not all_windows:
            return [np.zeros((0, len(self.id2label)), dtype=np.float32) for _ in sentences_tokens]

        # Length-sorted batching minimizes padding waste
        order = sorted(range(len(all_windows)), key=lambda i: len(all_windows[i][1].input_ids))
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 1
        win_logits: list[np.ndarray | None] = [None] * len(all_windows)

        with torch.inference_mode():
            for b_start in range(0, len(order), self.batch_size):
                batch_idx = order[b_start : b_start + self.batch_size]
                max_len = max(len(all_windows[i][1].input_ids) for i in batch_idx)
                input_ids = np.full((len(batch_idx), max_len), pad_id, dtype=np.int64)
                attn_mask = np.zeros((len(batch_idx), max_len), dtype=np.int64)
                for row, idx in enumerate(batch_idx):
                    w = all_windows[idx][1]
                    L = len(w.input_ids)
                    input_ids[row, :L] = w.input_ids
                    attn_mask[row, :L] = w.attention_mask
                out = (
                    self.model(
                        input_ids=torch.from_numpy(input_ids).to(self.device),
                        attention_mask=torch.from_numpy(attn_mask).to(self.device),
                    )
                    .logits.float()
                    .cpu()
                    .numpy()
                )
                for row, idx in enumerate(batch_idx):
                    L = len(all_windows[idx][1].input_ids)
                    win_logits[idx] = out[row, :L]

        by_sent: dict[int, list[tuple[list[int | None], np.ndarray]]] = {}
        for idx, (s_idx, w) in enumerate(all_windows):
            lg = win_logits[idx]
            assert lg is not None
            by_sent.setdefault(s_idx, []).append((w.word_ids, lg))

        results: list[np.ndarray] = []
        for s_idx, toks in enumerate(sentences_tokens):
            if not toks or s_idx not in by_sent:
                results.append(np.zeros((len(toks), len(self.id2label)), dtype=np.float32))
            else:
                wids_list = [x[0] for x in by_sent[s_idx]]
                lg_list = [x[1] for x in by_sent[s_idx]]
                results.append(merge_window_word_logits(len(toks), wids_list, lg_list))
        return results

    def predict_tags(
        self,
        sentences_tokens: Sequence[Sequence[str]],
        langs: Sequence[str] | None = None,
    ) -> tuple[list[list[str]], list[np.ndarray]]:
        logits_list = self.predict_word_logits(sentences_tokens)
        tags_list: list[list[str]] = []
        for i, lg in enumerate(logits_list):
            lang = langs[i] if langs is not None else "en"
            temp = self.temperatures.get(lang, 1.0)
            tags, _ = decode_word_logits_to_bio(lg, self.id2label, temperature=temp)
            tags_list.append(tags)
        return tags_list, logits_list

    def extract(self, chunks: Sequence[Chunk]) -> list[EntityMention]:
        tok_lists: list[list[str]] = []
        span_lists: list[list[tuple[str, int, int]]] = []
        langs: list[str] = []
        for ch in chunks:
            wt = word_tokenize(ch.text)
            span_lists.append(wt)
            tok_lists.append([t[0] for t in wt])
            langs.append(ch.language.value if isinstance(ch.language, Lang) else str(ch.language))

        logits_list = self.predict_word_logits(tok_lists)
        mentions: list[EntityMention] = []
        for ch, wt, lg, lang in zip(chunks, span_lists, logits_list, langs, strict=True):
            temp = self.temperatures.get(lang, 1.0)
            thr = self.confidence_thresholds.get(lang, 0.0)
            _, spans = decode_word_logits_to_bio(lg, self.id2label, temperature=temp)
            for s, e, ent_type, score in spans:
                if score < thr:
                    continue
                char_start = wt[s][1]
                char_end = wt[e - 1][2]
                et: EntityType | str = EntityType(ent_type) if ent_type in {"PER", "ORG", "LOC"} else ent_type
                mentions.append(
                    EntityMention(
                        chunk_id=ch.chunk_id,
                        surface=ch.text[char_start:char_end],
                        type=et,
                        char_start=char_start,
                        char_end=char_end,
                        score=score,
                        wikidata_qid=None,
                        model_version=self.version,
                    )
                )
        return mentions


class GLiNERExtractor:
    """Zero-shot span NER baseline (NER-0: ``urchade/gliner_multi-v2.1``)."""

    DEFAULT_LABELS: ClassVar[dict[str, str]] = {
        "person": "PER",
        "organization": "ORG",
        "location": "LOC",
    }

    def __init__(
        self,
        model_id: str = "urchade/gliner_multi-v2.1",
        threshold: float = 0.5,
        device: str | None = None,
        version: str = "ner-0-gliner-multi-v2.1",
    ) -> None:
        import torch
        from gliner import GLiNER

        self.version = version
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GLiNER.from_pretrained(model_id)
        if self.device.startswith("cuda"):
            self.model = self.model.to(self.device)
        self.prompt_labels = list(self.DEFAULT_LABELS.keys())

    def predict_tags(
        self, sentences_tokens: Sequence[Sequence[str]], batch_size: int = 16
    ) -> list[list[str]]:
        """Predict core IOB2 tags for pre-tokenized sentences."""
        from trifact.data.ner_common import spans_to_bio

        results: list[list[str]] = []
        texts: list[str] = []
        offsets_list: list[list[tuple[str, int, int]]] = []
        for toks in sentences_tokens:
            pos = 0
            offs: list[tuple[str, int, int]] = []
            for t in toks:
                offs.append((t, pos, pos + len(t)))
                pos += len(t) + 1
            offsets_list.append(offs)
            texts.append(" ".join(toks))

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch_offs = offsets_list[start : start + batch_size]
            batch_preds = self.model.batch_predict_entities(
                batch_texts, self.prompt_labels, threshold=self.threshold
            )
            for offs, ents in zip(batch_offs, batch_preds, strict=True):
                spans = [
                    (int(e["start"]), int(e["end"]), self.DEFAULT_LABELS[e["label"]])
                    for e in ents
                    if e.get("label") in self.DEFAULT_LABELS
                ]
                results.append(spans_to_bio(offs, spans))
        return results

    def extract(self, chunks: Sequence[Chunk]) -> list[EntityMention]:
        mentions: list[EntityMention] = []
        texts = [c.text for c in chunks]
        preds = self.model.batch_predict_entities(texts, self.prompt_labels, threshold=self.threshold)
        for ch, ents in zip(chunks, preds, strict=True):
            for e in ents:
                mapped = self.DEFAULT_LABELS.get(e.get("label", ""))
                if not mapped:
                    continue
                s, end = int(e["start"]), int(e["end"])
                mentions.append(
                    EntityMention(
                        chunk_id=ch.chunk_id,
                        surface=ch.text[s:end],
                        type=EntityType(mapped),
                        char_start=s,
                        char_end=end,
                        score=round(float(e.get("score", 0.5)), 6),
                        wikidata_qid=None,
                        model_version=self.version,
                    )
                )
        return mentions


# --------------------------------------------------------------------------- NER-5: Vocab trim + ONNX int8


def build_vocab_remap(
    tokenizer: Any,
    token_id_counts: Counter[int],
    max_vocab_size: int = 80000,
) -> tuple[list[int], dict[int, int]]:
    """Select kept token IDs (all special tokens + most frequent kk/ru/en corpus tokens)
    and return ``(kept_old_ids, old_to_new_id_map)``."""
    special_ids = sorted(set(getattr(tokenizer, "all_special_ids", [0, 1, 2, 3])))
    kept: list[int] = list(special_ids)
    kept_set: set[int] = set(kept)
    for tid, _ in token_id_counts.most_common():
        if len(kept) >= max_vocab_size:
            break
        if tid not in kept_set:
            kept.append(int(tid))
            kept_set.add(int(tid))
    remap = {old_id: new_id for new_id, old_id in enumerate(kept)}
    return kept, remap


def export_trimmed_onnx_ner(
    model_dir: Path,
    out_dir: Path,
    kept_old_ids: Sequence[int] | None = None,
    quantize_int8: bool = True,
) -> dict[str, Any]:
    """Export HF XLM-R token classifier with optional vocabulary trimming to ONNX (+ dynamic int8).

    Vocabulary trimming slices ``word_embeddings.weight`` to ``kept_old_ids`` and saves
    ``vocab_remap.json`` so the original SentencePiece tokenizer remains 100% lossless on
    kept kk/ru/en tokens while cutting embedding footprint by ~60-75%.
    """
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    model.eval()

    orig_vocab = int(model.config.vocab_size)
    orig_params = sum(p.numel() for p in model.parameters())
    unk_id = int(tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 3)

    if kept_old_ids is not None and len(kept_old_ids) < orig_vocab:
        kept_list = list(kept_old_ids)
        if unk_id not in kept_list:
            kept_list.insert(0, unk_id)
        remap = {int(old): int(new) for new, old in enumerate(kept_list)}
        emb = model.get_input_embeddings()
        new_weight = emb.weight.detach()[kept_list].clone()
        new_emb = torch.nn.Embedding(len(kept_list), new_weight.shape[1])
        new_emb.weight = torch.nn.Parameter(new_weight)
        model.set_input_embeddings(new_emb)
        model.config.vocab_size = len(kept_list)
        (out_dir / "vocab_remap.json").write_text(
            json.dumps({"unk_new_id": remap[unk_id], "old_to_new": {str(k): v for k, v in remap.items()}}),
            encoding="utf-8",
            newline="\n",
        )
    else:
        remap = {}

    trimmed_params = sum(p.numel() for p in model.parameters())
    tokenizer.save_pretrained(str(out_dir))
    meta = {
        "id2label": {str(k): str(v) for k, v in model.config.id2label.items()},
        "orig_vocab_size": orig_vocab,
        "new_vocab_size": int(model.config.vocab_size),
        "orig_params": orig_params,
        "trimmed_params": trimmed_params,
        "quantized_int8": quantize_int8,
    }
    (out_dir / "ner_onnx_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    # Copy calibration.json if present
    src_cal = model_dir / "calibration.json"
    if src_cal.is_file():
        (out_dir / "calibration.json").write_text(
            src_cal.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
        )

    fp32_onnx = out_dir / "model_fp32.onnx"
    dummy_ids = torch.ones((1, 16), dtype=torch.long)
    dummy_mask = torch.ones((1, 16), dtype=torch.long)
    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        str(fp32_onnx),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=14,
        dynamo=False,
    )

    final_onnx = out_dir / "model.onnx"
    if quantize_int8:
        quantize_dynamic(str(fp32_onnx), str(final_onnx), weight_type=QuantType.QInt8)
        fp32_onnx.unlink(missing_ok=True)
    else:
        fp32_onnx.replace(final_onnx)

    return {
        **meta,
        "onnx_path": str(final_onnx),
        "onnx_mb": round(final_onnx.stat().st_size / (1024 * 1024), 2),
        "export_seconds": round(time.perf_counter() - t0, 2),
    }


class OnnxNERExtractor:
    """ONNX Runtime dynamic int8 / fp16 NER extractor (NER-5)."""

    def __init__(
        self,
        model_dir: str | Path,
        version: str = "ner-5-onnx-int8",
        max_length: int = 256,
        batch_size: int = 32,
        providers: Sequence[str] | None = None,
        temperatures: dict[str, float] | None = None,
        confidence_thresholds: dict[str, float] | None = None,
        allow_unverified: bool = False,
    ) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.model_dir = Path(model_dir)
        self.version = version
        self.max_length = max_length
        self.batch_size = batch_size
        meta = json.loads((self.model_dir / "ner_onnx_meta.json").read_text(encoding="utf-8"))
        # Prevent rejected NER-5 optimization artifacts from being loaded in production (ADR-0009)
        if not allow_unverified:
            if meta.get("eligible") is False:
                raise RuntimeError(
                    f"Refusing to load rejected NER-5 ONNX artifact at {self.model_dir} "
                    "(failed <= 0.5 pp F1 parity gate). Use eligible HF champion or pass allow_unverified=True."
                )
            rep_file = self.model_dir / "ner5_report.json"
            if rep_file.is_file():
                rep = json.loads(rep_file.read_text(encoding="utf-8"))
                if rep.get("parity_gate", {}).get("passed") is False:
                    raise RuntimeError(
                        f"Refusing to load rejected NER-5 ONNX artifact at {self.model_dir} "
                        "(ner5_report.json parity_gate.passed=False)."
                    )
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), use_fast=True)
        self.id2label = {int(k): str(v) for k, v in meta["id2label"].items()}

        remap_path = self.model_dir / "vocab_remap.json"
        if remap_path.is_file():
            rm = json.loads(remap_path.read_text(encoding="utf-8"))
            self._unk_new_id: int | None = int(rm["unk_new_id"])
            self._old_to_new: dict[int, int] | None = {int(k): int(v) for k, v in rm["old_to_new"].items()}
        else:
            self._unk_new_id = None
            self._old_to_new = None

        self.temperatures = temperatures or {}
        self.confidence_thresholds = confidence_thresholds or {}
        cal_file = self.model_dir / "calibration.json"
        if cal_file.is_file() and not temperatures:
            cal = json.loads(cal_file.read_text(encoding="utf-8"))
            self.temperatures = {str(k): float(v) for k, v in cal.get("temperatures", {}).items()}
            if not confidence_thresholds:
                self.confidence_thresholds = {str(k): float(v) for k, v in cal.get("thresholds", {}).items()}

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_dir / "model.onnx"),
            sess_options=sess_opts,
            providers=list(providers or ["CPUExecutionProvider"]),
        )

    def _remap_ids(self, ids: list[int]) -> list[int]:
        if self._old_to_new is None:
            return ids
        unk = self._unk_new_id if self._unk_new_id is not None else 0
        return [self._old_to_new.get(i, unk) for i in ids]

    def predict_word_logits(self, sentences_tokens: Sequence[Sequence[str]]) -> list[np.ndarray]:
        all_windows: list[tuple[int, TokenWindow]] = []
        for s_idx, toks in enumerate(sentences_tokens):
            if not toks:
                continue
            for win in build_sliding_windows(toks, self.tokenizer, max_length=self.max_length):
                all_windows.append((s_idx, win))

        if not all_windows:
            return [np.zeros((0, len(self.id2label)), dtype=np.float32) for _ in sentences_tokens]

        order = sorted(range(len(all_windows)), key=lambda i: len(all_windows[i][1].input_ids))
        raw_pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 1
        pad_id = self._old_to_new.get(raw_pad, 0) if self._old_to_new is not None else raw_pad
        win_logits: list[np.ndarray | None] = [None] * len(all_windows)

        for b_start in range(0, len(order), self.batch_size):
            batch_idx = order[b_start : b_start + self.batch_size]
            max_len = max(len(all_windows[i][1].input_ids) for i in batch_idx)
            input_ids = np.full((len(batch_idx), max_len), pad_id, dtype=np.int64)
            attn_mask = np.zeros((len(batch_idx), max_len), dtype=np.int64)
            for row, idx in enumerate(batch_idx):
                w = all_windows[idx][1]
                remapped = self._remap_ids(w.input_ids)
                L = len(remapped)
                input_ids[row, :L] = remapped
                attn_mask[row, :L] = w.attention_mask
            out = self.session.run(["logits"], {"input_ids": input_ids, "attention_mask": attn_mask})[0]
            for row, idx in enumerate(batch_idx):
                L = len(all_windows[idx][1].input_ids)
                win_logits[idx] = out[row, :L].astype(np.float32)

        by_sent: dict[int, list[tuple[list[int | None], np.ndarray]]] = {}
        for idx, (s_idx, w) in enumerate(all_windows):
            lg = win_logits[idx]
            assert lg is not None
            by_sent.setdefault(s_idx, []).append((w.word_ids, lg))

        results: list[np.ndarray] = []
        for s_idx, toks in enumerate(sentences_tokens):
            if not toks or s_idx not in by_sent:
                results.append(np.zeros((len(toks), len(self.id2label)), dtype=np.float32))
            else:
                wids_list = [x[0] for x in by_sent[s_idx]]
                lg_list = [x[1] for x in by_sent[s_idx]]
                results.append(merge_window_word_logits(len(toks), wids_list, lg_list))
        return results

    def predict_tags(
        self,
        sentences_tokens: Sequence[Sequence[str]],
        langs: Sequence[str] | None = None,
    ) -> tuple[list[list[str]], list[np.ndarray]]:
        logits_list = self.predict_word_logits(sentences_tokens)
        tags_list: list[list[str]] = []
        for i, lg in enumerate(logits_list):
            lang = langs[i] if langs is not None else "en"
            temp = self.temperatures.get(lang, 1.0)
            tags, _ = decode_word_logits_to_bio(lg, self.id2label, temperature=temp)
            tags_list.append(tags)
        return tags_list, logits_list

    def extract(self, chunks: Sequence[Chunk]) -> list[EntityMention]:
        tok_lists: list[list[str]] = []
        span_lists: list[list[tuple[str, int, int]]] = []
        langs: list[str] = []
        for ch in chunks:
            wt = word_tokenize(ch.text)
            span_lists.append(wt)
            tok_lists.append([t[0] for t in wt])
            langs.append(ch.language.value if isinstance(ch.language, Lang) else str(ch.language))

        logits_list = self.predict_word_logits(tok_lists)
        mentions: list[EntityMention] = []
        for ch, wt, lg, lang in zip(chunks, span_lists, logits_list, langs, strict=True):
            temp = self.temperatures.get(lang, 1.0)
            thr = self.confidence_thresholds.get(lang, 0.0)
            _, spans = decode_word_logits_to_bio(lg, self.id2label, temperature=temp)
            for s, e, ent_type, score in spans:
                if score < thr:
                    continue
                char_start = wt[s][1]
                char_end = wt[e - 1][2]
                et: EntityType | str = EntityType(ent_type) if ent_type in {"PER", "ORG", "LOC"} else ent_type
                mentions.append(
                    EntityMention(
                        chunk_id=ch.chunk_id,
                        surface=ch.text[char_start:char_end],
                        type=et,
                        char_start=char_start,
                        char_end=char_end,
                        score=score,
                        wikidata_qid=None,
                        model_version=self.version,
                    )
                )
        return mentions


class CombinedExtractor:
    """Combines neural NER (PER/ORG/LOC) with rule-based DATE extraction (ADR-0006)."""

    def __init__(
        self,
        ner_extractor: Any,
        date_extractor: RuleBasedDateExtractor | None = None,
    ) -> None:
        self.ner_extractor = ner_extractor
        self.date_extractor = date_extractor or RuleBasedDateExtractor()
        self.version = f"{getattr(ner_extractor, 'version', 'ner')}+{self.date_extractor.version}"

    def extract(self, chunks: Sequence[Chunk]) -> list[EntityMention]:
        ner_mentions = list(self.ner_extractor.extract(chunks))
        date_mentions = list(self.date_extractor.extract(chunks))
        # Drop DATE mentions that overlap a higher-priority neural entity mention in the same chunk
        by_chunk: dict[str, list[EntityMention]] = {}
        for m in ner_mentions:
            by_chunk.setdefault(m.chunk_id, []).append(m)
        combined = list(ner_mentions)
        for dm in date_mentions:
            existing = by_chunk.get(dm.chunk_id, [])
            if any(dm.char_start < e.char_end and e.char_start < dm.char_end for e in existing):
                continue
            combined.append(dm)
        combined.sort(key=lambda m: (m.chunk_id, m.char_start))
        return combined
