"""Cloud-first NER training, curriculum learning, Optuna HPO, calibration, and NER-5 export.

Supports experiments NER-0 through NER-5 (architecture §7.2, ADR-0006, ADR-0015, ADR-0025):
- NER-0: GLiNER multi v2.1 zero-shot baseline evaluation.
- NER-1: XLM-R base supervised on gold_train (>=3 seeds).
- NER-2: XLM-R base silver -> gold curriculum (partial-annotation -100 masking + punctuation filter), Optuna HPO, >=3 seeds.
- NER-3: XLM-R large cloud quality ceiling.
- NER-4: Kazakh specialization on KazNERD-25 (full 25-class schema + core PER/ORG/LOC comparison).
- NER-5: Champion -> kk/ru/en vocabulary trimming + ONNX dynamic int8 export + parity/speed check.

Usage:
    python -m trifact.training.train_ner --exp NER-1 --seed 42 --out models/ner/ner1_s42
    python -m trifact.training.train_ner --exp NER-2 --hpo --n-trials 12 --out models/ner/ner2_hpo
    python -m trifact.training.train_ner --exp NER-5 --model models/ner/champion --out models/ner/champion_onnx
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from trifact.core.params import ROOT, load_params
from trifact.data.ner_common import map_tags
from trifact.evaluation.eval_ner import (
    evaluate_ner_predictions,
    extract_seen_surfaces,
    load_eval_ner_rows,
    strict_entity_metrics,
)
from trifact.extraction.ner import (
    CORE_BIO_LABELS,
    TokenClassificationExtractor,
    TokenWindow,
    build_sliding_windows,
    build_vocab_remap,
    clean_silver_tags,
    decode_word_logits_to_bio,
    evaluate_entity_calibration,
    export_trimmed_onnx_ner,
    fit_temperature_scaling,
    merge_window_word_logits,
)
from trifact.lake.stats import ResourceMonitor, git_sha

log = logging.getLogger("trifact.train_ner")
LABELS_YAML = ROOT / "configs" / "labels.yaml"


def set_all_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_ner_split(
    data_dir: Path,
    table: str,
    langs: Sequence[str] = ("kk", "ru", "en"),
    limit_per_lang: int | None = None,
    seed: int = 42,
    tag_col: str = "tags",
    source_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Load NER rows from ``data/task/ner/<lang>/<table>-*.parquet``."""
    rows: list[dict[str, Any]] = []
    for lg in langs:
        files = sorted((data_dir / lg).glob(f"{table}-*.parquet"))
        if not files:
            continue
        lf = pl.scan_parquet(files).select(
            ["sent_id", "doc_key", "source", "lang", "origin", "tokens", tag_col]
        )
        if source_filter is not None:
            lf = lf.filter(pl.col("source") == source_filter)
        if limit_per_lang is not None:
            # Deterministic subsampling without loading all silver rows into RAM
            df = lf.collect()
            if df.height > limit_per_lang:
                df = df.sample(n=limit_per_lang, seed=seed, shuffle=True)
        else:
            df = lf.collect()
        if tag_col != "tags":
            df = df.rename({tag_col: "tags"})
        rows.extend(df.to_dicts())
    return rows


def prepare_windows(
    rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    label2id: dict[str, int],
    max_length: int = 256,
    is_silver: bool = False,
    mask_unlabeled_capitals: bool = True,
) -> tuple[list[TokenWindow], list[tuple[int, int]]]:
    """Tokenize rows into sliding windows; returns ``(windows, [(row_idx, win_idx), ...])``."""
    windows: list[TokenWindow] = []
    mapping: list[tuple[int, int]] = []
    for r_idx, row in enumerate(rows):
        toks = row["tokens"]
        raw_tags = row["tags"]
        w_labels: Sequence[str | None]
        if is_silver:
            w_labels = clean_silver_tags(toks, raw_tags, mask_unlabeled_capitals=mask_unlabeled_capitals)
        else:
            w_labels = raw_tags
        wins = build_sliding_windows(
            toks, tokenizer, max_length=max_length, word_labels=w_labels, label2id=label2id
        )
        for w_idx, w in enumerate(wins):
            windows.append(w)
            mapping.append((r_idx, w_idx))
    return windows, mapping


def collate_windows(
    batch: Sequence[TokenWindow],
    pad_token_id: int,
) -> dict[str, Any]:
    import torch

    max_len = max(len(w.input_ids) for w in batch)
    bsz = len(batch)
    input_ids = np.full((bsz, max_len), pad_token_id, dtype=np.int64)
    attn_mask = np.zeros((bsz, max_len), dtype=np.int64)
    has_labels = batch[0].labels is not None
    labels = np.full((bsz, max_len), -100, dtype=np.int64) if has_labels else None

    for i, w in enumerate(batch):
        L = len(w.input_ids)
        input_ids[i, :L] = w.input_ids
        attn_mask[i, :L] = w.attention_mask
        if labels is not None and w.labels is not None:
            labels[i, :L] = w.labels

    out: dict[str, Any] = {
        "input_ids": torch.from_numpy(input_ids),
        "attention_mask": torch.from_numpy(attn_mask),
    }
    if labels is not None:
        out["labels"] = torch.from_numpy(labels)
    return out


def predict_on_rows(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    id2label: dict[int, str],
    device: str,
    max_length: int = 256,
    batch_size: int = 64,
    amp_dtype: Any = None,
) -> tuple[list[list[str]], list[np.ndarray]]:
    """Run batched inference on ``rows`` returning ``(pred_tags_list, word_logits_list)``."""
    import torch

    model.eval()
    all_windows: list[tuple[int, TokenWindow]] = []
    for r_idx, row in enumerate(rows):
        for w in build_sliding_windows(row["tokens"], tokenizer, max_length=max_length):
            all_windows.append((r_idx, w))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1
    order = sorted(range(len(all_windows)), key=lambda i: len(all_windows[i][1].input_ids))
    win_logits: list[np.ndarray | None] = [None] * len(all_windows)

    with torch.inference_mode():
        for b_start in range(0, len(order), batch_size):
            b_idx = order[b_start : b_start + batch_size]
            batch = collate_windows([all_windows[i][1] for i in b_idx], pad_id)
            in_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            if amp_dtype is not None and device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(input_ids=in_ids, attention_mask=mask).logits
            else:
                logits = model(input_ids=in_ids, attention_mask=mask).logits
            lg_np = logits.float().cpu().numpy()
            for row_i, idx in enumerate(b_idx):
                L = len(all_windows[idx][1].input_ids)
                win_logits[idx] = lg_np[row_i, :L]

    by_row: dict[int, list[tuple[list[int | None], np.ndarray]]] = {}
    for idx, (r_idx, w) in enumerate(all_windows):
        lg = win_logits[idx]
        assert lg is not None
        by_row.setdefault(r_idx, []).append((w.word_ids, lg))

    pred_tags: list[list[str]] = []
    word_logits: list[np.ndarray] = []
    n_labels = len(id2label)
    for r_idx, row in enumerate(rows):
        n_words = len(row["tokens"])
        if n_words == 0 or r_idx not in by_row:
            lg_arr = np.zeros((n_words, n_labels), dtype=np.float32)
        else:
            lg_arr = merge_window_word_logits(
                n_words, [x[0] for x in by_row[r_idx]], [x[1] for x in by_row[r_idx]]
            )
        tags, _ = decode_word_logits_to_bio(lg_arr, id2label, temperature=1.0)
        pred_tags.append(tags)
        word_logits.append(lg_arr)
    return pred_tags, word_logits


def run_overfit_sanity_check(
    model_name: str,
    rows: Sequence[dict[str, Any]],
    label_list: Sequence[str],
    device: str,
    max_length: int = 128,
    steps: int = 25,
) -> dict[str, float]:
    """Mandatory pre-run sanity check (trifact-training-hpo): overfit 64 examples to verify alignment."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    sample = list(rows[:64])
    label2id = {lab: i for i, lab in enumerate(label_list)}
    id2label = dict(enumerate(label_list))
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id
    ).to(device)
    windows, _ = prepare_windows(sample, tok, label2id, max_length=max_length)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 1
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    losses: list[float] = []
    for _ in range(steps):
        batch = collate_windows(windows[:32], pad_id)
        opt.zero_grad(set_to_none=True)
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        out.loss.backward()
        opt.step()
        losses.append(float(out.loss.item()))
    return {"initial_loss": round(losses[0], 4), "final_loss": round(losses[-1], 4), "steps": steps}


def _select_amp_dtype(device: str, precision: str) -> Any:
    import torch

    if not device.startswith("cuda"):
        return None
    if precision == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        log.warning("bf16 requested but unsupported on this GPU; falling back to fp16")
        return torch.float16
    if precision == "fp16":
        return torch.float16
    return None


def train_one_stage(
    model: Any,
    tokenizer: Any,
    train_windows: Sequence[TokenWindow],
    val_rows: Sequence[dict[str, Any]],
    id2label: dict[int, str],
    out_dir: Path,
    *,
    device: str,
    epochs: int,
    lr: float,
    batch_size: int,
    grad_accum: int = 1,
    warmup_ratio: float = 0.08,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    max_length: int = 256,
    precision: str = "fp16",
    seed: int = 42,
    select_lang: str = "kk",
    stage_name: str = "gold",
    resume_dir: Path | None = None,
    trial: Any = None,
) -> dict[str, Any]:
    """Train one curriculum stage with AMP, gradient clipping, checkpointing, and dev-F1 selection."""
    import torch
    from transformers import get_linear_schedule_with_warmup

    set_all_seeds(seed)
    amp_dtype = _select_amp_dtype(device, precision)
    use_scaler = amp_dtype == torch.float16 and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    grouped = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped, lr=lr)
    steps_per_epoch = max(1, math.ceil(len(train_windows) / (batch_size * grad_accum)))
    total_steps = steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * warmup_ratio), num_training_steps=total_steps
    )

    start_epoch = 0
    best_metric = -1.0
    history: list[dict[str, Any]] = []
    if resume_dir is not None and (resume_dir / "trainer_state.pt").is_file():
        state = torch.load(resume_dir / "trainer_state.pt", map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if use_scaler and "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state.get("epoch", 0))
        best_metric = float(state.get("best_metric", -1.0))
        history = list(state.get("history", []))
        log.info("Resumed %s stage from epoch %d (best_metric=%.4f)", stage_name, start_epoch, best_metric)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dir = out_dir / "best"
    last_dir = out_dir / "last"

    rng = random.Random(seed)
    t_stage0 = time.perf_counter()

    for ep in range(start_epoch, epochs):
        t_ep0 = time.perf_counter()
        model.train()
        order = list(range(len(train_windows)))
        rng.shuffle(order)
        # Mild length bucketing inside shuffled chunks of 1024 for throughput
        bucketed: list[int] = []
        for i in range(0, len(order), 1024):
            chunk = order[i : i + 1024]
            chunk.sort(key=lambda idx: len(train_windows[idx].input_ids))
            bucketed.extend(chunk)

        ep_loss = 0.0
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for step_i, b_start in enumerate(range(0, len(bucketed), batch_size), start=1):
            b_idx = bucketed[b_start : b_start + batch_size]
            batch = collate_windows([train_windows[i] for i in b_idx], pad_id)
            in_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)

            if amp_dtype is not None and device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    loss = model(input_ids=in_ids, attention_mask=mask, labels=labs).loss / grad_accum
            else:
                loss = model(input_ids=in_ids, attention_mask=mask, labels=labs).loss / grad_accum

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            ep_loss += float(loss.item()) * grad_accum
            n_batches += 1

            if step_i % grad_accum == 0 or b_start + batch_size >= len(bucketed):
                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        # Validation evaluation (strict entity F1 per language)
        pred_tags, _ = predict_on_rows(
            model, tokenizer, val_rows, id2label, device=device, max_length=max_length, amp_dtype=amp_dtype
        )
        val_by_lang: dict[str, float] = {}
        for lg in sorted({str(r["lang"]) for r in val_rows}):
            yt = [r["tags"] for r in val_rows if r["lang"] == lg]
            yp = [p for r, p in zip(val_rows, pred_tags, strict=True) if r["lang"] == lg]
            val_by_lang[lg] = strict_entity_metrics(yt, yp)["micro"]["f1"]

        sel_score = val_by_lang.get(
            select_lang,
            float(np.mean(list(val_by_lang.values()))) if val_by_lang else 0.0,
        )
        ep_secs = time.perf_counter() - t_ep0
        rec = {
            "stage": stage_name,
            "epoch": ep + 1,
            "train_loss": round(ep_loss / max(n_batches, 1), 5),
            "val_f1": {k: round(v, 5) for k, v in val_by_lang.items()},
            "select_metric": round(sel_score, 5),
            "seconds": round(ep_secs, 1),
            "samples_per_s": round(len(train_windows) / max(ep_secs, 1e-6), 1),
        }
        history.append(rec)
        log.info(
            "[%s] epoch %d/%d loss=%.4f val_f1=%s (%.1fs, %.1f samp/s)",
            stage_name,
            ep + 1,
            epochs,
            rec["train_loss"],
            rec["val_f1"],
            ep_secs,
            rec["samples_per_s"],
        )

        # Save last checkpoint (resumable)
        last_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(last_dir))
        tokenizer.save_pretrained(str(last_dir))
        torch.save(
            {
                "epoch": ep + 1,
                "best_metric": max(best_metric, sel_score),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if use_scaler else None,
                "history": history,
            },
            last_dir / "trainer_state.pt",
        )

        if sel_score > best_metric:
            best_metric = sel_score
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_dir))
            tokenizer.save_pretrained(str(best_dir))

        if trial is not None:
            import optuna

            trial.report(sel_score, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return {
        "stage": stage_name,
        "best_metric": round(best_metric, 5),
        "best_dir": str(best_dir),
        "last_dir": str(last_dir),
        "history": history,
        "seconds": round(time.perf_counter() - t_stage0, 1),
    }


def calibrate_model_on_val(
    model_dir: Path,
    val_rows: Sequence[dict[str, Any]],
    device: str,
    max_length: int = 256,
    min_precision: float = 0.85,
) -> dict[str, Any]:
    """Fit per-language temperature scaling on validation data ONLY and persist ``calibration.json``."""
    ext = TokenClassificationExtractor(model_dir, device=device, max_length=max_length)
    label2id = {v: k for k, v in ext.id2label.items()}
    logits_all = ext.predict_word_logits([r["tokens"] for r in val_rows])

    temps: dict[str, float] = {}
    thresholds: dict[str, float] = {}
    by_lang_report: dict[str, Any] = {}

    for lg in sorted({str(r["lang"]) for r in val_rows}):
        idxs = [i for i, r in enumerate(val_rows) if r["lang"] == lg]
        lg_logits = [logits_all[i] for i in idxs]
        lg_tags = [val_rows[i]["tags"] for i in idxs]
        lg_tag_ids = [[label2id.get(t, 0) for t in tags] for tags in lg_tags]

        fit = fit_temperature_scaling(lg_logits, lg_tag_ids)
        before = evaluate_entity_calibration(
            lg_logits, lg_tags, ext.id2label, temperature=1.0, min_precision=min_precision
        )
        after = evaluate_entity_calibration(
            lg_logits, lg_tags, ext.id2label, temperature=fit["temperature"], min_precision=min_precision
        )
        temps[lg] = fit["temperature"]
        thresholds[lg] = after["recommended_threshold"]["threshold"]
        by_lang_report[lg] = {
            "temperature_fit": fit,
            "before": before,
            "after": after,
        }

    cal_payload = {
        "temperatures": temps,
        "thresholds": thresholds,
        "min_precision_target": min_precision,
        "by_lang": by_lang_report,
    }
    (model_dir / "calibration.json").write_text(
        json.dumps(cal_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return cal_payload


def run_ner5_optimization(
    model_dir: Path,
    out_dir: Path,
    train_dir: Path,
    eval_dir: Path,
    max_vocab_size: int = 80000,
    eval_limit: int | None = None,
) -> dict[str, Any]:
    """NER-5: corpus-driven XLM-R vocab trimming + ONNX dynamic int8 export + parity & speed benchmark."""
    from transformers import AutoTokenizer

    from trifact.evaluation.gate import evaluate_ner5_optimization_gate
    from trifact.extraction.ner import OnnxNERExtractor

    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    counts: Counter[int] = Counter()
    for lg in ("kk", "ru", "en"):
        for fp in sorted((train_dir / lg).glob("gold_train-*.parquet")):
            df = pl.read_parquet(fp, columns=["tokens"]).head(40000)
            for toks in df["tokens"].to_list():
                ids = tok(toks, is_split_into_words=True, add_special_tokens=False)["input_ids"]
                counts.update(ids)
    kept_ids, _ = build_vocab_remap(tok, counts, max_vocab_size=max_vocab_size)
    export_info = export_trimmed_onnx_ner(model_dir, out_dir, kept_old_ids=kept_ids, quantize_int8=True)

    # Measure CPU throughput + parity on validation or eval slice
    eval_rows = load_eval_ner_rows(eval_dir, verify_frozen=True, limit_per_lang=eval_limit)
    toks_list = [r["tokens"] for r in eval_rows]
    langs_list = [str(r["lang"]) for r in eval_rows]

    t0 = time.perf_counter()
    pt_ext = TokenClassificationExtractor(model_dir, device="cpu", batch_size=32)
    pt_tags, _ = pt_ext.predict_tags(toks_list, langs=langs_list)
    pt_secs = time.perf_counter() - t0

    t1 = time.perf_counter()
    ox_ext = OnnxNERExtractor(out_dir, batch_size=32)
    ox_tags, _ = ox_ext.predict_tags(toks_list, langs=langs_list)
    ox_secs = time.perf_counter() - t1

    pt_f1: dict[str, float] = {}
    ox_f1: dict[str, float] = {}
    for lg in ("kk", "ru", "en"):
        yt = [r["tags"] for r in eval_rows if r["lang"] == lg]
        yp_pt = [p for r, p in zip(eval_rows, pt_tags, strict=True) if r["lang"] == lg]
        yp_ox = [p for r, p in zip(eval_rows, ox_tags, strict=True) if r["lang"] == lg]
        pt_f1[lg] = strict_entity_metrics(yt, yp_pt)["micro"]["f1"]
        ox_f1[lg] = strict_entity_metrics(yt, yp_ox)["micro"]["f1"]

    gate = evaluate_ner5_optimization_gate(pt_f1, ox_f1, max_drop_pp=0.5)
    return {
        "export": export_info,
        "pytorch_cpu_chunks_per_s": round(len(eval_rows) / max(pt_secs, 1e-6), 2),
        "onnx_int8_cpu_chunks_per_s": round(len(eval_rows) / max(ox_secs, 1e-6), 2),
        "speedup_cpu": round(pt_secs / max(ox_secs, 1e-6), 2),
        "pytorch_f1": pt_f1,
        "onnx_int8_f1": ox_f1,
        "parity_gate": gate,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", choices=["NER-0", "NER-1", "NER-2", "NER-3", "NER-4", "NER-5"], default="NER-1")
    ap.add_argument("--model", type=str, default=None, help="HF model id or local checkpoint path")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "task" / "ner")
    ap.add_argument("--eval-dir", type=Path, default=ROOT / "data" / "eval" / "ner")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "ner")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--silver-epochs", type=int, default=1)
    ap.add_argument("--silver-limit-per-lang", type=int, default=100000)
    ap.add_argument("--gold-limit-per-lang", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--silver-lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--sanity-check", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--eval-test", action="store_true")
    ap.add_argument("--hpo", action="store_true")
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--hpo-storage", type=str, default=None)
    ap.add_argument("--max-vocab-size", type=int, default=80000)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    params = load_params()
    ner_cfg = params["ner"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    epochs = args.epochs or int(ner_cfg["epochs"])
    lr = args.lr or float(ner_cfg["lr"])
    batch_size = args.batch_size or int(ner_cfg["batch_size"])
    max_length = args.max_length or int(ner_cfg["max_length"])

    if args.exp == "NER-5":
        src_model = Path(args.model) if args.model else args.out / "best"
        res = run_ner5_optimization(
            src_model, args.out, args.data_dir, args.eval_dir, max_vocab_size=args.max_vocab_size
        )
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "ner5_report.json").write_text(
            json.dumps(res, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        log.info("NER-5 result: %s", json.dumps(res["parity_gate"]))
        return 0 if res["parity_gate"]["passed"] else 1

    if args.exp == "NER-0":
        from trifact.extraction.ner import GLiNERExtractor

        t0 = time.perf_counter()
        ext = GLiNERExtractor(device=device)
        eval_rows = load_eval_ner_rows(
            args.eval_dir, verify_frozen=True, limit_per_lang=args.gold_limit_per_lang
        )
        preds = ext.predict_tags([r["tokens"] for r in eval_rows])
        seen = extract_seen_surfaces(args.data_dir)
        rep = evaluate_ner_predictions(
            eval_rows,
            preds,
            seen_surfaces_by_lang=seen,
            bootstrap_samples=int(params["evaluation"]["bootstrap_samples"]),
            ci=float(params["evaluation"]["ci"]),
            seed=args.seed,
            per_example_out=args.out / "ner0_per_example.jsonl",
        )
        rep["exp"] = "NER-0"
        rep["model"] = "urchade/gliner_multi-v2.1"
        rep["device"] = gpu_name
        rep["elapsed_s"] = round(time.perf_counter() - t0, 1)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "ner0_report.json").write_text(
            json.dumps(rep, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        return 0

    # Supervised experiments NER-1 .. NER-4
    labels_doc = yaml.safe_load(LABELS_YAML.read_text(encoding="utf-8"))
    if args.exp == "NER-4":
        label_list = tuple(labels_doc["tagsets"]["kaznerd"])
        langs: tuple[str, ...] = ("kk",)
        tag_col = "tags_full"
        source_filter: str | None = "kaznerd"
    else:
        label_list = CORE_BIO_LABELS
        langs = ("kk", "ru", "en")
        tag_col = "tags"
        source_filter = None

    label2id = {lab: i for i, lab in enumerate(label_list)}
    id2label = dict(enumerate(label_list))
    default_model = "FacebookAI/xlm-roberta-large" if args.exp == "NER-3" else str(ner_cfg["base_model"])
    model_id = args.model or default_model

    gold_train = load_ner_split(
        args.data_dir,
        "gold_train",
        langs=langs,
        limit_per_lang=args.gold_limit_per_lang,
        seed=args.seed,
        tag_col=tag_col,
        source_filter=source_filter,
    )
    gold_val = load_ner_split(
        args.data_dir,
        "gold_validation",
        langs=langs,
        limit_per_lang=args.gold_limit_per_lang,
        seed=args.seed,
        tag_col=tag_col,
        source_filter=source_filter,
    )

    if args.sanity_check:
        sc = run_overfit_sanity_check(model_id, gold_train, label_list, device=device)
        log.info("Overfit sanity check: %s", sc)

    if args.hpo:
        import optuna

        storage = args.hpo_storage or f"sqlite:///{(args.out / 'optuna_ner.db').as_posix()}"
        args.out.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(
            study_name=f"trifact_{args.exp.lower()}",
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(multivariate=True, seed=args.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
        )
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        tr_wins, _ = prepare_windows(gold_train, tok, label2id, max_length=max_length)

        def _objective(trial: optuna.Trial) -> float:
            t_lr = trial.suggest_float("lr", 1e-5, 6e-5, log=True)
            t_warmup = trial.suggest_float("warmup_ratio", 0.04, 0.15)
            t_wd = trial.suggest_float("weight_decay", 0.001, 0.05, log=True)
            t_bs = trial.suggest_categorical("batch_size", [16, 32])
            m = AutoModelForTokenClassification.from_pretrained(
                model_id, num_labels=len(label_list), id2label=id2label, label2id=label2id
            ).to(device)
            res = train_one_stage(
                m,
                tok,
                tr_wins,
                gold_val,
                id2label,
                args.out / f"trial_{trial.number}",
                device=device,
                epochs=min(epochs, 3),
                lr=t_lr,
                batch_size=t_bs,
                warmup_ratio=t_warmup,
                weight_decay=t_wd,
                max_length=max_length,
                precision=args.precision,
                seed=args.seed,
                trial=trial,
            )
            return float(res["best_metric"])

        study.optimize(_objective, n_trials=args.n_trials)
        summary = {
            "exp": args.exp,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
        }
        (args.out / "hpo_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        log.info("HPO complete: best_value=%.4f params=%s", study.best_value, study.best_params)
        return 0

    with ResourceMonitor() as mon:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(
            model_id, num_labels=len(label_list), id2label=id2label, label2id=label2id
        ).to(device)

        stages_log: list[dict[str, Any]] = []
        if args.exp == "NER-2":
            silver_rows = load_ner_split(
                args.data_dir,
                "silver_train",
                langs=langs,
                limit_per_lang=args.silver_limit_per_lang,
                seed=args.seed,
            )
            silver_wins, _ = prepare_windows(
                silver_rows,
                tok,
                label2id,
                max_length=max_length,
                is_silver=True,
                mask_unlabeled_capitals=True,
            )
            s_res = train_one_stage(
                model,
                tok,
                silver_wins,
                gold_val,
                id2label,
                args.out / "silver_stage",
                device=device,
                epochs=args.silver_epochs,
                lr=args.silver_lr,
                batch_size=batch_size,
                grad_accum=args.grad_accum,
                max_length=max_length,
                precision=args.precision,
                seed=args.seed,
                stage_name="silver",
            )
            stages_log.append(s_res)
            # Load best silver checkpoint before gold fine-tuning
            model = AutoModelForTokenClassification.from_pretrained(s_res["best_dir"]).to(device)

        gold_wins, _ = prepare_windows(gold_train, tok, label2id, max_length=max_length)
        g_res = train_one_stage(
            model,
            tok,
            gold_wins,
            gold_val,
            id2label,
            args.out,
            device=device,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            grad_accum=args.grad_accum,
            max_length=max_length,
            precision=args.precision,
            seed=args.seed,
            stage_name="gold",
            resume_dir=args.resume,
        )
        stages_log.append(g_res)

        best_dir = Path(g_res["best_dir"])
        cal_report = (
            calibrate_model_on_val(best_dir, gold_val, device=device, max_length=max_length)
            if args.calibrate
            else None
        )

        test_report = None
        if args.eval_test:
            eval_rows = load_eval_ner_rows(args.eval_dir, langs=langs, verify_frozen=True)
            ext = TokenClassificationExtractor(best_dir, device=device, max_length=max_length)
            pred_tags, _ = ext.predict_tags(
                [r["tokens"] for r in eval_rows], langs=[str(r["lang"]) for r in eval_rows]
            )
            if args.exp == "NER-4":
                # Also map KazNERD-25 predictions to core PER/ORG/LOC for direct comparison
                mapping = labels_doc["kaznerd"]
                pred_core = [map_tags(p, mapping) for p in pred_tags]
                test_report = {
                    "core_mapped": evaluate_ner_predictions(
                        eval_rows,
                        pred_core,
                        bootstrap_samples=int(params["evaluation"]["bootstrap_samples"]),
                        ci=float(params["evaluation"]["ci"]),
                        seed=args.seed,
                    )
                }
            else:
                seen = extract_seen_surfaces(args.data_dir)
                test_report = evaluate_ner_predictions(
                    eval_rows,
                    pred_tags,
                    seen_surfaces_by_lang=seen,
                    bootstrap_samples=int(params["evaluation"]["bootstrap_samples"]),
                    ci=float(params["evaluation"]["ci"]),
                    seed=args.seed,
                    per_example_out=args.out / "test_per_example.jsonl",
                )

    peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1) if device == "cuda" else 0.0
    run_summary = {
        "exp": args.exp,
        "model_id": model_id,
        "seed": args.seed,
        "git_sha": git_sha(ROOT),
        "device": gpu_name,
        "precision": args.precision,
        "peak_rss_mb": round(mon.peak_rss_mb, 1),
        "peak_vram_mb": peak_vram_mb,
        "wall_seconds": round(mon.seconds, 1),
        "stages": stages_log,
        "calibration": cal_report,
        "test_report": test_report,
    }
    (args.out / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    log.info(
        "Completed %s (seed=%d, best_val=%.4f, wall=%.1fs)",
        args.exp,
        args.seed,
        g_res["best_metric"],
        mon.seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
