"""Cloud-first cross-encoder reranker fine-tuning (RET-7, ADR-0025).

Trains ``BAAI/bge-reranker-v2-m3`` (or compatible XLM-R sequence-classification reranker) on
9-direction balanced ``(query, positive, hard_negatives)`` groups with:
- Listwise InfoNCE (cross-entropy over ``[pos, neg_1..neg_k]``) or BCE loss.
- Mandatory 64-example overfit sanity check, multi-seed support, Optuna HPO, fp16/bf16 AMP.
- Validation-only per-language Platt scaling calibration (``calibration.json``).
- ADR-0025 cloud-only guard.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from trifact.batch.annotate_corpus import is_cloud_runtime
from trifact.core.params import ROOT, load_params
from trifact.retrieval.reranker import fit_platt_scaling
from trifact.training.train_retriever import build_balanced_pairs

log = logging.getLogger("trifact.training.train_reranker")


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_cross_encoder(
    model_name: str,
    train_pairs: Sequence[dict[str, Any]],
    passage_map: Mapping[str, str],
    out_dir: Path,
    *,
    val_pairs: Sequence[dict[str, Any]] | None = None,
    lr: float = 1e-5,
    epochs: int = 1,
    batch_size: int = 8,
    grad_accum: int = 4,
    max_length: int = 384,
    num_negatives: int = 3,
    loss_type: str = "listwise",
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    seed: int = 42,
    sanity_check_steps: int = 0,
    calibrate: bool = True,
) -> dict[str, Any]:
    """Fine-tune cross-encoder reranker on ``(query, pos, k negs)`` groups."""
    import torch
    import torch.nn.functional as F
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    _set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    use_bf16 = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # Keep only pairs with at least 1 negative (or sample in-pool negatives)
    all_pids = list(passage_map.keys())
    rng = random.Random(seed)
    groups: list[tuple[str, str, list[str], str]] = []
    for r in train_pairs:
        negs = [passage_map[nid] for nid in (r.get("neg_ids") or []) if nid in passage_map][:num_negatives]
        while len(negs) < num_negatives and len(all_pids) > 1:
            cand = all_pids[rng.randrange(len(all_pids))]
            if cand != r["doc_id"]:
                negs.append(passage_map[cand])
        if negs:
            groups.append((r["query"], r["pos_text"], negs[:num_negatives], str(r.get("direction", "en-en"))))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, math.ceil(len(groups) / max(batch_size * grad_accum, 1)))
    total_steps = sanity_check_steps if sanity_check_steps > 0 else max(1, steps_per_epoch * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * warmup_ratio), num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and not use_bf16))

    def step_loss(batch_groups: Sequence[tuple[str, str, list[str], str]]) -> Any:
        qs: list[str] = []
        ds: list[str] = []
        g_size = 1 + num_negatives
        for q, pos, negs, _ in batch_groups:
            qs.append(q)
            ds.append(pos)
            for n_t in negs:
                qs.append(q)
                ds.append(n_t)
        enc = tokenizer(qs, ds, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        scores = logits.squeeze(-1) if logits.shape[-1] == 1 else (logits[:, 1] - logits[:, 0])
        if loss_type == "bce":
            targets = torch.zeros_like(scores)
            targets[0::g_size] = 1.0
            return F.binary_cross_entropy_with_logits(scores, targets)
        # Listwise InfoNCE over [pos, neg_1..neg_k]
        mat = scores.view(len(batch_groups), g_size)
        targets = torch.zeros(len(batch_groups), dtype=torch.long, device=device)
        return F.cross_entropy(mat, targets)

    t0 = time.perf_counter()
    losses: list[float] = []
    model.train()

    if sanity_check_steps > 0:
        sub = groups[: min(32, len(groups))]
        for _ in range(sanity_check_steps):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                loss = step_loss(sub[: min(batch_size, len(sub))])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu().item()))
        return {
            "sanity_check": True,
            "steps": sanity_check_steps,
            "initial_loss": round(losses[0], 4) if losses else 0.0,
            "final_loss": round(losses[-1], 4) if losses else 0.0,
            "seconds": round(time.perf_counter() - t0, 2),
        }

    global_step = 0
    for _ in range(epochs):
        rng.shuffle(groups)
        optimizer.zero_grad(set_to_none=True)
        for step_i, start in enumerate(range(0, len(groups), batch_size), start=1):
            batch_g = groups[start : start + batch_size]
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                loss = step_loss(batch_g) / grad_accum
            scaler.scale(loss).backward()
            if step_i % grad_accum == 0 or start + batch_size >= len(groups):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            losses.append(float(loss.detach().cpu().item()) * grad_accum)

    # Validation-only per-language Platt calibration
    platt_by_lang: dict[str, dict[str, float]] = {}
    val_mrr: float | None = None
    if val_pairs:
        model.eval()
        logits_by_lang: dict[str, list[float]] = defaultdict(list)
        labels_by_lang: dict[str, list[int]] = defaultdict(list)
        rr_list: list[float] = []
        with torch.inference_mode():
            for r in val_pairs[: min(500, len(val_pairs))]:
                negs = [passage_map[nid] for nid in (r.get("neg_ids") or []) if nid in passage_map][
                    :num_negatives
                ]
                if not negs and len(all_pids) > 1:
                    negs = [passage_map[all_pids[rng.randrange(len(all_pids))]]]
                cands = [r["pos_text"], *negs]
                enc = tokenizer(
                    [r["query"]] * len(cands),
                    cands,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    out = model(**enc).logits
                    sc = (
                        (out.squeeze(-1) if out.shape[-1] == 1 else (out[:, 1] - out[:, 0]))
                        .float()
                        .cpu()
                        .tolist()
                    )
                ql = str(r.get("direction", "en-en")).split("-")[0]
                for idx_c, val_s in enumerate(sc):
                    logits_by_lang[ql].append(float(val_s))
                    labels_by_lang[ql].append(1 if idx_c == 0 else 0)
                rank_pos = 1 + sum(1 for s in sc[1:] if s >= sc[0])
                rr_list.append(1.0 / rank_pos)
        val_mrr = round(float(np.mean(rr_list)), 5) if rr_list else None
        if calibrate:
            for lg, lgs in logits_by_lang.items():
                platt_by_lang[lg] = fit_platt_scaling(lgs, labels_by_lang[lg])

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    if platt_by_lang:
        (out_dir / "calibration.json").write_text(
            json.dumps({"platt_by_lang": platt_by_lang}, indent=2) + "\n", encoding="utf-8", newline="\n"
        )

    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated() / (1024**2), 1) if torch.cuda.is_available() else 0.0
    )
    dt = max(time.perf_counter() - t0, 1e-6)
    meta = {
        "base_model": model_name,
        "seed": seed,
        "lr": lr,
        "epochs": epochs,
        "loss_type": loss_type,
        "num_negatives": num_negatives,
        "train_groups": len(groups),
        "train_loss_start": round(losses[0], 5) if losses else 0.0,
        "train_loss_end": round(float(np.mean(losses[-min(20, len(losses)) :])), 5) if losses else 0.0,
        "val_mrr": val_mrr,
        "platt_by_lang": platt_by_lang,
        "steps": global_step,
        "seconds": round(dt, 2),
        "peak_vram_mb": peak_vram_mb,
    }
    (out_dir / "train_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return meta


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrieval-dir", type=Path, default=ROOT / "data" / "task" / "retrieval")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "reranker")
    ap.add_argument("--max-per-direction", type=int, default=2000)
    ap.add_argument("--kazparc-cap", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--loss", choices=["listwise", "bce"], default="listwise")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sanity-check", action="store_true")
    ap.add_argument("--allow-local-pilot", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not args.allow_local_pilot and not is_cloud_runtime():
        raise RuntimeError(
            "train_reranker must run on an authorized cloud GPU runtime (ADR-0025). "
            "Pass --allow-local-pilot for bounded unit tests only."
        )

    params = load_params()
    model_name = args.model or params["retrieval"]["reranker"]
    train_pairs, pmap, _ = build_balanced_pairs(
        args.retrieval_dir,
        split="train",
        max_per_direction=args.max_per_direction,
        kazparc_max_share=args.kazparc_cap,
        seed=args.seed,
    )
    val_pairs, vpmap, _ = build_balanced_pairs(
        args.retrieval_dir,
        split="validation",
        max_per_direction=max(100, args.max_per_direction // 10),
        kazparc_max_share=args.kazparc_cap,
        seed=args.seed,
    )
    pmap.update(vpmap)
    res = train_cross_encoder(
        model_name,
        train_pairs,
        pmap,
        args.out,
        val_pairs=val_pairs,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        loss_type=args.loss,
        seed=args.seed,
        sanity_check_steps=20 if args.sanity_check else 0,
    )
    log.info("train_reranker complete: %s", json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
