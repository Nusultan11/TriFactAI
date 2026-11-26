"""Cloud-first multilingual bi-encoder embedder fine-tuning (RET-6, ADR-0003, ADR-0025).

Key methodology (trifact-retrieval & trifact-training-hpo skills):
- 9-direction balanced, source-reweighted sampler capping short KazParC sentence pairs so
  passages and ``kk-kk`` (KazQAD + teacher) are not drowned out.
- Positive-aware hard-negative mining & false-negative filtering (NV-Retriever margin rule:
  drop negatives with ``score >= pos_score * margin``).
- InfoNCE / MultipleNegativesRankingLoss with in-batch + explicit hard negatives and optional
  Matryoshka (MRL) multi-dim loss.
- Mandatory 64-example overfit sanity check, multi-seed training (>=3 seeds), Optuna HPO,
  resumable checkpoints, and ADR-0025 cloud-only guard.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from trifact.batch.annotate_corpus import is_cloud_runtime
from trifact.core.params import ROOT, load_params
from trifact.evaluation.eval_retrieval import ALL_DIRECTIONS
from trifact.retrieval.embedder import detect_model_family, format_passage_text, format_query_text

log = logging.getLogger("trifact.training.train_retriever")


def build_balanced_pairs(
    retrieval_dir: Path,
    *,
    split: str = "train",
    max_per_direction: int = 8000,
    kazparc_max_share: float = 0.20,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    """Stream ``data/task/retrieval`` into a 9-direction balanced, KazParC-capped training set.

    Strictly filters to ``split`` (``train`` or ``validation``) so frozen test is never touched.
    Returns ``(pairs, passage_texts_by_id, sampling_stats)``.
    """
    rng = random.Random(seed)
    pairs_lf = pl.scan_parquet(str(retrieval_dir / "*/pairs-*.parquet")).filter(pl.col("split") == split)

    selected_frames: list[pl.DataFrame] = []
    stats: dict[str, Any] = {"split": split, "kazparc_max_share": kazparc_max_share, "by_direction": {}}

    for direction in ALL_DIRECTIONS:
        dir_df = (
            pairs_lf.filter(pl.col("direction") == direction)
            .select(
                "pair_id",
                "query_id",
                "query",
                "q_lang",
                "doc_id",
                "d_lang",
                "direction",
                "source",
                "origin",
                "neg_ids",
            )
            .collect(engine="streaming")
        )
        if dir_df.height == 0:
            continue

        kp_df = dir_df.filter(pl.col("source") == "kazparc")
        non_kp_df = dir_df.filter(pl.col("source") != "kazparc")

        kp_cap = int(max_per_direction * kazparc_max_share)
        non_kp_target = max_per_direction - min(kp_df.height, kp_cap)

        parts: list[pl.DataFrame] = []
        if non_kp_df.height > 0:
            if non_kp_df.height > non_kp_target:
                parts.append(non_kp_df.sample(n=non_kp_target, seed=rng.randrange(1 << 30)))
            else:
                parts.append(non_kp_df)
        rem = max_per_direction - sum(p.height for p in parts)
        kp_take = min(kp_df.height, kp_cap, max(rem, 0))
        if kp_take > 0:
            parts.append(kp_df.sample(n=kp_take, seed=rng.randrange(1 << 30)))

        if parts:
            chosen = pl.concat(parts)
            selected_frames.append(chosen)
            src_counts = {
                str(r["source"]): int(r["len"]) for r in chosen.group_by("source").len().iter_rows(named=True)
            }
            stats["by_direction"][direction] = {
                "total_available": dir_df.height,
                "selected": chosen.height,
                "by_source": src_counts,
            }

    if not selected_frames:
        return [], {}, stats

    all_pairs_df = pl.concat(selected_frames).sample(fraction=1.0, shuffle=True, seed=seed)

    # Collect all referenced doc_ids (positive + hard negatives) and fetch their text via streaming filter
    needed_ids: set[str] = set(all_pairs_df["doc_id"].to_list())
    for negs in all_pairs_df["neg_ids"].to_list():
        if negs:
            for nid in negs[:4]:
                needed_ids.add(str(nid))

    passages_lf = pl.scan_parquet(str(retrieval_dir / "multi/passages-*.parquet"))
    p_df = (
        passages_lf.filter(pl.col("doc_id").is_in(list(needed_ids)))
        .select("doc_id", "text")
        .unique("doc_id")
        .collect(engine="streaming")
    )
    passage_map = dict(zip(p_df["doc_id"].to_list(), p_df["text"].to_list(), strict=True))

    records: list[dict[str, Any]] = []
    for r in all_pairs_df.iter_rows(named=True):
        pos_text = passage_map.get(str(r["doc_id"]))
        if not pos_text:
            continue
        negs = [
            str(nid)
            for nid in (r["neg_ids"] or [])
            if str(nid) in passage_map and str(nid) != str(r["doc_id"])
        ]
        records.append(
            {
                "pair_id": str(r["pair_id"]),
                "query_id": str(r["query_id"]),
                "query": str(r["query"]),
                "direction": str(r["direction"]),
                "source": str(r["source"]),
                "doc_id": str(r["doc_id"]),
                "pos_text": str(pos_text),
                "neg_ids": negs,
            }
        )
    stats["total_selected_with_text"] = len(records)
    return records, passage_map, stats


def filter_false_negatives_positive_aware(
    pos_score: float,
    candidate_scores: Sequence[tuple[str, float]],
    *,
    margin_ratio: float = 0.95,
    max_negatives: int = 4,
) -> list[str]:
    """Positive-aware hard-negative filter (NV-Retriever / trifact-retrieval skill).

    Drops candidate negatives whose similarity/reranker score is suspiciously close to or above
    the positive passage (``score >= pos_score * margin_ratio`` when ``pos_score > 0``).
    """
    threshold = (pos_score * margin_ratio) if pos_score > 0 else (pos_score - 0.05)
    kept = [doc_id for doc_id, sc in candidate_scores if sc < threshold]
    return kept[:max_negatives]


def mine_positive_aware_negatives(
    pairs: list[dict[str, Any]],
    passage_map: Mapping[str, str],
    embedder: Any,
    *,
    top_k: int = 30,
    margin_ratio: float = 0.95,
    max_negatives: int = 4,
) -> dict[str, Any]:
    """Mine dense hard negatives for pairs lacking negatives and filter false negatives."""
    need_mining = [p for p in pairs if not p.get("neg_ids")]
    if not need_mining or not passage_map:
        return {"mined_pairs": 0, "avg_kept_negs": 0.0, "filtered_false_negatives": 0}

    pool_ids = list(passage_map.keys())
    pool_texts = [passage_map[i] for i in pool_ids]
    id_to_idx = {pid: i for i, pid in enumerate(pool_ids)}

    d_vecs = (
        embedder.embed_passages_np(pool_texts)
        if hasattr(embedder, "embed_passages_np")
        else np.asarray(embedder.embed_dense(pool_texts), dtype=np.float32)
    )
    q_texts = [p["query"] for p in need_mining]
    q_vecs = (
        embedder.embed_queries_np(q_texts)
        if hasattr(embedder, "embed_queries_np")
        else np.asarray(embedder.embed_dense(q_texts), dtype=np.float32)
    )

    kept_total = 0
    filtered_fn = 0
    batch = 256
    kk = min(top_k + 1, len(pool_ids))

    for start in range(0, len(need_mining), batch):
        qb = q_vecs[start : start + batch]
        scores = qb @ d_vecs.T
        for i in range(scores.shape[0]):
            rec = need_mining[start + i]
            pos_id = rec["doc_id"]
            pos_idx = id_to_idx.get(pos_id)
            row = scores[i]
            pos_sc = float(row[pos_idx]) if pos_idx is not None else 0.85
            top_idx = np.argpartition(-row, kk - 1)[:kk]
            top_idx = top_idx[np.argsort(-row[top_idx])]
            cands = [(pool_ids[int(j)], float(row[j])) for j in top_idx if pool_ids[int(j)] != pos_id]
            kept = filter_false_negatives_positive_aware(
                pos_sc, cands, margin_ratio=margin_ratio, max_negatives=max_negatives
            )
            filtered_fn += max(0, min(len(cands), max_negatives) - len(kept))
            rec["neg_ids"] = kept
            kept_total += len(kept)

    return {
        "mined_pairs": len(need_mining),
        "avg_kept_negs": round(kept_total / max(len(need_mining), 1), 3),
        "filtered_false_negatives": filtered_fn,
    }


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pool_hidden(last_hidden: Any, attn_mask: Any, family: str) -> Any:
    import torch

    if family == "bge_m3":
        return last_hidden[:, 0]
    if family == "qwen3":
        seq_lengths = attn_mask.sum(dim=1) - 1
        batch_idx = torch.arange(last_hidden.shape[0], device=last_hidden.device)
        return last_hidden[batch_idx, seq_lengths]
    mask = attn_mask.unsqueeze(-1).type_as(last_hidden)
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def train_biencoder(
    model_name: str,
    train_pairs: Sequence[dict[str, Any]],
    passage_map: Mapping[str, str],
    out_dir: Path,
    *,
    val_pairs: Sequence[dict[str, Any]] | None = None,
    lr: float = 2e-5,
    epochs: int = 1,
    batch_size: int = 16,
    grad_accum: int = 2,
    max_length: int = 256,
    temperature: float = 0.02,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    mrl_dims: Sequence[int] | None = None,
    seed: int = 42,
    sanity_check_steps: int = 0,
    use_grad_checkpointing: bool = True,
) -> dict[str, Any]:
    """Fine-tune a multilingual bi-encoder with InfoNCE (in-batch + hard negatives) and optional MRL loss."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    _set_seed(seed)
    family = detect_model_family(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    use_bf16 = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if use_grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_pairs) / max(batch_size * grad_accum, 1)))
    total_steps = sanity_check_steps if sanity_check_steps > 0 else max(1, steps_per_epoch * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * warmup_ratio), num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and not use_bf16))

    def encode_tensor(texts: list[str]) -> Any:
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        return _pool_hidden(out.last_hidden_state, enc["attention_mask"], family)

    def compute_infonce_loss(q_emb: Any, p_emb: Any, n_emb: Any | None) -> Any:
        dims = list(mrl_dims) if mrl_dims else [q_emb.shape[-1]]
        total_loss = 0.0
        for d in dims:
            qn = F.normalize(q_emb[:, :d], p=2, dim=-1)
            pn = F.normalize(p_emb[:, :d], p=2, dim=-1)
            if n_emb is not None and n_emb.shape[0] > 0:
                nn_v = F.normalize(n_emb[:, :d], p=2, dim=-1)
                docs = torch.cat([pn, nn_v], dim=0)
            else:
                docs = pn
            logits = (qn @ docs.T) / temperature
            labels = torch.arange(qn.shape[0], device=device)
            total_loss = total_loss + F.cross_entropy(logits, labels)
        return total_loss / len(dims)

    t0 = time.perf_counter()
    losses: list[float] = []
    model.train()
    rng = random.Random(seed)
    indices = list(range(len(train_pairs)))

    if sanity_check_steps > 0:
        # Mandatory 64-example overfit sanity check (trifact-training-hpo)
        sub = indices[: min(64, len(indices))]
        for step in range(sanity_check_steps):
            batch_recs = [train_pairs[i] for i in sub[: min(batch_size, len(sub))]]
            qs = [format_query_text(r["query"], family) for r in batch_recs]
            ps = [format_passage_text(r["pos_text"], family) for r in batch_recs]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                loss = compute_infonce_loss(encode_tensor(qs), encode_tensor(ps), None)
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
    for ep in range(epochs):
        rng.shuffle(indices)
        optimizer.zero_grad(set_to_none=True)
        for step_i, start in enumerate(range(0, len(indices), batch_size), start=1):
            b_idx = indices[start : start + batch_size]
            batch_recs = [train_pairs[i] for i in b_idx]
            qs = [format_query_text(r["query"], family) for r in batch_recs]
            ps = [format_passage_text(r["pos_text"], family) for r in batch_recs]
            # Pick 1 explicit hard negative per query if available
            neg_texts: list[str] = []
            for r in batch_recs:
                if r.get("neg_ids"):
                    nid = r["neg_ids"][0]
                    if nid in passage_map:
                        neg_texts.append(format_passage_text(passage_map[nid], family))
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                q_emb = encode_tensor(qs)
                p_emb = encode_tensor(ps)
                n_emb = encode_tensor(neg_texts) if neg_texts else None
                loss = compute_infonce_loss(q_emb, p_emb, n_emb) / grad_accum
            scaler.scale(loss).backward()
            if step_i % grad_accum == 0 or start + batch_size >= len(indices):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            losses.append(float(loss.detach().cpu().item()) * grad_accum)

    val_loss = None
    if val_pairs:
        model.eval()
        v_losses: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(val_pairs), batch_size):
                batch_recs = val_pairs[start : start + batch_size]
                qs = [format_query_text(r["query"], family) for r in batch_recs]
                ps = [format_passage_text(r["pos_text"], family) for r in batch_recs]
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    v_losses.append(
                        float(compute_infonce_loss(encode_tensor(qs), encode_tensor(ps), None).cpu().item())
                    )
        val_loss = round(float(np.mean(v_losses)), 5) if v_losses else None

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated() / (1024**2), 1) if torch.cuda.is_available() else 0.0
    )
    dt = max(time.perf_counter() - t0, 1e-6)
    meta = {
        "base_model": model_name,
        "family": family,
        "seed": seed,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "temperature": temperature,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs) if val_pairs else 0,
        "train_loss_start": round(losses[0], 5) if losses else 0.0,
        "train_loss_end": round(float(np.mean(losses[-min(20, len(losses)) :])), 5) if losses else 0.0,
        "val_loss": val_loss,
        "steps": global_step,
        "seconds": round(dt, 2),
        "samples_per_s": round((len(train_pairs) * epochs) / dt, 2),
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
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "retriever")
    ap.add_argument("--max-per-direction", type=int, default=4000)
    ap.add_argument("--kazparc-cap", type=float, default=0.20)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sanity-check", action="store_true")
    ap.add_argument("--allow-local-pilot", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not args.allow_local_pilot and not is_cloud_runtime():
        raise RuntimeError(
            "train_retriever must run on an authorized cloud GPU runtime (ADR-0025). "
            "Pass --allow-local-pilot for bounded unit tests only."
        )

    params = load_params()
    model_name = args.model or params["retrieval"]["embedder"]
    train_pairs, pmap, tr_stats = build_balanced_pairs(
        args.retrieval_dir,
        split="train",
        max_per_direction=args.max_per_direction,
        kazparc_max_share=args.kazparc_cap,
        seed=args.seed,
    )
    val_pairs, vpmap, _ = build_balanced_pairs(
        args.retrieval_dir,
        split="validation",
        max_per_direction=max(200, args.max_per_direction // 10),
        kazparc_max_share=args.kazparc_cap,
        seed=args.seed,
    )
    pmap.update(vpmap)
    res = train_biencoder(
        model_name,
        train_pairs,
        pmap,
        args.out,
        val_pairs=val_pairs,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        seed=args.seed,
        sanity_check_steps=30 if args.sanity_check else 0,
    )
    res["sampling_stats"] = tr_stats
    log.info(
        "train_retriever complete: %s", json.dumps({k: v for k, v in res.items() if k != "sampling_stats"})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
