"""Cloud-first bf16 LoRA fine-tuning and Step 0 compatibility checker for Qwen3.5 (ADR-0012, ADR-0025).

Supports:
- Step 0 compatibility check (``--compat-check``): verifies ``transformers`` ``qwen3_5`` loading,
  ``apply_chat_template(..., enable_thinking=False)``, actual linear module discovery via
  ``model.named_modules()``, PEFT LoRA attach & merge parity, and serving backend versions.
- GEN-1: ``Qwen/Qwen3.5-2B`` bf16 LoRA on grounded SFT dataset with assistant-only loss masking.
- GEN-2: SFT mixture ablations (``full``, ``no_cross_lingual``, ``no_unanswerable``, ``gold_only``,
  ``balanced_unanswerable``).
- GEN-3: Model size sweep (``0.8B``, ``2B``, ``4B``, ``9B``) with multi-seed support.

Hard guards (ADR-0012, ADR-0025):
- Refuses local execution unless ``--allow-local`` is explicitly passed for tests/compat checks.
- Refuses QLoRA and non-Ampere GPUs (e.g. Kaggle/Colab Tesla T4 sm75 lacks native bf16).
- Enforces ``params.yaml: cloud.budget_usd > 0`` before launching paid bf16 LoRA training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import polars as pl

from trifact.batch.annotate_corpus import is_cloud_runtime
from trifact.core.params import ROOT, load_params
from trifact.generation.prompt import build_grounded_prompt, sft_row_to_scored_chunks

log = logging.getLogger("trifact.training.train_llm")

# Candidate linear projection names across Qwen3.5 hybrid (Gated DeltaNet + Gated Attn + MLP) and Qwen3
STANDARD_LINEAR_TARGETS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
DELTANET_LINEAR_CANDIDATES: tuple[str, ...] = (
    "in_proj_qkv",
    "in_proj_qkvz",
    "in_proj_ba",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
)
EXCLUDED_LINEAR_NAMES: frozenset[str] = frozenset({"lm_head", "embed_tokens", "visual", "merger"})

ABLATION_MODES: tuple[str, ...] = (
    "full",
    "no_cross_lingual",
    "no_unanswerable",
    "gold_only",
    "balanced_unanswerable",
)


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_linear_target_modules(
    model: Any,
    *,
    include_vision: bool = False,
) -> dict[str, Any]:
    """Inspect ``model.named_modules()`` to discover all trainable ``nn.Linear`` leaf module names.

    Reconciles Qwen3.5 hybrid Gated DeltaNet + Gated Attention + MLP linear layers (ROADMAP 7.1).
    """
    from torch import nn

    leaf_counts: dict[str, int] = {}
    excluded_seen: dict[str, int] = {}
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not include_vision and any(seg in full_name.lower() for seg in ("visual", "vision", "merger")):
            excluded_seen[full_name.split(".")[-1]] = excluded_seen.get(full_name.split(".")[-1], 0) + 1
            continue
        leaf = full_name.split(".")[-1]
        if leaf in EXCLUDED_LINEAR_NAMES:
            excluded_seen[leaf] = excluded_seen.get(leaf, 0) + 1
            continue
        leaf_counts[leaf] = leaf_counts.get(leaf, 0) + 1

    recommended = sorted(leaf_counts.keys())
    return {
        "discovered_linear_modules": dict(sorted(leaf_counts.items())),
        "recommended_target_modules": recommended,
        "excluded_linear_modules": dict(sorted(excluded_seen.items())),
        "has_deltanet_projections": any(name in leaf_counts for name in DELTANET_LINEAR_CANDIDATES),
        "standard_targets_present": [m for m in STANDARD_LINEAR_TARGETS if m in leaf_counts],
    }


def check_bf16_lora_preconditions(
    params: dict[str, Any],
    *,
    allow_local: bool = False,
    require_ampere_gpu: bool = True,
    require_paid_budget: bool = True,
) -> dict[str, Any]:
    """Verify ADR-0012 (Ampere+ bf16 GPU) and ADR-0025 (cloud runtime + approved budget)."""
    import torch

    reasons: list[str] = []
    budget_usd = float(params.get("cloud", {}).get("budget_usd", 0.0))
    cloud_ok = is_cloud_runtime() or allow_local
    if not cloud_ok:
        reasons.append("ADR-0025: train_llm must run on cloud GPU (set --allow-local only for tests).")

    if require_paid_budget and budget_usd <= 0.0:
        reasons.append(
            f"ROADMAP 7.0 / ADR-0025: cloud.budget_usd={budget_usd} <= 0; "
            "approved paid cloud budget for Ampere+ GPU is required before bf16 LoRA training."
        )

    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "none"
    capability = torch.cuda.get_device_capability(0) if cuda_avail else (0, 0)
    bf16_ok = bool(cuda_avail and torch.cuda.is_bf16_supported() and capability[0] >= 8)

    if require_ampere_gpu:
        if not cuda_avail:
            reasons.append("ADR-0012: CUDA Ampere+ GPU required for bf16 LoRA (no CUDA device found).")
        elif capability[0] < 8:
            reasons.append(
                f"ADR-0012: GPU {gpu_name} (sm{capability[0]}{capability[1]}) lacks Ampere+ native bf16; "
                "never run Qwen3.5 bf16 LoRA on T4/sm75 or QLoRA."
            )
        elif not bf16_ok:
            reasons.append(f"ADR-0012: GPU {gpu_name} does not report bf16 support.")

    return {
        "eligible": len(reasons) == 0,
        "reasons": reasons,
        "cloud_runtime": is_cloud_runtime(),
        "budget_usd": budget_usd,
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "compute_capability": f"sm{capability[0]}{capability[1]}",
        "bf16_ampere_supported": bf16_ok,
    }


def apply_sft_ablation(
    df: pl.DataFrame,
    ablation: str = "full",
    *,
    seed: int = 42,
    max_examples: int | None = None,
) -> pl.DataFrame:
    """Apply GEN-2 SFT mixture ablation cleanly on ``data/task/sft`` rows."""
    if ablation not in ABLATION_MODES:
        raise ValueError(f"Unsupported ablation {ablation!r}; expected one of {ABLATION_MODES}")

    out = df
    if ablation == "no_cross_lingual":
        out = out.filter(pl.col("type") != "cross_lingual")
    elif ablation == "no_unanswerable":
        out = out.filter(pl.col("type") != "unanswerable")
    elif ablation == "gold_only":
        out = out.filter(pl.col("origin") == "gold")
    elif ablation == "balanced_unanswerable":
        # Address Stage 7 SFT audit finding: ru unanswerable is severely underrepresented (46 rows)
        unans = out.filter(pl.col("type") == "unanswerable")
        others = out.filter(pl.col("type") != "unanswerable")
        ru_unans = unans.filter(pl.col("lang") == "ru")
        if ru_unans.height > 0:
            target = min(max(ru_unans.height * 20, 1000), unans.filter(pl.col("lang") == "kk").height)
            reps = max(1, math.ceil(target / ru_unans.height))
            ru_up = pl.concat([ru_unans] * reps).head(target)
            out = pl.concat([others, unans.filter(pl.col("lang") != "ru"), ru_up])

    if max_examples is not None and max_examples > 0 and out.height > max_examples:
        # Stratified sample by language to preserve kk >= 30% share
        parts: list[pl.DataFrame] = []
        langs = sorted(out["lang"].unique().to_list())
        per_lang = max(1, max_examples // max(len(langs), 1))
        for lg in langs:
            sub = out.filter(pl.col("lang") == lg)
            sub = sub.with_columns(_r=pl.col("example_id").hash(seed=seed)).sort("_r").drop("_r")
            parts.append(sub.head(per_lang))
        out = pl.concat(parts).head(max_examples)

    return out


def build_sft_chat_messages(
    row: dict[str, Any], *, spotlight: bool = False, use_short_ids: bool = False
) -> list[dict[str, str]]:
    """Build exact chat messages for one SFT row (system + user + assistant)."""
    ev_chunks = sft_row_to_scored_chunks(row.get("evidence") or [])
    gp = build_grounded_prompt(
        question=str(row["question"]),
        context=ev_chunks,
        answer_lang=str(row["lang"]),
        use_short_ids=use_short_ids,
        spotlight=spotlight,
        merge_adjacent=False,
        dedup_threshold=1.01,
    )
    return [
        *gp.messages,
        {"role": "assistant", "content": str(row["answer"])},
    ]


def tokenize_assistant_only(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_seq_len: int = 4096,
    thinking: bool = False,
) -> dict[str, list[int]] | None:
    """Tokenize a single conversation and mask prompt tokens to ``-100`` (assistant-only loss).

    Returns ``None`` if the full sequence exceeds ``max_seq_len`` or has zero supervised tokens.
    """
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        raise ValueError("Expected chat messages ending with an assistant turn")

    prompt_msgs = messages[:-1]

    def _apply(msgs: list[dict[str, str]], add_gen: bool) -> str:
        try:
            return str(
                tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=add_gen,
                    enable_thinking=thinking,
                )
            )
        except TypeError:
            return str(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_gen))

    prompt_text = _apply(prompt_msgs, add_gen=True)
    full_text = _apply(messages, add_gen=False)

    prompt_ids = list(tokenizer.encode(prompt_text, add_special_tokens=False))
    full_ids = list(tokenizer.encode(full_text, add_special_tokens=False))

    if len(full_ids) > max_seq_len or len(full_ids) <= len(prompt_ids):
        return None

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    attention_mask = [1] * len(full_ids)
    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def run_qwen35_compat_check(
    model_name: str = "Qwen/Qwen3.5-2B",
    *,
    lora_r: int = 16,
    lora_alpha: int = 32,
    device: str | None = None,
) -> dict[str, Any]:
    """ROADMAP 7.1 Step 0 compatibility check for Qwen3.5 in ``transformers``, ``PEFT``, ``llama.cpp``, ``vLLM``."""
    import importlib.metadata

    import peft
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = (
        torch.bfloat16
        if (device == "cuda" and torch.cuda.is_bf16_supported())
        else (torch.float16 if device == "cuda" else torch.float32)
    )

    # Version inventory
    versions: dict[str, str | None] = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "vllm": None,
        "llama_cli": shutil.which("llama-cli"),
        "llama_server": shutil.which("llama-server"),
        "llama_quantize": shutil.which("llama-quantize"),
        "llama_imatrix": shutil.which("llama-imatrix"),
        "llama_perplexity": shutil.which("llama-perplexity"),
    }
    try:
        versions["vllm"] = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        versions["vllm"] = None

    if versions["llama_cli"]:
        try:
            proc = subprocess.run(
                [versions["llama_cli"], "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            versions["llama_cpp_version"] = (proc.stdout or proc.stderr).strip()[:200]
        except Exception as exc:  # noqa: BLE001
            versions["llama_cpp_version"] = f"error: {exc}"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    probe_msgs = [
        {"role": "system", "content": "You are TriFact AI."},
        {
            "role": "user",
            "content": "Evidence:\n[1] Astana is the capital of Kazakhstan.\n\nQuestion: What is the capital?",
        },
    ]
    non_thinking_supported = True
    try:
        rendered = str(
            tokenizer.apply_chat_template(
                probe_msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    except TypeError:
        non_thinking_supported = False
        rendered = str(tokenizer.apply_chat_template(probe_msgs, tokenize=False, add_generation_prompt=True))

    # Round-trip tokenization check on kk / ru / en
    rt_samples = {
        "kk": "Астана — Қазақстанның астанасы [c00000001].",
        "ru": "Астана — столица Казахстана [c00000001].",
        "en": "Astana is the capital of Kazakhstan [c00000001].",
    }
    roundtrip_ok = {
        lg: tokenizer.decode(tokenizer.encode(txt, add_special_tokens=False), skip_special_tokens=True) == txt
        for lg, txt in rt_samples.items()
    }

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model_type = getattr(model.config, "model_type", "unknown")
    linear_info = discover_linear_target_modules(model)

    # Verify PEFT LoRA attach and merge_and_unload
    targets = linear_info["recommended_target_modules"] or list(STANDARD_LINEAR_TARGETS)
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())

    peft_model.to(device)
    peft_model.eval()
    inputs = tokenizer(rendered, return_tensors="pt").to(device)
    with torch.no_grad():
        out_peft = peft_model.generate(
            **inputs, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    text_peft = tokenizer.decode(out_peft[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    merged = peft_model.merge_and_unload()
    merged.eval()
    with torch.no_grad():
        out_merged = merged.generate(
            **inputs, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    text_merged = tokenizer.decode(out_merged[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    return {
        "model_name": model_name,
        "model_type": model_type,
        "device": device,
        "dtype": str(dtype),
        "versions": versions,
        "non_thinking_template_supported": non_thinking_supported,
        "rendered_template_tail": rendered[-160:],
        "roundtrip_ok": roundtrip_ok,
        "linear_modules": linear_info,
        "peft_lora": {
            "r": lora_r,
            "alpha": lora_alpha,
            "target_modules": targets,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": round(100.0 * trainable / max(total, 1), 4),
            "merge_greedy_exact_match": text_peft == text_merged,
            "sample_output_peft": text_peft,
            "sample_output_merged": text_merged,
        },
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_adapter_manifest(out_dir: Path, meta: dict[str, Any]) -> Path:
    """Write ``adapter_manifest.json`` with SHA-256 checksums for clean-VM recovery (ADR-0025)."""
    files: dict[str, str] = {}
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "adapter_manifest.json":
            files[str(p.relative_to(out_dir)).replace("\\", "/")] = _sha256_file(p)
    manifest = {**meta, "files": files}
    path = out_dir / "adapter_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return path


def verify_adapter_manifest(out_dir: Path) -> dict[str, Any]:
    """Verify every file in ``adapter_manifest.json`` against its recorded SHA-256."""
    mpath = out_dir / "adapter_manifest.json"
    if not mpath.is_file():
        return {"verified": False, "reason": "adapter_manifest.json missing"}
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for rel, expected in (manifest.get("files") or {}).items():
        p = out_dir / rel
        if not p.is_file():
            mismatches.append(f"missing:{rel}")
        elif _sha256_file(p) != expected:
            mismatches.append(f"sha256_mismatch:{rel}")
    return {
        "verified": len(mismatches) == 0,
        "n_files": len(manifest.get("files") or {}),
        "mismatches": mismatches,
    }


def train_bf16_lora(
    model_name: str,
    sft_dir: Path,
    out_dir: Path,
    *,
    ablation: str = "full",
    seed: int = 42,
    lr: float = 2e-4,
    epochs: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_rslora: bool = False,
    max_seq_len: int = 4096,
    batch_size: int = 2,
    grad_accum: int = 8,
    max_train_examples: int | None = None,
    sanity_check_steps: int = 0,
    checkpoint_dir: Path | None = None,
    save_every_steps: int = 200,
    thinking: bool = False,
) -> dict[str, Any]:
    """Run bf16 LoRA SFT on an Ampere+ GPU with assistant-only loss and resumable checkpoints."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    _set_seed(seed)
    t0 = time.perf_counter()

    train_df = pl.read_parquet(str(sft_dir / "*/train-*.parquet"))
    train_df = apply_sft_ablation(train_df, ablation=ablation, seed=seed, max_examples=max_train_examples)
    rows = train_df.to_dicts()
    random.Random(seed).shuffle(rows)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded: list[dict[str, list[int]]] = []
    skipped_len = 0
    for row in rows:
        msgs = build_sft_chat_messages(row, spotlight=False, use_short_ids=False)
        item = tokenize_assistant_only(tokenizer, msgs, max_seq_len=max_seq_len, thinking=thinking)
        if item is None:
            skipped_len += 1
            continue
        encoded.append(item)

    if not encoded:
        raise RuntimeError(f"Zero SFT examples survived max_seq_len={max_seq_len} (skipped={skipped_len})")

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    linear_info = discover_linear_target_modules(model)
    targets = linear_info["recommended_target_modules"] or list(STANDARD_LINEAR_TARGETS)

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=targets,
        use_rslora=use_rslora,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = max(1, math.ceil(len(encoded) / max(batch_size * grad_accum, 1)))
    total_steps = sanity_check_steps if sanity_check_steps > 0 else max(1, steps_per_epoch * epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.05 * total_steps)),
        num_training_steps=total_steps,
    )

    pad_id = int(tokenizer.pad_token_id)
    global_step = 0
    loss_start: float | None = None
    loss_end = 0.0

    # Resume from checkpoint if present
    ckpt_state_path = (checkpoint_dir / "trainer_state.pt") if checkpoint_dir else None
    if ckpt_state_path and ckpt_state_path.is_file():
        state = torch.load(ckpt_state_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        global_step = int(state["global_step"])
        log.info("Resumed bf16 LoRA from checkpoint at step %d", global_step)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    n_epochs_run = 1 if sanity_check_steps > 0 else epochs
    for _ep in range(n_epochs_run):
        random.Random(seed + _ep).shuffle(encoded)
        optimizer.zero_grad(set_to_none=True)
        for idx in range(0, len(encoded), batch_size):
            batch = encoded[idx : idx + batch_size]
            max_l = max(len(x["input_ids"]) for x in batch)
            input_ids = torch.full((len(batch), max_l), pad_id, dtype=torch.long, device=device)
            attn_mask = torch.zeros((len(batch), max_l), dtype=torch.long, device=device)
            labels = torch.full((len(batch), max_l), -100, dtype=torch.long, device=device)
            for b_i, ex in enumerate(batch):
                ln = len(ex["input_ids"])
                input_ids[b_i, :ln] = torch.tensor(ex["input_ids"], dtype=torch.long, device=device)
                attn_mask[b_i, :ln] = 1
                labels[b_i, :ln] = torch.tensor(ex["labels"], dtype=torch.long, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                loss = out.loss / grad_accum
            loss.backward()

            cur_loss = float(out.loss.detach().item())
            if loss_start is None:
                loss_start = cur_loss
            loss_end = cur_loss

            if ((idx // batch_size) + 1) % grad_accum == 0 or (idx + batch_size) >= len(encoded):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if checkpoint_dir and save_every_steps > 0 and global_step % save_every_steps == 0:
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    torch.save(
                        {
                            "global_step": global_step,
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                        },
                        checkpoint_dir / "trainer_state.pt",
                    )

                if global_step >= total_steps:
                    break
        if global_step >= total_steps:
            break

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1) if device == "cuda" else 0.0
    summary = {
        "model_name": model_name,
        "ablation": ablation,
        "seed": seed,
        "dtype": "bf16",
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "target_modules": targets,
        "train_examples_used": len(encoded),
        "skipped_over_max_seq_len": skipped_len,
        "steps": global_step,
        "loss_start": round(loss_start or 0.0, 5),
        "loss_end": round(loss_end, 5),
        "peak_vram_mb": peak_vram_mb,
        "seconds": round(time.perf_counter() - t0, 2),
    }
    write_adapter_manifest(out_dir, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--compat-check", action="store_true", help="Run ROADMAP 7.1 Step 0 compatibility check only"
    )
    ap.add_argument("--model", default=None, help="Override params.yaml: llm.base_model")
    ap.add_argument("--sft-dir", type=Path, default=ROOT / "data" / "task" / "sft")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "llm_adapter")
    ap.add_argument("--ablation", choices=list(ABLATION_MODES), default="full")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-train-examples", type=int, default=None)
    ap.add_argument("--sanity-check-steps", type=int, default=0)
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--allow-local", action="store_true", help="Allow local run for unit/compat testing")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    params = load_params()
    llm_cfg = params["llm"]
    model_name = args.model or llm_cfg["base_model"]

    if args.compat_check:
        res = run_qwen35_compat_check(
            model_name=model_name,
            lora_r=int(llm_cfg["lora"]["r"]),
            lora_alpha=int(llm_cfg["lora"]["alpha"]),
        )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
            )
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    pre = check_bf16_lora_preconditions(params, allow_local=args.allow_local)
    if not pre["eligible"]:
        log.error("bf16 LoRA training blocked by preconditions: %s", pre["reasons"])
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps({"status": "blocked", "preconditions": pre}, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        raise RuntimeError("bf16 LoRA blocked: " + "; ".join(pre["reasons"]))

    summary = train_bf16_lora(
        model_name=model_name,
        sft_dir=args.sft_dir,
        out_dir=args.out,
        ablation=args.ablation,
        seed=args.seed if args.seed is not None else int(params["seed"]),
        lr=float(llm_cfg["lr"]),
        epochs=int(llm_cfg["epochs"]),
        lora_r=int(llm_cfg["lora"]["r"]),
        lora_alpha=int(llm_cfg["lora"]["alpha"]),
        lora_dropout=float(llm_cfg["lora"]["dropout"]),
        max_seq_len=int(llm_cfg["max_seq_len"]),
        max_train_examples=args.max_train_examples,
        sanity_check_steps=args.sanity_check_steps,
        checkpoint_dir=args.checkpoint_dir,
        thinking=bool(llm_cfg.get("thinking", False)),
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
