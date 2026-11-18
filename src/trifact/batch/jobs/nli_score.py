"""Kaggle job: NLI predictions for sentence pairs (label consistency of translated NLI, ADR-0015).

Input dataset: `pairs*.parquet` (columns `id`, `premise`, `hypothesis`) and `config.json`:
  {"model": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "revision": "<sha>",
   "batch_size": 128, "max_length": 256}
Output (/kaggle/working): `out/scores-g<gpu>.parquet` (id, p_entailment, p_neutral, p_contradiction) in the
model's label order mapped to names, and `summary.json`. One process per GPU (CUDA_VISIBLE_DEVICES).
Self-contained: no project imports.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

OUT = Path("/kaggle/working/out")


def worker(gpu: int, n: int) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg = json.loads(Path(glob.glob("/kaggle/input/**/config.json", recursive=True)[0]).read_text())
    files = sorted(glob.glob("/kaggle/input/**/pairs*.parquet", recursive=True))
    t = pa.concat_tables([pq.read_table(f) for f in files])
    ids = t.column("id").to_pylist()[gpu::n]
    prem = t.column("premise").to_pylist()[gpu::n]
    hyp = t.column("hypothesis").to_pylist()[gpu::n]
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg.get("revision"))
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"], revision=cfg.get("revision"), dtype=torch.float16
    )
    model = model.to("cuda:0").eval()
    names = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    order = sorted(range(len(ids)), key=lambda i: len(prem[i] or "") + len(hyp[i] or ""))
    probs: list[list[float]] = [[0.0] * len(names) for _ in ids]
    bs = cfg.get("batch_size", 128)
    t0 = time.time()
    for k in range(0, len(order), bs):
        b = order[k : k + bs]
        enc = tok([prem[i] or "" for i in b], [hyp[i] or "" for i in b], truncation=True,
                  max_length=cfg.get("max_length", 256), padding=True, return_tensors="pt").to("cuda:0")  # fmt: skip
        with torch.inference_mode():
            p = torch.softmax(model(**enc).logits.float(), dim=-1).cpu().tolist()
        for i, row in zip(b, p, strict=True):
            probs[i] = row
        if k % (200 * bs) == 0:
            print(f"[gpu{gpu}] {k}/{len(order)} {k / max(time.time() - t0, 1e-9):.0f} pairs/s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cols = {"id": ids, **{f"p_{nm}": [r[j] for r in probs] for j, nm in enumerate(names)}}
    pq.write_table(pa.table(cols), OUT / f"scores-g{gpu}.parquet", compression="zstd")
    Path(f"/kaggle/working/summary-g{gpu}.json").write_text(json.dumps(
        {"gpu": gpu, "pairs": len(ids), "seconds": round(time.time() - t0, 1), "labels": names}))  # fmt: skip


def main() -> None:
    if len(sys.argv) > 1:
        worker(int(sys.argv[1]), int(sys.argv[2]))
        return
    import torch

    n = max(1, torch.cuda.device_count())
    procs = [
        subprocess.Popen(
            [sys.executable, __file__, str(g), str(n)], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
        )
        for g in range(n)
    ]
    codes = [p.wait() for p in procs]
    parts = [json.loads(Path(f).read_text()) for f in sorted(glob.glob("/kaggle/working/summary-g*.json"))]
    Path("/kaggle/working/summary.json").write_text(
        json.dumps({"exit_codes": codes, "parts": parts}, indent=1)
    )
    if any(codes):
        raise SystemExit(f"worker failed: {codes}")


if __name__ == "__main__":
    main()
