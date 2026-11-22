"""Kaggle job: machine translation of sentences with NLLB-200, plus back-translation for quality control.

Input dataset: `<name>.parquet` files (columns `id`, `text`) and `config.json`:
  {"model": "facebook/nllb-200-distilled-1.3B", "back_model": "facebook/nllb-200-distilled-600M",
   "tasks": [{"file": "xnli_en.parquet", "src": "eng_Latn", "tgt": "kaz_Cyrl"}, ...],
   "beams": 2, "max_tokens": 4000, "max_new_tokens": 256, "time_budget_h": 11.0, "limit": null}
Output (/kaggle/working): `out/<file stem>-<tgt>-g<gpu>-<k>.parquet` (id, text, translation, back) and
`summary.json`. A batch that runs out of GPU memory is split in halves and retried.

One process per GPU (own model copy) takes every n-th sentence; parts are written every few thousand
sentences, so a kernel stopped by the time budget keeps what it translated (the driver re-runs
the remaining ids). Self-contained: no project imports.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

INPUT = Path(os.environ.get("TRIFACT_BATCH_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("TRIFACT_BATCH_WORK", "/kaggle/working"))
OUT = WORK / "out"
PART_ROWS = 8000


def find_inputs() -> tuple[dict[str, str], dict]:
    files = {f.name: str(f) for f in INPUT.rglob("*.parquet")}
    cfgs = sorted(INPUT.rglob("config.json"))
    if not files or not cfgs:
        raise SystemExit(f"inputs not found: {files} {cfgs}")
    return files, json.loads(cfgs[0].read_text(encoding="utf-8"))


def batches(lengths: list[int], max_tokens: int) -> list[list[int]]:
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out, cur, longest = [], [], 0
    for i in order:
        longest2 = max(longest, lengths[i])
        if cur and longest2 * (len(cur) + 1) > max_tokens:
            out.append(cur)
            cur, longest2 = [], lengths[i]
        cur.append(i)
        longest = longest2
    if cur:
        out.append(cur)
    return out


class Translator:
    def __init__(self, model_name: str, dev: str, revision: str | None = None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=revision, dtype=torch.float16)
            .to(dev)
            .eval()
        )
        self.dev = dev

    def _gen(self, texts: list[str], tgt_id: int, beams: int, max_new: int) -> list[str]:
        try:
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_new).to(
                self.dev
            )
            with self.torch.inference_mode():
                gen = self.model.generate(
                    **enc, forced_bos_token_id=tgt_id, num_beams=beams, max_new_tokens=max_new
                )
            return self.tok.batch_decode(gen, skip_special_tokens=True)
        except self.torch.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            if len(texts) == 1:
                if max_new <= 32:
                    return [""]  # untranslatable within memory: reported as an empty translation
                return self._gen(texts, tgt_id, 1, max_new // 2)
            h = len(texts) // 2
            return self._gen(texts[:h], tgt_id, beams, max_new) + self._gen(texts[h:], tgt_id, beams, max_new)

    def run(self, texts, src, tgt, beams, max_tokens, max_new, deadline, log):
        self.tok.src_lang = src
        tgt_id = self.tok.convert_tokens_to_ids(tgt)
        lengths = [len(x) for x in self.tok(texts, truncation=True, max_length=max_new)["input_ids"]]
        out: list[str | None] = [None] * len(texts)
        done = 0
        t0 = time.time()
        for bi, b in enumerate(batches(lengths, max_tokens)):
            if time.time() > deadline:
                log(f"time budget reached after {done} sentences")
                break
            for i, t in zip(b, self._gen([texts[i] for i in b], tgt_id, beams, max_new), strict=True):
                out[i] = t
            done += len(b)
            if bi % 50 == 0:
                log(f"{self.name} {src}->{tgt}: {done}/{len(texts)} {done / (time.time() - t0):.1f} sent/s")
        return out, time.time() - t0


def worker(gpu: int, n_gpu: int) -> None:
    files, cfg = find_inputs()
    dev = "cuda:0"  # CUDA_VISIBLE_DEVICES selects the GPU of this process
    t_start = time.time()
    deadline = t_start + cfg.get("time_budget_h", 11.0) * 3600
    logf = (WORK / f"log-g{gpu}.txt").open("a", buffering=1, encoding="utf-8")

    def log(msg: str) -> None:
        print(f"[gpu{gpu}] {msg}", flush=True)
        logf.write(msg + "\n")

    OUT.mkdir(parents=True, exist_ok=True)
    beams, mt, mn = cfg.get("beams", 2), cfg.get("max_tokens", 4000), cfg.get("max_new_tokens", 256)
    fwd = Translator(cfg["model"], dev, cfg.get("revision"))
    jobs = []
    for task in cfg["tasks"]:
        table = pq.read_table(files[task["file"]], columns=["id", "text"])
        if cfg.get("limit"):
            table = table.slice(0, cfg["limit"])
        ids = table.column("id").to_pylist()[gpu::n_gpu]
        texts = table.column("text").to_pylist()[gpu::n_gpu]
        # forward passes share 75% of the budget, back-translation the rest
        fwd_deadline = t_start + 0.75 * (deadline - t_start)
        trans, sec = fwd.run(texts, task["src"], task["tgt"], beams, mt, mn, fwd_deadline, log)
        jobs.append((task, ids, texts, trans, sec))
    del fwd.model
    fwd.torch.cuda.empty_cache()
    back_m = Translator(cfg["back_model"], dev, cfg.get("back_revision")) if cfg.get("back_model") else None
    summary = []
    for task, ids, texts, trans, sec in jobs:
        keep = [i for i, t in enumerate(trans) if t is not None]
        back: list[str | None] = [None] * len(texts)
        s_back = 0.0
        if back_m and keep:
            b, s_back = back_m.run(
                [trans[i] for i in keep], task["tgt"], task["src"], 1, mt, mn, deadline, log
            )
            for i, t in zip(keep, b, strict=True):
                back[i] = t
        stem = Path(task["file"]).stem
        for k in range(0, len(keep), PART_ROWS):
            idx = keep[k : k + PART_ROWS]
            pq.write_table(
                pa.table(
                    {
                        "id": [ids[i] for i in idx],
                        "text": [texts[i] for i in idx],
                        "translation": [trans[i] for i in idx],
                        "back": [back[i] for i in idx],
                    }
                ),
                OUT / f"{stem}-{task['tgt']}-g{gpu}-{k // PART_ROWS:04d}.parquet",
                compression="zstd",
            )
        summary.append(
            {
                "file": task["file"],
                "src": task["src"],
                "tgt": task["tgt"],
                "sentences": len(texts),
                "translated": len(keep),
                "back_translated": sum(x is not None for x in back),
                "seconds_forward": round(sec, 1),
                "seconds_back": round(s_back, 1),
                "forward_sent_per_s": round(len(keep) / max(sec, 1e-9), 2),
            }
        )
    (WORK / f"summary-g{gpu}.json").write_text(json.dumps({"gpu": gpu, "tasks": summary}, indent=1))
    log(json.dumps(summary))


def main() -> None:
    import subprocess
    import sys

    import torch

    if len(sys.argv) > 1:  # child: one GPU, made visible as cuda:0
        worker(int(sys.argv[1]), int(sys.argv[2]))
        return
    n = max(1, torch.cuda.device_count())
    # one process per GPU: model loading sets torch's global default dtype, which is not thread-safe
    procs = [
        subprocess.Popen(
            [sys.executable, __file__, str(g), str(n)], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
        )
        for g in range(n)
    ]
    codes = [p.wait() for p in procs]
    parts = [json.loads(f.read_text()) for f in sorted(WORK.glob("summary-g*.json"))]
    (WORK / "summary.json").write_text(
        json.dumps(
            {"gpus": n, "device": torch.cuda.get_device_name(0), "exit_codes": codes, "parts": parts},
            indent=1,
        )
    )
    if any(codes):
        raise SystemExit(f"worker failed: {codes}")


if __name__ == "__main__":
    main()
