"""Kaggle job: teacher LLM generation with llama.cpp (Qwen3.5-35B-A3B GGUF on 2x T4), ADR-0015 / ADR-0020.

Input dataset: `prompts*.parquet` (columns `id`, `system`, `user`, `max_tokens`, optional per-row `temperature`, e.g. 0
for judge prompts) and `config.json`:
  {"repo": "unsloth/Qwen3.5-35B-A3B-GGUF", "revision": "<sha>", "file": "Qwen3.5-35B-A3B-UD-IQ4_XS.gguf",
   "llama_cpp_ref": "<git tag>", "parallel": 16, "ctx_per_slot": 4096, "temperature": 0.7, "top_p": 0.9,
   "time_budget_h": 11.0, "limit": null}
Output (/kaggle/working): `out/gen-<k>.parquet` (id, text, finish_reason, prompt_tokens, completion_tokens,
seconds) every `PART` answers, `summary.json` with throughput. T4 has no bf16 and vLLM's MoE kernels need
sm80, so llama.cpp (CUDA, sm75) serves the 4-bit GGUF split over both GPUs. Thinking is disabled
(`enable_thinking: false`). Self-contained: no project imports.
"""

from __future__ import annotations

import concurrent.futures as cf
import glob
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

WORK = Path("/kaggle/working")
OUT = WORK / "out"
BUILD = Path("/tmp/llama.cpp")
PART = 2000
PORT = 8080


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def has_cuda(exe: Path) -> bool:
    r = subprocess.run([str(exe), "--list-devices"], capture_output=True, text=True, check=False)
    print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
    return "CUDA0" in (r.stdout + r.stderr)


def build(ref: str) -> Path:
    """Prebuilt CUDA 12.8 llama.cpp release if it sees the GPUs, else a source build against the local CUDA."""
    sh(
        "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; nvcc --version | tail -2 || true"
    )
    base = f"https://github.com/ggml-org/llama.cpp/releases/download/{ref}"
    pre = BUILD / "pre"
    pre.mkdir(parents=True, exist_ok=True)
    try:
        for name in (
            f"llama-{ref}-bin-ubuntu-cuda-12.8-x64.tar.gz",
            f"cudart-llama-{ref}-bin-ubuntu-cuda-12.8-x64.tar.gz",
        ):
            sh(f"curl -sSL --retry 5 -o /tmp/{name} {base}/{name} && tar -xzf /tmp/{name} -C {pre}")
        exe = next(pre.rglob("llama-server"))
        libs = sorted({str(p.parent) for p in pre.rglob("*.so*")})
        os.environ["LD_LIBRARY_PATH"] = ":".join([*libs, os.environ.get("LD_LIBRARY_PATH", "")])
        exe.chmod(0o755)
        if has_cuda(exe):
            return exe
        print("prebuilt binary sees no CUDA device: building from source", flush=True)
    except Exception as e:  # noqa: BLE001 - fall back to building from source
        print("prebuilt failed:", e, flush=True)
    src = BUILD / "src"
    sh(f"git clone --depth 1 --branch {ref} https://github.com/ggml-org/llama.cpp {src}")
    drv = next(iter(sorted(Path("/usr/lib/x86_64-linux-gnu").glob("libcuda.so*"))), None) or next(
        iter(sorted(Path("/usr/local/cuda/lib64/stubs").glob("libcuda.so*"))), None
    )
    sh(
        f"cmake -S {src} -B {src}/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF "
        f"-DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/usr/local/cuda -DCUDA_cuda_driver_LIBRARY={drv}"
    )
    sh(f"cmake --build {src}/build --target llama-server -j {os.cpu_count()}")
    exe = src / "build" / "bin" / "llama-server"
    if not has_cuda(exe):
        raise SystemExit("llama-server built from source sees no CUDA device")
    return exe


def model(cfg: dict) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(cfg["repo"], cfg["file"], revision=cfg["revision"], local_dir="/tmp/model")


def serve(exe: Path, gguf: str, cfg: dict) -> subprocess.Popen:
    n = cfg.get("parallel", 16)
    ctx = n * cfg.get("ctx_per_slot", 4096)
    log = open(WORK / "server.log", "w")  # noqa: SIM115 - lives for the job
    p = subprocess.Popen(
        [
            str(exe),
            "-m",
            gguf,
            "-ngl",
            "999",
            "--split-mode",
            "layer",
            "-c",
            str(ctx),
            "--parallel",
            str(n),
            "--port",
            str(PORT),
            "--jinja",
            "-fa",
            "on",
            "--cont-batching",
            "-b",
            str(cfg.get("batch", 2048)),
            "-ub",
            str(cfg.get("ubatch", 512)),
            "--no-webui",
            *cfg.get("extra_args", []),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    for _ in range(600):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return p
        except Exception:  # noqa: BLE001, S110 - server still loading
            pass
        if p.poll() is not None:
            raise SystemExit(f"llama-server exited: {(WORK / 'server.log').read_text()[-3000:]}")
        time.sleep(5)
    raise SystemExit("llama-server did not become healthy")


def ask(row: dict, cfg: dict) -> dict:
    body = {
        "messages": [m for m in ({"role": "system", "content": row["system"]} if row["system"] else None,
                                 {"role": "user", "content": row["user"]}) if m],
        "max_tokens": int(row["max_tokens"]),
        "temperature": cfg.get("temperature", 0.7) if row.get("temperature") is None else float(row["temperature"]),
        "top_p": cfg.get("top_p", 0.9), "chat_template_kwargs": {"enable_thinking": False},
    }  # fmt: skip
    t0 = time.time()
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.loads(r.read())
            c = d["choices"][0]
            return {"id": row["id"], "text": c["message"].get("content") or "", "finish_reason": c.get("finish_reason"),
                    "prompt_tokens": d["usage"]["prompt_tokens"], "completion_tokens": d["usage"]["completion_tokens"],
                    "seconds": round(time.time() - t0, 2)}  # fmt: skip
        except Exception as e:  # noqa: BLE001 - retried, then reported as an empty answer
            err = repr(e)
            time.sleep(5 * (attempt + 1))
    return {"id": row["id"], "text": "", "finish_reason": f"error: {err}", "prompt_tokens": 0,
            "completion_tokens": 0, "seconds": round(time.time() - t0, 2)}  # fmt: skip


def main() -> None:
    cfg = json.loads(Path(glob.glob("/kaggle/input/**/config.json", recursive=True)[0]).read_text())
    files = sorted(glob.glob("/kaggle/input/**/prompts*.parquet", recursive=True))
    rows = pa.concat_tables([pq.read_table(f) for f in files], promote_options="default").to_pylist()
    if cfg.get("limit"):
        rows = rows[: cfg["limit"]]
    t_start = time.time()
    deadline = t_start + cfg.get("time_budget_h", 11.0) * 3600
    exe = build(cfg["llama_cpp_ref"])
    t_build = time.time() - t_start
    gguf = model(cfg)
    t_model = time.time() - t_start - t_build
    if cfg.get("diagnose"):
        sh(f"{exe} --help 2>&1 | grep -i -E 'fit|gpu-layers|split-mode|flash' | head -20 || true")
        bench = exe.parent / "llama-bench"
        if bench.exists():
            sh(f"{bench} -m {gguf} -ngl 999 -p 512 -n 64 -fa 1 2>&1 | tail -6 || true")
        bb = exe.parent / "llama-batched-bench"
        if (
            bb.exists()
        ):  # prefill and decode throughput per batch / ubatch / parallel setting (config "bench")
            for extra in cfg.get("bench", ["-b 2048 -ub 512 -npp 400 -ntg 64 -npl 1,4,16 -kvu"]):
                sh(f"{bb} -m {gguf} -ngl 999 -fa on -c 65536 {extra} 2>&1 | tail -8 || true")
        if cfg.get("diagnose_only"):
            return
    server = serve(exe, gguf, cfg)
    sh("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv")
    t_ready = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    buf: list[dict] = []
    k = done = ptok = ctok = 0
    with cf.ThreadPoolExecutor(cfg.get("parallel", 16)) as pool:
        futs = []
        it = iter(rows)
        for r in it:  # keep 2x parallel requests in flight so the server batches stay full
            futs.append(pool.submit(ask, r, cfg))
            if len(futs) < 2 * cfg.get("parallel", 16):
                continue
            f = futs.pop(0)
            res = f.result()
            buf.append(res)
            done += 1
            ptok += res["prompt_tokens"]
            ctok += res["completion_tokens"]
            if len(buf) >= PART:
                pq.write_table(pa.Table.from_pylist(buf), OUT / f"gen-{k:05d}.parquet", compression="zstd")
                k += 1
                buf = []
                el = time.time() - t_ready
                print(
                    f"{done}/{len(rows)} {done / el:.2f} req/s, {ctok / el:.0f} gen tok/s, {ptok / el:.0f} prompt tok/s",
                    flush=True,
                )
            if time.time() > deadline:
                print("time budget reached", flush=True)
                break
        for f in futs:
            res = f.result()
            buf.append(res)
            done += 1
            ptok += res["prompt_tokens"]
            ctok += res["completion_tokens"]
    if buf:
        pq.write_table(pa.Table.from_pylist(buf), OUT / f"gen-{k:05d}.parquet", compression="zstd")
    el = time.time() - t_ready
    import torch

    summary = {"requests": len(rows), "done": done, "seconds_generation": round(el, 1),
               "seconds_build": round(t_build, 1), "seconds_model_download": round(t_model, 1),
               "req_per_s": round(done / el, 3), "completion_tok_per_s": round(ctok / el, 1),
               "prompt_tok_per_s": round(ptok / el, 1), "completion_tokens": ctok, "prompt_tokens": ptok,
               "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "config": cfg}  # fmt: skip
    (WORK / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)
    server.terminate()


if __name__ == "__main__":
    main()
