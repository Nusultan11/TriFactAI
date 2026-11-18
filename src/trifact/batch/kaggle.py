"""Run a batch job on a Kaggle GPU kernel (ADR-0014, ADR-0020): inputs as a private dataset, outputs downloaded.

python -m trifact.batch.kaggle run JOB --inputs DIR --out DIR [--config JSON] [--cpu] [--timeout-h H]
python -m trifact.batch.kaggle fetch JOB --out DIR      # download the output of the last run

A job is a self-contained script `src/trifact/batch/jobs/<JOB>.py` (no project imports: the kernel
image has torch/transformers but not this package). The driver

1. uploads `--inputs` (Parquet files and `config.json`) as the private dataset `<user>/trifact-<job>-in`
   (new version when it exists) and waits until Kaggle has processed it;
2. pushes the private kernel `<user>/trifact-<job>` with that dataset attached, GPU T4 x2, internet on
   (models come from the HF Hub, public weights only: no token leaves this machine);
3. polls the kernel status, then downloads `/kaggle/working` into `--out` and writes `run.json`
   (kernel version, wall time, GPU hours charged) for the cost record of the stage.

Nothing is sent but the input directory and the job script. Kaggle GPU time is free within the weekly
quota (`kaggle quota`); `params.yaml: cloud.budget_usd` stays 0.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("trifact.kaggle")
JOBS = Path(__file__).parent / "jobs"
POLL_S = 60


def kaggle(*args: str, check: bool = True) -> str:
    r = subprocess.run(
        ["kaggle", *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if check and r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed: {r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def username() -> str:
    for line in kaggle("config", "view").splitlines():
        if line.strip().startswith("- username:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("kaggle username not configured")


def push_dataset(user: str, name: str, inputs: Path) -> str:
    slug = f"trifact-{name}-in"
    ref = f"{user}/{slug}"
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for f in inputs.iterdir():
            if f.is_file():
                shutil.copy2(f, d / f.name)
        meta = {"title": slug, "id": ref, "licenses": [{"name": "other"}]}
        (d / "dataset-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        exists = kaggle("datasets", "status", ref, check=False).strip().lower().endswith("ready")
        if exists:
            log.info("new version of dataset %s", ref)
            kaggle(
                "datasets",
                "version",
                "-p",
                str(d),
                "-m",
                f"inputs {time.strftime('%Y-%m-%d %H:%M')}",
                "-q",
                "-t",
            )
        else:
            log.info("creating private dataset %s", ref)
            kaggle("datasets", "create", "-p", str(d), "-q", "-t")
    for _ in range(120):
        time.sleep(15)
        st = kaggle("datasets", "status", ref, check=False).strip().lower()
        if st.endswith("ready"):
            return ref
        log.info("dataset %s: %s", ref, st.splitlines()[-1] if st else "?")
    raise RuntimeError(f"dataset {ref} not ready")


def push_kernel(user: str, job: str, name: str, dataset: str, gpu: bool) -> str:
    ref = f"{user}/trifact-{name}"
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        shutil.copy2(JOBS / f"{job}.py", d / f"{job}.py")
        meta = {
            "id": ref, "title": f"trifact-{name}", "code_file": f"{job}.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [dataset], "kernel_sources": [],
            "competition_sources": [], "model_sources": [],
        }  # fmt: skip
        if gpu:
            meta["machine_shape"] = "NvidiaTeslaT4"
        (d / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        out = kaggle("kernels", "push", "-p", str(d))
    if "error" in out.lower():
        raise RuntimeError(f"kernel push of {ref} failed: {out.strip()}")
    log.info("pushed %s: %s", ref, out.strip().splitlines()[-1])
    return ref


def status(ref: str) -> str:
    out = kaggle("kernels", "status", ref, check=False)
    for word in ("complete", "error", "cancel", "running", "queued"):
        if word in out.lower():
            return word
    return out.strip()


def wait(ref: str, timeout_h: float) -> str:
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout_h * 3600:
        st = status(ref)
        if st != last:
            log.info("%s: %s (%.0f min)", ref, st, (time.time() - t0) / 60)
            last = st
        if st in ("complete", "error", "cancel"):
            return st
        time.sleep(POLL_S)
    return "timeout"


def fetch(ref: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", ref, "-p", str(out), "-o")


def run(
    job: str, inputs: Path, out: Path, gpu: bool, timeout_h: float, name: str | None = None
) -> dict[str, Any]:
    name = name or job
    user = username()
    quota0 = kaggle("quota", check=False)
    dataset = push_dataset(user, name, inputs)
    ref = push_kernel(user, job, name, dataset, gpu)
    t0 = time.time()
    time.sleep(30)
    st = wait(ref, timeout_h)
    fetch(ref, out)
    rec = {
        "job": job, "kernel": ref, "dataset": dataset, "status": st, "gpu": "T4x2" if gpu else None,
        "wall_hours": round((time.time() - t0) / 3600, 3), "quota_before": quota0.strip(),
        "quota_after": kaggle("quota", check=False).strip(), "cost_usd": 0.0,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }  # fmt: skip
    (out / "run.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    log.info("%s %s in %.2f h -> %s", ref, st, rec["wall_hours"], out)
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("job")
    r.add_argument("--inputs", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--cpu", action="store_true")
    r.add_argument("--timeout-h", type=float, default=13.0)
    r.add_argument(
        "--name",
        help="kernel / dataset name (default: the job), so one job script can run as several kernels",
    )
    f = sub.add_parser("fetch")
    f.add_argument("job")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--name")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "run":
        rec = run(args.job, args.inputs, args.out, not args.cpu, args.timeout_h, args.name)
        return 0 if rec["status"] == "complete" else 1
    fetch(f"{username()}/trifact-{args.name or args.job}", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
