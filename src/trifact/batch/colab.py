"""Run one bounded input shard on a hosted Colab L4 and checkpoint it to Google Drive."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from uuid import uuid4

import pyarrow.parquet as pq

from trifact.lake.stats import log_mlflow

JOBS = Path(__file__).parent / "jobs"
GPU_QUERY = "name,driver_version,memory.total,memory.used"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path) for path in sorted(root.rglob("*")) if path.is_file()
    }


def drive_root(path: Path) -> Path:
    """Require a mounted, existing project tree in a hosted Colab runtime."""
    mount = Path("/content/drive")
    root = path.resolve(strict=True)
    if not os.environ.get("COLAB_RELEASE_TAG") or not mount.is_mount() or not root.is_relative_to(mount):
        raise ValueError("Use a hosted Colab runtime with Google Drive mounted at /content/drive")
    if not (root / "dvc" / "files" / "md5").is_dir():
        raise ValueError(f"DVC object store missing under {root}")
    return root


def gpu_info() -> tuple[str, str, int, int]:
    result = (
        subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"], text=True
        )
        .strip()
        .splitlines()
    )
    if len(result) != 1:
        raise ValueError("This runner requires exactly one NVIDIA L4")
    name, driver, total, used = (field.strip() for field in result[0].split(","))
    if name not in {"NVIDIA L4", "Tesla L4"}:
        raise ValueError(f"Expected NVIDIA L4, received {name}")
    return name, driver, int(total), int(used)


def read_ids(paths: list[Path], limit: int | None = None) -> Counter:
    ids: Counter = Counter()
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(columns=["id"]):
            values = batch.column(0).to_pylist()
            if limit is not None:
                values = values[:limit]
                limit -= len(values)
            ids.update(values)
            if limit == 0:
                return ids
    return ids


def verify_rows(job: str, inputs: Path, outputs: Path, config: dict) -> int:
    limit = config.get("limit") or None
    if job == "translate":
        groups = [
            ([inputs / task["file"]], f"{Path(task['file']).stem}-{task['tgt']}-g*.parquet")
            for task in config["tasks"]
        ]
    elif job == "nli_score":
        groups = [(sorted(inputs.glob("pairs*.parquet")), "scores-*.parquet")]
    else:
        groups = [(sorted(inputs.glob("prompts*.parquet")), "gen-*.parquet")]
    if not groups or len({pattern for _, pattern in groups}) != len(groups):
        raise ValueError("Expected distinct, non-empty input groups")
    total = 0
    for sources, pattern in groups:
        expected = read_ids(sources, limit if job != "nli_score" else None)
        parts = sorted(outputs.glob(pattern))
        if not expected or None in expected or not all(n == 1 for n in expected.values()):
            raise ValueError(f"Expected unique, non-null input IDs for {pattern}")
        if read_ids(parts) != expected:
            raise ValueError(f"Output IDs do not match the input shard for {pattern}")
        for part in parts:
            if job == "teacher" and any(
                str(reason).startswith("error:")
                for reason in pq.read_table(part, columns=["finish_reason"]).column(0).to_pylist()
            ):
                raise ValueError(f"Generation transport failure in {part.name}")
            if job == "translate" and config.get("back_model"):
                values = pq.read_table(part, columns=["back"]).column(0)
                if values.null_count:
                    raise ValueError(f"Back-translation incomplete in {part.name}")
        total += len(expected)
    return total


def valid_checkpoint(path: Path, identity: dict) -> bool:
    if not (path / "run.json").is_file():
        return False
    receipt = json.loads((path / "run.json").read_text(encoding="utf-8"))
    return (
        receipt.get("identity") == identity
        and receipt.get("status") == "complete"
        and bool(receipt.get("outputs"))
        and (path / "summary.json").is_file()
        and receipt.get("summary_sha256") == sha256(path / "summary.json")
        and receipt.get("outputs") == file_hashes(path / "out")
    )


def execute(job: str, inputs: Path, root: Path, timeout_minutes: float) -> Path:
    import psutil

    project = JOBS.parents[3]
    source = JOBS / f"{job}.py"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()
    gpu, driver, memory_total, _ = gpu_info()
    if not inputs.is_dir() or not (inputs / "config.json").is_file():
        raise ValueError("Provide a directory with config.json and the input Parquet files")
    files = [inputs / "config.json", *sorted(inputs.glob("*.parquet"))]
    if len(files) == 1:
        raise ValueError("No input Parquet files found")
    identity = {
        "job": job,
        "git_sha": revision,
        "source_sha256": sha256(source),
        "runner_sha256": sha256(Path(__file__)),
        "lock_sha256": sha256(project / "uv.lock"),
        "inputs": {p.name: sha256(p) for p in files},
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    checkpoints = root / "workspace-data" / "colab" / job / key
    checkpoints.mkdir(parents=True, exist_ok=True)
    for checkpoint in sorted(checkpoints.glob("*/")):
        if valid_checkpoint(checkpoint, identity):
            return checkpoint

    attempt = checkpoints / uuid4().hex
    attempt.mkdir()
    start = time.monotonic()
    peak_ram = peak_vram = rows = 0
    receipt = {
        "identity": identity,
        "gpu": gpu,
        "driver": driver,
        "vram_total_mib": memory_total,
        "status": "running",
    }
    marker = attempt / "run.json"
    marker.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="trifact-colab-") as tmp:
        scratch = Path(tmp)
        staging = scratch / "inputs"
        staging.mkdir()
        for path in files:
            shutil.copyfile(path, staging / path.name)
        if file_hashes(staging) != identity["inputs"]:
            raise ValueError("Input shard changed while staging")
        config = json.loads((staging / "config.json").read_text(encoding="utf-8"))
        if job == "teacher":
            overrides = config.pop("colab", {})
            config.update(parallel=4, batch=2048, ubatch=512, split_mode="none")
            config.update(overrides)
            (staging / "config.json").write_text(json.dumps(config), encoding="utf-8")
        receipt["config"] = config
        # Stream diagnostics to persistent storage. Unpublished output remains in scratch.
        with (attempt / "execution.log").open("w", encoding="utf-8", buffering=1) as log:
            proc = subprocess.Popen(
                [sys.executable, str(source)],
                env={**os.environ, "TRIFACT_BATCH_INPUT": str(staging), "TRIFACT_BATCH_WORK": str(scratch)},
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            monitor = psutil.Process(proc.pid)
            try:
                while proc.poll() is None:
                    rss = 0
                    with contextlib.suppress(psutil.NoSuchProcess):
                        for child in [monitor, *monitor.children(recursive=True)]:
                            with contextlib.suppress(psutil.NoSuchProcess):
                                rss += child.memory_info().rss
                    peak_ram = max(peak_ram, rss)
                    peak_vram = max(peak_vram, gpu_info()[3])
                    if time.monotonic() - start >= timeout_minutes * 60:
                        raise TimeoutError(f"{job} exceeded {timeout_minutes} min")
                    time.sleep(2)
                if proc.returncode:
                    raise RuntimeError(f"{job} returned {proc.returncode}, see {attempt / 'execution.log'}")
                rows = verify_rows(job, staging, scratch / "out", config)
                hashes = file_hashes(scratch / "out")
                shutil.copytree(scratch / "out", attempt / "out")
                shutil.copyfile(scratch / "summary.json", attempt / "summary.json")
                if hashes != file_hashes(attempt / "out"):
                    raise ValueError("Output hash mismatch after copying to Drive")
                receipt.update(
                    status="complete", outputs=hashes, summary_sha256=sha256(attempt / "summary.json")
                )
            except BaseException as error:
                receipt.update(status="failed", error=repr(error))
                raise
            finally:
                # The teacher may leave llama-server running if the worker fails.
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                seconds = time.monotonic() - start
                metrics = {
                    "rows": rows,
                    "seconds": seconds,
                    "rows_per_second": rows / seconds,
                    "peak_process_tree_rss_bytes": peak_ram,
                    "peak_gpu_used_mib": peak_vram,
                    "gpu_hours": seconds / 3600,
                }
                receipt["metrics"] = metrics
                marker = attempt / "run.json"
                pending = marker.with_suffix(".tmp")
                pending.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
                pending.replace(marker)
                log_mlflow(
                    "batch",
                    f"colab-{job}",
                    {"checkpoint": key, "git_sha": revision, "gpu": gpu, "status": receipt["status"]},
                    metrics,
                    [marker],
                )
    return attempt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=("translate", "nli_score", "teacher"))
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--drive", required=True, type=Path)
    parser.add_argument("--timeout-minutes", type=float, required=True)
    args = parser.parse_args(argv)
    if not 0 < args.timeout_minutes < float("inf"):
        parser.error("--timeout-minutes must be finite and positive")
    print(execute(args.job, args.inputs.resolve(), drive_root(args.drive), args.timeout_minutes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
