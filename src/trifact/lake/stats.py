"""Run measurement for data stages: wall time, throughput, peak RSS of the whole process tree.

Peak RSS is sampled every `interval` seconds over the current process and all its children (pool
workers), so it is the number that has to stay under `params.yaml: lake.peak_ram_gb`.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import httpx
import psutil

from trifact.lake.manifest import RunStats

log = logging.getLogger(__name__)


def tree_rss(proc: psutil.Process) -> int:
    total = 0
    for p in [proc, *proc.children(recursive=True)]:
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


class ResourceMonitor:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak_rss = 0
        self.started = 0.0
        self.seconds = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._proc = psutil.Process(os.getpid())

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, tree_rss(self._proc))
            self._stop.wait(self.interval)

    def __enter__(self) -> Self:
        self.started = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(
        self, et: type[BaseException] | None, ev: BaseException | None, tb: TracebackType | None
    ) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_rss = max(self.peak_rss, tree_rss(self._proc))
        self.seconds = time.perf_counter() - self.started

    @property
    def peak_rss_mb(self) -> float:
        return self.peak_rss / 2**20

    def stats(self, rows: int, nbytes: int, workers: int) -> RunStats:
        secs = max(self.seconds or (time.perf_counter() - self.started), 1e-9)
        return RunStats(
            seconds=round(secs, 1),
            rows_per_s=round(rows / secs, 1),
            mb_per_s=round(nbytes / 2**20 / secs, 2),
            peak_rss_mb=round(self.peak_rss_mb, 1),
            workers=workers,
        )


def git_sha(root: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log_mlflow(
    experiment: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: list[Path] | None = None,
) -> bool:
    """Best-effort MLflow logging; the manifest is the primary record, MLflow the comparison view."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        log.warning("MLFLOW_TRACKING_URI is not set; run %s not logged to MLflow", run_name)
        return False
    try:
        httpx.get(f"{uri.rstrip('/')}/health", timeout=3).raise_for_status()
    except httpx.HTTPError as e:
        log.warning("MLflow at %s is not reachable (%s); run %s not logged", uri, e, run_name)
        return False
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
            mlflow.log_metrics(metrics)
            for a in artifacts or []:
                mlflow.log_artifact(str(a))
        return True
    except Exception as e:  # noqa: BLE001 - any MLflow failure must not lose a finished download
        log.warning("MLflow logging failed for %s: %s", run_name, e)
        return False
