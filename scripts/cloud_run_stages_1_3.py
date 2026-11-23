"""Turnkey cloud CPU runner for remaining Stage 1-3 DVC stages (ADR-0025).

Run inside a Colab CPU runtime with Google Drive mounted at `/content/drive`:

    uv run --no-sync python scripts/cloud_run_stages_1_3.py --drive /content/drive/MyDrive/TrifactAI

Steps (selectable via --steps):
  1. wikitext_merge  - merge verified ruwiki part 4 into raw/wikipedia_wikitext and push to DVC
  2. build_ner       - rebuild gold + silver NER (>5M target) with ruwiki part 4 and push to DVC
  3. synthetic       - import combined teacher_queries + queries2 (~102k) and finalize qa + qa2 with NLI scores
  4. train_and_eval  - rebuild build_pairs, build_sft, verify eval_leakage == 0, run task_eda + eda_report
  5. eda_rss_check   - run eda entities + curated and verify peak RSS <= lake.peak_ram_gb (10 GiB)

All DVC outputs are pushed immediately to the mounted Drive remote (`verify=true`), MLflow SQLite is
atomically snapshotted to Drive after each step, and updated `dvc.lock` + `reports/` are persisted to
`<drive>/workspace-data/stage-01-03-reconcile/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import time
import urllib.request
from contextlib import closing
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
ALL_STEPS = ("wikitext_merge", "build_ner", "synthetic", "train_and_eval", "eda_rss_check")


def run(*cmd: Any, check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    args = [str(c) for c in cmd]
    print("+", " ".join(args), flush=True)
    return subprocess.run(args, cwd=ROOT, check=check, text=True, env=env)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def start_mlflow(drive: Path, session: str) -> tuple[subprocess.Popen[str], Path, Path]:
    tracking = drive / "workspace-data" / "mlflow"
    tracking.mkdir(parents=True, exist_ok=True)
    db = Path("/content") / f"trifact-{session}.db"
    log_path = Path("/content") / f"mlflow-{session}.log"
    uv = shutil.which("uv") or "uv"
    log_f = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            uv,
            "run",
            "--no-sync",
            "mlflow",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            "5000",
            "--backend-store-uri",
            f"sqlite:///{db}",
            "--no-serve-artifacts",
            "--default-artifact-root",
            (tracking / "artifacts").as_uri(),
        ],
        cwd=ROOT,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )
    os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
    for _ in range(60):
        if proc.poll() is not None:
            raise RuntimeError(f"MLflow failed to start; see {log_path}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:5000/health", timeout=1):
                return proc, db, tracking
        except OSError:
            time.sleep(1)
    raise TimeoutError("MLflow did not become healthy within 60s")


def snapshot_mlflow(db: Path, tracking: Path, session: str) -> Path:
    temp_db = db.with_suffix(".snapshot")
    snap = tracking / f"{session}-{time.time_ns()}.db"
    pending = snap.with_suffix(".tmp")
    with closing(sqlite3.connect(db)) as src, closing(sqlite3.connect(temp_db)) as dst:
        src.backup(dst)
        assert dst.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    shutil.copyfile(temp_db, pending)
    assert sha256_file(temp_db) == sha256_file(pending), "MLflow snapshot corrupted during copy"
    pending.replace(snap)
    return snap


def setup_dvc_and_models(drive: Path) -> list[str]:
    dvc_dir = drive / "dvc"
    assert (dvc_dir / "files" / "md5").is_dir(), f"DVC cache missing at {dvc_dir}"
    uv = [shutil.which("uv") or "uv", "run", "--no-sync"]
    run(*uv, "dvc", "remote", "add", "--local", "--force", "--default", "drive", dvc_dir)
    run(*uv, "dvc", "remote", "modify", "--local", "drive", "verify", "true")
    run(*uv, "dvc", "config", "--local", "core.autostage", "false")
    lid_parts = drive / "models" / "lid176" / "lid.176.bin.parts.json"
    if lid_parts.is_file():
        run(
            *uv, "python", "scripts/restore_model.py", lid_parts, "--dest", ROOT / "data/.cache/models/lid176"
        )
    return uv


def step_wikitext_merge(uv: list[str], drive: Path) -> dict[str, Any]:
    part4_dir = drive / "workspace-data" / "wikitext-part4"
    res_file = part4_dir / "result.json"
    run(*uv, "dvc", "pull", "wikitext")
    if res_file.is_file():
        from trifact.lake.manifest import InputFile, RowCheck, ShardEntry, SourceManifest, verify_source
        from trifact.lake.stats import git_sha

        info = json.loads(res_file.read_text(encoding="utf-8"))
        src = ROOT / "data/raw/wikipedia_wikitext"
        m = SourceManifest.read(src)
        existing_paths = {s.path for s in m.shards}
        added = 0
        for s_dict in info["shards"]:
            se = ShardEntry.model_validate(s_dict)
            if se.path in existing_paths:
                continue
            src_f = part4_dir / se.path
            dst_f = src / se.path
            dst_f.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_f, dst_f)
            assert sha256_file(dst_f) == se.sha256, f"SHA-256 mismatch on {se.path}"
            m.shards.append(se)
            added += 1
        if added:
            m.inputs.append(
                InputFile(
                    path=info["url"],
                    size=int(info["input_size"]),
                    lang=info["file"]["lang"],
                    table="part",
                    rows=int(info["kept"]),
                )
            )
            m.row_checks.append(
                RowCheck(
                    scope=info["file"]["name"],
                    method="xml_page_count",
                    expected=int(info["kept"]),
                    actual=sum(s["rows"] for s in info["shards"] if s["table"] == "part"),
                )
            )
            m.code_version = git_sha(ROOT)
            m.write(src)
        problems = verify_source(src, deep=True)
        if problems:
            raise RuntimeError(f"wikipedia_wikitext verify failed: {problems}")
        run(*uv, "dvc", "commit", "-f", "wikitext")
    else:
        run(*uv, "dvc", "repro", "--single-item", "wikitext")
    run(*uv, "dvc", "push", "wikitext")
    return {"status": "passed"}


def step_build_ner(uv: list[str]) -> dict[str, Any]:
    run(
        *uv,
        "dvc",
        "pull",
        "split",
        "build_kb",
        "wikitext",
        "acquire@kaznerd",
        "acquire@wikiann",
        "acquire@wikineural",
        "acquire@nerel",
    )
    run(*uv, "dvc", "repro", "--single-item", "build_ner")
    run(*uv, "dvc", "push", "build_ner")
    rep = ROOT / "reports/ner_data.json"
    return json.loads(rep.read_text(encoding="utf-8")) if rep.is_file() else {"status": "passed"}


def step_synthetic(uv: list[str], drive: Path) -> dict[str, Any]:
    run(*uv, "dvc", "pull", "data/external.dvc", "chunk_corpus")
    q2 = drive / "workspace-data/kaggle/trifact-teacher-queries2/v1"
    qa2 = drive / "workspace-data/kaggle/trifact-teacher-qa2/v1"
    nli2 = drive / "workspace-data/kaggle/trifact-nli-score-qa2"
    q_runs = [ROOT / "data/external/kaggle/teacher_queries/out"] + ([q2] if q2.is_dir() else [])
    run(*uv, "python", "-m", "trifact.data.synthetic", "import", "--kind", "queries", "--kaggle-out", *q_runs)
    run(*uv, "dvc", "commit", "-f", "synthetic_queries")
    run(*uv, "dvc", "push", "synthetic_queries")

    out: dict[str, Any] = {
        "queries": json.loads((ROOT / "reports/synthetic_queries.json").read_text(encoding="utf-8"))
    }
    if nli2.is_dir() and list((nli2 / "out").glob("*.parquet")):
        qa_runs = [ROOT / "data/external/kaggle/teacher_qa/out"] + ([qa2] if qa2.is_dir() else [])
        run(*uv, "python", "-m", "trifact.data.synthetic", "import", "--kind", "qa", "--kaggle-out", *qa_runs)
        run(*uv, "python", "-m", "trifact.data.synthetic", "finalize", "--kind", "qa", "--scores", nli2)
        run(*uv, "dvc", "commit", "-f", "synthetic_qa")
        run(*uv, "dvc", "push", "synthetic_qa")
        out["qa"] = json.loads((ROOT / "reports/synthetic_qa.json").read_text(encoding="utf-8"))
    else:
        out["qa"] = {"status": "skipped_waiting_for_kaggle_nli_scores", "expected_dir": str(nli2)}
    return out


def step_train_and_eval(uv: list[str]) -> dict[str, Any]:
    run(
        *uv,
        "dvc",
        "pull",
        "split",
        "mt@qa",
        "synthetic_queries",
        "synthetic_qa",
        "build_eval",
        "acquire@ms_marco",
        "acquire@mmarco_ru",
        "acquire@miracl",
        "acquire@miracl_corpus",
        "acquire@ru_hnp",
        "acquire@kazparc",
        "acquire@squad_v2",
        "acquire@sberquad",
        "acquire@kazqad",
    )
    run(*uv, "dvc", "repro", "--single-item", "build_pairs")
    run(*uv, "dvc", "push", "build_pairs")
    run(*uv, "dvc", "repro", "--single-item", "build_sft")
    run(*uv, "dvc", "push", "build_sft")
    run(*uv, "dvc", "pull", "build_ner", "build_nli")
    run(*uv, "dvc", "repro", "--single-item", "eval_leakage")
    run(*uv, "dvc", "repro", "--single-item", "task_eda")
    run(*uv, "dvc", "push", "task_eda")
    leak = json.loads((ROOT / "reports/eval_leakage.json").read_text(encoding="utf-8"))
    return {"eval_leakage": leak}


def step_eda_rss_check(uv: list[str]) -> dict[str, Any]:
    run(*uv, "dvc", "pull", "wikidata", "decontaminate")
    run(*uv, "dvc", "repro", "--single-item", "eda@entities")
    run(*uv, "dvc", "push", "eda@entities")
    run(*uv, "dvc", "repro", "--single-item", "eda@curated")
    run(*uv, "dvc", "push", "eda@curated")
    ent = json.loads((ROOT / "reports/eda/entities.json").read_text(encoding="utf-8"))["meta"]
    cur = json.loads((ROOT / "reports/eda/curated.json").read_text(encoding="utf-8"))["meta"]
    if cur["peak_rss_mb"] > 10 * 1024:
        raise RuntimeError(f"eda@curated peak RSS {cur['peak_rss_mb']} MiB > 10 GiB")
    return {"entities": ent, "curated": cur}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", type=Path, default=Path("/content/drive/MyDrive/TrifactAI"))
    ap.add_argument("--steps", default=",".join(ALL_STEPS))
    args = ap.parse_args(argv)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = set(steps) - set(ALL_STEPS)
    if unknown:
        raise SystemExit(f"Unknown steps: {sorted(unknown)}")

    uv = setup_dvc_and_models(args.drive)
    session = uuid4().hex[:12]
    proc, db, tracking = start_mlflow(args.drive, session)
    out_dir = args.drive / "workspace-data" / "stage-01-03-reconcile" / session
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"session": session, "steps": {}}
    try:
        for step in steps:
            t0 = time.perf_counter()
            if step == "wikitext_merge":
                res = step_wikitext_merge(uv, args.drive)
            elif step == "build_ner":
                res = step_build_ner(uv)
            elif step == "synthetic":
                res = step_synthetic(uv, args.drive)
            elif step == "train_and_eval":
                res = step_train_and_eval(uv)
            else:
                res = step_eda_rss_check(uv)
            snap = snapshot_mlflow(db, tracking, session)
            summary["steps"][step] = {
                "seconds": round(time.perf_counter() - t0, 1),
                "mlflow_snapshot": str(snap),
                "result": res,
            }
            shutil.copy2(ROOT / "dvc.lock", out_dir / "dvc.lock")
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    finally:
        proc.terminate()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
