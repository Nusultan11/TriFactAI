"""Turnkey cloud CPU runner for remaining Stage 1-3 DVC stages (ADR-0025).

Run inside a Colab CPU runtime with Google Drive mounted at `/content/drive` (or an authorized
local/SSH DVC remote):

    uv run --no-sync python scripts/cloud_run_stages_1_3.py --drive /content/drive/MyDrive/TrifactAI

Steps (selectable via --steps):
  1. wikitext_merge  - merge verified ruwiki part 4 into raw/wikipedia_wikitext and push to DVC
  2. build_ner       - rebuild gold + silver NER (>5M target) with ruwiki part 4 and push to DVC
  3. synthetic       - import combined teacher_queries + queries2 and finalize qa + qa2 with NLI scores
  4. train_and_eval  - rebuild build_pairs, build_sft, verify eval_leakage == 0, run task_eda
  5. eda_rss_check   - run full raw, clean, curated and entities EDA and validate peak RSS

All DVC outputs are pushed immediately to the configured remote (`verify=true`), MLflow SQLite is
atomically snapshotted after each step, and updated `dvc.lock` + `reports/` are persisted to
`<out_dir>` (default `<drive>/workspace-data/stage-01-03-reconcile/<session>`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
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
REQUIRED_LEAKAGE_SETS = (
    "ner:sentence",
    "qa:query",
    "qa:passage",
    "retrieval:query",
    "retrieval:passage",
    "nli:nli",
)


def run(*cmd: Any, check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    args = [str(c) for c in cmd]
    print("+", " ".join(args), flush=True)
    return subprocess.run(args, cwd=ROOT, check=check, text=True, env=env)


def push_verified(uv: list[str], stage: str) -> None:
    run(*uv, "dvc", "push", stage)
    status = subprocess.run(
        [*uv, "dvc", "status", "--cloud", "--json", stage],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    if json.loads(status.stdout):
        raise RuntimeError(f"DVC objects for {stage} are not synchronized: {status.stdout.strip()}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_eval_leakage(leak_path: Path) -> dict[str, Any]:
    """Fail explicitly if eval_leakage.json is missing, incomplete, or has unintended overlap."""
    if not leak_path.is_file():
        raise RuntimeError(f"Missing leakage report: {leak_path}")
    leak = json.loads(leak_path.read_text(encoding="utf-8"))
    sets = leak.get("sets")
    if not isinstance(sets, dict) or not sets:
        raise RuntimeError(f"Invalid leakage report (missing 'sets'): {leak_path}")
    missing = [k for k in REQUIRED_LEAKAGE_SETS if k not in sets]
    if missing:
        raise RuntimeError(f"Incomplete leakage report; missing sets: {missing}")
    bad = {
        k: int(v.get("overlap", -1))
        for k, v in sets.items()
        if k != "retrieval:passage" and int(v.get("overlap", -1)) != 0
    }
    if bad:
        raise RuntimeError(f"Nonzero unintended evaluation leakage detected: {bad}")
    return leak


def validate_nli_scores_dir(nli_dir: Path) -> list[Path]:
    """Fail explicitly if required Kaggle NLI QA score parquets are missing."""
    out_dir = nli_dir / "out"
    files = sorted(out_dir.glob("*.parquet")) if out_dir.is_dir() else []
    if not files:
        raise RuntimeError(f"Missing required Kaggle NLI QA score parquets in {out_dir}")
    return files


def validate_rss_ceiling(meta: dict[str, Any], stage_name: str, max_ram_gb: float = 10.0) -> None:
    peak = meta.get("peak_rss_mb")
    if type(peak) not in (int, float) or not math.isfinite(peak) or peak <= 0:
        raise RuntimeError(f"{stage_name}: invalid or missing peak_rss_mb ({peak!r})")
    if peak > max_ram_gb * 1024:
        raise RuntimeError(f"{stage_name}: peak RSS {peak} MiB exceeds {max_ram_gb} GiB ceiling")


def start_mlflow(tracking_dir: Path, session: str) -> tuple[subprocess.Popen[str], Path, Path]:
    tracking_dir.mkdir(parents=True, exist_ok=True)
    db = Path("/content" if Path("/content").is_dir() else "/tmp") / f"trifact-{session}.db"
    log_path = db.with_suffix(".log")
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
            (tracking_dir / "artifacts").as_uri(),
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
                return proc, db, tracking_dir
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


def setup_dvc_and_models(drive: Path, dvc_remote_url: str | None = None) -> list[str]:
    uv = [shutil.which("uv") or "uv", "run", "--no-sync"]
    if dvc_remote_url:
        run(*uv, "dvc", "remote", "add", "--local", "--force", "--default", "drive", dvc_remote_url)
        if dvc_remote_url.startswith(("http://", "https://")):
            run(*uv, "dvc", "remote", "modify", "--local", "drive", "method", "PUT")
            token = os.environ.get("TRIFACT_BRIDGE_TOKEN")
            if token:
                run(*uv, "dvc", "remote", "modify", "--local", "drive", "auth", "custom")
                run(*uv, "dvc", "remote", "modify", "--local", "drive", "custom_auth_header", "Authorization")
                from dvc.repo import Repo

                with Repo(str(ROOT)) as repo, repo.config.edit("local") as config:
                    config["remote"]["drive"]["password"] = f"Bearer {token}"
    else:
        dvc_dir = drive / "dvc"
        assert (dvc_dir / "files" / "md5").is_dir(), f"DVC cache missing at {dvc_dir}"
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
    if not res_file.is_file():
        raise RuntimeError(f"Missing verified ruwiki part 4 metadata: {res_file}")
    run(*uv, "dvc", "pull", "wikitext")

    from trifact.lake.manifest import InputFile, RowCheck, ShardEntry, SourceManifest, verify_source
    from trifact.lake.stats import git_sha, log_mlflow

    info = json.loads(res_file.read_text(encoding="utf-8"))
    if not info.get("shards"):
        raise RuntimeError(f"Invalid ruwiki part 4 result.json (no shards): {res_file}")
    src = ROOT / "data/raw/wikipedia_wikitext"
    m = SourceManifest.read(src)
    existing_paths = {s.path: s for s in m.shards}

    # Verify all incoming part 4 artifacts before mutating the canonical dataset
    parsed_shards = [ShardEntry.model_validate(s_dict) for s_dict in info["shards"]]
    from trifact.core.params import load_params

    spec = load_params()["sources"]["wikipedia_wikitext"]
    expected = next(f for f in spec["files"] if f["wiki"] == "ruwiki" and "articles4." in f["name"])
    if (
        m.revision != str(spec["revision"])
        or info["file"] != expected
        or info["url"] != f"{spec['base_url']}/ruwiki/{m.revision}/{expected['name']}"
        or len({s.path for s in parsed_shards}) != len(parsed_shards)
        or any(
            s.lang != "ru" or not re.fullmatch(rf"ru/{s.table}-004-\d+\.parquet", s.path)
            for s in parsed_shards
        )
        or {s.table for s in parsed_shards} != {"part", "redirects"}
        or sum(s.rows for s in parsed_shards if s.table == "part") != info["kept"]
        or sum(s.rows for s in parsed_shards if s.table == "redirects") != info["redirects"]
    ):
        raise RuntimeError(f"Unexpected ruwiki part 4 inventory: {res_file}")
    for se in parsed_shards:
        src_f = part4_dir / se.path
        if not src_f.is_file():
            raise RuntimeError(f"Missing ruwiki part 4 shard: {src_f}")
        if sha256_file(src_f) != se.sha256:
            raise RuntimeError(f"SHA-256 mismatch on staged {src_f}")

    added = 0
    for se in parsed_shards:
        if se.path in existing_paths:
            if se != existing_paths[se.path]:
                raise RuntimeError(f"Existing canonical shard differs from verified part 4: {se.path}")
            continue
        src_f = part4_dir / se.path
        dst_f = src / se.path
        dst_f.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_f, dst_f)
        if sha256_file(dst_f) != se.sha256:
            raise RuntimeError(f"SHA-256 mismatch after copy on {dst_f}")
        m.shards.append(se)
        added += 1
    if added:
        if all(inp.path != info["url"] for inp in m.inputs):
            m.inputs.append(
                InputFile(
                    path=info["url"],
                    size=int(info["input_size"]),
                    lang=info["file"]["lang"],
                    table="part",
                    rows=int(info["kept"]),
                )
            )
        if all(rc.scope != info["file"]["name"] for rc in m.row_checks):
            m.row_checks.append(
                RowCheck(
                    scope=info["file"]["name"],
                    method="xml_page_count",
                    expected=int(info["kept"]),
                    actual=sum(s.rows for s in parsed_shards if s.table == "part"),
                )
            )
        m.code_version = git_sha(ROOT)
        m.write(src)

    problems = verify_source(src, deep=True)
    if problems:
        raise RuntimeError(f"wikipedia_wikitext verify failed: {problems}")
    run(*uv, "dvc", "commit", "-f", "wikitext")
    push_verified(uv, "wikitext")
    logged = log_mlflow(
        "data",
        "wikipedia_wikitext",
        {"revision": m.revision, "code_version": m.code_version, "part4_added": added},
        {"rows": sum(s.rows for s in m.shards if s.table == "part"), "shards": len(m.shards)},
    )
    if not logged:
        raise RuntimeError("Failed to record the Wikipedia merge in MLflow")
    return {
        "status": "passed",
        "shards_added": added,
        "total_shards": len(m.shards),
        "rows_by_lang": m.rows,
    }


def step_build_ner(uv: list[str]) -> dict[str, Any]:
    from trifact.lake.manifest import verify_dataset

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
    problems = verify_dataset(ROOT / "data/task/ner", deep=True)
    if problems:
        raise RuntimeError(f"task/ner verify failed: {problems}")
    rep = ROOT / "reports/ner_data.json"
    if not rep.is_file():
        raise RuntimeError(f"Missing NER report after build_ner: {rep}")
    report = json.loads(rep.read_text(encoding="utf-8"))
    from trifact.lake.manifest import DatasetManifest

    manifest = DatasetManifest.read(ROOT / "data/task/ner")
    silver = manifest.rows_of("silver_train")
    if set(silver) != {"kk", "ru", "en"} or silver != report["silver"]["kept_by_lang"]:
        raise RuntimeError("Silver NER report does not match the output manifest")
    push_verified(uv, "build_ner")
    return report


def step_synthetic(uv: list[str]) -> dict[str, Any]:
    import polars as pl

    from trifact.lake.manifest import DatasetManifest, verify_dataset

    run(*uv, "dvc", "pull", "data/external.dvc", "data/stage3_teacher.dvc", "chunk_corpus")
    validate_nli_scores_dir(ROOT / "data/stage3_teacher/nli_score_qa_v4")
    run(*uv, "dvc", "repro", "--single-item", "synthetic_queries")
    q_problems = verify_dataset(ROOT / "data/task/synthetic/queries", deep=True)
    if q_problems:
        raise RuntimeError(f"task/synthetic/queries verify failed: {q_problems}")
    q_kept = pl.read_parquet(
        sorted(str(p) for p in (ROOT / "data/task/synthetic/queries").glob("*/kept-*.parquet"))
    )
    if q_kept.is_empty():
        raise RuntimeError("task/synthetic/queries kept table is empty")
    if q_kept.select(pl.concat_str("id", "q_lang", separator="|")).is_duplicated().any():
        raise RuntimeError("Duplicate (id, q_lang) found in task/synthetic/queries")
    push_verified(uv, "synthetic_queries")

    run(*uv, "dvc", "repro", "--single-item", "synthetic_qa")
    qa_dir = ROOT / "data/task/synthetic/qa"
    qa_problems = verify_dataset(qa_dir, deep=True)
    if qa_problems:
        raise RuntimeError(f"task/synthetic/qa verify failed: {qa_problems}")
    qa_manifest = DatasetManifest.read(qa_dir)
    if "NLI filter pending." in qa_manifest.description:
        raise RuntimeError("task/synthetic/qa manifest still indicates NLI filter pending")
    qa_kept = pl.read_parquet(sorted(str(p) for p in qa_dir.glob("*/kept-*.parquet")))
    if qa_kept.is_empty():
        raise RuntimeError("task/synthetic/qa kept table is empty")
    if qa_kept["id"].is_duplicated().any():
        raise RuntimeError("Duplicate id found in task/synthetic/qa")
    push_verified(uv, "synthetic_qa")

    return {
        "queries": json.loads((ROOT / "reports/synthetic_queries.json").read_text(encoding="utf-8")),
        "qa": json.loads((ROOT / "reports/synthetic_qa.json").read_text(encoding="utf-8")),
    }


def step_train_and_eval(uv: list[str]) -> dict[str, Any]:
    from trifact.lake.manifest import verify_dataset

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
    p_probs = verify_dataset(ROOT / "data/task/retrieval", deep=True)
    if p_probs:
        raise RuntimeError(f"task/retrieval verify failed: {p_probs}")

    run(*uv, "dvc", "repro", "--single-item", "build_sft")
    s_probs = verify_dataset(ROOT / "data/task/sft", deep=True)
    if s_probs:
        raise RuntimeError(f"task/sft verify failed: {s_probs}")

    run(*uv, "dvc", "pull", "build_ner", "build_nli")
    run(*uv, "dvc", "repro", "--single-item", "eval_leakage")
    leak = validate_eval_leakage(ROOT / "reports/eval_leakage.json")
    push_verified(uv, "build_pairs")
    push_verified(uv, "build_sft")

    run(*uv, "dvc", "repro", "--single-item", "task_eda")
    push_verified(uv, "task_eda")
    return {
        "eval_leakage": leak,
        "retrieval": json.loads((ROOT / "reports/retrieval_data.json").read_text(encoding="utf-8")),
        "sft": json.loads((ROOT / "reports/sft_data.json").read_text(encoding="utf-8")),
    }


def step_eda_rss_check(uv: list[str]) -> dict[str, Any]:
    from trifact.core.params import load_params

    params = load_params()
    max_ram_gb = float(params["lake"]["peak_ram_gb"])
    raw = ["wikidata", "wikitext"]
    raw.extend(
        f"acquire@{name}" for name in params["sources"] if name not in ("wikidata", "wikipedia_wikitext")
    )
    results = {}
    for layer, inputs in (
        ("entities", ["wikidata"]),
        ("raw", raw),
        ("clean", ["clean"]),
        ("curated", ["dedup", "decontaminate"]),
    ):
        run(*uv, "dvc", "pull", *inputs)
        stage = f"eda@{layer}"
        run(*uv, "dvc", "repro", "--single-item", "--force", stage)
        report = json.loads((ROOT / f"reports/eda/{layer}.json").read_text(encoding="utf-8"))
        meta = report["meta"]
        if meta["layer"] != layer or not report["profiles"]:
            raise RuntimeError(f"Incomplete EDA profiles for {layer}")
        validate_rss_ceiling(meta, stage, max_ram_gb)
        push_verified(uv, stage)
        results[layer] = meta
    run(*uv, "dvc", "pull", "task_eda")
    if not (ROOT / "reports/corpus_stats.json").is_file():
        run(*uv, "dvc", "repro", "--single-item", "--force", "corpus_stats")
    run(*uv, "dvc", "repro", "--single-item", "eda_report")
    push_verified(uv, "eda_report")
    return results


def sync_reports_to(out_dir: Path) -> None:
    rep_dst = out_dir / "reports"
    rep_dst.mkdir(parents=True, exist_ok=True)
    for rel in (
        "reports/ner_data.json",
        "reports/synthetic_queries.json",
        "reports/synthetic_qa.json",
        "reports/retrieval_data.json",
        "reports/sft_data.json",
        "reports/eval_leakage.json",
        "reports/corpus_stats.json",
        "reports/eda/raw.json",
        "reports/eda/clean.json",
        "reports/eda/curated.json",
        "reports/eda/entities.json",
        "reports/eda/task.json",
        "reports/eda/report.md",
        "reports/eda/findings.json",
    ):
        src = ROOT / rel
        if src.is_file():
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", type=Path, default=Path("/content/drive/MyDrive/TrifactAI"))
    ap.add_argument("--dvc-remote", default=None, help="Override DVC remote URL (default: <drive>/dvc)")
    ap.add_argument(
        "--out-dir", type=Path, default=None, help="Output directory for summary, dvc.lock, reports"
    )
    ap.add_argument("--steps", default=",".join(ALL_STEPS))
    args = ap.parse_args(argv)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    if not steps or len(steps) != len(set(steps)) or (set(steps) - set(ALL_STEPS)):
        raise SystemExit("Specify one or more distinct known steps")

    uv = setup_dvc_and_models(args.drive, args.dvc_remote)
    session = uuid4().hex[:12]
    out_dir = args.out_dir or (args.drive / "workspace-data" / "stage-01-03-reconcile" / session)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracking_dir = (
        (args.drive / "workspace-data" / "mlflow") if args.out_dir is None else (out_dir / "mlflow")
    )
    proc, db, tracking = start_mlflow(tracking_dir, session)
    summary: dict[str, Any] = {"session": session, "steps": {}}
    try:
        for step in steps:
            t0 = time.perf_counter()
            if step == "wikitext_merge":
                res = step_wikitext_merge(uv, args.drive)
            elif step == "build_ner":
                res = step_build_ner(uv)
            elif step == "synthetic":
                res = step_synthetic(uv)
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
            sync_reports_to(out_dir)
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    finally:
        proc.terminate()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
