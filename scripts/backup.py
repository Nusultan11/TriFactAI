"""Snapshot the project: full git history + working state that git does not hold.

Creates in the backup directory:
  <timestamp>_<tag>.bundle    git bundle with every ref (restore: git clone <bundle> TrifactAI)
  <timestamp>_<tag>.zip       uncommitted diff, untracked files, local state (.env, .shrimp/, .agents/settings.local.json)
Data and model weights are backed up with `dvc push`, not here.

Usage: python scripts/backup.py --tag stage-01 [--dest DIR] [--keep 30] [--git-only]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT.parent / f"{ROOT.name}_backups"
LOCAL_STATE = [".env", ".shrimp", ".agents/settings.local.json"]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout


def add_path(zf: zipfile.ZipFile, path: Path) -> int:
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    for f in files:
        zf.write(f, f.relative_to(ROOT).as_posix())
    return len(files)


def prune(dest: Path, keep: int) -> list[str]:
    stamps = sorted(
        {p.name.split("_", 2)[0] + "_" + p.name.split("_", 2)[1] for p in dest.glob("*_*_*.bundle")}
    )
    removed = []
    for stamp in stamps[:-keep] if keep > 0 else []:
        for p in dest.glob(f"{stamp}_*"):
            p.unlink()
            removed.append(p.name)
    return removed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="label, e.g. stage-01 or before-refactor")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--keep", type=int, default=30, help="number of snapshots to keep")
    ap.add_argument("--git-only", action="store_true", help="exclude working state and local credentials")
    args = ap.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.tag):
        sys.exit("tag may contain only letters, digits, '.', '_' and '-'")
    args.dest.mkdir(parents=True, exist_ok=True)
    base = args.dest / f"{datetime.now():%Y%m%d_%H%M%S}_{args.tag}"

    bundle = base.parent / f"{base.name}.bundle"
    pending = bundle.with_suffix(".bundle.tmp")
    if bundle.exists() or pending.exists():
        sys.exit(f"snapshot already exists: {bundle}")
    git("bundle", "create", str(pending), "--all")
    git("bundle", "verify", str(pending))
    pending.replace(bundle)
    if args.git_only:
        print(f"bundle: {bundle}")
        return 0

    state = base.parent / f"{base.name}.zip"
    with zipfile.ZipFile(state, "w", zipfile.ZIP_DEFLATED) as zf:
        diff = git("diff", "HEAD", "--binary")
        if diff:
            zf.writestr("uncommitted.patch", diff)
        untracked = [p for p in git("ls-files", "--others", "--exclude-standard").splitlines() if p]
        n_files = sum(add_path(zf, ROOT / p) for p in untracked)
        n_files += sum(add_path(zf, ROOT / p) for p in LOCAL_STATE if (ROOT / p).exists())
        zf.writestr(
            "SNAPSHOT.txt",
            f"tag: {args.tag}\nhead: {git('rev-parse', 'HEAD').strip()}\n"
            f"branch: {git('branch', '--show-current').strip()}\n"
            f"uncommitted_patch: {bool(diff)}\nfiles: {n_files}\n",
        )

    removed = prune(args.dest, args.keep)
    print(f"bundle: {bundle}\nstate:  {state}")
    if diff:
        print("warning: uncommitted changes were saved as uncommitted.patch")
    if removed:
        print(f"pruned: {len(removed)} old files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
