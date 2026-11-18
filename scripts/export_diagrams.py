"""Inspect the C4 model and regenerate docs/architecture/diagrams/*.mmd from workspace.dsl.

Diagrams are never edited by hand: change workspace.dsl, then run this script.
Requires Docker (image structurizr/structurizr). Fails if the inspection reports any error or warning.

Usage: python scripts/export_diagrams.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / "docs" / "architecture"
DIAGRAMS = ARCH / "diagrams"
IMAGE = "structurizr/structurizr"
MOUNT = "/usr/local/structurizr"


def structurizr(*args: str, extra_mount: Path | None = None) -> str:
    cmd = ["docker", "run", "--rm", "-v", f"{ARCH.as_posix()}:{MOUNT}"]
    if extra_mount:
        cmd += ["-v", f"{extra_mount.as_posix()}:/out"]
    cmd += [IMAGE, *args]
    env = {**os.environ, "MSYS_NO_PATHCONV": "1"}  # keep container paths intact under Git Bash
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
    out = res.stdout + res.stderr
    if res.returncode != 0:
        sys.exit(f"structurizr {args[0]} failed:\n{out}")
    return out


def main() -> int:
    findings = [
        line
        for line in structurizr("inspect", "-w", f"{MOUNT}/workspace.dsl").splitlines()
        if line.startswith(("ERROR", "WARNING"))
    ]
    if findings:
        sys.exit("inspection findings:\n" + "\n".join(findings))

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        structurizr("export", "-w", f"{MOUNT}/workspace.dsl", "-f", "mermaid", "-o", "/out", extra_mount=out)
        exported = sorted(out.glob("structurizr-*.mmd"))
        if not exported:
            sys.exit("export produced no diagrams")
        for old in DIAGRAMS.glob("*.mmd"):
            old.unlink()
        DIAGRAMS.mkdir(exist_ok=True)
        for f in exported:
            shutil.copyfile(f, DIAGRAMS / f.name.removeprefix("structurizr-"))
    print(f"inspection clean; exported {len(exported)} diagrams to {DIAGRAMS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
