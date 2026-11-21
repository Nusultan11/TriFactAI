"""Mirror the shared Agent Skills from `.agents/skills` (Codex, source of truth) to `.claude/skills` (Claude Code).

python scripts/sync_skills.py           # copy changed files, remove stale ones
python scripts/sync_skills.py --check   # exit 1 if the mirror differs (pre-commit, tests/test_skills.py)

Only skill directories named `trifact-*` are managed, so Claude-only skills in `.claude/skills` stay untouched.
Symlinks are not used: the repository has `core.symlinks = false` on Windows, where a link becomes a text file.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".agents" / "skills"
MIRROR = ROOT / ".claude" / "skills"
PREFIX = "trifact-"


def _files(base: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for skill in sorted(base.glob(f"{PREFIX}*")):
        if skill.is_dir():
            for f in skill.rglob("*"):
                if f.is_file():
                    out[f.relative_to(base).as_posix()] = f.read_bytes().replace(b"\r\n", b"\n")
    return out


def diff(source: Path = SOURCE, mirror: Path = MIRROR) -> list[str]:
    """Relative paths that are missing, stale or different in the mirror."""
    src, dst = _files(source), _files(mirror)
    return sorted({p for p in src.keys() | dst.keys() if src.get(p) != dst.get(p)})


def sync(source: Path = SOURCE, mirror: Path = MIRROR) -> list[str]:
    changed = diff(source, mirror)
    for rel in changed:
        src, dst = source / rel, mirror / rel
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes().replace(b"\r\n", b"\n"))
        else:
            dst.unlink()
    for skill in mirror.glob(f"{PREFIX}*"):
        if skill.is_dir() and not (source / skill.name).is_dir():
            shutil.rmtree(skill)
    return changed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report differences without writing")
    args = ap.parse_args(argv)
    if args.check:
        changed = diff()
        for rel in changed:
            print(f"out of sync: {rel}", file=sys.stderr)
        if changed:
            print("run: python scripts/sync_skills.py", file=sys.stderr)
        return 1 if changed else 0
    for rel in sync():
        print(f"synced: {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
