"""Fetch and verify Stage 5 Kaggle 2x T4 outputs (`nurs2005/trifact-stage5-final`) after `kaggle auth login`.

Downloads `/kaggle/working/ner_champion` and `stage5_final_summary.json`, verifies every SHA-256
in `ner_champion/model_meta.json`, copies artifacts to Google Drive persistent storage, and
prints the reconciled NER-0/1, NER-3 (3-seed), and NER-4 (3-seed + neural DATE) results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_champion_meta(champ_dir: Path) -> dict[str, str]:
    meta_path = champ_dir / "model_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing model_meta.json in {champ_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for name, expected in meta.get("files_sha256", {}).items():
        p = champ_dir / name
        if not p.is_file():
            mismatches.append(f"missing:{name}")
        elif sha256_file(p) != expected:
            mismatches.append(f"sha256_mismatch:{name}")
    if mismatches:
        raise ValueError(f"Champion checkpoint integrity check failed: {mismatches}")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", default="nurs2005/trifact-stage5-final")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "stage5" / "kaggle_final")
    ap.add_argument("--model-dest", type=Path, default=ROOT / "models" / "ner")
    ap.add_argument(
        "--drive-dest", type=Path, default=Path(r"G:\Мой диск\TrifactAI\workspace-data\stage-05-extraction")
    )
    args = ap.parse_args()

    st = subprocess.run(
        ["kaggle", "kernels", "status", args.slug], capture_output=True, text=True, check=True
    )
    print(st.stdout.strip())
    if "COMPLETE" not in st.stdout.upper():
        raise RuntimeError(f"Kernel {args.slug} is not COMPLETE yet.")

    args.out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "kernels", "output", args.slug, "-p", str(args.out)], check=True)

    champ_src = args.out / "ner_champion"
    meta = verify_champion_meta(champ_src)
    print(f"Verified champion checkpoint ({len(meta['files_sha256'])} files SHA-256 OK).")

    if args.model_dest.exists():
        shutil.rmtree(args.model_dest)
    shutil.copytree(champ_src, args.model_dest)
    print(f"Installed verified champion into {args.model_dest}")

    if args.drive_dest.parent.exists():
        args.drive_dest.mkdir(parents=True, exist_ok=True)
        drive_champ = args.drive_dest / "models" / "ner_champion"
        if drive_champ.exists():
            shutil.rmtree(drive_champ)
        shutil.copytree(champ_src, drive_champ)
        summary_src = args.out / "stage5_final_summary.json"
        if summary_src.is_file():
            shutil.copy2(summary_src, args.drive_dest / "stage5_final_summary.json")
        print(f"Archived verified champion and summary to {args.drive_dest}")

    summary_path = args.out / "stage5_final_summary.json"
    if summary_path.is_file():
        print(summary_path.read_text(encoding="utf-8")[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
