"""Restore and verify a model from its Google Drive parts manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory


def restore(manifest: Path, destination: Path) -> Path:
    model = json.loads(manifest.read_text(encoding="utf-8"))
    output = destination / Path(model["name"]).name
    destination.mkdir(parents=True, exist_ok=True)
    if output.exists():
        with output.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if output.stat().st_size != model["size"] or digest != model["sha256"]:
            raise ValueError(f"Existing model differs from the manifest: {output}")
        return output

    digest = hashlib.sha256()
    total = 0
    with TemporaryDirectory(dir=destination) as scratch:
        pending = Path(scratch) / output.name
        with pending.open("wb") as stream:
            for part in model["parts"]:
                name = Path(part["name"])
                if name.name != part["name"]:
                    raise ValueError(f"Expected a filename, received {name}")
                part_digest = hashlib.sha256()
                size = 0
                with (manifest.parent / name).open("rb") as source:
                    while block := source.read(4 * 2**20):
                        part_digest.update(block)
                        digest.update(block)
                        stream.write(block)
                        size += len(block)
                if size != part["size"] or part_digest.hexdigest() != part["sha256"]:
                    raise ValueError(f"Model part differs from manifest: {name}")
                total += size
        if total != model["size"] or digest.hexdigest() != model["sha256"]:
            raise ValueError(f"Restored model differs from manifest: {output}")
        pending.replace(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    print(restore(args.manifest, args.dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
