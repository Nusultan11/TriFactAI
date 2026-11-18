"""Lake maintenance: integrity check and volume report over source manifests.

python -m trifact.lake verify [--layer raw] [--quick]   re-hash every shard, check row checks; exit 1 on problems
python -m trifact.lake report [--layer raw] [--out reports/raw.json]   volumes, row checks, throughput per source
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from trifact.core.params import ROOT, load_params
from trifact.lake.manifest import MANIFEST_NAME, SourceManifest, verify_source


def _sources(root: Path, layer: str) -> list[Path]:
    return sorted(p.parent for p in (root / layer).glob(f"*/{MANIFEST_NAME}"))


def report(root: Path, layer: str) -> dict[str, dict]:
    out = {}
    for src in _sources(root, layer):
        m = SourceManifest.read(src)
        out[m.name] = {
            "role": m.role,
            "revision": m.revision,
            "license": m.license.license,
            "use": m.license.use.value,
            "release": m.license.release,
            "partial": m.partial,
            "rows": m.rows,
            "bytes": m.bytes,
            "shards": len(m.shards),
            "row_checks": [f"{c.method}: {c.scope} {c.actual}/{c.expected}" for c in m.row_checks],
            "row_checks_ok": all(c.ok for c in m.row_checks),
            "stats": m.stats.model_dump() if m.stats else None,
        }
    return out


def markdown(rep: dict[str, dict]) -> str:
    lines = [
        "| Источник | Строк по языкам | Parquet, GB | Шардов | Проверка строк | Throughput, строк/с | MB/s | Пик RSS, MB | Лицензия |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in rep.items():
        rows = ", ".join(f"{k} {v:,}".replace(",", " ") for k, v in r["rows"].items())
        st = r["stats"] or {}
        methods = sorted({c.split(":")[0] for c in r["row_checks"]})
        lines.append(
            f"| {name} | {rows} | {sum(r['bytes'].values()) / 1e9:.2f} | {r['shards']} | "
            f"{'✓' if r['row_checks_ok'] else '✗'} {', '.join(methods)} | {st.get('rows_per_s', '-')} | "
            f"{st.get('mb_per_s', '-')} | {st.get('peak_rss_mb', '-')} | {r['license']} ({r['use']}) |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["verify", "report"])
    ap.add_argument("--layer", default="raw")
    ap.add_argument("--root", type=Path)
    ap.add_argument("--quick", action="store_true", help="verify sizes and row counts, skip sha256")
    ap.add_argument("--out", type=Path, help="report: write JSON here")
    args = ap.parse_args(argv)
    root = args.root or ROOT / load_params()["lake"]["root"]
    if args.command == "verify":
        problems = []
        for src in _sources(root, args.layer):
            found = verify_source(src, deep=not args.quick)
            print(f"{src.name}: {'OK' if not found else f'{len(found)} problem(s)'}")
            problems += found
        for p in problems:
            print("  " + p, file=sys.stderr)
        return 1 if problems else 0
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    rep = report(root, args.layer)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(markdown(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
