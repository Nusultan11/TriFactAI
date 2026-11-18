"""Lake maintenance: integrity check and volume report over source manifests.

python -m trifact.lake verify [--layer raw] [--quick]   re-hash every shard, check row checks; exit 1 on problems
    (layer raw: source manifests; other layers: every derived dataset.yaml, e.g. clean/<source>,
    curated/dedup, curated/corpus, plus its data card)
python -m trifact.lake report [--layer raw] [--out reports/raw.json]   volumes, row checks, throughput per source
python -m trifact.lake mlflow [--layer clean]   log every manifest of a layer as an MLflow run (backfill for
    stages that ran while the tracking server was down; the manifest stays the primary record)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from trifact.core.params import ROOT, load_params
from trifact.lake.manifest import DATASET_NAME, MANIFEST_NAME, SourceManifest, verify_dataset, verify_source


def _sources(root: Path, layer: str) -> list[Path]:
    return sorted(p.parent for p in (root / layer).glob(f"*/{MANIFEST_NAME}"))


def _datasets(root: Path, layer: str) -> list[Path]:
    """Derived datasets of a layer: directories holding dataset.yaml (at most two levels deep)."""
    base = root / layer
    return sorted({p.parent for pat in (f"*/{DATASET_NAME}", f"*/*/{DATASET_NAME}") for p in base.glob(pat)})


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


def _numbers(d: dict, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[key] = float(v)
        elif isinstance(v, dict) and len(out) < 400:
            out.update(_numbers(v, key + "."))
    return out


def backfill(root: Path, layer: str) -> int:
    from trifact.lake.manifest import DatasetManifest
    from trifact.lake.stats import log_mlflow

    if layer == "raw":
        for src in _sources(root, layer):
            m = SourceManifest.read(src)
            st = m.stats.model_dump() if m.stats else {}
            ok = log_mlflow("data", f"{m.name}-manifest", {"revision": m.revision, "code_version": m.code_version},
                            {"rows": float(sum(m.rows.values())), **{k: float(v) for k, v in st.items()}})  # fmt: skip
            print(m.name, "logged" if ok else "not logged")
        return 0
    for d in _datasets(root, layer):
        m = DatasetManifest.read(d)
        metrics = {"rows_out": float(sum(m.rows_of().values())), "rows_in": float(sum(m.rows_in.values())),
                   **{f"drop.{k}": float(sum(v.values())) for k, v in m.drops.items()}, **_numbers(m.metrics, "m.")}  # fmt: skip
        if m.stats:
            metrics.update({f"run.{k}": float(v) for k, v in m.stats.model_dump().items()})
        exp = {"clean": "curation", "curated": "curation", "task": "task", "eval": "eval"}.get(layer, layer)
        ok = log_mlflow(exp, m.name, {"fingerprint": m.fingerprint, "code_version": m.code_version, "stage": m.stage,
                                      "partial": m.partial}, dict(list(metrics.items())[:500]))  # fmt: skip
        print(m.name, "logged" if ok else "not logged")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["verify", "report", "mlflow"])
    ap.add_argument("--layer", default="raw")
    ap.add_argument("--root", type=Path)
    ap.add_argument("--quick", action="store_true", help="verify sizes and row counts, skip sha256")
    ap.add_argument("--out", type=Path, help="report: write JSON here")
    args = ap.parse_args(argv)
    root = args.root or ROOT / load_params()["lake"]["root"]
    if args.command == "mlflow":
        return backfill(root, args.layer)
    if args.command == "verify":
        problems = []
        if args.layer == "raw":
            checks = [(src.name, verify_source, src) for src in _sources(root, args.layer)]
        else:
            checks = [
                (d.relative_to(root).as_posix(), verify_dataset, d) for d in _datasets(root, args.layer)
            ]
        if not checks:
            print(f"no manifests under {root / args.layer}", file=sys.stderr)
            return 1
        for name, check, path in checks:
            found = check(path, deep=not args.quick)
            print(f"{name}: {'OK' if not found else f'{len(found)} problem(s)'}")
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
