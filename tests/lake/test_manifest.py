from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from trifact.lake.layout import sha256_file
from trifact.lake.manifest import LicenseDecision, RowCheck, ShardEntry, SourceManifest, verify_source


def _source(tmp_path: Path, use: str = "allowed") -> Path:
    src = tmp_path / "raw" / "demo"
    (src / "kk").mkdir(parents=True)
    shards = []
    for i in range(2):
        p = src / "kk" / f"part-{i:05d}.parquet"
        pq.write_table(pa.table({"id": list(range(10 * i, 10 * i + 10))}), p)
        shards.append(
            ShardEntry(
                path=f"kk/{p.name}",
                lang="kk",
                table="part",
                rows=10,
                bytes=p.stat().st_size,
                sha256=sha256_file(p),
            )
        )
    SourceManifest(
        name="demo",
        role="corpus",
        origin="hf",
        hf_id="org/demo",
        revision="a" * 40,
        license=LicenseDecision(license="CC-BY-4.0", use=use, release=True, evidence="test"),
        code_version="test",
        shards=shards,
        row_checks=[RowCheck(scope="all", method="parquet_footer", expected=20, actual=20)],
    ).write(src)
    return src


def test_manifest_roundtrip_and_derived_totals(tmp_path: Path) -> None:
    src = _source(tmp_path)
    m = SourceManifest.read(src)
    assert m.rows == {"kk": 20}
    assert m.license.use == "allowed"
    text = (src / "source.yaml").read_text(encoding="utf-8")
    assert "row_checks_ok: true" in text and "sha256" in text
    assert verify_source(src) == []


def test_verify_detects_corruption_missing_and_unknown_shards(tmp_path: Path) -> None:
    src = _source(tmp_path)
    p0 = src / "kk" / "part-00000.parquet"
    data = bytearray(p0.read_bytes())
    data[10] ^= 0xFF  # same size, different content
    p0.write_bytes(bytes(data))
    (src / "kk" / "part-00001.parquet").rename(src / "kk" / "part-00009.parquet")
    (src / "kk" / "x.parquet.tmp").write_bytes(b"")
    problems = "\n".join(verify_source(src))
    assert "sha256 mismatch" in problems
    assert "missing on disk: kk/part-00001.parquet" in problems
    assert "not in manifest: kk/part-00009.parquet" in problems
    assert "leftover temp file" in problems


def test_verify_flags_failed_row_check_and_blocked_license(tmp_path: Path) -> None:
    src = _source(tmp_path, use="blocked")
    m = SourceManifest.read(src)
    m.row_checks = [RowCheck(scope="all", method="hf_dataset_info", expected=21, actual=20)]
    m.write(src)
    problems = "\n".join(verify_source(src, deep=False))
    assert "row check failed" in problems
    assert "blocked" in problems


def test_missing_manifest(tmp_path: Path) -> None:
    assert verify_source(tmp_path / "nothing") != []
