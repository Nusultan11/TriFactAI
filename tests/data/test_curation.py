from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from trifact.data.curation import StageDir, UnitWriter, fingerprint, iter_batches
from trifact.data.task_common import dataset_input_of
from trifact.lake.layout import LakeConfig
from trifact.lake.manifest import DatasetManifest, InheritedLicense, verify_dataset


def _manifest(sd: StageDir, shards: list[dict]) -> DatasetManifest:
    return DatasetManifest(
        name="clean/demo",
        layer="clean",
        stage="clean",
        source="demo",
        description="test",
        code_version="test",
        fingerprint=sd.fp,
        params={"a": 1},
        inputs=[],
        licenses=[InheritedLicense(source="demo", license="CC-BY-4.0", use="allowed", release=True)],
        rows_in={"kk": 3},
        drops={"too_short": {"kk": 1}},
        shards=shards,
    )


def test_fingerprint_changes_with_params_code_inputs(tmp_path: Path) -> None:
    code, inp = tmp_path / "code.py", tmp_path / "source.yaml"
    code.write_text("x = 1\n")
    inp.write_text("rows: 1\n")
    base = fingerprint("clean", {"a": 1}, [code], [inp])
    assert base == fingerprint("clean", {"a": 1}, [code], [inp])
    code.write_bytes(b"x = 1\r\n")  # CRLF checkout must not change the fingerprint
    assert base == fingerprint("clean", {"a": 1}, [code], [inp])
    assert base != fingerprint("clean", {"a": 2}, [code], [inp])
    assert base != fingerprint("dedup", {"a": 1}, [code], [inp])
    inp.write_text("rows: 2\n")
    assert base != fingerprint("clean", {"a": 1}, [code], [inp])


def test_dataset_input_accepts_relative_paths_from_dvc(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "data/clean/demo"
    out.mkdir(parents=True)
    stage = StageDir(out, "stage-fingerprint")
    _manifest(stage, []).write(out)
    monkeypatch.chdir(tmp_path)

    dep = dataset_input_of(tmp_path / "data", Path("data/clean/demo"))

    assert dep.path == "clean/demo/dataset.yaml"


def test_stage_dir_resume_and_finish(tmp_path: Path) -> None:
    out, cfg = tmp_path / "clean" / "demo", LakeConfig(root=tmp_path)
    sd = StageDir(out, "fp1")
    sd.prepare()
    w = UnitWriter(out, 0, "part", cfg)
    w.write("kk", pa.table({"text": ["a", "b"]}))
    sd.mark(0, {"shards": w.close()})
    # unit 1 crashed half-way: its file exists but no marker
    w1 = UnitWriter(out, 1, "part", cfg)
    w1.write("kk", pa.table({"text": ["c"]}))
    w1.close()
    assert set(StageDir(out, "fp1").done()) == {0}

    sd = StageDir(out, "fp1")
    sd.prepare()  # same fingerprint: keeps finished units
    assert set(sd.done()) == {0}
    sd.clear_unit(1)
    assert not list(out.rglob("part-00001-*"))
    assert list(out.rglob("part-00000-*"))

    shards = [s for r in sd.done().values() for s in r["shards"]]
    sd.finish(_manifest(sd, shards))
    assert not (out / "_work").exists()
    assert sd.complete()
    assert verify_dataset(out) == []
    assert not StageDir(out, "fp2").complete()

    StageDir(out, "fp2").prepare()  # new fingerprint wipes the old output
    assert not list(out.rglob("*.parquet"))


def test_verify_dataset_detects_damage(tmp_path: Path) -> None:
    out = tmp_path / "ds"
    sd = StageDir(out, "fp")
    sd.prepare()
    w = UnitWriter(out, 0, "part", LakeConfig(root=tmp_path))
    w.write("kk", pa.table({"text": ["a", "b", "c"]}))
    m = _manifest(sd, w.close())
    sd.finish(m)
    assert m.rows_of() == {"kk": 3}
    card = (out / "card.md").read_text(encoding="utf-8")
    assert "too_short" in card and "CC-BY-4.0" in card
    assert DatasetManifest.read(out).fingerprint == "fp"

    (out / "card.md").write_text("edited", encoding="utf-8")
    assert any("card.md" in p for p in verify_dataset(out))
    m.write(out)
    shard = next(out.rglob("part-*.parquet"))
    shard.write_bytes(shard.read_bytes()[:-10])
    assert verify_dataset(out)


def test_iter_batches_limit(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    p = tmp_path / "x.parquet"
    pq.write_table(pa.table({"i": list(range(1000))}), p, row_group_size=100)
    assert sum(b.num_rows for b in iter_batches(p, None, 250)) == 250
    assert sum(b.num_rows for b in iter_batches(p, ["i"], None)) == 1000


def test_lake_cli_verifies_derived_layers(tmp_path: Path) -> None:
    from trifact.lake.__main__ import main

    out = tmp_path / "curated" / "dedup"
    sd = StageDir(out, "fp")
    sd.prepare()
    w = UnitWriter(out, 0, "part", LakeConfig(root=tmp_path))
    w.write("kk", pa.table({"text": ["a"]}))
    sd.finish(_manifest(sd, w.close()))
    assert main(["verify", "--layer", "curated", "--root", str(tmp_path)]) == 0
    shard = next(out.rglob("part-*.parquet"))
    shard.write_bytes(shard.read_bytes()[:-10])
    assert main(["verify", "--layer", "curated", "--root", str(tmp_path)]) == 1
    assert main(["verify", "--layer", "clean", "--root", str(tmp_path)]) == 1  # nothing there
