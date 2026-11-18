from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trifact.data import acquire as acq
from trifact.lake.layout import LakeConfig, sha256_file
from trifact.lake.manifest import SourceManifest, verify_source

REV = "0123456789abcdef0123456789abcdef01234567"
LICENSE = {"license": "CC-BY-4.0", "use": "allowed", "release": True, "evidence": "test"}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A fake HF dataset repository with every supported format."""
    r = tmp_path / "repo"
    (r / "kk").mkdir(parents=True)
    pq.write_table(
        pa.table({"id": [str(i) for i in range(30)], "text": ["a"] * 30}),
        r / "kk" / "train-00000-of-00002.parquet",
        row_group_size=7,
    )
    pq.write_table(
        pa.table({"id": [str(i) for i in range(30, 45)], "text": ["b"] * 15}),
        r / "kk" / "train-00001-of-00002.parquet",
    )
    pq.write_table(
        pa.table({"id": ["t1", "t2"], "text": ["c", "d"]}), r / "kk" / "test-00000-of-00001.parquet"
    )
    (r / "pairs_valid_kk_en.csv").write_text(
        'id,text\n1,"line one\nline two"\n2,"he said ""hi"""\n3,\n', encoding="utf-8", newline="\n"
    )
    (r / "collection.tsv").write_text(
        '1\tтекст с "кавычкой"\n2\t second\tкусок\n', encoding="utf-8", newline="\n"
    )
    with gzip.open(r / "docs-0.jsonl.gz", "wt", encoding="utf-8") as f:
        for i in range(25):  # field `extra` appears late: the schema must be unified over the file
            row: dict[str, Any] = {"docid": f"d{i}", "text": "т" * i}
            if i > 21:
                row["extra"] = {"k": i}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    t = pa.table({"tokens": [["a", "b"], ["c"]], "tags": [[0, 1], [2]]})
    with pa.ipc.new_stream(r / "validation.arrow", t.schema) as w:
        w.write_table(t)
    return r


SPEC: dict[str, Any] = {
    "hf_id": "org/fake",
    "revision": REV,
    "role": "corpus",
    "license": LICENSE,
    "files": [
        {"glob": "kk/*.parquet", "lang": "kk", "config": "c1"},
        {"glob": "*.csv", "lang": "multi", "table": "pairs"},
        {"glob": "*.tsv", "lang": "ru", "table": "collection", "columns": ["pid", "text"], "split": "train"},
        {"glob": "*.jsonl.gz", "lang": "en", "table": "docs", "split": "train"},
        {"glob": "*.arrow", "lang": "kk", "table": "ner", "config": "c2"},
    ],
}


@pytest.fixture
def fake_hub(repo: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {"downloads": [], "fail": set()}

    def files(_repo: str, _rev: str) -> list[acq.RemoteFile]:
        return [
            acq.RemoteFile(p.relative_to(repo).as_posix(), p.stat().st_size, sha256_file(p))
            for p in repo.rglob("*")
            if p.is_file()
        ]

    def download(repo_id: str, filename: str, repo_type: str, revision: str, local_dir: Path) -> str:
        assert revision == REV
        calls["downloads"].append(filename)
        if filename in calls["fail"]:
            raise ConnectionError(f"network dropped while fetching {filename}")
        dst = Path(local_dir) / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(repo / filename, dst)
        return str(dst)

    import huggingface_hub

    monkeypatch.setattr(acq, "hf_files", files)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(
        acq, "readme_counts", lambda *a: {("c1", "train"): 45, ("c1", "test"): 2, ("c2", "validation"): 2}
    )
    monkeypatch.setattr(acq, "server_counts", lambda *a: {})
    return calls


def _cfg(tmp_path: Path) -> LakeConfig:
    return LakeConfig(root=tmp_path / "lake", shard_rows=20, row_group_rows=8)


def test_split_and_format_detection() -> None:
    assert acq.split_of("20231101.kk/train-00000-of-00021.parquet") == ("train", "train")
    assert acq.split_of("kazparc/08_kazparc_valid_kk_en.csv") == ("valid", "validation")
    assert acq.split_of("data/kaz_Cyrl/test/000_00000.parquet") == ("test", "test")
    assert acq.split_of("topics.miracl-v1.0-ru-test-a.tsv") == ("test-a", "test-a")
    assert acq.split_of("data/val_ru-00000-of-00001.parquet") == ("val", "validation")
    assert acq.split_of("russian_collection.tsv") == (None, None)
    assert acq.file_format("a/b.jsonl.gz") == "jsonl"
    with pytest.raises(ValueError):
        acq.file_format("x.zip")


def test_plan_uses_globs_meta_split_and_rejects_empty_globs() -> None:
    files = [
        acq.RemoteFile(p, 1, None) for p in ["data/train_ru-0.parquet", "data/val_ru-0.parquet", "README.md"]
    ]
    spec = {
        "files": [
            {"glob": "data/*_ru-*.parquet", "lang": "ru", "config": "default", "meta_split": "{split}_ru"}
        ]
    }
    planned = acq.plan(spec, files)
    assert [(p.split, p.meta_split) for p in planned] == [("train", "train_ru"), ("validation", "val_ru")]
    with pytest.raises(ValueError, match="matches no file"):
        acq.plan({"files": [{"glob": "*.csv", "lang": "kk"}]}, files)


def test_acquire_writes_layout_manifest_and_checks_rows(tmp_path: Path, fake_hub: dict[str, Any]) -> None:
    m, problems = acq.acquire("fake", SPEC, _cfg(tmp_path), tmp_path / "staging", workers=3)
    assert problems == []
    src = tmp_path / "lake" / "raw" / "fake"
    assert m.rows == {"en": 25, "kk": 49, "multi": 3, "ru": 2}
    assert sorted(p.relative_to(src).as_posix() for p in src.rglob("*.parquet")) == [
        "en/docs-00000.parquet",
        "en/docs-00001.parquet",
        "kk/ner-00000.parquet",
        "kk/part-00000.parquet",
        "kk/part-00001.parquet",
        "kk/part-00002.parquet",
        "kk/part-00003.parquet",
        "multi/pairs-00000.parquet",
        "ru/collection-00000.parquet",
    ]
    assert not (src / "_work").exists()
    assert not any((tmp_path / "staging").rglob("*.parquet"))  # staged downloads are removed
    checks = {(c.method, c.scope.split(" files")[0]): (c.expected, c.actual) for c in m.row_checks}
    assert checks[("hf_dataset_info", "config=c1 split=train")] == (45, 45)
    assert checks[("parquet_footer", "table=part lang=kk split=train")] == (45, 45)
    assert checks[("csv_record_count", "table=pairs lang=multi split=validation")] == (3, 3)
    assert checks[("line_count", "table=collection lang=ru split=train")] == (2, 2)
    assert all(c.ok for c in m.row_checks)

    kk = pa.concat_tables(pq.read_table(p) for p in sorted((src / "kk").glob("part-*.parquet"))).to_pydict()
    assert set(zip(kk["_config"], kk["_split"], strict=True)) == {("c1", "test"), ("c1", "train")}
    assert kk["id"][:2] == ["t1", "t2"] and kk["id"][2:] == [str(i) for i in range(45)]  # file order kept
    csv_rows = pq.read_table(src / "multi" / "pairs-00000.parquet").to_pydict()
    assert csv_rows["text"] == ["line one\nline two", 'he said "hi"', ""]
    tsv = pq.read_table(src / "ru" / "collection-00000.parquet").to_pydict()
    assert tsv["text"] == ['текст с "кавычкой"', " second\tкусок"]
    docs = pa.concat_tables(pq.read_table(p) for p in sorted((src / "en").glob("*.parquet")))
    assert docs["extra"].to_pylist()[-1] == {"k": 24} and docs["extra"].to_pylist()[0] is None

    saved = SourceManifest.read(src)
    assert saved.revision == REV and saved.license.use == "allowed" and not saved.partial
    assert {i.path for i in saved.inputs} >= {"collection.tsv", "validation.arrow"}
    assert verify_source(src) == []


def test_interrupted_run_resumes_without_redoing_finished_files(
    tmp_path: Path, fake_hub: dict[str, Any]
) -> None:
    fake_hub["fail"].add("docs-0.jsonl.gz")
    with pytest.raises(ConnectionError):
        acq.acquire("fake", SPEC, _cfg(tmp_path), tmp_path / "staging", workers=1)
    work = tmp_path / "lake" / "raw" / "fake" / "_work"
    finished = {json.loads(p.read_text())["idx"] for p in work.glob("0*.json")}
    assert {0, 1, 2, 3, 4} <= finished and 5 not in finished  # files before the failing one are kept

    fake_hub["fail"].clear()
    fake_hub["downloads"].clear()
    m, problems = acq.acquire("fake", SPEC, _cfg(tmp_path), tmp_path / "staging", workers=1)
    assert problems == []
    redone = ["docs-0.jsonl.gz"] + ([] if 6 in finished else ["validation.arrow"])
    assert fake_hub["downloads"] == redone
    assert m.rows == {"en": 25, "kk": 49, "multi": 3, "ru": 2}

    fake_hub["downloads"].clear()  # a third run finds a verified manifest and does nothing
    acq.acquire("fake", SPEC, _cfg(tmp_path), tmp_path / "staging", workers=1)
    assert fake_hub["downloads"] == []

    revised = {**SPEC, "license": {**LICENSE, "use": "research_only", "release": False}}
    m, _ = acq.acquire("fake", revised, _cfg(tmp_path), tmp_path / "staging", workers=1)
    assert fake_hub["downloads"] == [] and m.license.use == "research_only"
    assert SourceManifest.read(tmp_path / "lake" / "raw" / "fake").license.release is False


def test_metadata_mismatch_is_reported(
    tmp_path: Path, fake_hub: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(acq, "readme_counts", lambda *a: {("c1", "train"): 46})
    _, problems = acq.acquire("fake", SPEC, _cfg(tmp_path), tmp_path / "staging", workers=2)
    assert any("row check failed config=c1 split=train: 45 != 46" in p for p in problems)


def test_corrupted_download_is_rejected(
    tmp_path: Path, fake_hub: dict[str, Any], repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = acq.hf_files

    def tampered(*a: Any) -> list[acq.RemoteFile]:
        return [
            acq.RemoteFile(f.path, f.size, "0" * 64 if f.path == "collection.tsv" else f.sha256)
            for f in good(*a)
        ]

    monkeypatch.setattr(acq, "hf_files", tampered)
    with pytest.raises(RuntimeError, match="sha256 differs"):
        acq.acquire("fake", SPEC, _cfg(tmp_path), tmp_path / "staging", workers=1)


def test_blocked_license_is_never_downloaded(tmp_path: Path, fake_hub: dict[str, Any]) -> None:
    spec = {**SPEC, "license": {**LICENSE, "use": "blocked"}}
    with pytest.raises(PermissionError):
        acq.acquire("fake", spec, _cfg(tmp_path), tmp_path / "staging", workers=1)
    assert fake_hub["downloads"] == []


def test_branch_names_are_not_accepted_as_revision(tmp_path: Path, fake_hub: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="commit SHA"):
        acq.acquire("fake", {**SPEC, "revision": "main"}, _cfg(tmp_path), tmp_path / "staging", workers=1)


def test_smoke_limit_marks_manifest_partial(tmp_path: Path, fake_hub: dict[str, Any]) -> None:
    m, problems = acq.acquire("fake", SPEC, _cfg(tmp_path), tmp_path / "staging", workers=2, limit=5)
    assert problems == []
    assert m.partial and m.rows["kk"] == 5 + 5 + 2 + 2
    assert all(c.ok for c in m.row_checks)
