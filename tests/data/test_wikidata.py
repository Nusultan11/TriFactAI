from __future__ import annotations

import bz2
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trifact.data import wikidata as wd
from trifact.lake.layout import LakeConfig


def entity(i: int) -> dict[str, Any]:
    e: dict[str, Any] = {
        "type": "item",
        "id": f"Q{i}",
        "lastrevid": 1000 + i,
        "labels": {
            "en": {"language": "en", "value": f"Item {i}"},
            "kk-cyrl": {"language": "kk-cyrl", "value": f"Нысан {i}"},
        },
        "descriptions": {"ru": {"language": "ru", "value": "описание"}},
        "aliases": {
            "kk": [{"language": "kk", "value": "a1"}],
            "kk-kz": [{"language": "kk-kz", "value": "a1"}, {"language": "kk-kz", "value": "a2"}],
        },
        "claims": {
            "P31": [
                {"rank": "normal", "mainsnak": {"snaktype": "value", "datavalue": {"value": {"id": "Q5"}}}},
                {
                    "rank": "deprecated",
                    "mainsnak": {"snaktype": "value", "datavalue": {"value": {"id": "Q1"}}},
                },
                {"rank": "normal", "mainsnak": {"snaktype": "novalue"}},
            ]
        },
        "sitelinks": {
            "enwiki": {"site": "enwiki", "title": f"Item {i}"},
            "dewiki": {"site": "dewiki", "title": "x"},
        },
    }
    if i % 3 == 0:
        e["labels"]["mul"] = {"language": "mul", "value": f"Mul {i}"}
    return e


def make_dump(path: Path, n: int) -> Path:
    lines = (
        ["["]
        + [json.dumps(entity(i), ensure_ascii=False) + ("," if i < n else "") for i in range(1, n + 1)]
        + ["]"]
    )
    path.write_bytes(bz2.compress(("\n".join(lines) + "\n").encode("utf-8")))
    return path


def test_entity_row_folds_language_variants_and_skips_deprecated_claims() -> None:
    row = wd.entity_row(entity(3))
    assert row["qid"] == "Q3" and row["p31"] == ["Q5"] and row["p279"] == []
    assert row["label_kk"] == "Нысан 3" and row["label_en"] == "Item 3" and row["label_ru"] is None
    assert row["label_mul"] == "Mul 3" and row["description_ru"] == "описание"
    assert row["aliases_kk"] == ["a1", "a2"]
    assert row["sitelink_enwiki"] == "Item 3" and row["sitelink_kkwiki"] is None and row["n_sitelinks"] == 2
    pa.RecordBatch.from_pylist([row], schema=wd.SCHEMA)  # fits the table schema


def test_count_entities_ignores_array_brackets() -> None:
    assert wd.count_entities(b'[\n{"id":1},\n{"id":2}\n]\n') == 2
    assert wd.count_entities(b'{"id":1},\n') == 1


def _parse(tmp_path: Path, dump: Path, **kw: Any) -> tuple[list[Any], int]:
    cfg = LakeConfig(root=tmp_path / "lake", shard_rows=40, row_group_rows=16)
    return wd.parse_dump(dump, tmp_path / "lake" / "raw" / "wikidata", cfg, workers=2, **kw)


def test_parse_dump_shards_by_entity_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wd, "READ_BYTES", 4096)  # many small chunks, some straddling shard borders
    dump = make_dump(tmp_path / "dump.json.bz2", 101)
    shards, counted = _parse(tmp_path, dump)
    assert counted == 101
    assert [s.rows for s in shards] == [40, 40, 21]
    assert [s.path for s in shards] == [f"multi/entities-{k:05d}.parquet" for k in range(3)]
    qids = pa.concat_tables(pq.read_table(tmp_path / "lake/raw/wikidata" / s.path) for s in shards)["qid"]
    assert qids.to_pylist() == [f"Q{i}" for i in range(1, 102)]


def test_parse_resumes_from_finished_shards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wd, "READ_BYTES", 4096)
    dump = make_dump(tmp_path / "dump.json.bz2", 101)
    calls = {"n": 0}
    real = wd.ShardWriter.close

    def crash_on_second_shard(self: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt  # simulated kill while finishing shard 1
        return real(self)

    monkeypatch.setattr(wd.ShardWriter, "close", crash_on_second_shard)
    with pytest.raises(KeyboardInterrupt):
        _parse(tmp_path, dump)
    progress = tmp_path / "lake/raw/wikidata/_progress"
    assert [p.name for p in progress.glob("*.json")] == ["entities-00000.json"]
    first = (tmp_path / "lake/raw/wikidata/multi/entities-00000.parquet").stat().st_mtime_ns

    monkeypatch.setattr(wd.ShardWriter, "close", real)
    shards, counted = _parse(tmp_path, dump)
    assert counted == 101 and [s.rows for s in shards] == [40, 40, 21]
    assert (
        tmp_path / "lake/raw/wikidata/multi/entities-00000.parquet"
    ).stat().st_mtime_ns == first  # not rewritten
    again, _ = _parse(tmp_path, dump)  # everything finished: markers only, no parsing
    assert again == shards


def test_limit_stops_early(tmp_path: Path) -> None:
    shards, counted = _parse(tmp_path, make_dump(tmp_path / "dump.json.bz2", 101), limit=55)
    assert counted == 55 and sum(s.rows for s in shards) == 55


def _server(
    data: bytes, fail_ranges: set[int] | None = None, sha1: str | None = None
) -> tuple[httpx.MockTransport, list[str]]:
    seen: list[str] = []
    fail = fail_ranges if fail_ranges is not None else set()

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("sha1sums.txt"):
            return httpx.Response(200, text=f"{sha1 or hashlib.sha1(data).hexdigest()}  dump.json.bz2\n")
        if req.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(data))})
        lo, hi = map(int, req.headers["range"].removeprefix("bytes=").split("-"))
        seen.append(req.headers["range"])
        if lo in fail:
            return httpx.Response(429)
        return httpx.Response(206, content=data[lo : hi + 1])

    return httpx.MockTransport(handler), seen


def test_download_resumes_and_verifies_sha1(tmp_path: Path) -> None:
    data = bytes(range(256)) * 40  # 10240 bytes, 10 chunks of 1024
    url, sums = "https://dumps.test/dump.json.bz2", "https://dumps.test/sha1sums.txt"
    transport, seen = _server(data, fail_ranges={5 * 1024})
    with pytest.raises(RuntimeError, match="chunk 5 failed"):
        wd.download(
            url,
            tmp_path / "dump.json.bz2",
            1,
            checksum_url=sums,
            retries=2,
            chunk_bytes=1024,
            backoff=0,
            transport=transport,
        )
    state = json.loads((tmp_path / "dump.json.bz2.state.json").read_text())
    assert 5 not in state["done"] and {0, 1, 2, 3, 4} <= set(state["done"])

    transport, seen = _server(data)
    wd.download(
        url,
        tmp_path / "dump.json.bz2",
        3,
        checksum_url=sums,
        chunk_bytes=1024,
        backoff=0,
        transport=transport,
    )
    assert f"bytes={0}-{1023}" not in seen and "bytes=5120-6143" in seen
    assert (tmp_path / "dump.json.bz2").read_bytes() == data
    assert (tmp_path / "dump.json.bz2.verified").exists()


def test_download_rejects_wrong_checksum(tmp_path: Path) -> None:
    transport, _ = _server(b"x" * 3000, sha1="0" * 40)
    with pytest.raises(RuntimeError, match="sha1 mismatch"):
        wd.download(
            "https://dumps.test/dump.json.bz2",
            tmp_path / "dump.json.bz2",
            2,
            checksum_url="https://dumps.test/sha1sums.txt",
            chunk_bytes=1024,
            backoff=0,
            transport=transport,
        )
