import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trifact.batch.colab import file_hashes, sha256, valid_checkpoint, verify_rows


def test_checkpoint_requires_unchanged_outputs(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    part = out / "scores-g0.parquet"
    part.write_bytes(b"contents")
    summary = tmp_path / "summary.json"
    summary.write_text("{}", encoding="utf-8")
    identity = {"job": "nli_score"}
    receipt = {
        "status": "complete",
        "identity": identity,
        "outputs": file_hashes(out),
        "summary_sha256": sha256(summary),
    }
    (tmp_path / "run.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert valid_checkpoint(tmp_path, identity)
    assert not valid_checkpoint(tmp_path, {"job": "translate"})
    part.write_bytes(b"corrupt!")
    assert not valid_checkpoint(tmp_path, identity)


def test_shard_with_duplicate_output_ids_is_incomplete(tmp_path):
    inputs, out = tmp_path / "inputs", tmp_path / "out"
    inputs.mkdir()
    out.mkdir()
    pq.write_table(pa.table({"id": ["1", "2"]}), inputs / "pairs.parquet")
    part = out / "scores-g0.parquet"
    pq.write_table(pa.table({"id": ["1", "1"]}), part)
    with pytest.raises(ValueError, match="Output IDs do not match"):
        verify_rows("nli_score", inputs, out, {})
    pq.write_table(pa.table({"id": ["2", "1"]}), part)
    assert verify_rows("nli_score", inputs, out, {}) == 2


def test_translation_requires_back_translation_when_configured(tmp_path):
    inputs, out = tmp_path / "inputs", tmp_path / "out"
    inputs.mkdir()
    out.mkdir()
    pq.write_table(pa.table({"id": [1]}), inputs / "text.parquet")
    pq.write_table(pa.table({"id": [1], "back": [None]}), out / "text-kaz_Cyrl-g0-0000.parquet")
    config = {"back_model": "pinned-back-model", "tasks": [{"file": "text.parquet", "tgt": "kaz_Cyrl"}]}
    with pytest.raises(ValueError, match="Back-translation incomplete"):
        verify_rows("translate", inputs, out, config)
