from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from trifact.batch.annotate_corpus import (
    discover_chunk_shards,
    parse_shard_spec,
    run_annotate_corpus,
    validate_mention_offsets,
)
from trifact.core.schemas import Chunk, EntityMention, EntityType
from trifact.extraction.dates import RuleBasedDateExtractor
from trifact.extraction.linking import InMemoryKBIndex, KBEntity, WikidataEntityLinker
from trifact.extraction.ner import (
    CORE_ID2LABEL,
    CORE_LABEL2ID,
    CombinedExtractor,
    align_word_labels_to_subwords,
    build_vocab_remap,
    clean_silver_tags,
    decode_word_logits_to_bio,
    evaluate_entity_calibration,
    fit_temperature_scaling,
    merge_window_word_logits,
)


def test_clean_silver_tags_and_subword_alignment() -> None:
    tokens = ["Қазақстан", "астанасы", "Астана", ".", "Алматы"]
    tags = ["B-LOC", "O", "O", "B-LOC", "B-LOC"]
    cleaned = clean_silver_tags(tokens, tags, mask_unlabeled_capitals=True)
    # Token 0 kept, token 2 ('Астана', unlabeled capitalized non-initial) masked to None -> -100,
    # token 3 ('.', punctuation-only silver span) reset to 'O'
    assert cleaned == ["B-LOC", "O", None, "O", "B-LOC"]

    word_ids = [None, 0, 0, 1, 2, 3, 4, 4, None]
    aligned = align_word_labels_to_subwords(word_ids, cleaned, CORE_LABEL2ID)
    assert aligned == [
        -100,
        CORE_LABEL2ID["B-LOC"],
        -100,
        CORE_LABEL2ID["O"],
        -100,
        CORE_LABEL2ID["O"],
        CORE_LABEL2ID["B-LOC"],
        -100,
        -100,
    ]


def test_sliding_window_merge_and_calibration() -> None:
    # Two overlapping windows over 4 words
    w1_ids = [None, 0, 1, 2, None]
    w2_ids = [None, 1, 2, 3, None]
    n_labels = len(CORE_ID2LABEL)
    lg1 = np.zeros((5, n_labels), dtype=np.float32)
    lg2 = np.zeros((5, n_labels), dtype=np.float32)
    lg1[1, CORE_LABEL2ID["B-PER"]] = 5.0
    lg1[2, CORE_LABEL2ID["I-PER"]] = 5.0
    lg2[2, CORE_LABEL2ID["O"]] = 5.0
    lg2[3, CORE_LABEL2ID["B-LOC"]] = 5.0
    merged = merge_window_word_logits(4, [w1_ids, w2_ids], [lg1, lg2])
    tags, spans = decode_word_logits_to_bio(merged, CORE_ID2LABEL, temperature=1.0)
    assert tags == ["B-PER", "I-PER", "O", "B-LOC"]
    assert len(spans) == 2

    # Temperature scaling lowers NLL on overconfident logits
    rng = np.random.default_rng(42)
    overconf = rng.normal(size=(200, n_labels)).astype(np.float32) * 5.0
    gold_ids = [0] * 200
    for i in range(200):
        gold_ids[i] = (
            int(np.argmax(overconf[i])) if i % 5 != 0 else (int(np.argmax(overconf[i])) + 1) % n_labels
        )
    fit = fit_temperature_scaling([overconf], [gold_ids])
    assert fit["temperature"] > 1.0
    assert fit["nll_after"] < fit["nll_before"]

    cal = evaluate_entity_calibration([merged], [tags], CORE_ID2LABEL, temperature=1.0)
    assert cal["calibration"]["n"] == 2


def test_vocab_remap_preserves_special_tokens() -> None:
    class _DummyTok:
        all_special_ids = (0, 1, 2, 3)

    kept, remap = build_vocab_remap(_DummyTok(), Counter({50: 100, 12: 50, 99: 10}), max_vocab_size=6)
    assert kept[:4] == [0, 1, 2, 3]
    assert 50 in remap and 12 in remap and 99 not in remap


class _StubNERExtractor:
    version = "stub-ner-v1"

    def extract(self, chunks: Sequence[Chunk]) -> list[EntityMention]:
        out: list[EntityMention] = []
        for ch in chunks:
            idx = ch.text.find("Астанада")
            if idx >= 0:
                out.append(
                    EntityMention(
                        chunk_id=ch.chunk_id,
                        surface="Астанада",
                        type=EntityType.LOC,
                        char_start=idx,
                        char_end=idx + len("Астанада"),
                        score=0.92,
                        model_version=self.version,
                    )
                )
        return out


def test_annotate_corpus_sharded_and_idempotent(tmp_path: Path) -> None:
    in_dir = tmp_path / "chunks" / "wikipedia" / "kk"
    in_dir.mkdir(parents=True)
    schema = pa.schema(
        [
            ("chunk_id", pa.string()),
            ("doc_id", pa.string()),
            ("source", pa.string()),
            ("lang", pa.string()),
            ("ordinal", pa.int32()),
            ("char_start", pa.int32()),
            ("char_end", pa.int32()),
            ("text", pa.string()),
        ]
    )
    for idx in range(2):
        pq.write_table(
            pa.table(
                {
                    "chunk_id": [f"doc{idx}#0"],
                    "doc_id": [f"doc{idx}"],
                    "source": ["wikipedia"],
                    "lang": ["kk"],
                    "ordinal": [0],
                    "char_start": [0],
                    "char_end": [40],
                    "text": ["Астанада 2024 жылы үлкен конференция өтті."],
                },
                schema=schema,
            ),
            in_dir / f"part-0000{idx}.parquet",
        )

    assert parse_shard_spec("0/2") == (0, 2)
    assert len(discover_chunk_shards(tmp_path / "chunks", "0/2")) == 1
    assert len(discover_chunk_shards(tmp_path / "chunks", "1/2")) == 1

    kb = InMemoryKBIndex()
    kb.add_entity(KBEntity(qid="Q1520", ner_type="LOC", n_sitelinks=180, label_kk="Астана"))
    linker = WikidataEntityLinker(kb)
    extractor = CombinedExtractor(_StubNERExtractor(), RuleBasedDateExtractor(include_relative=True))

    out_dir = tmp_path / "annotations"
    s1 = run_annotate_corpus(tmp_path / "chunks", out_dir, extractor, linker, shard_spec="0/2")
    assert s1["shards_processed"] == 1
    assert s1["shards_skipped"] == 0
    assert s1["total_chunks"] == 1
    assert s1["by_type"]["LOC"] == 1
    assert s1["by_type"]["DATE"] == 1
    assert s1["linked_mentions"] == 1

    # Verify output Parquet schema & offsets
    out_pq = out_dir / "wikipedia" / "kk" / "part-00000.parquet"
    df = pl.read_parquet(out_pq)
    assert df.height == 2
    loc_row = df.filter(pl.col("type") == "LOC").to_dicts()[0]
    assert loc_row["wikidata_qid"] == "Q1520"
    assert validate_mention_offsets(
        EntityMention(
            **{
                k: loc_row[k]
                for k in (
                    "chunk_id",
                    "surface",
                    "type",
                    "char_start",
                    "char_end",
                    "score",
                    "wikidata_qid",
                    "model_version",
                )
            }
        ),
        "Астанада 2024 жылы үлкен конференция өтті.",
    )

    # Idempotent rerun skips finished shard
    s2 = run_annotate_corpus(tmp_path / "chunks", out_dir, extractor, linker, shard_spec="0/2")
    assert s2["shards_skipped"] == 1
