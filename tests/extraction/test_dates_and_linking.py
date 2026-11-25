from __future__ import annotations

from pathlib import Path

import polars as pl

from trifact.core.schemas import Chunk, EntityMention, EntityType, Lang, SourceType
from trifact.extraction.dates import RuleBasedDateExtractor, evaluate_ru_en_reference
from trifact.extraction.linking import (
    InMemoryKBIndex,
    KBEntity,
    SQLiteKBIndex,
    WikidataEntityLinker,
    build_sqlite_kb_index,
    evaluate_entity_linking,
    strip_kazakh_suffixes,
)


def _chunk(text: str, lang: Lang, cid: str = "c1") -> Chunk:
    return Chunk(
        document_id="doc1",
        chunk_id=cid,
        ordinal=0,
        language=lang,
        char_start=0,
        char_end=len(text),
        text=text,
        source_type=SourceType.CORPUS,
    )


def test_date_extractor_multilingual_offsets_and_reference() -> None:
    ext = RuleBasedDateExtractor(include_relative=True)
    chunks = [
        _chunk(
            "1991 жылғы 16 желтоқсанда Қазақстан тәуелсіздігін жариялады, биыл мерейтой.", Lang.KK, "c_kk"
        ),
        _chunk("Договор подписан 15.03.2023 в Астане и продлён до 2025 года.", Lang.RU, "c_ru"),
        _chunk("The report from December 16, 1991 and Q1 2024 was archived on 2024-11-01.", Lang.EN, "c_en"),
    ]
    mentions = ext.extract(chunks)
    by_chunk = {c.chunk_id: c for c in chunks}
    assert len(mentions) >= 6
    for m in mentions:
        assert m.type == EntityType.DATE
        txt = by_chunk[m.chunk_id].text
        assert txt[m.char_start : m.char_end] == m.surface

    ref = evaluate_ru_en_reference(include_relative=True)
    assert ref["by_lang"]["ru"]["f1"] == 1.0
    assert ref["by_lang"]["en"]["f1"] == 1.0
    assert ref["failures"] == []


def test_kazakh_suffix_stripping() -> None:
    assert "астана" in strip_kazakh_suffixes("Астанада")
    assert "қазақстан" in strip_kazakh_suffixes("Қазақстанның")
    assert "ақш" in strip_kazakh_suffixes("АҚШ-тың")
    assert "алматы" in strip_kazakh_suffixes("Алматыдан")
    assert "назарбаев" in strip_kazakh_suffixes("Назарбаевпен")


def test_wikidata_linking_disambiguation_kk_fallback_and_nil() -> None:
    kb = InMemoryKBIndex()
    kb.add_entity(
        KBEntity(
            qid="Q1520",
            ner_type="LOC",
            n_sitelinks=180,
            label_kk="Астана",
            label_ru="Астана",
            label_en="Astana",
            desc_en="capital city of Kazakhstan",
        ),
        aliases=[("Нұр-Сұлтан", 0.9)],
    )
    kb.add_entity(
        KBEntity(
            qid="Q232",
            ner_type="LOC",
            n_sitelinks=220,
            label_kk="",  # Simulate 0.38% kk label gap -> relies on ru/en fallback
            label_ru="Казахстан",
            label_en="Kazakhstan",
            desc_ru="государство в Центральной Азии",
        ),
        aliases=[("Қазақстан", 1.0)],
    )
    kb.add_entity(
        KBEntity(
            qid="Q318193",
            ner_type="PER",
            n_sitelinks=65,
            label_kk="Абай Құнанбайұлы",
            label_ru="Абай Кунанбаев",
            label_en="Abai Qunanbaiuly",
            desc_en="Kazakh poet and philosopher",
        ),
        aliases=[("Абай", 0.95)],
    )
    kb.add_entity(
        KBEntity(
            qid="Q305025",
            ner_type="LOC",
            n_sitelinks=12,
            label_kk="Абай",
            label_ru="Абай",
            label_en="Abay",
            desc_en="town in Karaganda Region",
        ),
    )

    linker = WikidataEntityLinker(kb, candidates_k=10, nil_threshold=0.35, use_kk_suffix_strip=True)

    # Inflected Kazakh mention + suffix stripping
    qid, _, _ = linker.link_one("Астанада", "Астанада үлкен жиын өтті.", "kk", "LOC")
    assert qid == "Q1520"

    # Homonym disambiguation by NER type & context
    qid_per, _, _ = linker.link_one("Абай", "Абай — қазақтың ұлы ақыны (poet).", "kk", "PER")
    qid_loc, _, _ = linker.link_one("Абай", "Абай town in Karaganda Region.", "kk", "LOC")
    assert qid_per == "Q318193"
    assert qid_loc == "Q305025"

    # NIL prediction for unknown entity
    qid_nil, _, _ = linker.link_one("Зоргландія-9999", "Unknown fictional place.", "kk", "LOC")
    assert qid_nil is None

    # Quantify Kazakh suffix-stripping recall lift
    eval_examples = [
        {
            "surface": "Астанада",
            "context": "Астанада",
            "lang": "kk",
            "ner_type": "LOC",
            "gold_qid": "Q1520",
            "gold_n_sitelinks": 180,
        },
        {
            "surface": "Қазақстанның",
            "context": "Қазақстанның",
            "lang": "kk",
            "ner_type": "LOC",
            "gold_qid": "Q232",
            "gold_n_sitelinks": 220,
        },
        {
            "surface": "Абай",
            "context": "poet",
            "lang": "kk",
            "ner_type": "PER",
            "gold_qid": "Q318193",
            "gold_n_sitelinks": 65,
        },
        {"surface": "Несуществующий", "context": "нет", "lang": "kk", "ner_type": "ORG", "gold_qid": None},
    ]
    ev = evaluate_entity_linking(
        eval_examples, kb, candidates_k=10, reference_provenance="synthetic_unit_test"
    )
    assert ev["human_verified"] is False
    kk = ev["by_lang"]["kk"]
    assert kk["kk_suffix_strip_recall_gain"] > 0.0
    assert kk["candidate_recall@10"] == 1.0
    assert kk["accuracy@1"] == 1.0


def test_sqlite_kb_index_build_and_link(tmp_path: Path) -> None:
    multi = tmp_path / "kb" / "multi"
    multi.mkdir(parents=True)
    pl.DataFrame(
        {
            "qid": ["Q1520", "Q99999"],
            "ner_type": ["LOC", "O"],
            "n_sitelinks": [180, 1],
            "label_kk": ["Астана", "Тест"],
            "label_ru": ["Астана", "Тест"],
            "label_en": ["Astana", "Test"],
            "description_kk": ["Қазақстан астанасы", ""],
            "description_ru": ["столица Казахстана", ""],
            "description_en": ["capital of Kazakhstan", ""],
            "aliases_kk": [["Нұр-Сұлтан"], []],
            "aliases_ru": [["Нур-Султан"], []],
            "aliases_en": [["Nur-Sultan"], []],
        }
    ).write_parquet(multi / "entities-00000.parquet")

    db_path = tmp_path / "kb_index.sqlite"
    info = build_sqlite_kb_index(tmp_path / "kb", db_path, min_sitelinks=1, core_only=True)
    assert info["n_entities"] == 1  # O filtered out

    idx = SQLiteKBIndex(db_path)
    try:
        linker = WikidataEntityLinker(idx)
        ch = _chunk("Астанада конференция өтті.", Lang.KK, "c1")
        m = EntityMention(
            chunk_id="c1",
            surface="Астанада",
            type=EntityType.LOC,
            char_start=0,
            char_end=8,
            score=0.95,
            model_version="test",
        )
        linked = linker.link([m], [ch])
        assert linked[0].wikidata_qid == "Q1520"
    finally:
        idx.close()
