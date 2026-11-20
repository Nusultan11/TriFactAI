from __future__ import annotations

import itertools

from trifact.ingestion.chunker import chunk_text
from trifact.ingestion.segment import sentences, tokenize


def words(texts: list[str]) -> list[int]:
    return [len(t.split()) for t in texts]


def test_sentences_kk_ru_en_and_offsets() -> None:
    text = "Алматы — ең ірі қала. Ол 1854 жылы құрылған!\nМосква — столица. London is big."
    spans = sentences(text)
    assert [text[a:b] for a, b in spans] == [
        "Алматы — ең ірі қала.",
        "Ол 1854 жылы құрылған!",
        "Москва — столица.",
        "London is big.",
    ]
    assert tokenize("Н.Ә. Назарбаев's book", 10)[0] == ("Н", 10, 11)


def test_chunks_respect_budget_overlap_and_cover_text() -> None:
    text = " ".join(f"Sentence number {i} has five words." for i in range(60))
    chunks = chunk_text(text, words, max_tokens=40, overlap_tokens=12)
    assert all(c.tokens <= 40 for c in chunks)
    assert chunks[0].start == 0 and chunks[-1].end == len(text)
    for a, b in itertools.pairwise(chunks):
        assert b.start < a.end  # sentence overlap
        assert text[b.start : a.end].strip()  # overlap is whole sentences of the previous chunk


def test_overlong_sentence_is_windowed() -> None:
    text = "word " * 500
    chunks = chunk_text(text.strip(), words, max_tokens=100, overlap_tokens=0)
    assert len(chunks) >= 5
    assert all(c.tokens <= 100 for c in chunks)


def test_chunk_many_equals_per_document() -> None:
    from trifact.ingestion.chunker import chunk_many

    texts = ["", "One short sentence.", " ".join(f"Sentence number {i} has five words." for i in range(30))]
    assert chunk_many(texts, words, 40, 12) == [chunk_text(t, words, 40, 12) for t in texts]


def test_document_chunker_sections_pages_offsets() -> None:
    from trifact.core.schemas import Lang
    from trifact.ingestion.chunker import DocumentChunker
    from trifact.ingestion.parser import Block, assemble

    body = " ".join(f"Sentence number {i} has five words." for i in range(30))
    doc = assemble(
        [Block("heading", "Intro", 1, 1), Block("text", body, 1), Block("heading", "Next", 2, 2),
         Block("text", body, 2), Block("text", body, 3)],
        3,
        {"format": "pdf"},
    )  # fmt: skip
    chunker = DocumentChunker(words, lambda t: (Lang.EN, 1.0), max_tokens=40, overlap_tokens=10)
    chunks = chunker.split("d1", doc)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert doc.text[c.char_start : c.char_end] == c.text == c.text.strip()
        assert c.page == doc.page_at(c.char_start)
        assert c.source_type.value == "uploaded_pdf"
    # a chunk never crosses a heading
    next_at = doc.text.index("Next")
    assert all(c.char_end <= next_at or c.char_start >= next_at for c in chunks)
    assert {c.section for c in chunks} == {"Intro", "Next"}
    assert any(c.page == 3 for c in chunks)
    flat = DocumentChunker(words, lambda t: (Lang.EN, 1.0), 40, 10, sections=False).split("d1", doc)
    assert any(c.char_start < next_at < c.char_end for c in flat)


def test_script_language_fallback() -> None:
    from trifact.ingestion.langdetect import script_lang

    assert script_lang("Қазақстан Республикасы — мемлекет") == "kk"
    assert script_lang("Российская Федерация — государство") == "ru"
    assert script_lang("The Republic of Kazakhstan") == "en"
