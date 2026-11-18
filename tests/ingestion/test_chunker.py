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
