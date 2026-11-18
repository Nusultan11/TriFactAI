"""Ports between the pipeline stages.

Each stage depends on these protocols, not on concrete models, so every
component can be swapped and evaluated on its own (baseline vs fine-tuned).
"""

from __future__ import annotations

from typing import Protocol, Sequence

from trifact.core.schemas import (
    Answer,
    AskRequest,
    Chunk,
    Claim,
    EntityMention,
    Lang,
    RetrievalScope,
    ScoredChunk,
)


class DocumentParser(Protocol):
    def parse(self, raw: bytes, filename: str) -> "ParsedDocument": ...


class ParsedDocument(Protocol):
    text: str
    pages: list[tuple[int, int, int]]  # (page_no, char_start, char_end)
    sections: list[tuple[str, int, int]]  # (title, char_start, char_end)


class LanguageDetector(Protocol):
    def detect(self, text: str) -> tuple[Lang, float]: ...


class Chunker(Protocol):
    def split(self, document_id: str, parsed: ParsedDocument) -> list[Chunk]: ...


class EntityExtractor(Protocol):
    version: str

    def extract(self, chunks: Sequence[Chunk]) -> list[EntityMention]: ...


class EntityLinker(Protocol):
    def link(self, mentions: Sequence[EntityMention], chunks: Sequence[Chunk]) -> list[EntityMention]: ...


class Embedder(Protocol):
    version: str

    def embed_dense(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_sparse(self, texts: Sequence[str]) -> list[dict[int, float]]: ...


class Retriever(Protocol):
    def search(
        self,
        query: str,
        top_k: int,
        scope: RetrievalScope,
        workspace_id: str | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]: ...


class Reranker(Protocol):
    version: str

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]: ...


class Generator(Protocol):
    version: str

    def generate(self, question: str, context: Sequence[ScoredChunk], answer_lang: Lang) -> str: ...


class EvidenceVerifier(Protocol):
    def verify(self, raw_answer: str, context: Sequence[ScoredChunk]) -> list[Claim]: ...


class QueryPipeline(Protocol):
    def answer(self, request: AskRequest, workspace_id: str) -> Answer: ...
