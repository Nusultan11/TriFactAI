"""Domain schemas shared by every TriFact AI runtime (API, worker, inference, offline jobs).

This module must not import anything from other ``trifact`` subpackages.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Lang(str, Enum):
    KK = "kk"
    RU = "ru"
    EN = "en"


class DocumentStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class SourceType(str, Enum):
    UPLOADED_PDF = "uploaded_pdf"
    UPLOADED_DOCX = "uploaded_docx"
    UPLOADED_TXT = "uploaded_txt"
    UPLOADED_HTML = "uploaded_html"
    CORPUS = "corpus"  # offline-built knowledge corpus (Wikipedia, FineWeb-2 kk, ...), ADR-0013


class RetrievalScope(str, Enum):
    DOCUMENTS = "documents"  # workspace_chunks only (default)
    CORPUS = "corpus"  # corpus_chunks only
    ALL = "all"  # both collections, fused by RRF


class EntityType(str, Enum):
    """Core label set shared by all three languages (ADR-0006).

    PER/ORG/LOC come from the trained model; DATE from the rule-based extractor.
    """

    PER = "PER"
    ORG = "ORG"
    LOC = "LOC"
    DATE = "DATE"


class Verdict(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_INFO = "not_enough_info"


# ---------------------------------------------------------------- ingestion


class Document(BaseModel):
    id: str
    workspace_id: str
    title: str | None = None
    filename: str
    mime: str
    lang: Lang | None = None
    pages: int | None = None
    sha256: str
    source_type: SourceType
    status: DocumentStatus = DocumentStatus.QUEUED
    uploaded_at: datetime
    error: str | None = None


class Chunk(BaseModel):
    """Unit of retrieval and citation. Page/section/offsets make citations exact."""

    document_id: str
    chunk_id: str
    ordinal: int
    language: Lang
    page: int | None = None
    section: str | None = None
    char_start: int
    char_end: int
    text: str
    source_type: SourceType
    corpus_source: str | None = None  # e.g. "wikipedia" when source_type == CORPUS
    token_count: int | None = None


# ---------------------------------------------------------------- extraction


class EntityMention(BaseModel):
    chunk_id: str
    surface: str
    type: EntityType | str  # str allows extended KazNERD classes (NER-3)
    char_start: int
    char_end: int
    score: float
    wikidata_qid: str | None = None
    model_version: str


# ---------------------------------------------------------------- retrieval


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    ranks: dict[str, int] = Field(default_factory=dict)  # e.g. {"dense": 3, "bm25": 11, "rerank": 1}


# ---------------------------------------------------------------- answering


class AskRequest(BaseModel):
    question: str
    answer_lang: Lang
    scope: RetrievalScope = RetrievalScope.DOCUMENTS
    document_ids: list[str] | None = None
    entity_filter: list[str] | None = None  # optional, never required (ADR-0004)


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    page: int | None = None
    quote: str


class Claim(BaseModel):
    text: str
    citations: list[Citation]
    verdict: Verdict | None = None
    nli_score: float | None = None


class RetrievalStats(BaseModel):
    scope: RetrievalScope
    candidates: int
    used: int


class Answer(BaseModel):
    answer: str
    answer_lang: Lang
    claims: list[Claim]
    refused: bool
    retrieval: RetrievalStats
    model_versions: dict[str, str]
    latency_ms: dict[str, int]
