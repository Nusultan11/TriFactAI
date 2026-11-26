"""Two-tier Qdrant store and hybrid retriever (ADR-0002, ADR-0004, ADR-0013).

Tiers:
- ``corpus_chunks__<embedder>@<ver>`` (alias ``corpus_chunks``): static offline corpus,
  fp16 dense vectors on disk, configurable quantization in RAM (``none`` | ``scalar`` |
  ``binary`` | ``turbo``) with ``rescore`` + ``oversampling``, sparse IDF vectors, payload
  on disk, and payload indexes (``lang``, ``source``, ``document_id``, ``entity_ids``).
- ``workspace_chunks__<embedder>@<ver>`` (alias ``workspace_chunks``): user-uploaded chunks,
  in-RAM vectors, strict ``workspace_id`` tenant isolation (``is_tenant=True``).

Implements ``trifact.core.interfaces.Retriever``.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from trifact.core.interfaces import Embedder, Reranker
from trifact.core.schemas import Chunk, Lang, RetrievalScope, ScoredChunk, SourceType
from trifact.retrieval.hybrid import apply_entity_soft_boost, fuse_scored_chunks

log = logging.getLogger("trifact.retrieval.store")

CORPUS_ALIAS = "corpus_chunks"
WORKSPACE_ALIAS = "workspace_chunks"
_POINT_NAMESPACE = uuid.UUID("7f3c8b7e-4b61-4d5e-9d8a-3e0d6e5a9f01")


def slugify_model(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_").lower()


def versioned_collection_name(tier: str, embedder_name: str, version: str) -> str:
    """Return canonical versioned collection name ``<tier>__<embedder>@<version>`` (ADR-0013)."""
    if tier not in {CORPUS_ALIAS, WORKSPACE_ALIAS}:
        raise ValueError(f"Unknown tier {tier!r}; expected {CORPUS_ALIAS!r} or {WORKSPACE_ALIAS!r}")
    return f"{tier}__{slugify_model(embedder_name)}@{slugify_model(version)}"


def chunk_point_id(chunk_id: str) -> str:
    """Deterministic UUID5 point ID for a stable ``chunk_id``."""
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


def _build_quantization_config(quantization: str) -> Any:
    q = (quantization or "none").lower()
    if q == "none":
        return None
    if q == "scalar":
        return models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            )
        )
    if q == "binary":
        return models.BinaryQuantization(binary=models.BinaryQuantizationConfig(always_ram=True))
    if q == "turbo":
        # Qdrant >=1.15 supports 2-bit binary encoding; fall back to 1-bit binary if older client
        encoding_enum = getattr(models, "BinaryQuantizationEncoding", None)
        if encoding_enum is not None and hasattr(encoding_enum, "TWO_BITS"):
            return models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(
                    encoding=encoding_enum.TWO_BITS,
                    always_ram=True,
                )
            )
        return models.BinaryQuantization(binary=models.BinaryQuantizationConfig(always_ram=True))
    raise ValueError(f"Unsupported quantization={quantization!r}; expected none | scalar | binary | turbo")


def _build_datatype(datatype: str) -> Any:
    dt = (datatype or "float16").lower()
    if dt == "float16":
        return models.Datatype.FLOAT16
    if dt == "float32":
        return models.Datatype.FLOAT32
    if dt == "uint8":
        return models.Datatype.UINT8
    raise ValueError(f"Unsupported datatype={datatype!r}")


@dataclass
class CorpusIndexConfig:
    dim: int = 1024
    datatype: str = "float16"
    vectors_on_disk: bool = True
    quantization: str = "binary"
    rescore: bool = True
    oversampling: float = 3.0
    hnsw_m: int = 16
    hnsw_ef_construct: int = 128
    hnsw_on_disk: bool = False
    hnsw_ef_search: int = 128

    @classmethod
    def from_params(cls, cfg: Mapping[str, Any], dim: int = 1024) -> CorpusIndexConfig:
        hnsw = cfg.get("hnsw", {}) or {}
        return cls(
            dim=int(cfg.get("mrl_dim") or dim),
            datatype=str(cfg.get("datatype", "float16")),
            vectors_on_disk=bool(cfg.get("vectors_on_disk", True)),
            quantization=str(cfg.get("quantization", "binary")),
            rescore=bool(cfg.get("rescore", True)),
            oversampling=float(cfg.get("oversampling", 3.0)),
            hnsw_m=int(hnsw.get("m", 16)),
            hnsw_ef_construct=int(hnsw.get("ef_construct", 128)),
            hnsw_on_disk=bool(hnsw.get("on_disk", False)),
            hnsw_ef_search=int(hnsw.get("ef_search", 128)),
        )


class QdrantStore:
    """Lifecycle and CRUD manager for TriFact's two-tier Qdrant collections (ADR-0013)."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        *,
        url: str | None = None,
        path: str | None = None,
        in_memory: bool = False,
        timeout: float = 60.0,
    ):
        if client is not None:
            self.client = client
        elif in_memory:
            self.client = QdrantClient(location=":memory:")
        elif path is not None:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(url=url or "http://localhost:6333", timeout=timeout)

    def create_corpus_collection(
        self,
        collection_name: str,
        cfg: CorpusIndexConfig,
        *,
        disable_hnsw_for_bulk: bool = False,
        recreate: bool = False,
    ) -> None:
        """Create a versioned corpus collection and its payload indexes BEFORE bulk upload."""
        if self.client.collection_exists(collection_name):
            if not recreate:
                return
            self.client.delete_collection(collection_name)

        hnsw_cfg = (
            models.HnswConfigDiff(m=0, ef_construct=cfg.hnsw_ef_construct, on_disk=cfg.hnsw_on_disk)
            if disable_hnsw_for_bulk
            else models.HnswConfigDiff(
                m=cfg.hnsw_m, ef_construct=cfg.hnsw_ef_construct, on_disk=cfg.hnsw_on_disk
            )
        )
        opt_cfg = (
            models.OptimizersConfigDiff(indexing_threshold=0)
            if disable_hnsw_for_bulk
            else models.OptimizersConfigDiff(indexing_threshold=20_000)
        )
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=cfg.dim,
                    distance=models.Distance.COSINE,
                    on_disk=cfg.vectors_on_disk,
                    datatype=_build_datatype(cfg.datatype),
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
            },
            quantization_config=_build_quantization_config(cfg.quantization),
            hnsw_config=hnsw_cfg,
            optimizers_config=opt_cfg,
            on_disk_payload=True,
        )
        # Create payload indexes BEFORE bulk upload (qdrant-at-scale.md §Bulk Collection Build Order)
        for field_name in ("lang", "source", "document_id", "entity_ids"):
            self._create_payload_index_safe(collection_name, field_name, models.PayloadSchemaType.KEYWORD)

    def create_workspace_collection(
        self,
        collection_name: str,
        dim: int = 1024,
        *,
        recreate: bool = False,
    ) -> None:
        """Create a versioned workspace collection with in-RAM vectors and tenant isolation."""
        if self.client.collection_exists(collection_name):
            if not recreate:
                return
            self.client.delete_collection(collection_name)

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                    on_disk=False,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
            },
            on_disk_payload=False,
        )
        # workspace_id with is_tenant=True for tenant isolation
        self._create_payload_index_safe(
            collection_name,
            "workspace_id",
            models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True),
        )
        for field_name in ("document_id", "lang", "entity_ids"):
            self._create_payload_index_safe(collection_name, field_name, models.PayloadSchemaType.KEYWORD)

    def _create_payload_index_safe(self, collection_name: str, field_name: str, schema: Any) -> None:
        try:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema,
            )
        except (RuntimeError, ValueError, NotImplementedError, AttributeError) as exc:
            # Local :memory: Qdrant client ignores payload indexes gracefully
            log.debug("create_payload_index(%s, %s) skipped: %s", collection_name, field_name, exc)

    def enable_hnsw(self, collection_name: str, cfg: CorpusIndexConfig) -> None:
        """Re-enable HNSW and quantization after bulk loading completes."""
        self.client.update_collection(
            collection_name=collection_name,
            hnsw_config=models.HnswConfigDiff(
                m=cfg.hnsw_m,
                ef_construct=cfg.hnsw_ef_construct,
                on_disk=cfg.hnsw_on_disk,
            ),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20_000),
            quantization_config=_build_quantization_config(cfg.quantization),
        )

    def switch_alias(self, alias_name: str, target_collection: str) -> dict[str, Any]:
        """Atomically switch ``alias_name`` to ``target_collection`` (zero-downtime blue/green & rollback)."""
        if not self.client.collection_exists(target_collection):
            raise ValueError(
                f"Cannot switch alias {alias_name!r}: collection {target_collection!r} does not exist"
            )
        prev = self.get_alias_target(alias_name)
        ops: list[Any] = []
        if prev is not None and prev != target_collection:
            ops.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias_name)))
        if prev != target_collection:
            ops.append(
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=target_collection,
                        alias_name=alias_name,
                    )
                )
            )
            self.client.update_collection_aliases(change_aliases_operations=ops)
        return {"alias": alias_name, "previous": prev, "current": target_collection}

    def get_alias_target(self, alias_name: str) -> str | None:
        try:
            aliases = self.client.get_aliases().aliases
        except (RuntimeError, ValueError, OSError):
            return None
        for a in aliases:
            if a.alias_name == alias_name:
                return str(a.collection_name)
        return None

    def count(self, collection_or_alias: str) -> int:
        return int(self.client.count(collection_name=collection_or_alias, exact=True).count)

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: Sequence[Chunk],
        dense_vectors: Sequence[Sequence[float]],
        sparse_vectors: Sequence[Mapping[int, float]] | None = None,
        *,
        workspace_id: str | None = None,
        entity_ids_by_chunk: Mapping[str, Sequence[str]] | None = None,
        embedder_version: str = "",
        batch_size: int = 256,
    ) -> int:
        """Upsert chunks with dense + sparse vectors and full citation/provenance payload."""
        if len(chunks) != len(dense_vectors):
            raise ValueError("chunks and dense_vectors must have equal length")
        if sparse_vectors is not None and len(sparse_vectors) != len(chunks):
            raise ValueError("sparse_vectors must match chunks length")

        total = 0
        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            points: list[models.PointStruct] = []
            for i in range(start, end):
                c = chunks[i]
                vec: dict[str, Any] = {"dense": [float(x) for x in dense_vectors[i]]}
                if sparse_vectors is not None:
                    sv = sparse_vectors[i]
                    if sv:
                        keys = list(sv.keys())
                        vals = [float(sv[k]) for k in keys]
                        vec["sparse"] = models.SparseVector(indices=keys, values=vals)
                    else:
                        vec["sparse"] = models.SparseVector(indices=[], values=[])
                payload: dict[str, Any] = {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "ordinal": c.ordinal,
                    "lang": c.language.value if isinstance(c.language, Lang) else str(c.language),
                    "page": c.page,
                    "section": c.section,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                    "text": c.text,
                    "source_type": c.source_type.value
                    if isinstance(c.source_type, SourceType)
                    else str(c.source_type),
                    "source": c.corpus_source or c.source_type.value,
                    "token_count": c.token_count,
                    "embedder_version": embedder_version,
                }
                if workspace_id is not None:
                    payload["workspace_id"] = workspace_id
                if entity_ids_by_chunk is not None and c.chunk_id in entity_ids_by_chunk:
                    payload["entity_ids"] = list(entity_ids_by_chunk[c.chunk_id])
                points.append(models.PointStruct(id=chunk_point_id(c.chunk_id), vector=vec, payload=payload))
            self.client.upsert(collection_name=collection_name, points=points, wait=True)
            total += len(points)
        return total

    def delete_workspace_documents(
        self,
        collection_or_alias: str,
        workspace_id: str,
        document_ids: Sequence[str] | None = None,
    ) -> None:
        """Delete chunks for a workspace (or specific documents inside that workspace)."""
        must: list[Any] = [
            models.FieldCondition(key="workspace_id", match=models.MatchValue(value=workspace_id))
        ]
        if document_ids:
            must.append(
                models.FieldCondition(key="document_id", match=models.MatchAny(any=list(document_ids)))
            )
        self.client.delete(
            collection_name=collection_or_alias,
            points_selector=models.FilterSelector(filter=models.Filter(must=must)),
            wait=True,
        )


def _payload_to_chunk(payload: Mapping[str, Any]) -> Chunk:
    lang_raw = str(payload.get("lang", "en"))
    try:
        lang = Lang(lang_raw)
    except ValueError:
        lang = Lang.EN
    st_raw = str(payload.get("source_type", SourceType.CORPUS.value))
    try:
        stype = SourceType(st_raw)
    except ValueError:
        stype = SourceType.CORPUS
    return Chunk(
        document_id=str(payload["document_id"]),
        chunk_id=str(payload["chunk_id"]),
        ordinal=int(payload.get("ordinal", 0)),
        language=lang,
        page=payload.get("page"),
        section=payload.get("section"),
        char_start=int(payload.get("char_start", 0)),
        char_end=int(payload.get("char_end", len(str(payload.get("text", ""))))),
        text=str(payload.get("text", "")),
        source_type=stype,
        corpus_source=payload.get("source"),
        token_count=payload.get("token_count"),
    )


class QdrantHybridRetriever:
    """Two-tier dense + sparse + fusion (+ optional reranker) retriever (ADR-0002, ADR-0004, ADR-0013).

    Satisfies ``trifact.core.interfaces.Retriever``.
    """

    def __init__(
        self,
        store: QdrantStore,
        embedder: Embedder,
        *,
        reranker: Reranker | None = None,
        corpus_collection: str = CORPUS_ALIAS,
        workspace_collection: str = WORKSPACE_ALIAS,
        prefetch_k: int = 100,
        candidates_k: int = 50,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        leg_weights: Mapping[str, float] | None = None,
        rescore: bool = True,
        oversampling: float = 3.0,
        hnsw_ef: int | None = None,
        exact: bool = False,
    ):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.corpus_collection = corpus_collection
        self.workspace_collection = workspace_collection
        self.prefetch_k = int(prefetch_k)
        self.candidates_k = int(candidates_k)
        self.fusion_method = fusion_method
        self.rrf_k = int(rrf_k)
        self.leg_weights = dict(leg_weights) if leg_weights else None
        self.rescore = bool(rescore)
        self.oversampling = float(oversampling)
        self.hnsw_ef = hnsw_ef
        self.exact = bool(exact)

    def _build_filter(
        self,
        *,
        workspace_id: str | None = None,
        document_ids: Sequence[str] | None = None,
        lang: str | None = None,
        source: str | None = None,
        entity_filter: Sequence[str] | None = None,
    ) -> models.Filter | None:
        must: list[Any] = []
        if workspace_id is not None:
            must.append(
                models.FieldCondition(key="workspace_id", match=models.MatchValue(value=workspace_id))
            )
        if document_ids:
            must.append(
                models.FieldCondition(key="document_id", match=models.MatchAny(any=list(document_ids)))
            )
        if lang:
            must.append(models.FieldCondition(key="lang", match=models.MatchValue(value=lang)))
        if source:
            must.append(models.FieldCondition(key="source", match=models.MatchValue(value=source)))
        # Entity filter is ONLY applied when explicitly requested by user (ADR-0004)
        if entity_filter:
            must.append(
                models.FieldCondition(key="entity_ids", match=models.MatchAny(any=list(entity_filter)))
            )
        return models.Filter(must=must) if must else None

    def _search_collection(
        self,
        collection: str,
        dense_q: Sequence[float],
        sparse_q: Mapping[int, float],
        q_filter: models.Filter | None,
        *,
        use_quant_params: bool = False,
        top_k: int = 50,
    ) -> tuple[list[ScoredChunk], dict[str, list[str]]]:
        search_params = models.SearchParams(
            exact=self.exact,
            hnsw_ef=self.hnsw_ef,
            quantization=(
                models.QuantizationSearchParams(rescore=self.rescore, oversampling=self.oversampling)
                if (use_quant_params and not self.exact)
                else None
            ),
        )
        dense_res = self.store.client.query_points(
            collection_name=collection,
            query=list(dense_q),
            using="dense",
            query_filter=q_filter,
            limit=self.prefetch_k,
            search_params=search_params,
            with_payload=True,
        ).points

        dense_chunks: list[ScoredChunk] = []
        chunk_entities: dict[str, list[str]] = {}
        for rank_1, pt in enumerate(dense_res, start=1):
            payload = pt.payload or {}
            c = _payload_to_chunk(payload)
            if payload.get("entity_ids"):
                chunk_entities[c.chunk_id] = list(payload["entity_ids"])
            dense_chunks.append(ScoredChunk(chunk=c, score=float(pt.score), ranks={"dense": rank_1}))

        sparse_chunks: list[ScoredChunk] = []
        if sparse_q:
            keys = list(sparse_q.keys())
            vals = [float(sparse_q[k]) for k in keys]
            sparse_res = self.store.client.query_points(
                collection_name=collection,
                query=models.SparseVector(indices=keys, values=vals),
                using="sparse",
                query_filter=q_filter,
                limit=self.prefetch_k,
                with_payload=True,
            ).points
            for rank_1, pt in enumerate(sparse_res, start=1):
                payload = pt.payload or {}
                c = _payload_to_chunk(payload)
                if payload.get("entity_ids"):
                    chunk_entities[c.chunk_id] = list(payload["entity_ids"])
                sparse_chunks.append(ScoredChunk(chunk=c, score=float(pt.score), ranks={"sparse": rank_1}))

        legs: dict[str, Sequence[ScoredChunk]] = {"dense": dense_chunks}
        if sparse_chunks:
            legs["sparse"] = sparse_chunks
        fused = fuse_scored_chunks(
            legs,
            method=self.fusion_method,
            rrf_k=self.rrf_k,
            weights=self.leg_weights,
            top_k=top_k,
        )
        return fused, chunk_entities

    def search(
        self,
        query: str,
        top_k: int = 8,
        scope: RetrievalScope = RetrievalScope.DOCUMENTS,
        workspace_id: str | None = None,
        document_ids: Sequence[str] | None = None,
        *,
        lang: str | None = None,
        source: str | None = None,
        entity_filter: Sequence[str] | None = None,
        entity_boost_ids: Sequence[str] | None = None,
        entity_boost_weight: float = 0.05,
        use_reranker: bool = True,
    ) -> list[ScoredChunk]:
        if scope in {RetrievalScope.DOCUMENTS, RetrievalScope.ALL} and not workspace_id:
            raise ValueError(f"workspace_id is required when scope={scope.value!r}")

        if hasattr(self.embedder, "embed_queries"):
            dense_q = self.embedder.embed_queries([query])[0]
        else:
            dense_q = self.embedder.embed_dense([query])[0]

        if hasattr(self.embedder, "embed_sparse_queries"):
            sparse_q = self.embedder.embed_sparse_queries([query])[0]
        else:
            sparse_q = self.embedder.embed_sparse([query])[0]

        all_entities: dict[str, list[str]] = {}
        if scope == RetrievalScope.DOCUMENTS:
            wf = self._build_filter(
                workspace_id=workspace_id,
                document_ids=document_ids,
                lang=lang,
                source=source,
                entity_filter=entity_filter,
            )
            candidates, all_entities = self._search_collection(
                self.workspace_collection,
                dense_q,
                sparse_q,
                wf,
                use_quant_params=False,
                top_k=max(self.candidates_k, top_k),
            )
        elif scope == RetrievalScope.CORPUS:
            cf = self._build_filter(
                document_ids=document_ids,
                lang=lang,
                source=source,
                entity_filter=entity_filter,
            )
            candidates, all_entities = self._search_collection(
                self.corpus_collection,
                dense_q,
                sparse_q,
                cf,
                use_quant_params=True,
                top_k=max(self.candidates_k, top_k),
            )
        else:
            # RetrievalScope.ALL: search both collections and fuse via RRF (ADR-0013)
            wf = self._build_filter(
                workspace_id=workspace_id,
                document_ids=document_ids,
                lang=lang,
                source=source,
                entity_filter=entity_filter,
            )
            cf = self._build_filter(lang=lang, source=source, entity_filter=entity_filter)
            ws_res, ws_ents = self._search_collection(
                self.workspace_collection,
                dense_q,
                sparse_q,
                wf,
                use_quant_params=False,
                top_k=self.candidates_k,
            )
            cp_res, cp_ents = self._search_collection(
                self.corpus_collection,
                dense_q,
                sparse_q,
                cf,
                use_quant_params=True,
                top_k=self.candidates_k,
            )
            all_entities.update(ws_ents)
            all_entities.update(cp_ents)
            candidates = fuse_scored_chunks(
                {"workspace": ws_res, "corpus": cp_res},
                method="rrf",
                rrf_k=self.rrf_k,
                top_k=max(self.candidates_k, top_k),
            )

        if entity_boost_ids:
            candidates = apply_entity_soft_boost(
                candidates, all_entities, entity_boost_ids, boost=entity_boost_weight
            )

        if use_reranker and self.reranker is not None and candidates:
            return self.reranker.rerank(query, candidates[: self.candidates_k], top_k=top_k)
        return candidates[:top_k]
