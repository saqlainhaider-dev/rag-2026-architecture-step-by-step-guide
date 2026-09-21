"""
Retrieval: dense (Qdrant), sparse (BM25), hybrid RRF, optional local rerank.
"""

from __future__ import annotations

from typing import Literal

from langchain_openai import OpenAIEmbeddings
from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag.bm25_index import search_bm25
from rag.config import (
    COLLECTION_NAME,
    DEFAULT_RERANK,
    EMBEDDING_MODEL,
    RERANK_CANDIDATES,
    get_qdrant_client,
)
from rag.fusion import reciprocal_rank_fusion
from rag.rerank import rerank_chunks
from rag.types import RetrievedChunk

RetrievalMode = Literal["dense", "bm25", "hybrid"]
AccessFilter = Literal["any", "public", "internal"]

__all__ = [
    "RetrievedChunk",
    "RetrievalMode",
    "retrieve",
    "retrieve_dense",
    "retrieve_bm25",
    "retrieve_hybrid",
]


def _access_query_filter(access_filter: str | None) -> Filter | None:
    if not access_filter or access_filter == "any":
        return None
    return Filter(
        must=[FieldCondition(key="access", match=MatchValue(value=access_filter))]
    )


def _chunk_from_payload(payload: dict, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        text=str(payload.get("text", "")),
        score=float(score),
        doc_id=str(payload.get("doc_id", "")),
        section=str(payload.get("section", "")),
        source=str(payload.get("source", "")),
        access=str(payload.get("access", "")),
        chunk_id=str(payload.get("chunk_id", "")),
    )


def retrieve_dense(
    query: str,
    k: int = 5,
    *,
    access_filter: str | None = None,
) -> list[RetrievedChunk]:
    """Embed the query and return nearest chunks from Qdrant."""
    client = get_qdrant_client()
    try:
        if not client.collection_exists(COLLECTION_NAME):
            raise SystemExit(
                "Collection missing. Start Qdrant (docker compose up -d) "
                "then run: python -m rag.ingest"
            )
    except SystemExit:
        raise
    except Exception as exc:
        client.close()
        raise SystemExit(
            "Cannot reach Qdrant at localhost:6333. "
            "Run: docker compose up -d\n"
            f"Details: {exc}"
        ) from exc

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector = embeddings.embed_query(query)
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=_access_query_filter(access_filter),
        limit=k,
        with_payload=True,
    ).points
    client.close()

    return [
        _chunk_from_payload(hit.payload or {}, float(hit.score or 0.0))
        for hit in hits
    ]


def retrieve_bm25(
    query: str,
    k: int = 5,
    *,
    access_filter: str | None = None,
) -> list[RetrievedChunk]:
    """Keyword search over the same chunk corpus written at ingest time."""
    return [
        _chunk_from_payload(rec, score)
        for rec, score in search_bm25(query, k=k, access_filter=access_filter)
    ]


def retrieve_hybrid(
    query: str,
    k: int = 3,
    candidate_k: int = 5,
    *,
    rerank: bool = False,
    rerank_candidates: int = RERANK_CANDIDATES,
    access_filter: str | None = None,
) -> list[RetrievedChunk]:
    """
    Dense + BM25 → RRF. Optionally expand the fused shortlist and rerank.

    When rerank=True we fuse `rerank_candidates` chunks, then keep top `k`
    after the local cross-encoder (see docs/RERANKER.md).
    """
    pool = rerank_candidates if rerank else k
    per_retriever = max(candidate_k, pool)
    dense = retrieve_dense(query, k=per_retriever, access_filter=access_filter)
    sparse = retrieve_bm25(query, k=per_retriever, access_filter=access_filter)
    fused = reciprocal_rank_fusion([dense, sparse], limit=pool)
    if rerank:
        return rerank_chunks(query, fused, top_n=k)
    return fused


def retrieve(
    query: str,
    k: int = 3,
    mode: RetrievalMode = "hybrid",
    *,
    rerank: bool | None = None,
    access_filter: str | None = None,
) -> list[RetrievedChunk]:
    """Public entrypoint used by smoke_test and answer."""
    use_rerank = DEFAULT_RERANK if rerank is None else rerank
    if mode == "dense":
        chunks = retrieve_dense(
            query,
            k=max(k, RERANK_CANDIDATES) if use_rerank else k,
            access_filter=access_filter,
        )
        return rerank_chunks(query, chunks, top_n=k) if use_rerank else chunks[:k]
    if mode == "bm25":
        chunks = retrieve_bm25(
            query,
            k=max(k, RERANK_CANDIDATES) if use_rerank else k,
            access_filter=access_filter,
        )
        return rerank_chunks(query, chunks, top_n=k) if use_rerank else chunks[:k]
    return retrieve_hybrid(
        query, k=k, rerank=use_rerank, access_filter=access_filter
    )


def retrieve_many(
    queries: list[str],
    k: int = 3,
    mode: RetrievalMode = "hybrid",
    *,
    rerank: bool | None = None,
    access_filter: str | None = None,
    rerank_query: str | None = None,
) -> list[RetrievedChunk]:
    """
    Run retrieval for one or more queries (Step 5 decompose) and fuse.

    Rerank (if on) uses rerank_query or the first query so multi-query
    shortlists are judged against the user's intent.
    """
    use_rerank = DEFAULT_RERANK if rerank is None else rerank
    cleaned = [q.strip() for q in queries if q and q.strip()]
    if not cleaned:
        return []
    if len(cleaned) == 1:
        return retrieve(
            cleaned[0], k=k, mode=mode, rerank=use_rerank, access_filter=access_filter
        )

    # Pull a wider pool per sub-query, then fuse across lists
    per_k = RERANK_CANDIDATES if use_rerank else k
    lists = [
        retrieve(q, k=per_k, mode=mode, rerank=False, access_filter=access_filter)
        for q in cleaned
    ]
    fused = reciprocal_rank_fusion(lists, limit=per_k)
    if use_rerank:
        return rerank_chunks(rerank_query or cleaned[0], fused, top_n=k)
    return fused[:k]
