"""
Retrieval: dense (Qdrant), sparse (BM25), hybrid RRF, optional local rerank.
"""

from __future__ import annotations

from typing import Literal

from langchain_openai import OpenAIEmbeddings

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

__all__ = [
    "RetrievedChunk",
    "RetrievalMode",
    "retrieve",
    "retrieve_dense",
    "retrieve_bm25",
    "retrieve_hybrid",
]


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


def retrieve_dense(query: str, k: int = 5) -> list[RetrievedChunk]:
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
        limit=k,
        with_payload=True,
    ).points
    client.close()

    return [
        _chunk_from_payload(hit.payload or {}, float(hit.score or 0.0))
        for hit in hits
    ]


def retrieve_bm25(query: str, k: int = 5) -> list[RetrievedChunk]:
    """Keyword search over the same chunk corpus written at ingest time."""
    return [
        _chunk_from_payload(rec, score) for rec, score in search_bm25(query, k=k)
    ]


def retrieve_hybrid(
    query: str,
    k: int = 3,
    candidate_k: int = 5,
    *,
    rerank: bool = False,
    rerank_candidates: int = RERANK_CANDIDATES,
) -> list[RetrievedChunk]:
    """
    Dense + BM25 → RRF. Optionally expand the fused shortlist and rerank.

    When rerank=True we fuse `rerank_candidates` chunks, then keep top `k`
    after the local cross-encoder (see docs/RERANKER.md).
    """
    pool = rerank_candidates if rerank else k
    # Each retriever should contribute enough for a useful fused pool
    per_retriever = max(candidate_k, pool)
    dense = retrieve_dense(query, k=per_retriever)
    sparse = retrieve_bm25(query, k=per_retriever)
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
) -> list[RetrievedChunk]:
    """Public entrypoint used by smoke_test and answer."""
    use_rerank = DEFAULT_RERANK if rerank is None else rerank
    if mode == "dense":
        chunks = retrieve_dense(query, k=max(k, RERANK_CANDIDATES) if use_rerank else k)
        return rerank_chunks(query, chunks, top_n=k) if use_rerank else chunks[:k]
    if mode == "bm25":
        chunks = retrieve_bm25(query, k=max(k, RERANK_CANDIDATES) if use_rerank else k)
        return rerank_chunks(query, chunks, top_n=k) if use_rerank else chunks[:k]
    return retrieve_hybrid(query, k=k, rerank=use_rerank)
