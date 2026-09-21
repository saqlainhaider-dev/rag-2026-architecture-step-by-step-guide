"""
Retrieval: dense (Qdrant), sparse (BM25), and hybrid via RRF fusion.
"""

from __future__ import annotations

from typing import Literal

from langchain_openai import OpenAIEmbeddings

from rag.bm25_index import search_bm25
from rag.config import COLLECTION_NAME, EMBEDDING_MODEL, get_qdrant_client
from rag.fusion import reciprocal_rank_fusion
from rag.types import RetrievedChunk

RetrievalMode = Literal["dense", "bm25", "hybrid"]

# Re-export for existing imports: `from rag.retrieve import RetrievedChunk`
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


def retrieve_hybrid(query: str, k: int = 3, candidate_k: int = 5) -> list[RetrievedChunk]:
    """
    Dense + BM25, fused with RRF, then truncated to k.

    candidate_k is how many each retriever contributes before fusion.
    """
    dense = retrieve_dense(query, k=candidate_k)
    sparse = retrieve_bm25(query, k=candidate_k)
    return reciprocal_rank_fusion([dense, sparse], limit=k)


def retrieve(
    query: str,
    k: int = 3,
    mode: RetrievalMode = "hybrid",
) -> list[RetrievedChunk]:
    """Public entrypoint used by smoke_test and answer."""
    if mode == "dense":
        return retrieve_dense(query, k=k)
    if mode == "bm25":
        return retrieve_bm25(query, k=k)
    return retrieve_hybrid(query, k=k)
