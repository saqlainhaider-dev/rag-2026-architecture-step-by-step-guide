"""
Shared dense retrieval against Qdrant.

Used by smoke_test (Step 1) and answer (Step 2).
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_openai import OpenAIEmbeddings

from rag.config import COLLECTION_NAME, EMBEDDING_MODEL, get_qdrant_client


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float
    doc_id: str
    section: str
    source: str
    access: str
    chunk_id: str

    @property
    def citation(self) -> str:
        return f"{self.doc_id} → {self.section} ({self.source})"


def retrieve(query: str, k: int = 3) -> list[RetrievedChunk]:
    """Embed the query and return the top-k nearest chunks from Qdrant."""
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

    chunks: list[RetrievedChunk] = []
    for hit in hits:
        payload = hit.payload or {}
        chunks.append(
            RetrievedChunk(
                text=str(payload.get("text", "")),
                score=float(hit.score or 0.0),
                doc_id=str(payload.get("doc_id", "")),
                section=str(payload.get("section", "")),
                source=str(payload.get("source", "")),
                access=str(payload.get("access", "")),
                chunk_id=str(payload.get("chunk_id", "")),
            )
        )
    return chunks
