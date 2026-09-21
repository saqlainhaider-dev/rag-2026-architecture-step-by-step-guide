"""
Step 4 — Local cross-encoder reranking.

See docs/RERANKER.md for why we use a local model instead of Cohere/Voyage.
"""

from __future__ import annotations

from functools import lru_cache

from rag.config import RERANK_BATCH_SIZE, RERANK_MODEL
from rag.types import RetrievedChunk


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """Lazy-load so import/answer still works if you never enable rerank."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is required for reranking.\n"
            "Install: pip install sentence-transformers\n"
            "See docs/RERANKER.md"
        ) from exc

    # device="cpu" keeps behavior predictable on laptops without a GPU
    return CrossEncoder(RERANK_MODEL, device="cpu")


def rerank_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    top_n: int = 3,
) -> list[RetrievedChunk]:
    """
    Score each (query, chunk) pair with a cross-encoder and return top_n.

    Unlike bi-encoders (separate embeddings), the cross-encoder reads query
    and passage jointly, then outputs a relevance logit/score.
    """
    if not chunks:
        return []
    if len(chunks) == 1:
        return list(chunks)

    model = _get_cross_encoder()
    pairs = [(query, chunk.text) for chunk in chunks]
    scores = model.predict(pairs, batch_size=RERANK_BATCH_SIZE)

    ranked = sorted(
        zip(chunks, scores, strict=True),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    results: list[RetrievedChunk] = []
    for chunk, score in ranked[:top_n]:
        results.append(
            RetrievedChunk(
                text=chunk.text,
                score=float(score),
                doc_id=chunk.doc_id,
                section=chunk.section,
                source=chunk.source,
                access=chunk.access,
                chunk_id=chunk.chunk_id,
            )
        )
    return results
