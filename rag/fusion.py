"""Rank fusion helpers (Step 3)."""

from __future__ import annotations

from collections import defaultdict

from rag.config import RRF_K
from rag.types import RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    *,
    k: int = RRF_K,
    limit: int = 3,
) -> list[RetrievedChunk]:
    """
    Merge multiple ranked lists with Reciprocal Rank Fusion.

    RRF(d) = sum_i 1 / (k + rank_i(d))   where rank is 1-based.
    Raw dense/BM25 scores are ignored on purpose — they are not comparable.
    """
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, RetrievedChunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.chunk_id] += 1.0 / (k + rank)
            # Keep the first-seen chunk payload; score replaced below
            if chunk.chunk_id not in best:
                best[chunk.chunk_id] = chunk

    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    results: list[RetrievedChunk] = []
    for chunk_id, rrf_score in fused[:limit]:
        base = best[chunk_id]
        results.append(
            RetrievedChunk(
                text=base.text,
                score=rrf_score,
                doc_id=base.doc_id,
                section=base.section,
                source=base.source,
                access=base.access,
                chunk_id=base.chunk_id,
            )
        )
    return results
