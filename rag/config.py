"""Shared settings for the Acme RAG guide."""

from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[1]

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "acme_kb"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
VECTOR_SIZE = 1536

# Step 3 — sparse index + fusion
BM25_CORPUS_PATH = ROOT / "indexes" / "bm25_corpus.json"
RRF_K = 60  # standard RRF constant; dampens top-rank dominance a bit
DEFAULT_RETRIEVAL_MODE = "hybrid"

# Step 4 — local cross-encoder rerank (see docs/RERANKER.md)
RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_CANDIDATES = 10  # fuse this many, then rerank down to final k
RERANK_BATCH_SIZE = 8
DEFAULT_RERANK = True

# Step 5 — query orchestration (classify / rewrite / route)
DEFAULT_ORCHESTRATE = True

# Step 6 — context engineering (see docs/CONTEXT.md)
PARENTS_PATH = ROOT / "indexes" / "parents.json"
CONTEXT_MAX_CHARS = 3500
DEFAULT_EXPAND_PARENTS = True
DEFAULT_COMPRESS_CONTEXT = False


def get_qdrant_client() -> QdrantClient:
    """Connect to Qdrant running in Docker (see docker-compose.yml)."""
    return QdrantClient(url=QDRANT_URL, check_compatibility=False)
