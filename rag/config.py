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


def get_qdrant_client() -> QdrantClient:
    """Connect to Qdrant running in Docker (see docker-compose.yml)."""
    return QdrantClient(url=QDRANT_URL, check_compatibility=False)
