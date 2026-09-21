"""Shared Qdrant connection settings for Step 1."""

from __future__ import annotations

from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "acme_kb"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1536


def get_qdrant_client() -> QdrantClient:
    """Connect to Qdrant running in Docker (see docker-compose.yml)."""
    return QdrantClient(url=QDRANT_URL, check_compatibility=False)
