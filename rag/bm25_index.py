"""BM25 sparse index over the same chunks as Qdrant (Step 3)."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

from rag.config import BM25_CORPUS_PATH

# Tiny stoplist so queries like "What is SEV-1?" aren't dominated by "what"/"is"
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "to",
        "of",
        "in",
        "on",
        "for",
        "and",
        "or",
        "what",
        "which",
        "who",
        "how",
        "do",
        "does",
        "did",
        "can",
        "i",
        "my",
        "me",
    }
)


def tokenize(text: str) -> list[str]:
    """Simple alphanumeric tokenizer; keeps hyphenated tokens like sev-1."""
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]

def save_bm25_corpus(records: list[dict]) -> Path:
    """Persist chunk texts + metadata for keyword search (same units as vectors)."""
    path = BM25_CORPUS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return path


def load_bm25_corpus(path: Path | None = None) -> list[dict]:
    path = path or BM25_CORPUS_PATH
    if not path.exists():
        raise SystemExit(
            f"BM25 corpus missing at {path}. Re-run: python -m rag.ingest"
        )
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_index() -> tuple[BM25Okapi, tuple[dict, ...]]:
    records = tuple(load_bm25_corpus())
    tokenized = [tokenize(r["text"]) for r in records]
    return BM25Okapi(tokenized), records


def clear_bm25_cache() -> None:
    _load_index.cache_clear()


def search_bm25(
    query: str,
    k: int = 5,
    *,
    access_filter: str | None = None,
) -> list[tuple[dict, float]]:
    """Return top-k (record, bm25_score) pairs, optionally filtered by access."""
    bm25, records = _load_index()
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(
        zip(records, scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    results: list[tuple[dict, float]] = []
    for rec, score in ranked:
        if score <= 0:
            continue
        if access_filter and access_filter != "any":
            if str(rec.get("access", "")) != access_filter:
                continue
        results.append((rec, float(score)))
        if len(results) >= k:
            break
    return results
