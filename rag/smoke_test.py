"""
Retrieval smoke test — dense / BM25 / hybrid, optional local rerank compare.
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from rag.config import DEFAULT_RERANK, DEFAULT_RETRIEVAL_MODE
from rag.retrieve import (
    RetrievalMode,
    retrieve,
    retrieve_bm25,
    retrieve_dense,
    retrieve_hybrid,
)

load_dotenv()


def _print_chunks(title: str, chunks) -> None:
    print(f"{title}")
    if not chunks:
        print("  (no hits)\n")
        return
    for rank, chunk in enumerate(chunks, start=1):
        print(f"  #{rank}  score={chunk.score:.4f}  [{chunk.doc_id} → {chunk.section}]")
        preview = chunk.text.replace("\n", " ")
        print(f"      {preview[:160]}{'...' if len(preview) > 160 else ''}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test retrieval modes")
    parser.add_argument(
        "query",
        nargs="?",
        default="What is SEV-1?",
    )
    parser.add_argument("-k", type=int, default=3)
    parser.add_argument(
        "--mode",
        choices=["dense", "bm25", "hybrid"],
        default=DEFAULT_RETRIEVAL_MODE,
        help="Retrieval mode when not using --compare",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print dense vs BM25 vs hybrid vs hybrid+rerank",
    )
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RERANK,
        help="Apply local cross-encoder rerank (default: on). See docs/RERANKER.md",
    )
    args = parser.parse_args()

    print(f"Query: {args.query!r}\n")

    if args.compare:
        _print_chunks("Dense:", retrieve_dense(args.query, k=args.k))
        _print_chunks("BM25:", retrieve_bm25(args.query, k=args.k))
        _print_chunks("Hybrid (RRF):", retrieve_hybrid(args.query, k=args.k, rerank=False))
        _print_chunks(
            "Hybrid + local rerank:",
            retrieve_hybrid(args.query, k=args.k, rerank=True),
        )
        return

    mode: RetrievalMode = args.mode  # type: ignore[assignment]
    chunks = retrieve(args.query, k=args.k, mode=mode, rerank=args.rerank)
    label = f"{mode}" + (" + rerank" if args.rerank else "")
    _print_chunks(f"{label}:", chunks)


if __name__ == "__main__":
    main()
