"""
Retrieval smoke test — dense / BM25 / hybrid (RRF), with optional side-by-side compare.
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from rag.config import DEFAULT_RETRIEVAL_MODE
from rag.retrieve import RetrievalMode, retrieve, retrieve_bm25, retrieve_dense, retrieve_hybrid

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
        help="Print dense vs BM25 vs hybrid side by side",
    )
    args = parser.parse_args()

    print(f"Query: {args.query!r}\n")

    if args.compare:
        _print_chunks("Dense:", retrieve_dense(args.query, k=args.k))
        _print_chunks("BM25:", retrieve_bm25(args.query, k=args.k))
        _print_chunks("Hybrid (RRF):", retrieve_hybrid(args.query, k=args.k))
        return

    mode: RetrievalMode = args.mode  # type: ignore[assignment]
    chunks = retrieve(args.query, k=args.k, mode=mode)
    _print_chunks(f"{mode}:", chunks)


if __name__ == "__main__":
    main()
