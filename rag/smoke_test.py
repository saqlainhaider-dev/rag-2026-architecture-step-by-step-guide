"""
Step 1 smoke test — print nearest chunks from Qdrant (retrieval only).
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from rag.retrieve import retrieve

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test dense retrieval (Qdrant)")
    parser.add_argument(
        "query",
        nargs="?",
        default="How long do I have to request a refund?",
    )
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    chunks = retrieve(args.query, k=args.k)
    print(f"Query: {args.query!r}\n")
    for rank, chunk in enumerate(chunks, start=1):
        print(f"#{rank}  score={chunk.score:.4f}  [{chunk.doc_id} → {chunk.section}]")
        print(f"    access={chunk.access}  source={chunk.source}")
        preview = chunk.text.replace("\n", " ")
        print(f"    {preview[:220]}{'...' if len(preview) > 220 else ''}\n")


if __name__ == "__main__":
    main()
