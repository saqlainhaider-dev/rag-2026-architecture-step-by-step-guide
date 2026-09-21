"""
Step 1 smoke test — embed a query and print nearest chunks from Qdrant.
No LLM answer generation yet; we only validate retrieval quality of the index.
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from rag.config import COLLECTION_NAME, EMBEDDING_MODEL, get_qdrant_client

load_dotenv()


def search(query: str, k: int = 3) -> None:
    client = get_qdrant_client()
    try:
        if not client.collection_exists(COLLECTION_NAME):
            raise SystemExit(
                "Collection missing. Start Qdrant (docker compose up -d) "
                "then run: python -m rag.ingest"
            )
    except Exception as exc:
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

    print(f"Query: {query!r}\n")
    for rank, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        print(
            f"#{rank}  score={hit.score:.4f}  "
            f"[{payload.get('doc_id')} → {payload.get('section')}]"
        )
        print(
            f"    access={payload.get('access')}  source={payload.get('source')}"
        )
        preview = str(payload.get("text", "")).replace("\n", " ")
        print(f"    {preview[:220]}{'...' if len(preview) > 220 else ''}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test dense retrieval (Qdrant)")
    parser.add_argument(
        "query",
        nargs="?",
        default="How long do I have to request a refund?",
    )
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()
    search(args.query, k=args.k)


if __name__ == "__main__":
    main()
