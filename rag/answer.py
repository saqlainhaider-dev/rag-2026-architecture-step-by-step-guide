"""
Step 2 — Baseline RAG: retrieve → prompt → LLM answer with citations.
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from rag.config import CHAT_MODEL, DEFAULT_RERANK, DEFAULT_RETRIEVAL_MODE
from rag.retrieve import RetrievedChunk, RetrievalMode, retrieve

load_dotenv()

SYSTEM_PROMPT = """You are Acme's internal knowledge assistant.

Answer ONLY using the provided context chunks.
If the context is insufficient, say you don't know based on the available documents.
Be concise. Do not invent policy details, prices, or procedures.
When you use a fact, cite it inline like [1], [2] matching the chunk numbers.
"""


def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{i}] {chunk.citation}\n"
            f"access={chunk.access} score={chunk.score:.4f}\n"
            f"{chunk.text}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return (
        f"Question: {question}\n\n"
        f"Context:\n{format_context(chunks)}\n\n"
        "Answer with inline citations like [1] where appropriate."
    )


def answer_question(
    question: str,
    k: int = 3,
    mode: RetrievalMode = "hybrid",
    *,
    rerank: bool | None = None,
) -> tuple[str, list[RetrievedChunk]]:
    chunks = retrieve(question, k=k, mode=mode, rerank=rerank)
    if not chunks:
        return "I don't know — no documents were retrieved.", []

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, chunks)},
    ]
    response = llm.invoke(messages)
    content = response.content if isinstance(response.content, str) else str(response.content)
    return content, chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG: retrieve + generate")
    parser.add_argument(
        "question",
        nargs="?",
        default="Does the Pro plan include SSO?",
    )
    parser.add_argument("-k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument(
        "--mode",
        choices=["dense", "bm25", "hybrid"],
        default=DEFAULT_RETRIEVAL_MODE,
        help="Retrieval mode (default: hybrid)",
    )
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RERANK,
        help="Local cross-encoder rerank (default: on). See docs/RERANKER.md",
    )
    args = parser.parse_args()

    mode: RetrievalMode = args.mode  # type: ignore[assignment]
    text, chunks = answer_question(
        args.question, k=args.k, mode=mode, rerank=args.rerank
    )

    print(f"Question: {args.question}")
    print(f"Mode: {mode}  rerank={args.rerank}\n")
    print("Retrieved:")
    for i, chunk in enumerate(chunks, start=1):
        print(f"  [{i}] score={chunk.score:.4f}  {chunk.citation}")
    print(f"\nAnswer:\n{text}\n")
    if chunks:
        print("Sources:")
        for i, chunk in enumerate(chunks, start=1):
            print(f"  [{i}] {chunk.citation}")


if __name__ == "__main__":
    main()
