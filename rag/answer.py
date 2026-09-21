"""
RAG answer path: optional orchestration → retrieve → LLM (+ citations).
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from rag.config import (
    CHAT_MODEL,
    DEFAULT_ORCHESTRATE,
    DEFAULT_RERANK,
    DEFAULT_RETRIEVAL_MODE,
)
from rag.orchestrate import RetrievalPlan, orchestrate, search_queries_for_plan
from rag.retrieve import RetrievedChunk, RetrievalMode, retrieve, retrieve_many

load_dotenv()

SYSTEM_PROMPT = """You are Acme's internal knowledge assistant.

Answer ONLY using the provided context chunks.
If the context is insufficient, say you don't know based on the available documents.
Be concise. Do not invent policy details, prices, or procedures.
When you use a fact, cite it inline like [1], [2] matching the chunk numbers.
"""

CHITCHAT_SYSTEM = """You are Acme's friendly internal assistant.
Respond briefly to greetings or small talk. Do not invent company policy.
Offer to help with product, billing, or on-call questions.
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


def _print_plan(plan: RetrievalPlan) -> None:
    print("Orchestration plan:")
    print(f"  intent={plan.intent}  retrieve={plan.should_retrieve}  "
          f"access={plan.access_filter}")
    print(f"  rewritten_query={plan.rewritten_query!r}")
    if plan.sub_queries:
        print(f"  sub_queries={plan.sub_queries}")
    if plan.reasoning:
        print(f"  reasoning={plan.reasoning}")
    print()


def answer_question(
    question: str,
    k: int = 3,
    mode: RetrievalMode = "hybrid",
    *,
    rerank: bool | None = None,
    orchestrate_query: bool | None = None,
) -> tuple[str, list[RetrievedChunk], RetrievalPlan | None]:
    use_orch = DEFAULT_ORCHESTRATE if orchestrate_query is None else orchestrate_query
    plan: RetrievalPlan | None = None

    if use_orch:
        plan = orchestrate(question)
        if not plan.should_retrieve:
            llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
            response = llm.invoke(
                [
                    {"role": "system", "content": CHITCHAT_SYSTEM},
                    {"role": "user", "content": question},
                ]
            )
            content = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            return content, [], plan

        queries = search_queries_for_plan(plan, question)
        chunks = retrieve_many(
            queries,
            k=k,
            mode=mode,
            rerank=rerank,
            access_filter=plan.access_filter,
            rerank_query=question,
        )
    else:
        chunks = retrieve(question, k=k, mode=mode, rerank=rerank)

    if not chunks:
        return "I don't know — no documents were retrieved.", [], plan

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, chunks)},
    ]
    response = llm.invoke(messages)
    content = response.content if isinstance(response.content, str) else str(response.content)
    return content, chunks, plan


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG: orchestrate + retrieve + generate")
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
    parser.add_argument(
        "--orchestrate",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ORCHESTRATE,
        help="Classify/rewrite/route before retrieve (default: on)",
    )
    args = parser.parse_args()

    mode: RetrievalMode = args.mode  # type: ignore[assignment]
    text, chunks, plan = answer_question(
        args.question,
        k=args.k,
        mode=mode,
        rerank=args.rerank,
        orchestrate_query=args.orchestrate,
    )

    print(f"Question: {args.question}")
    print(f"Mode: {mode}  rerank={args.rerank}  orchestrate={args.orchestrate}\n")
    if plan is not None:
        _print_plan(plan)
    if chunks:
        print("Retrieved:")
        for i, chunk in enumerate(chunks, start=1):
            print(f"  [{i}] score={chunk.score:.4f}  {chunk.citation}")
        print()
    print(f"Answer:\n{text}\n")
    if chunks:
        print("Sources:")
        for i, chunk in enumerate(chunks, start=1):
            print(f"  [{i}] {chunk.citation}")


if __name__ == "__main__":
    main()
