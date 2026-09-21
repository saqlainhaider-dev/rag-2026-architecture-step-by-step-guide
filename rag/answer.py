"""
RAG answer path: orchestrate → retrieve → context → guards → LLM.
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from rag.config import (
    CHAT_MODEL,
    CONTEXT_MAX_CHARS,
    DEFAULT_COMPRESS_CONTEXT,
    DEFAULT_EXPAND_PARENTS,
    DEFAULT_GUARDS,
    DEFAULT_ORCHESTRATE,
    DEFAULT_RERANK,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_ROLE,
)
from rag.context import ContextBlock, build_context, format_context_blocks
from rag.guardrails import (
    GROUNDED_SYSTEM_PROMPT,
    UserRole,
    build_guarded_user_prompt,
    check_grounding,
    filter_blocks_for_role,
    filter_chunks_for_role,
    grounding_refusal,
    resolve_access_filter,
    scan_injection,
)
from rag.orchestrate import RetrievalPlan, orchestrate, search_queries_for_plan
from rag.retrieve import RetrievedChunk, RetrievalMode, retrieve, retrieve_many

load_dotenv()

SYSTEM_PROMPT = """You are Acme's internal knowledge assistant.

Answer ONLY using the provided context blocks.
If the context is insufficient, say you don't know based on the available documents.
Be concise. Do not invent policy details, prices, or procedures.
When you use a fact, cite it inline like [1], [2] matching the block numbers.
"""

CHITCHAT_SYSTEM = """You are Acme's friendly internal assistant.
Respond briefly to greetings or small talk. Do not invent company policy.
Offer to help with product, billing, or on-call questions.
"""


def _print_plan(plan: RetrievalPlan) -> None:
    print("Orchestration plan:")
    print(
        f"  intent={plan.intent}  retrieve={plan.should_retrieve}  "
        f"access={plan.access_filter}"
    )
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
    expand_parents: bool | None = None,
    compress: bool | None = None,
    max_chars: int = CONTEXT_MAX_CHARS,
    role: UserRole = DEFAULT_ROLE,
    guards: bool | None = None,
) -> tuple[str, list[RetrievedChunk], RetrievalPlan | None, list[ContextBlock]]:
    use_orch = DEFAULT_ORCHESTRATE if orchestrate_query is None else orchestrate_query
    use_expand = DEFAULT_EXPAND_PARENTS if expand_parents is None else expand_parents
    use_compress = DEFAULT_COMPRESS_CONTEXT if compress is None else compress
    use_guards = DEFAULT_GUARDS if guards is None else guards
    plan: RetrievalPlan | None = None
    injection_hits: list[str] = []

    if use_guards:
        injection_hits = scan_injection(question)

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
            return content, [], plan, []

        access = (
            resolve_access_filter(role, plan.access_filter)
            if use_guards
            else plan.access_filter
        )
        queries = search_queries_for_plan(plan, question)
        chunks = retrieve_many(
            queries,
            k=k,
            mode=mode,
            rerank=rerank,
            access_filter=access,
            rerank_query=question,
        )
    else:
        access = resolve_access_filter(role, "any") if use_guards else None
        chunks = retrieve(
            question, k=k, mode=mode, rerank=rerank, access_filter=access
        )

    if use_guards:
        chunks = filter_chunks_for_role(chunks, role)

    if not chunks:
        msg = "I don't know — no documents were retrieved."
        if use_guards and role == "public":
            msg = (
                "I don't know based on documents available for your role "
                "(public). Internal runbooks are not visible."
            )
        return msg, [], plan, []

    blocks = build_context(
        chunks,
        question=question,
        expand_parents=use_expand,
        max_chars=max_chars,
        compress=use_compress,
    )
    if use_guards:
        blocks = filter_blocks_for_role(blocks, role)

    if not blocks:
        return (
            "I don't know — no permitted context remained after access checks.",
            chunks,
            plan,
            [],
        )

    system = GROUNDED_SYSTEM_PROMPT if use_guards else SYSTEM_PROMPT
    if use_guards:
        user = build_guarded_user_prompt(question, blocks)
    else:
        user = (
            f"Question: {question}\n\n"
            f"Context:\n{format_context_blocks(blocks)}\n\n"
            "Answer with inline citations like [1] where appropriate."
        )

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    response = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    content = response.content if isinstance(response.content, str) else str(response.content)

    if use_guards:
        ok, reason = check_grounding(content, blocks)
        if not ok:
            content = grounding_refusal(reason)

    # Attach injection note for CLI visibility via a side channel: prepend is noisy;
    # callers that care can re-scan. We stash nothing; main() re-scans for print.
    _ = injection_hits
    return content, chunks, plan, blocks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG: orchestrate + retrieve + context + guards + generate"
    )
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
    parser.add_argument(
        "--expand-parents",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_EXPAND_PARENTS,
        help="Expand retrieved children to full parent sections (default: on)",
    )
    parser.add_argument(
        "--compress",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_COMPRESS_CONTEXT,
        help="Extractive compression into the char budget (default: off)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=CONTEXT_MAX_CHARS,
        help=f"Context char budget (default: {CONTEXT_MAX_CHARS})",
    )
    parser.add_argument(
        "--role",
        choices=["public", "internal"],
        default=DEFAULT_ROLE,
        help="Simulated caller role for ACL (default: internal)",
    )
    parser.add_argument(
        "--guards",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_GUARDS,
        help="ACL + injection hardening + grounding (default: on)",
    )
    args = parser.parse_args()

    mode: RetrievalMode = args.mode  # type: ignore[assignment]
    role: UserRole = args.role  # type: ignore[assignment]
    text, chunks, plan, blocks = answer_question(
        args.question,
        k=args.k,
        mode=mode,
        rerank=args.rerank,
        orchestrate_query=args.orchestrate,
        expand_parents=args.expand_parents,
        compress=args.compress,
        max_chars=args.max_chars,
        role=role,
        guards=args.guards,
    )

    print(f"Question: {args.question}")
    print(
        f"Mode: {mode}  role={role}  guards={args.guards}  "
        f"rerank={args.rerank}  orchestrate={args.orchestrate}  "
        f"expand_parents={args.expand_parents}  compress={args.compress}\n"
    )
    if args.guards:
        hits = scan_injection(args.question)
        if hits:
            print(f"Guardrail warning: possible injection phrases matched: {hits}\n")
    if plan is not None:
        _print_plan(plan)
        if args.guards:
            print(
                f"Effective access_filter: "
                f"{resolve_access_filter(role, plan.access_filter)}\n"
            )
    if chunks:
        print("Retrieved (children):")
        for i, chunk in enumerate(chunks, start=1):
            print(
                f"  [{i}] score={chunk.score:.4f}  {chunk.citation}  "
                f"access={chunk.access}  ({len(chunk.text)} chars)"
            )
        print()
    if blocks:
        print("Context blocks (prompt):")
        for i, block in enumerate(blocks, start=1):
            kind = "parent" if block.expanded else "chunk"
            print(
                f"  [{i}] {block.citation}  [{kind}]  access={block.access}  "
                f"({len(block.text)} chars)"
            )
        print()
    print(f"Answer:\n{text}\n")
    if blocks:
        print("Sources:")
        for i, block in enumerate(blocks, start=1):
            print(f"  [{i}] {block.citation}")
        if args.guards:
            ok, reason = check_grounding(text, blocks)
            print(f"\nGrounding: {'pass' if ok else 'fail'} ({reason})")


if __name__ == "__main__":
    main()
