"""
RAG answer path: orchestrate → retrieve → context → guards → LLM (+ traces).
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
    DEFAULT_TRACE,
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
from rag.observe import new_trace, timed, write_trace
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
    trace: bool | None = None,
) -> tuple[str, list[RetrievedChunk], RetrievalPlan | None, list[ContextBlock]]:
    use_orch = DEFAULT_ORCHESTRATE if orchestrate_query is None else orchestrate_query
    use_expand = DEFAULT_EXPAND_PARENTS if expand_parents is None else expand_parents
    use_compress = DEFAULT_COMPRESS_CONTEXT if compress is None else compress
    use_guards = DEFAULT_GUARDS if guards is None else guards
    use_trace = DEFAULT_TRACE if trace is None else trace
    use_rerank = DEFAULT_RERANK if rerank is None else rerank

    plan: RetrievalPlan | None = None
    chunks: list[RetrievedChunk] = []
    blocks: list[ContextBlock] = []
    content = ""
    grounding_meta: dict | None = None

    req = new_trace(
        question,
        role=role,
        mode=mode,
        orchestrate=use_orch,
        rerank=use_rerank,
        expand_parents=use_expand,
        compress=use_compress,
        guards=use_guards,
        k=k,
    )

    try:
        if use_guards:
            with timed(req, "injection_scan"):
                hits = scan_injection(question)
            req.flags["injection_hits"] = hits

        if use_orch:
            with timed(req, "orchestrate"):
                plan = orchestrate(question)
            if plan is not None:
                req.plan = plan.model_dump()

            if plan is not None and not plan.should_retrieve:
                with timed(req, "llm_chitchat"):
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
                req.answer_preview = content[:500]
                if use_trace:
                    write_trace(req)
                return content, [], plan, []

            access = (
                resolve_access_filter(role, plan.access_filter if plan else "any")
                if use_guards
                else (plan.access_filter if plan else "any")
            )
            req.flags["access_filter"] = access
            queries = search_queries_for_plan(plan, question) if plan else [question]
            with timed(req, "retrieve"):
                chunks = retrieve_many(
                    queries,
                    k=k,
                    mode=mode,
                    rerank=use_rerank,
                    access_filter=access,
                    rerank_query=question,
                )
        else:
            access = resolve_access_filter(role, "any") if use_guards else None
            req.flags["access_filter"] = access
            with timed(req, "retrieve"):
                chunks = retrieve(
                    question,
                    k=k,
                    mode=mode,
                    rerank=use_rerank,
                    access_filter=access,
                )

        if use_guards:
            with timed(req, "acl_filter_chunks"):
                chunks = filter_chunks_for_role(chunks, role)

        req.retrieved = [
            {
                "doc_id": c.doc_id,
                "section": c.section,
                "access": c.access,
                "score": round(c.score, 4),
            }
            for c in chunks
        ]

        if not chunks:
            content = "I don't know — no documents were retrieved."
            if use_guards and role == "public":
                content = (
                    "I don't know based on documents available for your role "
                    "(public). Internal runbooks are not visible."
                )
            req.answer_preview = content
            if use_trace:
                write_trace(req)
            return content, [], plan, []

        with timed(req, "context"):
            blocks = build_context(
                chunks,
                question=question,
                expand_parents=use_expand,
                max_chars=max_chars,
                compress=use_compress,
            )
            if use_guards:
                blocks = filter_blocks_for_role(blocks, role)

        req.context = [
            {
                "doc_id": b.doc_id,
                "section": b.section,
                "access": b.access,
                "expanded": b.expanded,
                "chars": len(b.text),
            }
            for b in blocks
        ]

        if not blocks:
            content = (
                "I don't know — no permitted context remained after access checks."
            )
            req.answer_preview = content
            if use_trace:
                write_trace(req)
            return content, chunks, plan, []

        system = GROUNDED_SYSTEM_PROMPT if use_guards else SYSTEM_PROMPT
        if use_guards:
            user = build_guarded_user_prompt(question, blocks)
        else:
            user = (
                f"Question: {question}\n\n"
                f"Context:\n{format_context_blocks(blocks)}\n\n"
                "Answer with inline citations like [1] where appropriate."
            )
        req.totals["prompt_chars"] = len(system) + len(user)

        with timed(req, "llm_generate"):
            llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
            response = llm.invoke(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )
            content = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )

        if use_guards:
            with timed(req, "grounding"):
                ok, reason = check_grounding(content, blocks)
                grounding_meta = {"ok": ok, "reason": reason}
                if not ok:
                    content = grounding_refusal(reason)
            req.grounding = grounding_meta

        req.answer_preview = content[:500]
        if use_trace:
            path = write_trace(req)
            req.flags["trace_path"] = str(path)
        return content, chunks, plan, blocks

    except Exception as exc:
        req.error = str(exc)
        if use_trace:
            write_trace(req)
        raise


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
    parser.add_argument(
        "--trace",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TRACE,
        help="Append JSONL observability trace (default: on)",
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
        trace=args.trace,
    )

    print(f"Question: {args.question}")
    print(
        f"Mode: {mode}  role={role}  guards={args.guards}  "
        f"rerank={args.rerank}  orchestrate={args.orchestrate}  "
        f"expand_parents={args.expand_parents}  compress={args.compress}  "
        f"trace={args.trace}\n"
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
    if args.trace:
        from rag.config import TRACE_LOG_PATH

        print(f"\nTrace appended → {TRACE_LOG_PATH}")


if __name__ == "__main__":
    main()
