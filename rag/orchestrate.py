"""
Step 5 — Query orchestration: classify, rewrite, route (and light decompose).

Turns a raw user message into a RetrievalPlan before hybrid search runs.
"""

from __future__ import annotations

from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from rag.config import CHAT_MODEL

Intent = Literal["chitchat", "docs_qa", "oncall", "billing", "product"]
AccessFilter = Literal["any", "public", "internal"]


class RetrievalPlan(BaseModel):
    """Structured plan produced before retrieval."""

    intent: Intent = Field(description="High-level intent of the user message")
    should_retrieve: bool = Field(
        description="False for greetings/thanks/chitchat that need no docs"
    )
    rewritten_query: str = Field(
        description="Search-friendly rewrite of the question (keyword-rich, concise)"
    )
    access_filter: AccessFilter = Field(
        description=(
            "'internal' for SRE/on-call/runbook topics, 'public' for customer FAQ/"
            "billing/product, 'any' if unclear"
        )
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description=(
            "If the user asked multiple distinct things, one short search query "
            "per facet; otherwise empty and use rewritten_query only"
        ),
    )
    reasoning: str = Field(default="", description="One short sentence why this plan")


ORCHESTRATOR_SYSTEM = """You plan retrieval for Acme's internal knowledge assistant.

Corpus covers: refund/billing policy, product FAQ (plans, SSO, API), on-call runbooks (SEV, rollback).

Rules:
- Greets, thanks, small talk ONLY (hi/thanks/how are you) → should_retrieve=false, access_filter=any, rewritten_query can echo the message.
- Specific factual questions about Acme (even if the answer may be unknown) → should_retrieve=true so grounding can abstain from docs.
- On-call / SEV / PagerDuty / rollback / incident → intent=oncall, access_filter=internal.
- Refunds / billing / money back → intent=billing, access_filter=public.
- Plans / SSO / API / pricing / export → intent=product, access_filter=public.
- Mixed or unclear doc questions → intent=docs_qa, access_filter=any.
- rewritten_query: concise, keyword-rich, good for search (keep terms like SEV-1, SSO, acmectl).
- sub_queries: only when the user clearly asks 2+ separate facts; max 3; else [].
"""


def _fallback_plan(question: str) -> RetrievalPlan:
    """Fail open: retrieve with the original question."""
    return RetrievalPlan(
        intent="docs_qa",
        should_retrieve=True,
        rewritten_query=question.strip(),
        access_filter="any",
        sub_queries=[],
        reasoning="fallback: orchestrator unavailable or empty input",
    )


def orchestrate(question: str) -> RetrievalPlan:
    """LLM structured plan; falls back to the raw question on failure."""
    q = (question or "").strip()
    if not q:
        return _fallback_plan(question)

    try:
        llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
        planner = llm.with_structured_output(RetrievalPlan)
        plan = planner.invoke(
            [
                {"role": "system", "content": ORCHESTRATOR_SYSTEM},
                {"role": "user", "content": q},
            ]
        )
        if not isinstance(plan, RetrievalPlan):
            return _fallback_plan(q)
        if plan.should_retrieve and not plan.rewritten_query.strip():
            plan.rewritten_query = q
        return plan
    except Exception:
        return _fallback_plan(q)


def search_queries_for_plan(plan: RetrievalPlan, original: str) -> list[str]:
    """Queries to run against the index for this plan."""
    if plan.sub_queries:
        return [s.strip() for s in plan.sub_queries if s.strip()][:3]
    q = (plan.rewritten_query or original).strip()
    return [q] if q else [original.strip()]


def main() -> None:
    """Inspect a plan: python -m rag.orchestrate \"your question\""""
    import argparse
    import json

    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Inspect query orchestration plan")
    parser.add_argument(
        "question",
        nargs="?",
        default="how long till i get my money back on software?",
    )
    args = parser.parse_args()
    plan = orchestrate(args.question)
    print(json.dumps(plan.model_dump(), indent=2))


if __name__ == "__main__":
    main()
