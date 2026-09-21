"""
Step 8 — Golden-set evaluation runner.

  python -m rag.eval
  python -m rag.eval --rerank   # slower / more expensive
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from rag.answer import answer_question
from rag.config import GOLDEN_SET_PATH
from rag.guardrails import UserRole

load_dotenv()


def _contains_any(text: str, needles: list[str] | None) -> bool:
    if not needles:
        return True
    lower = text.lower()
    return any(n.lower() in lower for n in needles)


def _is_abstain(text: str) -> bool:
    lower = text.lower()
    return any(
        p in lower
        for p in (
            "don't know",
            "do not know",
            "doesn't know",
            "don't have information",
            "do not have information",
            "no information",
            "not available",
            "no documents",
            "insufficient",
            "not in the",
            "aren't in the available",
            "are not in the available",
        )
    )


def evaluate_case(case: dict[str, Any], *, rerank: bool, guards: bool) -> dict[str, Any]:
    role: UserRole = case.get("role", "internal")  # type: ignore[assignment]
    question = case["question"]
    answer, chunks, plan, blocks = answer_question(
        question,
        k=3,
        mode="hybrid",
        rerank=rerank,
        orchestrate_query=True,
        expand_parents=True,
        compress=False,
        role=role,
        guards=guards,
    )

    failures: list[str] = []
    retrieved = bool(chunks) or bool(blocks)
    expect_retrieve = case.get("expect_retrieve")
    if expect_retrieve is True and not retrieved and not case.get("expect_abstain"):
        # ACL cases may retrieve irrelevant public docs then abstain
        pass
    if expect_retrieve is False and retrieved:
        failures.append("expected no retrieval (chitchat) but got chunks/blocks")

    expect_doc = case.get("expect_doc_id")
    if expect_doc:
        docs = {c.doc_id for c in chunks} | {b.doc_id for b in blocks}
        if expect_doc not in docs:
            failures.append(f"expected doc_id={expect_doc} in retrieval/context, got {sorted(docs)}")

    forbid_doc = case.get("forbid_doc_id")
    if forbid_doc:
        docs = {c.doc_id for c in chunks} | {b.doc_id for b in blocks}
        if forbid_doc in docs:
            failures.append(f"forbidden doc_id={forbid_doc} leaked into context")

    section_need = case.get("expect_section_contains")
    if section_need:
        sections = " ".join(
            [c.section for c in chunks] + [b.section for b in blocks]
        )
        if section_need.lower() not in sections.lower():
            failures.append(f"expected section containing {section_need!r}")

    if case.get("expect_abstain"):
        if not _is_abstain(answer):
            failures.append("expected abstain / don't-know style answer")
    else:
        needles = case.get("expect_answer_contains_any")
        if needles and not _contains_any(answer, needles):
            failures.append(f"answer missing any of {needles}: {answer[:180]!r}")

    if case.get("expect_abstain") is False and case.get("expect_retrieve") is not False:
        if not re.search(r"\[\d+\]", answer) and not _is_abstain(answer):
            # soft: chitchat excluded above
            if plan and plan.should_retrieve:
                failures.append("expected citation [n] in grounded answer")

    return {
        "id": case["id"],
        "pass": not failures,
        "failures": failures,
        "answer_preview": answer[:240],
        "retrieved_docs": sorted({c.doc_id for c in chunks}),
        "context_docs": sorted({b.doc_id for b in blocks}),
        "role": role,
    }


def load_golden(path: Path = GOLDEN_SET_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run golden-set RAG evaluation")
    parser.add_argument(
        "--golden",
        type=Path,
        default=GOLDEN_SET_PATH,
        help="Path to golden JSON",
    )
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable local reranker (slower; default off for eval)",
    )
    parser.add_argument(
        "--guards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep guardrails on (default: on)",
    )
    args = parser.parse_args()

    cases = load_golden(args.golden)
    print(f"Running {len(cases)} golden cases (rerank={args.rerank}, guards={args.guards})\n")

    results = [
        evaluate_case(case, rerank=args.rerank, guards=args.guards) for case in cases
    ]
    passed = sum(1 for r in results if r["pass"])
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"[{status}] {r['id']}  docs={r['context_docs'] or r['retrieved_docs']}")
        if r["failures"]:
            for f in r["failures"]:
                print(f"       - {f}")
        print(f"       answer: {r['answer_preview']!r}\n")

    print(f"Summary: {passed}/{len(results)} passed")
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
