"""
Step 6 — Context engineering: parent expansion, ordering, budget, optional compress.

Retrieve on child chunks; pack fuller parent sections into the LLM prompt.
See docs/CONTEXT.md.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache

from rag.config import CONTEXT_MAX_CHARS, PARENTS_PATH
from rag.types import RetrievedChunk


@dataclass(frozen=True)
class ContextBlock:
    """One prompt unit after context engineering (usually a parent section)."""

    text: str
    score: float
    doc_id: str
    section: str
    source: str
    access: str
    parent_id: str
    expanded: bool
    chunk_ids: tuple[str, ...]

    @property
    def citation(self) -> str:
        return f"{self.doc_id} → {self.section} ({self.source})"


def parent_key(doc_id: str, section: str) -> str:
    return f"{doc_id}::{section}"


def save_parents(parents: dict[str, dict]) -> None:
    PARENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARENTS_PATH.write_text(json.dumps(parents, indent=2), encoding="utf-8")


@lru_cache(maxsize=1)
def _load_parents() -> dict[str, dict]:
    if not PARENTS_PATH.exists():
        return {}
    return json.loads(PARENTS_PATH.read_text(encoding="utf-8"))


def clear_parents_cache() -> None:
    _load_parents.cache_clear()


def expand_to_parents(chunks: list[RetrievedChunk]) -> list[ContextBlock]:
    """
    Map retrieved children → unique parent sections, preserving rerank order.

    First time a parent_id appears wins (highest-ranked child for that section).
    """
    parents = _load_parents()
    blocks: list[ContextBlock] = []
    seen: set[str] = set()

    for chunk in chunks:
        pid = parent_key(chunk.doc_id, chunk.section)
        if pid in seen:
            # Same parent already included via an earlier (higher) child
            continue
        seen.add(pid)

        parent = parents.get(pid)
        if parent and parent.get("text"):
            text = str(parent["text"])
            expanded = text.strip() != chunk.text.strip()
            blocks.append(
                ContextBlock(
                    text=text,
                    score=chunk.score,
                    doc_id=chunk.doc_id,
                    section=chunk.section,
                    source=chunk.source,
                    access=chunk.access,
                    parent_id=pid,
                    expanded=expanded,
                    chunk_ids=(chunk.chunk_id,),
                )
            )
        else:
            blocks.append(
                ContextBlock(
                    text=chunk.text,
                    score=chunk.score,
                    doc_id=chunk.doc_id,
                    section=chunk.section,
                    source=chunk.source,
                    access=chunk.access,
                    parent_id=pid,
                    expanded=False,
                    chunk_ids=(chunk.chunk_id,),
                )
            )
    return blocks


def _sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def extractive_compress(text: str, question: str, *, max_chars: int) -> str:
    """
    Keep sentences that overlap the question; fall back to a head truncate.

    Cheap teaching stand-in for LLM compression — no extra API call.
    """
    if len(text) <= max_chars:
        return text

    q_tokens = {t for t in re.findall(r"[a-z0-9]+", question.lower()) if len(t) > 2}
    scored: list[tuple[int, str]] = []
    for sent in _sentence_split(text):
        s_tokens = set(re.findall(r"[a-z0-9]+", sent.lower()))
        overlap = len(q_tokens & s_tokens)
        scored.append((overlap, sent))

    scored.sort(key=lambda item: item[0], reverse=True)
    kept: list[str] = []
    total = 0
    for overlap, sent in scored:
        if overlap <= 0 and kept:
            continue
        if total + len(sent) + 1 > max_chars:
            break
        kept.append(sent)
        total += len(sent) + 1

    if not kept:
        return text[: max_chars - 1] + "…"
    # Restore rough document order
    order = {s: i for i, s in enumerate(_sentence_split(text))}
    kept.sort(key=lambda s: order.get(s, 0))
    return " ".join(kept)


def fit_budget(
    blocks: list[ContextBlock],
    *,
    max_chars: int = CONTEXT_MAX_CHARS,
    question: str = "",
    compress: bool = False,
) -> list[ContextBlock]:
    """Keep rerank order; drop lowest-ranked blocks that don't fit the budget."""
    if max_chars <= 0:
        return blocks

    fitted: list[ContextBlock] = []
    used = 0
    for block in blocks:
        text = block.text
        if compress:
            room = max(max_chars - used, 200)
            text = extractive_compress(text, question, max_chars=room)

        cost = len(text)
        if fitted and used + cost > max_chars:
            break
        if not fitted and cost > max_chars:
            text = text[: max_chars - 1] + "…"
            cost = len(text)

        fitted.append(
            ContextBlock(
                text=text,
                score=block.score,
                doc_id=block.doc_id,
                section=block.section,
                source=block.source,
                access=block.access,
                parent_id=block.parent_id,
                expanded=block.expanded,
                chunk_ids=block.chunk_ids,
            )
        )
        used += cost
    return fitted


def build_context(
    chunks: list[RetrievedChunk],
    *,
    question: str = "",
    expand_parents: bool = True,
    max_chars: int = CONTEXT_MAX_CHARS,
    compress: bool = False,
) -> list[ContextBlock]:
    """Full Step 6 packing pipeline for the answer prompt."""
    blocks = expand_to_parents(chunks) if expand_parents else [
        ContextBlock(
            text=c.text,
            score=c.score,
            doc_id=c.doc_id,
            section=c.section,
            source=c.source,
            access=c.access,
            parent_id=parent_key(c.doc_id, c.section),
            expanded=False,
            chunk_ids=(c.chunk_id,),
        )
        for c in chunks
    ]
    return fit_budget(
        blocks, max_chars=max_chars, question=question, compress=compress
    )


def format_context_blocks(blocks: list[ContextBlock]) -> str:
    parts: list[str] = []
    for i, block in enumerate(blocks, start=1):
        flag = "parent" if block.expanded else "chunk"
        parts.append(
            f"[{i}] {block.citation} ({flag})\n"
            f"access={block.access} score={block.score:.4f}\n"
            f"{block.text}"
        )
    return "\n\n".join(parts)
