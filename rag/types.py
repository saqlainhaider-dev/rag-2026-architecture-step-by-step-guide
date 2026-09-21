"""Shared retrieval result types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float
    doc_id: str
    section: str
    source: str
    access: str
    chunk_id: str

    @property
    def citation(self) -> str:
        return f"{self.doc_id} → {self.section} ({self.source})"
