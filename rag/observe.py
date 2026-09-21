"""
Step 8 — Lightweight observability: JSONL request traces.

Not a full APM (Langfuse/Phoenix/Datadog). Enough to inspect latency,
stages, retrieval IDs, and rough cost proxies after each answer().
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from rag.config import TRACE_LOG_PATH


@dataclass
class StageTiming:
    name: str
    ms: float


@dataclass
class RequestTrace:
    trace_id: str
    ts: str
    question: str
    role: str
    mode: str
    flags: dict[str, Any] = field(default_factory=dict)
    stages: list[StageTiming] = field(default_factory=list)
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    context: list[dict[str, Any]] = field(default_factory=list)
    plan: dict[str, Any] | None = None
    grounding: dict[str, Any] | None = None
    answer_preview: str = ""
    error: str | None = None
    totals: dict[str, Any] = field(default_factory=dict)

    def add_stage(self, name: str, ms: float) -> None:
        self.stages.append(StageTiming(name=name, ms=round(ms, 2)))

    def finalize(self) -> None:
        stage_ms = sum(s.ms for s in self.stages)
        # Very rough cost proxy: chars/4 ≈ tokens; gpt-4o-mini ballpark
        prompt_chars = self.totals.get("prompt_chars", 0)
        answer_chars = len(self.answer_preview)
        est_input_tokens = prompt_chars / 4
        est_output_tokens = answer_chars / 4
        est_usd = (est_input_tokens * 0.15 + est_output_tokens * 0.60) / 1_000_000
        self.totals.update(
            {
                "latency_ms": round(stage_ms, 2),
                "est_input_tokens": int(est_input_tokens),
                "est_output_tokens": int(est_output_tokens),
                "est_usd_gpt4o_mini": round(est_usd, 6),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        self.finalize()
        data = asdict(self)
        return data


@contextmanager
def timed(trace: RequestTrace, name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        trace.add_stage(name, (time.perf_counter() - start) * 1000)


def new_trace(question: str, *, role: str, mode: str, **flags: Any) -> RequestTrace:
    return RequestTrace(
        trace_id=str(uuid.uuid4())[:8],
        ts=datetime.now(timezone.utc).isoformat(),
        question=question,
        role=role,
        mode=mode,
        flags=flags,
    )


def write_trace(trace: RequestTrace, path: Path | None = None) -> Path:
    path = path or TRACE_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
    return path
