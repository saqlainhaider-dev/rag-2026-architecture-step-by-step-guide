# Step 8 — Evaluation & observability

Close the loop: **measure** quality offline and **inspect** each live request.

## Evaluation (golden set)

File: `evals/golden.json`

Each case can assert:

- expected / forbidden `doc_id`
- section substring
- answer must contain a phrase (or must abstain)
- chitchat should skip retrieval

Run:

```bash
python -m rag.eval              # rerank off (faster CI-style)
python -m rag.eval --rerank     # full stack
```

Exit code `1` if any case fails — useful as a regression gate before merging changes to chunking, prompts, or fusion.

## Observability (JSONL traces)

Every `answer_question(..., trace=True)` appends one line to `logs/traces.jsonl` (gitignored):

- `trace_id`, timestamps
- stage timings (`orchestrate`, `retrieve`, `context`, `llm_generate`, `grounding`, …)
- retrieved / context doc ids
- orchestration plan snapshot
- grounding result
- rough token/USD proxy for `gpt-4o-mini`

```bash
python -m rag.answer "Does Pro include SSO?" --role public --no-rerank
tail -n 1 logs/traces.jsonl | python -m json.tool
```

This is intentionally simple (file-based). In production you’d ship the same fields to Langfuse, Phoenix, OpenTelemetry, Datadog, etc.

## What “good” looks like

| Signal | Meaning |
|--------|---------|
| Golden pass rate | Did a change break refund/SSO/ACL? |
| Retrieve stage ms | Index / hybrid latency |
| LLM stage ms | Usually dominates cost/latency |
| Forbidden doc in context | ACL regression |
