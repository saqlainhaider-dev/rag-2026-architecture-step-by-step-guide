# Step 5 — Query orchestration

Before hybrid retrieval, an LLM turns the raw user message into a **RetrievalPlan**.

## What the plan contains

| Field | Role |
|-------|------|
| `intent` | chitchat / docs_qa / oncall / billing / product |
| `should_retrieve` | Skip the index for greetings/thanks |
| `rewritten_query` | Keyword-rich search string |
| `access_filter` | `public` / `internal` / `any` (metadata filter) |
| `sub_queries` | Optional decompose for multi-intent questions |

## Pipeline

```text
user message → orchestrate (structured LLM call)
             → if chitchat: reply without RAG
             → else: retrieve(rewritten / sub_queries, access_filter)
             → rerank → answer
```

Fail-open: if the planner errors, we retrieve with the original question.

## Try it

```bash
python -m rag.orchestrate "hey there"
python -m rag.orchestrate "how long till i get my money back on software?"
python -m rag.answer "What is SEV-1 and how do I rollback with acmectl?"
python -m rag.answer "hi" --no-orchestrate   # forces retrieval path A/B
```
