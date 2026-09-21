# Why we use a local cross-encoder for Step 4

This guide reranks with a **local** model, not Cohere / Voyage (or similar APIs).

## Decision

| Option | Pros | Cons |
|--------|------|------|
| **Local cross-encoder** (`BAAI/bge-reranker-base`) | No paid API key; same “score (query, passage)” idea as production; works offline after download | First run downloads weights; slower on CPU; you manage deps (`sentence-transformers` / torch) |
| Hosted (Cohere Rerank, Voyage Rerank, …) | Fast, managed, strong quality | Needs an API key + ongoing cost (~$0.001–0.003 per query at typical shortlist sizes) |

We chose **local** because this is a learning repo and we do not assume a rerank vendor key. The pipeline shape is identical: hybrid shortlist → rerank → top-n for the LLM. Swapping in Cohere/Voyage later is a thin client change inside `rag/rerank.py`.

## How it fits the pipeline

```text
query → dense + BM25 → RRF (larger candidate list)
      → local cross-encoder scores each (query, chunk)
      → keep top-n → LLM
```

First-stage retrieval maximizes **recall**. The cross-encoder maximizes **precision at the top** of the prompt.

## Model

- Default: `BAAI/bge-reranker-base` (see `RERANK_MODEL` in `rag/config.py`)
- Loaded via `sentence_transformers.CrossEncoder`
- Runs on CPU by default; use CUDA/MPS if available

## Cost note (hosted, for later)

Rough production ballpark (verify on vendor pricing pages):

- Cohere: billed per “search” (query + up to ~100 docs) — often ~$2 / 1k searches for Fast-tier models
- Voyage: billed on processed tokens — often ~$0.02–0.05 / 1M tokens after a free allowance

For this Acme toy corpus, hybrid alone is often already good; rerank still teaches the production pattern.
