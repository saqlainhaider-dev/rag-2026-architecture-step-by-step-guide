# RAG 2026 Architecture — Step-by-Step Guide

Build a production-shaped RAG system one layer at a time. Each git commit is one step: explain → implement → test → study.

## Steps

| Step | Status | What you get |
|------|--------|--------------|
| **1. Ingestion & Indexing** | Done (this commit) | Docs → chunks → embeddings → Qdrant |
| 2. Baseline RAG | Next | Retrieve → LLM answer + citations |
| 3. Hybrid retrieval + fusion | Planned | BM25 + dense + RRF |
| 4. Reranking | Planned | Cross-encoder / reranker |
| 5. Query orchestration | Planned | Rewrite / route |
| 6. Context engineering | Planned | Parent chunks / compression |
| 7. Guardrails | Planned | Grounding / ACL / injection |
| 8. Eval + observability | Planned | Golden set / traces / cost |

## Domain

Small fake **Acme** knowledge base (refund policy, product FAQ, on-call runbook) under `data/acme/`.

## Prerequisites

- Python 3.11+
- Docker (for Qdrant)
- OpenAI API key

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY

docker compose up -d   # Qdrant on localhost:6333
```

## Step 1 — run it

```bash
python -m rag.ingest
python -m rag.smoke_test "Does Pro plan include SSO?"
```

Qdrant dashboard: http://localhost:6333/dashboard

## Layout

```text
data/acme/           # source markdown corpus
rag/ingest.py        # chunk + embed + upsert
rag/smoke_test.py    # dense retrieval check (no LLM answer yet)
rag/config.py        # Qdrant URL, collection, embedding model
docker-compose.yml   # Qdrant server
```

## Commit convention

Each step lands as its own commit, e.g. `step 1: ingestion and indexing with Qdrant`.
