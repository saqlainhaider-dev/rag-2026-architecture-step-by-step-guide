# RAG 2026 Architecture — Step-by-Step Guide

Build a production-shaped RAG system one layer at a time. Each git commit is one step.

**How we work:** [docs/WORKFLOW.md](docs/WORKFLOW.md) — explain → confirm → implement → test → study guide → commit + push.

**Reranker choice:** [docs/RERANKER.md](docs/RERANKER.md) — local cross-encoder (no Cohere/Voyage key required).

## Steps

| Step | Status | What you get |
|------|--------|--------------|
| **1. Ingestion & Indexing** | Done | Docs → chunks → embeddings → Qdrant |
| **2. Baseline RAG** | Done | Retrieve → LLM answer + citations |
| **3. Hybrid retrieval + fusion** | Done | BM25 + dense + RRF |
| **4. Reranking** | Done | Local cross-encoder (`bge-reranker-base`) |
| 5. Query orchestration | Next | Rewrite / route |
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

## Step 1 — ingest + retrieval check

```bash
python -m rag.ingest
python -m rag.smoke_test "Does Pro plan include SSO?"
```

## Step 2 — baseline RAG answer

```bash
python -m rag.answer "Does the Pro plan include SSO?"
python -m rag.answer "How long do I have to request a software refund?"
python -m rag.answer "What is Acme's office coffee brand?"   # expect don't-know
```

## Step 3 — hybrid retrieval + RRF fusion

Re-ingest (writes BM25 corpus + Qdrant), then compare modes:

```bash
python -m rag.ingest
python -m rag.smoke_test "What is SEV-1?" --compare
python -m rag.answer "What is SEV-1?" --no-rerank
python -m rag.answer "What is SEV-1?" --mode dense --no-rerank
```

## Step 4 — local cross-encoder rerank

Uses `BAAI/bge-reranker-base` on CPU (first run downloads the model). Why local: [docs/RERANKER.md](docs/RERANKER.md).

```bash
python -m rag.smoke_test "What is SEV-1?" --compare          # includes hybrid vs hybrid+rerank
python -m rag.answer "What is SEV-1?"                        # rerank on by default
python -m rag.answer "What is SEV-1?" --no-rerank            # A/B without rerank
```

Qdrant dashboard: http://localhost:6333/dashboard

## Layout

```text
data/acme/           # source markdown corpus
docs/RERANKER.md     # why local cross-encoder vs Cohere/Voyage
rag/ingest.py        # chunk + embed + upsert + BM25 corpus
rag/bm25_index.py    # sparse keyword index
rag/fusion.py        # Reciprocal Rank Fusion
rag/rerank.py        # local cross-encoder
rag/retrieve.py      # dense / bm25 / hybrid / +rerank
rag/smoke_test.py    # retrieval check (+ --compare)
rag/answer.py        # retrieve → LLM → citations
rag/types.py         # RetrievedChunk
rag/config.py        # models, RRF_K, RERANK_*
docker-compose.yml   # Qdrant server
```

## Commit convention

See [docs/WORKFLOW.md](docs/WORKFLOW.md). Short form: one curriculum step → one commit (`step N: …`).
