# Step 6 — Context engineering

After retrieval + rerank, pack what the LLM **reads**.

## Ideas in this step

| Technique | What we do |
|-----------|------------|
| **Child vs parent** | Embed/search small chunks; store full `##` sections as parents |
| **Expansion** | Replace each hit with its parent section (dedupe by `parent_id`) |
| **Ordering** | Keep rerank order (first child for a parent wins) |
| **Budget** | Drop lowest-ranked blocks once `CONTEXT_MAX_CHARS` is exceeded |
| **Compress** (optional) | Extractive sentence keep overlapping the question |

```text
reranked children → expand parents → fit budget → prompt → LLM
```

## Why

Children are good for **matching**. Parents are better for **answering** (constraints often sit one paragraph away).

## Try

```bash
python -m rag.ingest   # rebuilds children + parents.json
python -m rag.answer "How long do I have to request a software refund?" 
python -m rag.answer "..." --no-expand-parents
python -m rag.answer "..." --compress --max-chars 800
```

See `rag/context.py` and `PARENTS_PATH` in `rag/config.py`.
