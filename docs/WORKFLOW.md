# How we build this guide

Each architecture layer follows the same loop. Do not skip ahead or dump a full RAG in one commit.

## The loop

1. **Explain** — what the step is, why it matters, how it fits the bigger picture, what “proper” looks like, what we will / won’t build yet  
2. **Confirm** — learner says they understand (or asks questions)  
3. **Implement** — only that step’s code, keep it runnable  
4. **Test** — smoke checks that prove the step works  
5. **Study guide** — short walkthrough with code references  
6. **Commit + push** — one commit per step, then move on  

## Commit convention

```text
step N: short description of the layer

Optional body: what landed and why (1–2 sentences).
```

Examples:

- `step 1: ingestion and indexing with Qdrant`
- `step 2: baseline retrieve-and-generate with citations`

Do **not** mix Step N+1 work into the Step N commit.

## What’s conventional (and what we chose)

| Practice | Common in industry / open source | This repo |
|----------|----------------------------------|-----------|
| One logical change per commit | Yes — keeps history reviewable | One **step** = one commit |
| Document the process | `CONTRIBUTING.md` or `docs/` | This file (`docs/WORKFLOW.md`) |
| Tutorial / learning path | README table of contents | `README.md` steps table |
| Don’t commit secrets | `.gitignore` + `.env.example` | Already set up |
| Feature branches + PRs | Common for teams | Optional later; `main` is fine while learning solo |
| Conventional Commits (`feat:`, `fix:`) | Popular on many teams | We use `step N:` because the unit of work is a **curriculum step**, not a product feature |

So: **team product repos** often use Conventional Commits + PRs; **step-by-step teaching repos** often use numbered step commits (what we’re doing). Both are conventional in their context.

## Agent / pair-programming rule

When working with an AI pair in this repo:

- Never implement the next step until the learner confirms the explanation  
- After each finished step: study guide → commit → push  
- Prefer small, testable diffs over a complete architecture dump  
