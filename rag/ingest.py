"""
Step 1 — Ingestion & Indexing

Load markdown docs → section-aware chunks → embed → persist in Qdrant.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag.bm25_index import clear_bm25_cache, save_bm25_corpus
from rag.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    VECTOR_SIZE,
    get_qdrant_client,
)
from rag.context import clear_parents_cache, parent_key, save_parents

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "acme"


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Pull simple key: value lines from the top of an Acme markdown doc."""
    meta: dict[str, str] = {}
    lines = text.splitlines()
    body_start = 0
    if lines and lines[0].startswith("# "):
        meta["title"] = lines[0][2:].strip()
        body_start = 1

    for i in range(body_start, min(len(lines), body_start + 8)):
        line = lines[i].strip()
        if not line:
            body_start = i + 1
            continue
        m = re.match(r"^(Last updated|Owner|Access):\s*(.+)$", line, re.I)
        if m:
            key = m.group(1).lower().replace(" ", "_")
            meta[key] = m.group(2).strip()
            body_start = i + 1
        elif line.startswith("## "):
            break
        else:
            break
    return meta, "\n".join(lines[body_start:]).strip()


def split_by_headings(body: str, doc_title: str) -> list[tuple[str, str]]:
    """Split on ## headings so chunks stay section-aligned when possible."""
    parts = re.split(r"(?m)^(## .+)$", body)
    sections: list[tuple[str, str]] = []
    if parts and parts[0].strip():
        sections.append((doc_title, parts[0].strip()))

    i = 1
    while i < len(parts):
        heading = parts[i].lstrip("# ").strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if content:
            sections.append((heading, f"## {heading}\n\n{content}"))
        i += 2
    return sections


def chunk_section(section_text: str) -> list[str]:
    """
    Split long sections into smaller *child* chunks for retrieval.

    Full section text is stored separately as a *parent* (Step 6 expansion).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=220,
        chunk_overlap=40,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(section_text) or [section_text]


def chunk_id_to_uuid(chunk_id: str) -> str:
    """Qdrant point IDs must be UUID or unsigned int — derive a stable UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def load_documents(data_dir: Path = DATA_DIR) -> tuple[list[Document], dict[str, dict]]:
    """Return child Documents plus a parent-section map for context expansion."""
    docs: list[Document] = []
    parents: dict[str, dict] = {}

    for path in sorted(data_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = parse_front_matter(raw)
        title = meta.get("title", path.stem)
        sections = split_by_headings(body, title)

        for section_title, section_text in sections:
            pid = parent_key(path.stem, section_title)
            parents[pid] = {
                "parent_id": pid,
                "text": section_text,
                "doc_id": path.stem,
                "section": section_title,
                "source": str(path.relative_to(ROOT)),
                "title": title,
                "access": meta.get("access", "public"),
                "owner": meta.get("owner", ""),
                "last_updated": meta.get("last_updated", ""),
            }
            for chunk_i, chunk in enumerate(chunk_section(section_text)):
                chunk_id = hashlib.sha1(
                    f"{path.name}:{section_title}:{chunk_i}:{chunk[:64]}".encode()
                ).hexdigest()[:16]
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "chunk_id": chunk_id,
                            "parent_id": pid,
                            "doc_id": path.stem,
                            "source": str(path.relative_to(ROOT)),
                            "title": title,
                            "section": section_title,
                            "last_updated": meta.get("last_updated", ""),
                            "owner": meta.get("owner", ""),
                            "access": meta.get("access", "public"),
                            "chunk_index": chunk_i,
                        },
                    )
                )
    return docs, parents


def build_index(
    docs: list[Document] | None = None,
    parents: dict[str, dict] | None = None,
) -> int:
    if docs is None or parents is None:
        docs, parents = load_documents()
    if not docs:
        raise SystemExit(f"No markdown docs found in {DATA_DIR}")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    texts = [d.page_content for d in docs]
    vectors = embeddings.embed_documents(texts)

    client = get_qdrant_client()
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=chunk_id_to_uuid(d.metadata["chunk_id"]),
            vector=vector,
            payload={
                "text": d.page_content,
                **d.metadata,
            },
        )
        for d, vector in zip(docs, vectors, strict=True)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    count = client.count(collection_name=COLLECTION_NAME, exact=True).count
    client.close()

    # Same chunks for BM25 so hybrid fusion compares identical units
    bm25_records = [
        {
            "text": d.page_content,
            **d.metadata,
        }
        for d in docs
    ]
    bm25_path = save_bm25_corpus(bm25_records)
    clear_bm25_cache()
    print(f"Wrote BM25 corpus ({len(bm25_records)} chunks) → {bm25_path}")

    save_parents(parents or {})
    clear_parents_cache()
    print(f"Wrote {len(parents or {})} parent sections → parents store")
    return count


def main() -> None:
    docs, parents = load_documents()
    print(f"Loaded {len(docs)} child chunks / {len(parents)} parents from {DATA_DIR}")
    for d in docs:
        print(
            f"  - {d.metadata['doc_id']} | {d.metadata['section']} "
            f"| access={d.metadata['access']} | {len(d.page_content)} chars"
        )

    count = build_index(docs, parents)
    print(f"\nIndexed {count} vectors → Qdrant@{COLLECTION_NAME}")
    print("Ingest complete (dense + BM25 + parents).")


if __name__ == "__main__":
    main()
