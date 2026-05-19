"""Simple RAG core: chunking, embedding, hybrid retrieval, and on-disk persistence.

Framework-agnostic — no Streamlit imports here so the pipeline stays portable.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 50))
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))


@dataclass
class RAGStore:
    """In-memory store for chunks, their source document, and parallel indexes."""

    chunks: List[str] = field(default_factory=list)
    doc_names: List[str] = field(default_factory=list)
    pdf_metadata: dict = field(default_factory=dict)
    index: Optional[faiss.Index] = None
    bm25: Optional[BM25Okapi] = None

    @property
    def documents(self) -> List[str]:
        return sorted(set(self.doc_names))

    def has_document(self, name: str) -> bool:
        return name in self.pdf_metadata

    def add_document(self, doc_name: str, chunks: List[str], meta: dict) -> None:
        self.chunks.extend(chunks)
        self.doc_names.extend([doc_name] * len(chunks))
        self.pdf_metadata[doc_name] = meta


def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - chunk_overlap)
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step)]


def build_indexes(store: RAGStore, model: SentenceTransformer) -> None:
    """(Re)build both the dense FAISS index and the sparse BM25 index."""
    if not store.chunks:
        store.index = None
        store.bm25 = None
        return

    embeddings = np.asarray(
        model.encode(store.chunks, show_progress_bar=False, batch_size=32),
        dtype="float32",
    )
    faiss.normalize_L2(embeddings)
    store.index = faiss.IndexFlatIP(embeddings.shape[1])
    store.index.add(embeddings)

    tokenized = [chunk.lower().split() for chunk in store.chunks]
    store.bm25 = BM25Okapi(tokenized)


def search(
    store: RAGStore,
    model: SentenceTransformer,
    query: str,
    k: int,
    *,
    hybrid: bool = False,
    bm25_weight: float = 0.4,
    doc_filter: Optional[List[str]] = None,
    min_score: float = 0.0,
) -> List[Tuple[int, str, str, float]]:
    """Return [(chunk_index, chunk_text, doc_name, score), ...] sorted by relevance."""
    if not store.chunks or store.index is None:
        return []

    n = len(store.chunks)
    query_emb = np.asarray(model.encode([query]), dtype="float32")
    faiss.normalize_L2(query_emb)
    dense_scores, dense_idx = store.index.search(query_emb, n)
    dense_map = {int(i): float(s) for i, s in zip(dense_idx[0], dense_scores[0])}

    if hybrid and store.bm25 is not None:
        bm25_raw = np.asarray(store.bm25.get_scores(query.lower().split()))
        denom = float(bm25_raw.max()) if bm25_raw.max() > 0 else 1.0
        bm25_norm = bm25_raw / denom
        scored = [
            (i, (1 - bm25_weight) * dense_map.get(i, 0.0) + bm25_weight * float(bm25_norm[i]))
            for i in range(n)
        ]
    else:
        scored = [(i, dense_map.get(i, 0.0)) for i in range(n)]

    scored.sort(key=lambda x: x[1], reverse=True)

    results: List[Tuple[int, str, str, float]] = []
    for idx, score in scored:
        if score < min_score:
            continue
        if doc_filter and store.doc_names[idx] not in doc_filter:
            continue
        results.append((idx, store.chunks[idx], store.doc_names[idx], score))
        if len(results) >= k:
            break
    return results


def save_store(store: RAGStore, name: str) -> Path:
    """Persist FAISS index + chunks + metadata under data/<name>/."""
    if store.index is None:
        raise ValueError("Cannot save an empty store.")
    folder = DATA_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    faiss.write_index(store.index, str(folder / "faiss.index"))
    (folder / "meta.json").write_text(
        json.dumps(
            {
                "chunks": store.chunks,
                "doc_names": store.doc_names,
                "pdf_metadata": store.pdf_metadata,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return folder


def load_store(name: str) -> RAGStore:
    """Load a previously saved store. BM25 is rebuilt from chunks (cheap)."""
    folder = DATA_DIR / name
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    store = RAGStore(
        chunks=meta["chunks"],
        doc_names=meta["doc_names"],
        pdf_metadata=meta["pdf_metadata"],
    )
    store.index = faiss.read_index(str(folder / "faiss.index"))
    if store.chunks:
        tokenized = [chunk.lower().split() for chunk in store.chunks]
        store.bm25 = BM25Okapi(tokenized)
    return store


def list_saved_indexes() -> List[str]:
    if not DATA_DIR.exists():
        return []
    return sorted(
        p.name for p in DATA_DIR.iterdir()
        if p.is_dir() and (p / "faiss.index").exists()
    )


def delete_store(name: str) -> None:
    folder = DATA_DIR / name
    if folder.exists():
        shutil.rmtree(folder)
