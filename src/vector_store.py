# src/vector_store.py
"""
Vector Store — builds, caches, and queries a FAISS index.

Design decisions:
- In-memory cache avoids re-reading from disk on every query.
- Cache is invalidated on every write or delete so it never serves stale data.
- add_to_vector_store() merges new chunks into the existing index so uploading
  a second document does not wipe the first.
- build_vector_store() always starts fresh (full rebuild).
"""

import faiss
import pickle
import numpy as np
from langchain_core.documents import Document
from typing import List, Tuple
from src.config import VECTOR_DB_DIR, FAISS_INDEX_NAME, TOP_K_RESULTS
from src.embeddings import embed_texts, embed_query


FAISS_FILE = VECTOR_DB_DIR / f"{FAISS_INDEX_NAME}.faiss"
DOCS_FILE  = VECTOR_DB_DIR / f"{FAISS_INDEX_NAME}.pkl"

# ── In-memory cache ────────────────────────────────────────────────────────────
# Loaded once from disk, then reused for every query in the session.
# Cleared whenever the index is rebuilt, extended, or deleted.
_cache: dict = {"index": None, "chunks": None}


def _clear_cache() -> None:
    _cache["index"]  = None
    _cache["chunks"] = None


def _populate_cache(index: faiss.Index, chunks: List[Document]) -> None:
    _cache["index"]  = index
    _cache["chunks"] = chunks


# ── Write operations ───────────────────────────────────────────────────────────

def build_vector_store(chunks: List[Document]) -> None:
    """
    Build a brand-new FAISS index from the given chunks.
    Wipes any existing index. Use add_to_vector_store() to append instead.
    """
    if not chunks:
        raise ValueError("No chunks provided — cannot build an empty vector store.")

    print(f"\n  🔨 Building vector store from {len(chunks)} chunks...")

    vectors   = np.array(embed_texts([c.page_content for c in chunks]), dtype=np.float32)
    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    _save(index, chunks)
    _populate_cache(index, chunks)

    print(f"  📦 FAISS index: {index.ntotal} vectors ({dimension}D)")
    print(f"  💾 Saved → {FAISS_FILE.name} + {DOCS_FILE.name}")


def add_to_vector_store(new_chunks: List[Document]) -> None:
    """
    Merge new_chunks into the existing index without wiping old entries.
    If no index exists yet, creates one (behaves like build_vector_store).
    """
    if not new_chunks:
        return

    if vector_store_exists():
        index, existing_chunks = load_vector_store()
        print(f"  ➕ Merging {len(new_chunks)} new chunks into "
              f"existing {len(existing_chunks)}")
        all_chunks = existing_chunks + new_chunks
    else:
        index      = None
        all_chunks = new_chunks
        print(f"  🆕 No existing index — creating one from {len(new_chunks)} chunks")

    new_vectors = np.array(
        embed_texts([c.page_content for c in new_chunks]),
        dtype=np.float32
    )

    if index is not None:
        index.add(new_vectors)
    else:
        dimension = new_vectors.shape[1]
        index     = faiss.IndexFlatIP(dimension)
        index.add(new_vectors)

    _save(index, all_chunks)
    _populate_cache(index, all_chunks)

    print(f"  ✅ Index now has {index.ntotal} total vectors")


def delete_vector_store() -> None:
    """Delete persisted files and clear the cache."""
    if FAISS_FILE.exists():
        FAISS_FILE.unlink()
    if DOCS_FILE.exists():
        DOCS_FILE.unlink()
    _clear_cache()
    print("  🗑️  Vector store deleted.")


# ── Read operations ────────────────────────────────────────────────────────────

def load_vector_store(force_reload: bool = False) -> Tuple[faiss.Index, List[Document]]:
    """
    Return (index, chunks), reading from the in-memory cache when possible.

    Args:
        force_reload: Bypass cache and re-read from disk. Useful when another
                      process may have updated the files externally.
    """
    if not force_reload and _cache["index"] is not None:
        return _cache["index"], _cache["chunks"]

    if not FAISS_FILE.exists() or not DOCS_FILE.exists():
        raise FileNotFoundError(
            "No vector store found. Please upload and index documents first."
        )

    index = faiss.read_index(str(FAISS_FILE))
    with open(DOCS_FILE, "rb") as f:
        chunks = pickle.load(f)

    print(f"  📂 Loaded from disk: {index.ntotal} vectors")
    _populate_cache(index, chunks)
    return index, chunks


def similarity_search(query: str, k: int = TOP_K_RESULTS) -> List[Document]:
    """
    Return the top-k most relevant chunks for query.
    Uses the cached index — no disk read after the first call.
    Attaches similarity_score to each chunk's metadata.
    """
    index, chunks = load_vector_store()
    query_vector  = np.array([embed_query(query)], dtype=np.float32)
    distances, indices = index.search(query_vector, k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        doc = chunks[idx]
        doc.metadata["similarity_score"] = float(distances[0][i])
        results.append(doc)
    return results


def vector_store_exists() -> bool:
    """True if both the FAISS index file and the chunk pickle exist on disk."""
    return FAISS_FILE.exists() and DOCS_FILE.exists()


# ── Internal helpers ───────────────────────────────────────────────────────────

def _save(index: faiss.Index, chunks: List[Document]) -> None:
    """Persist index and chunks to disk atomically."""
    faiss.write_index(index, str(FAISS_FILE))
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(chunks, f)