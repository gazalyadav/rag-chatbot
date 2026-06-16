# src/vector_store.py
import faiss
import pickle
import numpy as np
from pathlib import Path
from langchain.schema import Document
from typing import List, Tuple
from src.config import VECTOR_DB_DIR, FAISS_INDEX_NAME, TOP_K_RESULTS
from src.embeddings import embed_texts, embed_query

FAISS_FILE = VECTOR_DB_DIR / f"{FAISS_INDEX_NAME}.faiss"
DOCS_FILE  = VECTOR_DB_DIR / f"{FAISS_INDEX_NAME}.pkl"

def build_vector_store(chunks: List[Document]) -> None:
    if not chunks:
        raise ValueError("No chunks provided.")

    print(f"\n  🔨 Building vector store from {len(chunks)} chunks...")
    texts      = [chunk.page_content for chunk in chunks]
    embeddings = embed_texts(texts)
    vectors    = np.array(embeddings, dtype=np.float32)

    dimension = vectors.shape[1]
    index     = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    print(f"  📦 FAISS index built: {index.ntotal} vectors, {dimension}D")
    faiss.write_index(index, str(FAISS_FILE))
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    print(f"  💾 Saved → {FAISS_FILE.name} + {DOCS_FILE.name}")

def load_vector_store() -> Tuple[faiss.Index, List[Document]]:
    if not FAISS_FILE.exists() or not DOCS_FILE.exists():
        raise FileNotFoundError("No vector store found. Please index documents first.")

    index = faiss.read_index(str(FAISS_FILE))
    with open(DOCS_FILE, "rb") as f:
        chunks = pickle.load(f)

    print(f"  📂 Loaded vector store: {index.ntotal} vectors")
    return index, chunks

def similarity_search(query: str, k: int = TOP_K_RESULTS) -> List[Document]:
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
    return FAISS_FILE.exists() and DOCS_FILE.exists()

def delete_vector_store() -> None:
    if FAISS_FILE.exists(): FAISS_FILE.unlink()
    if DOCS_FILE.exists():  DOCS_FILE.unlink()
    print("  🗑️  Vector store deleted.")