# src/embeddings.py
"""
Embedding Model — wraps Sentence Transformers for vector generation.
Model is loaded once and cached to avoid repeated 3-second startup costs.
"""

from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from typing import List
from src.config import EMBEDDING_MODEL_NAME


# ── Module-level cache — model loads once, reused forever ─────────────────────
_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    """
    Load and cache the embedding model.
    First call takes ~3 seconds. All subsequent calls are instant.

    Returns:
        Loaded SentenceTransformer model.
    """
    global _embedding_model

    if _embedding_model is None:
        print(f"  🔄 Loading embedding model: {EMBEDDING_MODEL_NAME}")
        print(f"     (First load ~3 seconds — cached after that)")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"  ✅ Embedding model loaded")

    return _embedding_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Convert a list of strings into embedding vectors.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (each is a List[float] of 384 dimensions).
    """
    model = get_embedding_model()

    # encode() returns a numpy array — convert to Python list for FAISS
    embeddings = model.encode(
        texts,
        show_progress_bar=True,     # shows progress for large batches
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize — improves cosine similarity
    )

    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string for similarity search.
    Separate function because queries don't need batch processing.

    Args:
        query: The user's question string.

    Returns:
        Single embedding vector (List[float], 384 dimensions).
    """
    model = get_embedding_model()

    embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embedding.tolist()