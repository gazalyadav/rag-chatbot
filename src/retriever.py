# src/retriever.py
from langchain.schema import Document
from typing import List
from src.vector_store import similarity_search
from src.config import TOP_K_RESULTS

def retrieve_relevant_chunks(query: str, k: int = TOP_K_RESULTS) -> List[Document]:
    """
    Retrieve the top-K most relevant chunks for a given query.
    Returns chunks with metadata for source citation.
    """
    print(f"\n  🔍 Retrieving top-{k} chunks for: '{query[:60]}...'")
    results = similarity_search(query, k=k)

    print(f"  📎 Retrieved {len(results)} chunks:")
    for i, doc in enumerate(results):
        score    = doc.metadata.get("similarity_score", 0)
        filename = doc.metadata.get("filename", "unknown")
        page     = doc.metadata.get("page", "?")
        print(f"     [{i+1}] {filename} | Page {page} | Score {score:.4f}")

    return results

def format_context_for_prompt(chunks: List[Document]) -> str:
    """
    Format retrieved chunks into a structured context string for the LLM prompt.
    Each chunk is labelled with its source for citation tracking.
    """
    context_parts = []

    for i, chunk in enumerate(chunks):
        filename = chunk.metadata.get("filename", "unknown")
        page     = chunk.metadata.get("page", "?")

        context_parts.append(
            f"[Source {i+1}: {filename}, Page {page}]\n"
            f"{chunk.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)