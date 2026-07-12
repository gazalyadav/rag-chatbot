# src/text_splitter.py
"""
Text Splitter — chunks Documents into smaller pieces for embedding.
Uses RecursiveCharacterTextSplitter to respect natural language boundaries.
Metadata (source, page, filename) is preserved on every chunk.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split a list of Documents into smaller chunks.
    Each chunk inherits the metadata of its parent document.

    Args:
        documents: List of Document objects from document_loader.py

    Returns:
        List of chunked Document objects with preserved metadata
        and added 'chunk_index' field for ordering.
    """

    # RecursiveCharacterTextSplitter tries separators in order:
    # ["\n\n", "\n", " ", ""] — always prefers natural breaks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,          # max chars per chunk (500)
        chunk_overlap=CHUNK_OVERLAP,    # overlap between chunks (50)
        length_function=len,            # character count (not token count)
        separators=["\n\n", "\n", " ", ""],  # priority order
    )

    all_chunks = []

    for doc in documents:
        # Split this document's text into chunks
        chunks = splitter.split_text(doc.page_content)

        for i, chunk_text in enumerate(chunks):
            # Skip chunks that are too short to be meaningful
            if len(chunk_text.strip()) < 20:
                continue

            # Create a new Document for each chunk
            # CRITICAL: copy parent metadata + add chunk_index
            chunk_doc = Document(
                page_content=chunk_text.strip(),
                metadata={
                    **doc.metadata,          # inherit source, filename, page, filetype
                    "chunk_index": i,        # position within the parent document
                    "chunk_total": len(chunks),  # total chunks from this doc
                }
            )
            all_chunks.append(chunk_doc)

    print(f"  ✂️  Split {len(documents)} document(s) → {len(all_chunks)} chunks")
    print(f"     Chunk size: {CHUNK_SIZE} chars | Overlap: {CHUNK_OVERLAP} chars")

    return all_chunks


def get_chunk_stats(chunks: List[Document]) -> dict:
    """
    Returns statistics about the chunks for debugging.
    Useful for tuning chunk size/overlap parameters.
    """
    if not chunks:
        return {}

    sizes = [len(c.page_content) for c in chunks]

    return {
        "total_chunks" : len(chunks),
        "avg_size"     : round(sum(sizes) / len(sizes)),
        "min_size"     : min(sizes),
        "max_size"     : max(sizes),
        "sources"      : list({c.metadata["filename"] for c in chunks}),
    }