# src/rag_chain.py
"""
RAG Chain — the end-to-end orchestrator.
Wires together: retrieval → prompt building → LLM generation → memory.
This is the single entry point the Streamlit UI calls for every query.
"""

from typing import Generator, List, Tuple
from langchain.schema import Document

from src.retriever      import retrieve_relevant_chunks
from src.prompt_builder import build_rag_prompt, extract_sources_from_chunks
from src.llm_handler    import generate_response, generate_streaming
from src.memory_manager import ConversationMemory
from src.vector_store   import vector_store_exists
from src.config         import TOP_K_RESULTS


def run_rag_query(
    query          : str,
    memory         : ConversationMemory,
    k              : int  = TOP_K_RESULTS,
    stream         : bool = False
) -> Tuple[str, List[dict]]:
    """
    Run a complete RAG query — retrieval + generation + memory update.
    Use this for non-streaming responses (testing, API use).

    Args:
        query  : The user's question.
        memory : ConversationMemory instance (shared across turns).
        k      : Number of chunks to retrieve.
        stream : If True, use streaming mode (returns generator instead).

    Returns:
        Tuple of (answer_string, sources_list)
        sources_list contains citation info for UI display.
    """

    # Guard: check vector store exists before querying
    if not vector_store_exists():
        return (
            "⚠️ No documents have been indexed yet. "
            "Please upload and process documents first.",
            []
        )

    # ── Step 1: Add user query to memory ──────────────────────────────────────
    memory.add_user_message(query)

    # ── Step 2: Retrieve relevant chunks ──────────────────────────────────────
    chunks = retrieve_relevant_chunks(query, k=k)

    if not chunks:
        answer = "I could not find any relevant information in the documents."
        memory.add_assistant_message(answer)
        return answer, []

    # ── Step 3: Build the RAG prompt ──────────────────────────────────────────
    chat_history = memory.get_formatted_history()
    prompt       = build_rag_prompt(
        query          = query,
        context_chunks = chunks,
        chat_history   = chat_history
    )

    # ── Step 4: Generate the response ─────────────────────────────────────────
    answer = generate_response(prompt)

    # ── Step 5: Save assistant response to memory ─────────────────────────────
    memory.add_assistant_message(answer)

    # ── Step 6: Extract source citations for UI display ───────────────────────
    sources = extract_sources_from_chunks(chunks)

    return answer, sources


def run_rag_streaming(
    query  : str,
    memory : ConversationMemory,
    k      : int = TOP_K_RESULTS
) -> Tuple[Generator, List[dict], List[Document]]:
    """
    Run a RAG query with streaming LLM output.
    Use this in Streamlit for the real-time typing effect.

    Args:
        query  : The user's question.
        memory : ConversationMemory instance.
        k      : Number of chunks to retrieve.

    Returns:
        Tuple of (token_generator, sources_list, chunks)
        - token_generator : iterate this to get tokens as they stream
        - sources_list    : citation info for UI display
        - chunks          : raw chunks (needed to save answer to memory)

    Usage in Streamlit:
        generator, sources, chunks = run_rag_streaming(query, memory)
        full_answer = ""
        for token in generator:
            full_answer += token
            # update UI with token
        memory.add_assistant_message(full_answer)
    """

    if not vector_store_exists():
        def error_gen():
            yield "⚠️ No documents indexed yet. Please upload documents first."
        return error_gen(), [], []

    # Add user message to memory
    memory.add_user_message(query)

    # Retrieve relevant chunks
    chunks = retrieve_relevant_chunks(query, k=k)

    if not chunks:
        def no_results_gen():
            yield "I could not find relevant information in the documents."
        return no_results_gen(), [], []

    # Build prompt with history
    chat_history = memory.get_formatted_history()
    prompt       = build_rag_prompt(
        query          = query,
        context_chunks = chunks,
        chat_history   = chat_history
    )

    # Return the streaming generator + sources
    # NOTE: caller must save the full answer to memory after streaming completes
    sources = extract_sources_from_chunks(chunks)
    return generate_streaming(prompt), sources, chunks