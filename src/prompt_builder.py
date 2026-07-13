# src/prompt_builder.py
"""
Prompt Builder — constructs structured RAG prompts for the LLM.
The prompt design is critical: it enforces grounding, citations,
and honest fallback when the answer isn't in the retrieved context.
"""

from langchain_core.documents import Document
from typing import List


# ── System instruction — defines LLM behaviour ────────────────────────────────
SYSTEM_PROMPT = """You are an intelligent document assistant. Your job is to answer questions accurately based ONLY on the provided document context.

STRICT RULES:
1. Answer ONLY from the provided context below. Do NOT use outside knowledge.
2. Always cite your sources using [Source N] notation (e.g., [Source 1], [Source 2]).
3. If the answer is not found in the context, respond with:
   "I could not find information about this in the provided documents."
4. Be concise, clear, and factual.
5. If multiple sources support the answer, cite all of them.
"""


def build_rag_prompt(
    query: str,
    context_chunks: List[Document],
    chat_history: str = ""
) -> str:
    """
    Build a complete RAG prompt combining:
      - System instructions
      - Retrieved document context with source labels
      - Conversation history (for multi-turn memory)
      - Current user query

    Args:
        query         : The user's current question.
        context_chunks: Retrieved Document objects from the vector store.
        chat_history  : Formatted string of previous conversation turns.

    Returns:
        A single formatted prompt string ready for the LLM.
    """

    # Format each retrieved chunk with a source label
    context_sections = []
    for i, chunk in enumerate(context_chunks):
        filename = chunk.metadata.get("filename", "unknown")
        page     = chunk.metadata.get("page", "?")
        score    = chunk.metadata.get("similarity_score", 0)

        context_sections.append(
            f"[Source {i+1}: {filename} | Page {page} | "
            f"Relevance: {score:.3f}]\n"
            f"{chunk.page_content}"
        )

    context_block = "\n\n---\n\n".join(context_sections)

    # Include conversation history only if it exists
    history_block = ""
    if chat_history.strip():
        history_block = f"""
CONVERSATION HISTORY:
{chat_history}
"""

    # Assemble the full prompt
    prompt = f"""{SYSTEM_PROMPT}

DOCUMENT CONTEXT:
{context_block}
{history_block}
CURRENT QUESTION: {query}

ANSWER (cite sources using [Source N] notation):"""

    return prompt


def format_chat_history(messages: List[dict]) -> str:
    """
    Format the chat history list into a readable string for the prompt.

    Args:
        messages: List of dicts with 'role' and 'content' keys.
                  e.g. [{"role": "user", "content": "..."},
                         {"role": "assistant", "content": "..."}]

    Returns:
        Formatted string of the conversation history.
    """
    if not messages:
        return ""

    formatted = []
    for msg in messages:
        role    = "Human"    if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        formatted.append(f"{role}: {content}")

    return "\n".join(formatted)


def extract_sources_from_chunks(chunks: List[Document]) -> List[dict]:
    """
    Extract clean source citation info from retrieved chunks.
    Used by the UI to display source cards below each answer.

    Args:
        chunks: Retrieved Document objects.

    Returns:
        List of source dicts with display-ready citation info.
    """
    sources = []
    for i, chunk in enumerate(chunks):
        sources.append({
            "index"    : i + 1,
            "filename" : chunk.metadata.get("filename", "unknown"),
            "page"     : chunk.metadata.get("page", "?"),
            "filetype" : chunk.metadata.get("filetype", "?"),
            "score"    : round(chunk.metadata.get("similarity_score", 0), 4),
            "preview"  : chunk.page_content[:150] + "..."  # snippet for UI
        })
    return sources