# src/prompt_builder.py
"""
Prompt Builder — constructs structured RAG prompts for the LLM.

Design decisions:
- A character budget caps both the context block and the history block before
  they are assembled into the final prompt. This prevents silent context-window
  overflow when top-k is large or conversations are long.
- Budget is character-based (~4 chars ≈ 1 token for English) to avoid adding
  a heavy tokenizer dependency. Tune MAX_CONTEXT_CHARS / MAX_HISTORY_CHARS if
  you switch to a model with a different context window.
- Truncation always keeps at least the single best chunk so the model always
  has something to ground its answer in.
- History truncation keeps the most recent turns because recent context is more
  relevant to the current question than the start of a long conversation.
"""

from langchain_core.documents import Document
from typing import List


# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an intelligent document assistant. \
Answer questions accurately based ONLY on the provided document context.

STRICT RULES:
1. Answer ONLY from the provided context. Do NOT use outside knowledge.
2. Always cite sources using [Source N] notation (e.g. [Source 1], [Source 2]).
3. If the answer is not in the context, say exactly:
   "I could not find information about this in the provided documents."
4. Be concise, clear, and factual.
5. If multiple sources support the answer, cite all of them.
"""

# ── Budget constants ───────────────────────────────────────────────────────────
MAX_CONTEXT_CHARS  = 6_000   # total chars across all retrieved chunks
MAX_HISTORY_CHARS  = 2_000   # total chars of prior conversation
MAX_HISTORY_TURNS  = 6       # max messages to even consider from history


# ── Internal helpers ───────────────────────────────────────────────────────────

def _truncate_context(
    chunks: List[Document],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> List[Document]:
    """
    Keep chunks in relevance order until the character budget is reached.
    Always keeps at least the first (most relevant) chunk, even if it alone
    exceeds the budget, so the model is never given an empty context.
    """
    kept  = []
    total = 0
    for chunk in chunks:
        length = len(chunk.page_content)
        if total + length > max_chars and kept:
            break
        kept.append(chunk)
        total += length
    return kept


def _truncate_history(history: str, max_chars: int = MAX_HISTORY_CHARS) -> str:
    """
    Trim history to max_chars, keeping the most recent portion.
    Aligns the cut to the next newline to avoid mid-line breaks.
    """
    if len(history) <= max_chars:
        return history

    trimmed     = history[-max_chars:]
    newline_idx = trimmed.find("\n")
    if newline_idx != -1:
        trimmed = trimmed[newline_idx + 1:]
    return "[...earlier conversation truncated...]\n" + trimmed


# ── Public API ─────────────────────────────────────────────────────────────────

def build_rag_prompt(
    query          : str,
    context_chunks : List[Document],
    chat_history   : str = "",
) -> str:
    """
    Assemble the full RAG prompt sent to the LLM.

    Order:
        system instructions
        → retrieved document context  (budget-capped)
        → conversation history        (budget-capped, optional)
        → current user question

    Args:
        query         : The user's current question.
        context_chunks: Retrieved Document objects, highest-relevance first.
        chat_history  : Pre-formatted string of prior turns from
                        format_chat_history(). Empty string = no history.

    Returns:
        A single string ready to pass to the LLM.
    """
    context_chunks = _truncate_context(context_chunks)
    chat_history   = _truncate_history(chat_history)

    # Build labelled context sections
    sections = []
    for i, chunk in enumerate(context_chunks):
        filename = chunk.metadata.get("filename", "unknown")
        page     = chunk.metadata.get("page", "?")
        score    = chunk.metadata.get("similarity_score", 0.0)
        sections.append(
            f"[Source {i+1}: {filename} | Page {page} | Relevance: {score:.3f}]\n"
            f"{chunk.page_content}"
        )

    context_block = "\n\n---\n\n".join(sections)

    history_block = (
        f"\nCONVERSATION HISTORY:\n{chat_history}\n"
        if chat_history.strip()
        else ""
    )

    return (
        f"{SYSTEM_PROMPT}\n"
        f"DOCUMENT CONTEXT:\n{context_block}\n"
        f"{history_block}\n"
        f"CURRENT QUESTION: {query}\n\n"
        f"ANSWER (cite sources using [Source N] notation):"
    )


def format_chat_history(
    messages  : List[dict],
    max_turns : int = MAX_HISTORY_TURNS,
) -> str:
    """
    Convert a list of message dicts into a plain-text history string.
    Only the most recent max_turns messages are included — older turns
    are dropped here before the character-level truncation in build_rag_prompt.

    Args:
        messages : [{"role": "user"|"assistant", "content": "..."}, ...]
        max_turns: Maximum number of most-recent messages to include.

    Returns:
        Multi-line string of "Human: ..." / "Assistant: ..." turns.
    """
    if not messages:
        return ""

    recent = messages[-max_turns:] if len(messages) > max_turns else messages
    lines  = []
    for msg in recent:
        role = "Human" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")

    return "\n".join(lines)


def extract_sources_from_chunks(chunks: List[Document]) -> List[dict]:
    """
    Extract display-ready citation metadata from retrieved chunks.
    Used by the Streamlit UI to render source cards below each answer.

    Args:
        chunks: Retrieved Document objects (with similarity_score in metadata).

    Returns:
        List of dicts, one per chunk, ready for st.expander rendering.
    """
    sources = []
    for i, chunk in enumerate(chunks):
        text    = chunk.page_content
        preview = text[:150] + ("..." if len(text) > 150 else "")
        sources.append({
            "index"    : i + 1,
            "filename" : chunk.metadata.get("filename", "unknown"),
            "page"     : chunk.metadata.get("page", "?"),
            "filetype" : chunk.metadata.get("filetype", "?"),
            "score"    : round(chunk.metadata.get("similarity_score", 0.0), 4),
            "preview"  : preview,
        })
    return sources