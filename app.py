# app.py
"""
Streamlit UI — RAG-Based Intelligent Chatbot
The main entry point for the web application.
Run with: streamlit run app.py
"""

import streamlit as st
import tempfile
import os
from pathlib import Path

from src.config import (
    APP_TITLE, APP_ICON, APP_SUBTITLE,
    SUPPORTED_EXTENSIONS, TOP_K_RESULTS, OLLAMA_MODEL
)
from src.document_loader import load_documents
from src.text_splitter   import split_documents
from src.vector_store    import (
    build_vector_store, vector_store_exists, delete_vector_store
)
from src.rag_chain       import run_rag_streaming
from src.memory_manager  import ConversationMemory
from src.llm_handler     import check_ollama_connection


# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ── Session State Initialisation ──────────────────────────────────────────────
# Session state persists across Streamlit reruns within the same browser session

def init_session_state():
    """Initialise all session state variables on first load."""
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()

    if "messages" not in st.session_state:
        st.session_state.messages = []      # list of {role, content, sources}

    if "docs_indexed" not in st.session_state:
        st.session_state.docs_indexed = 0

    if "indexed_filenames" not in st.session_state:
        st.session_state.indexed_filenames = []

    if "ollama_ok" not in st.session_state:
        st.session_state.ollama_ok = False


init_session_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption(APP_SUBTITLE)
    st.divider()

    # ── Ollama Status ──────────────────────────────────────────────────────────
    st.subheader("🔌 LLM Status")

    if st.button("Check Ollama Connection", use_container_width=True):
        with st.spinner("Checking..."):
            st.session_state.ollama_ok = check_ollama_connection()

    if st.session_state.ollama_ok:
        st.success(f"✅ Connected — {OLLAMA_MODEL}")
    else:
        st.warning("⚠️ Not connected — click above to check")
        st.code("ollama serve", language="bash")

    st.divider()

    # ── Document Upload ────────────────────────────────────────────────────────
    st.subheader("📁 Upload Documents")

    uploaded_files = st.file_uploader(
        label="Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload the documents you want to chat with."
    )

    if uploaded_files:
        if st.button(
            f"⚡ Index {len(uploaded_files)} Document(s)",
            use_container_width=True,
            type="primary"
        ):
            with st.spinner("Processing documents..."):
                try:
                    # Save uploaded files to a temp directory
                    temp_dir   = tempfile.mkdtemp()
                    temp_paths = []

                    for uploaded_file in uploaded_files:
                        temp_path = os.path.join(
                            temp_dir, uploaded_file.name
                        )
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        temp_paths.append(temp_path)

                    # Progress feedback
                    progress = st.progress(0, text="Loading documents...")

                    # Stage 1: Load
                    docs = load_documents(temp_paths)
                    progress.progress(33, text="Chunking text...")

                    # Stage 2: Split
                    chunks = split_documents(docs)
                    progress.progress(66, text="Building vector index...")

                    # Stage 3: Build vector store
                    build_vector_store(chunks)
                    progress.progress(100, text="Done!")

                    # Update session state
                    st.session_state.docs_indexed      = len(docs)
                    st.session_state.indexed_filenames = [
                        f.name for f in uploaded_files
                    ]

                    st.success(
                        f"✅ Indexed {len(docs)} document(s) "
                        f"→ {len(chunks)} chunks"
                    )

                except Exception as e:
                    st.error(f"❌ Indexing failed: {e}")

    st.divider()

    # ── Index Status ───────────────────────────────────────────────────────────
    st.subheader("📊 Index Status")

    if vector_store_exists():
        st.success("✅ Vector store ready")
        if st.session_state.indexed_filenames:
            for fname in st.session_state.indexed_filenames:
                st.caption(f"📄 {fname}")
    else:
        st.info("No documents indexed yet")

    # Clear index button
    if vector_store_exists():
        if st.button(
            "🗑️ Clear Index & Chat",
            use_container_width=True
        ):
            delete_vector_store()
            st.session_state.messages            = []
            st.session_state.memory              = ConversationMemory()
            st.session_state.docs_indexed        = 0
            st.session_state.indexed_filenames   = []
            st.rerun()

    st.divider()

    # ── Settings ───────────────────────────────────────────────────────────────
    st.subheader("⚙️ Settings")

    top_k = st.slider(
        "Top-K Results",
        min_value=1,
        max_value=8,
        value=TOP_K_RESULTS,
        help="Number of document chunks retrieved per query."
    )

    if st.button(
        "🗑️ Clear Chat History",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.session_state.memory   = ConversationMemory()
        st.rerun()


# ── Main Chat Area ────────────────────────────────────────────────────────────

st.title(f"{APP_ICON} {APP_TITLE}")
st.caption(APP_SUBTITLE)

# ── Welcome message if no chat yet ────────────────────────────────────────────
if not st.session_state.messages:
    st.info(
        "👋 Welcome! To get started:\n\n"
        "1. Click **Check Ollama Connection** in the sidebar\n"
        "2. Upload your documents (PDF, DOCX, or TXT)\n"
        "3. Click **Index Documents**\n"
        "4. Start asking questions below!"
    )

# ── Render existing chat messages ─────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show source citations for assistant messages
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(
                f"📎 {len(message['sources'])} Source(s) Used",
                expanded=False
            ):
                for source in message["sources"]:
                    st.markdown(
                        f"**[Source {source['index']}]** "
                        f"`{source['filename']}` — "
                        f"Page **{source['page']}** | "
                        f"Score: `{source['score']}`"
                    )
                    st.caption(f"*...{source['preview']}*")
                    st.divider()

# ── Chat Input ────────────────────────────────────────────────────────────────
if query := st.chat_input("Ask a question about your documents..."):

    # Guard: check Ollama is running
    if not st.session_state.ollama_ok:
        st.warning(
            "⚠️ Please check Ollama connection first "
            "(sidebar → Check Ollama Connection)"
        )
        st.stop()

    # Guard: check documents are indexed
    if not vector_store_exists():
        st.warning(
            "⚠️ Please upload and index documents first "
            "(sidebar → Upload Documents)"
        )
        st.stop()

    # ── Display user message ───────────────────────────────────────────────────
    st.session_state.messages.append({
        "role"    : "user",
        "content" : query,
        "sources" : []
    })

    with st.chat_message("user"):
        st.markdown(query)

    # ── Generate and stream assistant response ─────────────────────────────────
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response        = ""

        try:
            # Get streaming generator + sources from RAG chain
            token_generator, sources, chunks = run_rag_streaming(
                query  = query,
                memory = st.session_state.memory,
                k      = top_k
            )

            # Stream tokens into the UI in real time
            for token in token_generator:
                full_response += token
                response_placeholder.markdown(full_response + "▌")

            # Final render without cursor
            response_placeholder.markdown(full_response)

            # Save full response to memory
            st.session_state.memory.add_assistant_message(full_response)

            # Show source citations
            if sources:
                with st.expander(
                    f"📎 {len(sources)} Source(s) Used",
                    expanded=True
                ):
                    for source in sources:
                        st.markdown(
                            f"**[Source {source['index']}]** "
                            f"`{source['filename']}` — "
                            f"Page **{source['page']}** | "
                            f"Score: `{source['score']}`"
                        )
                        st.caption(f"*...{source['preview']}*")
                        st.divider()

        except Exception as e:
            full_response = f"❌ Error generating response: {e}"
            response_placeholder.markdown(full_response)
            sources = []

        # Save to message history for display on rerun
        st.session_state.messages.append({
            "role"    : "assistant",
            "content" : full_response,
            "sources" : sources
        })