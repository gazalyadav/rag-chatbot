# tests/test_pipeline.py
"""
Smoke test — verifies all dependencies are installed and importable.
Run with: python tests/test_pipeline.py
"""

import sys

def test_imports():
    results = []

    checks = [
        ("streamlit",                 "Streamlit UI"),
        ("langchain",                 "LangChain core"),
        ("langchain_community",       "LangChain community"),
        ("sentence_transformers",     "Sentence Transformers"),
        ("faiss",                     "FAISS vector DB"),
        ("fitz",                      "PyMuPDF (PDF parser)"),
        ("docx",                      "python-docx (DOCX parser)"),
        ("ollama",                    "Ollama LLM client"),
        ("dotenv",                    "python-dotenv"),
        ("pydantic",                  "Pydantic"),
    ]

    print("\n" + "="*55)
    print("   RAG CHATBOT — DEPENDENCY SMOKE TEST")
    print("="*55)

    all_passed = True
    for module, label in checks:
        try:
            __import__(module)
            status = "✅ PASS"
        except ImportError as e:
            status = f"❌ FAIL  ({e})"
            all_passed = False
        print(f"  {status:<14}  {label}")

    print("="*55)

    if all_passed:
        print("  🎉 All dependencies installed correctly!")
    else:
        print("  ⚠️  Fix failing imports before proceeding.")
        sys.exit(1)

    # Additional check: config loads without errors
    print("\n  Checking config module...")
    try:
        sys.path.insert(0, ".")
        from src.config import (
            CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME,
            TOP_K_RESULTS, OLLAMA_MODEL, VECTOR_DB_DIR
        )
        print(f"  ✅ Config loaded")
        print(f"     Chunk size    : {CHUNK_SIZE} tokens")
        print(f"     Overlap       : {CHUNK_OVERLAP} tokens")
        print(f"     Embedding     : {EMBEDDING_MODEL_NAME}")
        print(f"     Top-K results : {TOP_K_RESULTS}")
        print(f"     LLM model     : {OLLAMA_MODEL}")
        print(f"     Vector DB dir : {VECTOR_DB_DIR}")
    except Exception as e:
        print(f"  ❌ Config error: {e}")
        sys.exit(1)

    print("\n  ✅ Phase 0 complete. Ready to build Phase 1.\n")

if __name__ == "__main__":
    test_imports()


# Add to tests/test_pipeline.py

def test_ingestion_pipeline():
    """End-to-end test of Phase 1 pipeline."""
    import os, sys
    sys.path.insert(0, ".")

    print("\n" + "="*55)
    print("   PHASE 1 — INGESTION PIPELINE TEST")
    print("="*55)

    # Create a small test TXT file
    test_file = "data/test_sample.txt"
    os.makedirs("data", exist_ok=True)
    with open(test_file, "w") as f:
        f.write("""Artificial intelligence is transforming industries worldwide.
Machine learning models can now process vast amounts of data.
Natural language processing enables computers to understand human text.
Vector databases store embeddings for fast similarity search.
Retrieval-Augmented Generation combines search with language models.""")

    from src.document_loader import load_document
    from src.text_splitter   import split_documents, get_chunk_stats
    from src.vector_store    import build_vector_store, similarity_search

    # Stage 1: Load
    docs = load_document(test_file)
    print(f"\n  Stage 1 — Loaded {len(docs)} document(s)")

    # Stage 2: Split
    chunks = split_documents(docs)
    stats  = get_chunk_stats(chunks)
    print(f"  Stage 2 — {stats}")

    # Stage 3: Build vector store
    build_vector_store(chunks)

    # Stage 4: Search
    results = similarity_search("What is RAG?", k=2)
    print(f"\n  Stage 4 — Top result:")
    print(f"  '{results[0].page_content[:100]}...'")
    print(f"  Source: {results[0].metadata['filename']}, "
          f"Score: {results[0].metadata['similarity_score']:.4f}")

    print("\n  ✅ Phase 1 pipeline working end-to-end!\n")

if __name__ == "__main__":
    test_imports()
    test_ingestion_pipeline()

def test_rag_chain():
    """Test the full RAG chain end-to-end."""
    import sys
    sys.path.insert(0, ".")

    print("\n" + "="*55)
    print("   PHASE 2 — RAG CHAIN TEST")
    print("="*55)

    from src.llm_handler    import check_ollama_connection
    from src.memory_manager import ConversationMemory
    from src.rag_chain      import run_rag_query

    # Check Ollama is running
    print("\n  Checking Ollama...")
    if not check_ollama_connection():
        print("  ⚠️  Skipping LLM test — start Ollama first")
        print("     Run: ollama serve  (in a separate terminal)")
        return

    # Test memory
    memory = ConversationMemory(window_size=3)
    memory.add_user_message("Hello")
    memory.add_assistant_message("Hi! How can I help?")
    print(f"\n  ✅ Memory working: {memory}")

    # Test full RAG query (uses vector store from Phase 1 test)
    print("\n  Running RAG query...")
    answer, sources = run_rag_query(
        query  = "What is artificial intelligence?",
        memory = memory
    )

    print(f"\n  ✅ Answer received ({len(answer)} chars):")
    print(f"  '{answer[:200]}...'")
    print(f"\n  📎 Sources cited: {len(sources)}")
    for s in sources:
        print(f"     [Source {s['index']}] {s['filename']} | "
              f"Page {s['page']} | Score {s['score']}")

    print(f"\n  Memory after query: {memory}")
    print("\n  ✅ Phase 2 RAG chain working end-to-end!\n")


if __name__ == "__main__":
    test_imports()
    test_ingestion_pipeline()
    test_rag_chain()