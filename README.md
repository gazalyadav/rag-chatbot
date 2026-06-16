# rag-chatbot
RAG-Based Intelligent Chatbot

# RAG-Based Intelligent Chatbot

A domain-specific Retrieval-Augmented Generation (RAG) chatbot that processes
custom document sets (PDF, TXT, DOCX), indexes them into a vector database,
and uses a local LLM to generate grounded, hallucination-free answers with
source citations.

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Orchestration | LangChain |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Database | FAISS |
| Local LLM | Llama 3 via Ollama |
| Document Parsing | PyMuPDF + python-docx |

## Architecture
Document Upload → Chunking → Embeddings → FAISS Index

User Query → Embedding → Similarity Search → Top-K Chunks

Retrieved Context + Query → Llama 3 → Answer + Citations


## Setup & Installation

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com) installed

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 4. Pull the LLM model
```bash
ollama pull llama3
```

### 5. Run the app
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Launch the app
streamlit run app.py
```

## Project Structure
rag_chatbot/

├── app.py                  # Streamlit UI

├── requirements.txt

├── src/

│   ├── config.py           # Central configuration

│   ├── document_loader.py  # PDF, DOCX, TXT parsing

│   ├── text_splitter.py    # Document chunking

│   ├── embeddings.py       # Vector embeddings

│   ├── vector_store.py     # FAISS operations

│   ├── retriever.py        # Similarity search

│   ├── prompt_builder.py   # RAG prompt templates

│   ├── llm_handler.py      # Ollama LLM interface

│   ├── memory_manager.py   # Conversation memory

│   └── rag_chain.py        # End-to-end orchestrator

└── tests/

└── test_pipeline.py
