# src/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project Paths ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# ── Document Ingestion ─────────────────────────────────────
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx"]

# ── Text Splitting ─────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

# ── Embedding Model ────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ── Vector Database ────────────────────────────────────────
FAISS_INDEX_NAME = "rag_index"

# ── Retrieval ──────────────────────────────────────────────
TOP_K_RESULTS = 4

# ── LLM (Ollama) ──────────────────────────────────────────
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS  = 1024

# ── Memory ─────────────────────────────────────────────────
MEMORY_WINDOW_SIZE = 5

# ── UI ─────────────────────────────────────────────────────
APP_TITLE    = "RAG Intelligent Chatbot"
APP_ICON     = "🤖"
APP_SUBTITLE = "Ask questions about your uploaded documents"