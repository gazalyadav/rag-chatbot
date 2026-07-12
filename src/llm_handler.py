# src/llm_handler.py
"""
LLM Handler — supports both Groq (cloud) and Ollama (local).
Groq is used for deployment; Ollama for local development.
"""

import os
from typing import Generator
from src.config import LLM_TEMPERATURE, LLM_MAX_TOKENS, OLLAMA_MODEL

# ── Detect which backend to use ───────────────────────────────────────────────
# If GROQ_API_KEY is set in environment → use Groq (cloud)
# Otherwise → fall back to Ollama (local)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
USE_GROQ     = bool(GROQ_API_KEY)
GROQ_MODEL   = "llama-3.1-8b-instant"   # Llama 3 8B on Groq — free tier


def check_ollama_connection() -> bool:
    """Check LLM backend connection — Groq or Ollama."""

    if USE_GROQ:
        # Verify Groq API key works
        try:
            from groq import Groq
            client   = Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model    = GROQ_MODEL,
                messages = [{"role": "user", "content": "Hi"}],
                max_tokens = 5
            )
            print(f"  ✅ Groq connected | Model: {GROQ_MODEL}")
            return True
        except Exception as e:
            print(f"  ❌ Groq connection failed: {e}")
            return False
    else:
        # Fall back to Ollama
        try:
            import ollama
            response  = ollama.list()
            if isinstance(response, dict):
                available = [m.get("name", m.get("model", ""))
                             for m in response.get("models", [])]
            else:
                available = [m.model for m in response.models]

            model_available = any(OLLAMA_MODEL in m for m in available)
            if not model_available:
                print(f"  ⚠️  Model '{OLLAMA_MODEL}' not found.")
                return False

            print(f"  ✅ Ollama connected | Model: {OLLAMA_MODEL}")
            return True
        except Exception as e:
            print(f"  ❌ Ollama not running: {e}")
            return False


def generate_response(prompt: str) -> str:
    """Generate a complete response from Groq or Ollama."""

    if USE_GROQ:
        try:
            from groq import Groq
            client   = Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model    = GROQ_MODEL,
                messages = [{"role": "user", "content": prompt}],
                max_tokens  = LLM_MAX_TOKENS,
                temperature = LLM_TEMPERATURE,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ Groq error: {e}"
    else:
        try:
            import ollama
            response = ollama.generate(
                model  = OLLAMA_MODEL,
                prompt = prompt,
                options = {
                    "temperature" : LLM_TEMPERATURE,
                    "num_predict" : LLM_MAX_TOKENS,
                }
            )
            if isinstance(response, dict):
                return response.get("response", "")
            return response.response
        except Exception as e:
            return f"❌ Ollama error: {e}"


def generate_streaming(prompt: str) -> Generator[str, None, None]:
    """Generate a streaming response from Groq or Ollama."""

    if USE_GROQ:
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            stream = client.chat.completions.create(
                model    = GROQ_MODEL,
                messages = [{"role": "user", "content": prompt}],
                max_tokens  = LLM_MAX_TOKENS,
                temperature = LLM_TEMPERATURE,
                stream      = True,
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    yield token
        except Exception as e:
            yield f"❌ Groq error: {e}"
    else:
        try:
            import ollama
            stream = ollama.generate(
                model  = OLLAMA_MODEL,
                prompt = prompt,
                stream = True,
                options = {
                    "temperature" : LLM_TEMPERATURE,
                    "num_predict" : LLM_MAX_TOKENS,
                }
            )
            for chunk in stream:
                if isinstance(chunk, dict):
                    token = chunk.get("response", "")
                else:
                    token = chunk.response
                if token:
                    yield token
        except Exception as e:
            yield f"❌ Ollama error: {e}"