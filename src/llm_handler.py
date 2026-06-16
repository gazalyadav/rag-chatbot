# src/llm_handler.py
import ollama
from typing import Generator
from src.config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS
)


def check_ollama_connection() -> bool:
    """Verify Ollama is running and the model is available."""
    try:
        response  = ollama.list()

        # Handle both old (object) and new (dict) Ollama API formats
        if isinstance(response, dict):
            available = [m.get("name", m.get("model", ""))
                         for m in response.get("models", [])]
        else:
            available = [m.model for m in response.models]

        model_available = any(OLLAMA_MODEL in m for m in available)

        if not model_available:
            print(f"  ⚠️  Model '{OLLAMA_MODEL}' not found.")
            print(f"     Run: ollama pull {OLLAMA_MODEL}")
            print(f"     Available: {available}")
            return False

        print(f"  ✅ Ollama connected | Model: {OLLAMA_MODEL}")
        return True

    except Exception as e:
        print(f"  ❌ Ollama not running: {e}")
        print(f"     Start it with: ollama serve")
        return False


def generate_response(prompt: str) -> str:
    """
    Generate a complete response from the LLM.
    Blocks until the full response is ready.
    """
    try:
        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={
                "temperature" : LLM_TEMPERATURE,
                "num_predict" : LLM_MAX_TOKENS,
            }
        )

        # Handle both old (object) and new (dict) Ollama API formats
        if isinstance(response, dict):
            return response.get("response", "")
        return response.response

    except ollama.ResponseError as e:
        return f"❌ LLM Error: {e}. Is Ollama running? Try: ollama serve"
    except Exception as e:
        return f"❌ Unexpected error: {e}"


def generate_streaming(prompt: str) -> Generator[str, None, None]:
    """
    Generate a streaming response — yields tokens as they are produced.
    Use this in Streamlit for the real-time typing effect.
    """
    try:
        stream = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            stream=True,
            options={
                "temperature" : LLM_TEMPERATURE,
                "num_predict" : LLM_MAX_TOKENS,
            }
        )

        for chunk in stream:
            # Handle both old (object) and new (dict) Ollama API formats
            if isinstance(chunk, dict):
                token = chunk.get("response", "")
            else:
                token = chunk.response

            if token:
                yield token

    except ollama.ResponseError as e:
        yield f"❌ LLM Error: {e}. Is Ollama running? Try: ollama serve"
    except Exception as e:
        yield f"❌ Unexpected error: {e}"