# src/memory_manager.py
"""
Memory Manager — maintains multi-turn conversation history.
Uses a sliding window to prevent context window overflow.
Stores messages as simple dicts compatible with Streamlit session_state.
"""

from typing import List, Optional
from src.config import MEMORY_WINDOW_SIZE
from src.prompt_builder import format_chat_history


class ConversationMemory:
    """
    Manages conversation history for multi-turn RAG interactions.

    Stores the last N turns (user + assistant pairs) to maintain
    context across follow-up questions without overflowing the LLM's
    context window.
    """

    def __init__(self, window_size: int = MEMORY_WINDOW_SIZE):
        """
        Args:
            window_size: Number of conversation TURNS to remember.
                         1 turn = 1 user message + 1 assistant response.
        """
        self.window_size : int        = window_size
        self.messages    : List[dict] = []   # full history for UI display
        self._turn_count : int        = 0

    def add_user_message(self, content: str) -> None:
        """Add a user message to history."""
        self.messages.append({
            "role"    : "user",
            "content" : content
        })

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant response to history."""
        self.messages.append({
            "role"    : "assistant",
            "content" : content
        })
        self._turn_count += 1

    def get_window_messages(self) -> List[dict]:
        """
        Get the most recent N turns for the LLM prompt.
        Each turn = 1 user + 1 assistant message = 2 items.
        We exclude the most recent user message (it's the current query).
        """
        # window_size turns × 2 messages per turn
        window = self.window_size * 2

        # Exclude the last message (current user query, not yet answered)
        history = self.messages[:-1] if self.messages else []

        # Return only the most recent window
        return history[-window:] if len(history) > window else history

    def get_formatted_history(self) -> str:
        """
        Get conversation history formatted as a string for the LLM prompt.

        Returns:
            Formatted history string, or empty string if no history.
        """
        window_messages = self.get_window_messages()
        return format_chat_history(window_messages)

    def get_all_messages(self) -> List[dict]:
        """Get the complete message history for UI display."""
        return self.messages

    def clear(self) -> None:
        """Reset the conversation history."""
        self.messages    = []
        self._turn_count = 0
        print("  🗑️  Conversation memory cleared.")

    def is_empty(self) -> bool:
        """Check if there is any conversation history."""
        return len(self.messages) == 0

    @property
    def turn_count(self) -> int:
        """Number of complete turns (user + assistant pairs)."""
        return self._turn_count

    def __repr__(self) -> str:
        return (
            f"ConversationMemory("
            f"turns={self._turn_count}, "
            f"messages={len(self.messages)}, "
            f"window={self.window_size})"
        )