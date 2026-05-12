"""
conversation_memory.py
──────────────────────
Phase 2 — conversation history management.

Keeps a rolling window of the last MAX_HISTORY_TURNS user/assistant
pairs so follow-up questions like "explain that more simply" work
correctly without the LLM losing context.

Memory is in-process only — intentionally not persisted to disk.
On PDF swap, memory should be cleared via reset().
"""

from dataclasses import dataclass, field
from config import MAX_HISTORY_TURNS


@dataclass
class ConversationMemory:
    """
    Manages a sliding window of conversation turns.

    Each turn is a dict with "role" ("user" or "assistant") and "content".
    Older turns are dropped when the window exceeds MAX_HISTORY_TURNS pairs.
    """
    _history: list[dict] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        """Record a user message."""
        self._history.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        """Record an assistant response."""
        self._history.append({"role": "assistant", "content": content})
        self._trim()

    def get_history(self) -> list[dict]:
        """
        Return the current history as a list of role/content dicts,
        ready to be inserted into the OpenAI messages array.
        """
        return list(self._history)

    def reset(self) -> None:
        """
        Clear all history.
        Call this on PDF swap so old context doesn't bleed into new sessions.
        """
        self._history.clear()
        print("  [Memory] Conversation history cleared")

    def turn_count(self) -> int:
        """Return number of complete user/assistant turn pairs."""
        return len(self._history) // 2

    def _trim(self) -> None:
        """
        Keep only the last MAX_HISTORY_TURNS pairs (pairs = 2 messages each).
        Drops oldest messages first.
        """
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]
