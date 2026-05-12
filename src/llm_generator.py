"""
llm_generator.py
────────────────
Uses Claude Sonnet for final answer generation — best reasoning quality.
Conversation history is passed as the messages array directly.
"""

import anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_SONNET_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE
from context_builder import BuiltContext

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

_SYSTEM_PROMPT = """You are a focused study revision assistant. Your sole purpose is to help
the student understand and revise topics from their uploaded study materials.

Behaviour rules:
1. Answer ONLY from the provided context. Do not add information not present in the context.
2. Be CONCISE — this is revision, not a lecture. Bullet points and short paragraphs preferred.
3. Always cite your source at the end:  📄 Source: <PDF name>, Page <number>, Section: <section>
4. If multiple sources contributed, list all of them.
5. If web context was used, add:  🌐 Web supplementary content was used.
6. If the context lacks enough information, say so clearly. Do NOT make up an answer.
7. Keep responses under 300 words unless the topic genuinely requires more depth."""


def generate_response(
    refined_query: str,
    built_context: BuiltContext,
    history: list[dict],
) -> str:
    """
    Build the message array and call Claude Sonnet with streaming.

    Anthropic's messages API takes history as alternating user/assistant turns.
    The current query + context goes in as the final user message.
    """
    user_turn = (
        f"{built_context.context_text}\n\n"
        f"Question: {refined_query}"
    )

    # history is already in {"role": "user/assistant", "content": "..."} format
    messages = [
        *history,
        {"role": "user", "content": user_turn},
    ]

    print(f"  [LLM] Calling {CLAUDE_SONNET_MODEL} (history turns: {len(history) // 2})")
    print("\n" + "─" * 55)

    full_response = ""

    with _client.messages.stream(
        model=CLAUDE_SONNET_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        system=_SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_response += text

    print("\n" + "─" * 55 + "\n")
    return full_response