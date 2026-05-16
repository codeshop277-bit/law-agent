"""
llm_generator.py
────────────────
Uses Claude Sonnet for final answer generation — best reasoning quality.
Conversation history is passed as the messages array directly.
"""

import anthropic
from .config import ANTHROPIC_API_KEY, CLAUDE_SONNET_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_MAX_TOKENS_BROAD
from .context_builder import BuiltContext

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

_BROAD_INSTRUCTION = """
8. This is a broad compile/timeline/summary request.
   - Format the response as a numbered chronological list.
   - Group events by unit or chapter using the source metadata.
   - Cover all units mentioned in the query — do not stop at the first unit.
   - You may exceed 300 words for complete coverage across all units."""


def generate_response(
    refined_query: str,
    built_context: BuiltContext,
    history: list[dict],
    is_broad: bool = False,
) -> str:
    user_turn = (
        f"{built_context.context_text}\n\n"
        f"Question: {refined_query}"
    )

    messages = [
        *history,
        {"role": "user", "content": user_turn},
    ]

    # Append broad instruction to system prompt when needed
    system = _SYSTEM_PROMPT + (_BROAD_INSTRUCTION if is_broad else "")

    # Allow more tokens for broad queries so output isn't cut off mid-way
    max_tokens = LLM_MAX_TOKENS_BROAD if is_broad else LLM_MAX_TOKENS

    print(f"  [LLM] Calling {CLAUDE_SONNET_MODEL} (history turns: {len(history) // 2}, broad={is_broad})")
    print("\n" + "─" * 55)

    full_response = ""

    with _client.messages.stream(
        model=CLAUDE_SONNET_MODEL,
        max_tokens=max_tokens,
        temperature=LLM_TEMPERATURE,
        system=system,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_response += text

    print("\n" + "─" * 55 + "\n")
    return full_response