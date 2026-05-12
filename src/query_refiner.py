"""
query_refiner.py
────────────────
Uses Claude Haiku — fast and cheap for this lightweight task.
"""

import json
import anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_HAIKU_MODEL

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

_REFINE_PROMPT = """You are a query preprocessing assistant for a study revision tool.
Given a raw user question, return a JSON object with two fields:

1. "refined_query"  — grammar and spelling corrected. Keep meaning identical.
2. "expanded_terms" — list of 3-5 synonyms or alternate phrasings for better search recall.

Respond ONLY with valid JSON. No preamble, no markdown fences.

Example input:  "wat is the difrnce between tcp and udp protcol"
Example output:
{
  "refined_query": "What is the difference between TCP and UDP protocols?",
  "expanded_terms": ["TCP vs UDP", "Transmission Control Protocol", "User Datagram Protocol", "transport layer protocols"]
}"""


def refine_query(raw_query: str) -> tuple[str, list[str]]:
    response = _client.messages.create(
        model=CLAUDE_HAIKU_MODEL,
        max_tokens=300,
        temperature=0,
        system=_REFINE_PROMPT,
        messages=[{"role": "user", "content": raw_query}],
    )

    raw_output = response.content[0].text.strip()

    try:
        parsed        = json.loads(raw_output)
        refined_query = parsed.get("refined_query", raw_query)
        expanded      = parsed.get("expanded_terms", [])
    except json.JSONDecodeError:
        print("  [QueryRefiner] JSON parse failed — using raw query")
        refined_query = raw_query
        expanded      = []

    print(f"  [QueryRefiner] Refined : {refined_query}")
    print(f"  [QueryRefiner] Expanded: {expanded}")
    return refined_query, expanded