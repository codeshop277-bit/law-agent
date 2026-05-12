"""
web_fallback.py
───────────────
Uses DuckDuckGo (free, no API key) for web search.
Claude Haiku summarises the raw results into clean context.
"""

import anthropic
from duckduckgo_search import DDGS
from .config import ANTHROPIC_API_KEY, CLAUDE_HAIKU_MODEL, RELEVANCE_THRESHOLD, WEB_FALLBACK_MAX_RESULTS

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def should_use_fallback(top_chunks: list[dict]) -> bool:
    if not top_chunks:
        print("  [WebFallback] No chunks retrieved — triggering fallback")
        return True

    best_score = top_chunks[0].get("rerank_score", 0.0)
    trigger    = best_score < RELEVANCE_THRESHOLD

    if trigger:
        print(f"  [WebFallback] Best score {best_score:.2f} < {RELEVANCE_THRESHOLD} — triggering fallback")
    else:
        print(f"  [WebFallback] Best score {best_score:.2f} ≥ threshold — PDF context sufficient")

    return trigger


def fetch_web_context(query: str) -> str:
    """
    1. DuckDuckGo fetches raw search results (free, no key needed).
    2. Claude Haiku summarises them into clean, concise study context.
    """
    print(f"  [WebFallback] Searching DuckDuckGo for: '{query}'")

    # ── Step 1: Fetch raw results via DuckDuckGo ──────────────────────────
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=WEB_FALLBACK_MAX_RESULTS))
    except Exception as e:
        print(f"  [WebFallback] DuckDuckGo search failed: {e}")
        return ""

    if not results:
        print("  [WebFallback] No results found")
        return ""

    # Format raw results for Claude to summarise
    raw_text = ""
    for i, r in enumerate(results, 1):
        raw_text += f"Result {i}: {r.get('title', '')}\n{r.get('body', '')}\n\n"

    # ── Step 2: Claude Haiku summarises into study-friendly context ────────
    response = _client.messages.create(
        model=CLAUDE_HAIKU_MODEL,
        max_tokens=500,
        system=(
            "You are a study assistant. Summarise the following web search results "
            "into concise, accurate, student-friendly bullet points relevant to the query. "
            "Include only factual content. Do not add anything not present in the results."
        ),
        messages=[{
            "role": "user",
            "content": f"Query: {query}\n\nSearch results:\n{raw_text}"
        }],
    )

    summary = response.content[0].text.strip()
    print(f"  [WebFallback] Web context summarised ({len(summary)} chars)")
    return summary


def build_fallback_context(web_text: str) -> str:
    if not web_text:
        return ""
    return (
        "\n\n--- SUPPLEMENTARY WEB CONTEXT ---\n"
        "(This information is from the web, not from your study PDFs.)\n\n"
        f"{web_text}\n"
        "--- END WEB CONTEXT ---\n"
    )