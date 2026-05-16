"""
reranker.py
───────────
Uses Claude Haiku — scores all 20 candidates in one call.
"""

import json
import anthropic
from .config import ANTHROPIC_API_KEY, CLAUDE_HAIKU_MODEL, RERANK_TOP_N

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

_RERANK_PROMPT = """You are a relevance scoring assistant for a study revision tool.
Score each chunk for relevance to the query on a scale of 0.0 to 1.0.

  1.0   — directly and fully answers the query
  0.7-0.9 — highly relevant, covers the topic or contains key facts about it
  0.4-0.6 — partially relevant, mentions related concepts or events
  0.0-0.3 — not relevant

IMPORTANT: If the query is a broad compile, timeline, or summary request across
multiple units or chapters, score ANY chunk that contains historical events,
dates, facts, or topic content from those units as 0.7 or above.
Do not penalise chunks for being partial — they are meant to be combined.

Return ONLY a valid JSON array with "index" and "score" fields. No preamble, no markdown.

Example: [{"index": 0, "score": 0.95}, {"index": 1, "score": 0.3}]"""


def rerank_chunks(query: str, candidates: list[dict],  top_n: int = RERANK_TOP_N) -> list[dict]:
    if not candidates:
        return []

    chunks_input = [
        {"index": i, "text": m.get("metadata", {}).get("text", "")[:600]}
        for i, m in enumerate(candidates)
    ]

    prompt = f"Query: {query}\n\nChunks:\n{json.dumps(chunks_input, indent=2)}"

    response = _client.messages.create(
        model=CLAUDE_HAIKU_MODEL,
        max_tokens=500,
        temperature=0,
        system=_RERANK_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_output = response.content[0].text.strip()

    try:
        scores    = json.loads(raw_output)
        score_map = {item["index"]: item["score"] for item in scores}
    except (json.JSONDecodeError, KeyError):
        print("  [Reranker] Parse failed — falling back to Pinecone order")
        score_map = {i: 1.0 - (i * 0.05) for i in range(len(candidates))}

    scored = []
    for idx, match in enumerate(candidates):
        enriched = {
        "id":           match["id"],
        "score":        match["score"],
        "metadata":     match["metadata"],
        "rerank_score": score_map.get(idx, 0.0),
        }
        scored.append(enriched)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    top_chunks = scored[:top_n]

    print(f"  [Reranker] {len(candidates)} candidates → top {len(top_chunks)} selected")
    for c in top_chunks:
        meta = c.get("metadata", {})
        print(f"    score={c['rerank_score']:.2f}  [{meta.get('pdf_name')} p{meta.get('page_number')}] {meta.get('section_heading', '')}")

    return top_chunks