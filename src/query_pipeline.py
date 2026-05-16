"""
query_pipeline.py
─────────────────
Phase 2 orchestrator — ties together all query handling steps.
Supports broad queries (compile, summarize, timeline) via multi-query retrieval.
"""

import json
import anthropic

from .vector_store import similarity_search
from .embedder import embed_query
from .query_refiner import refine_query
from .reranker import rerank_chunks
from .web_fallback import should_use_fallback, fetch_web_context, build_fallback_context
from .context_builder import build_context
from .llm_generator import generate_response
from .conversation_memory import ConversationMemory
from .config import ANTHROPIC_API_KEY, CLAUDE_HAIKU_MODEL, RERANK_TOP_N

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

_DECOMPOSE_PROMPT = """You are a query decomposition assistant for a study revision tool.
If the user query is broad — asks to compile, summarize, list events, create timeline,
or covers multiple chapters/units — break it into specific focused sub-queries, one per unit or topic.
If the query is already specific and focused, return it as a single item list.
Return ONLY a valid JSON array of query strings. No preamble, no markdown fences.

Example input: "compile important events from unit 1 to 6"
Example output: ["important events unit 1", "important events unit 2", "important events unit 3", "important events unit 4", "important events unit 5", "important events unit 6"]

Example input: "What is the Sangam age?"
Example output: ["What is the Sangam age?"]"""


class QueryPipeline:

    def __init__(self, index):
        self.index  = index
        self.memory = ConversationMemory()

    def run(self, raw_query: str) -> tuple[str, list[dict], bool]:
        print(f"\n{'='*55}")
        print(f"  PHASE 2 — QUERY PIPELINE")
        print(f"  Raw query: {raw_query}")
        print(f"{'='*55}\n")

        # ── Step 1 & 2: Refine + expand query ─────────────────────────────
        print("[Step 1/7] Refining query...")
        refined_query, expanded_terms = refine_query(raw_query)

        search_query = refined_query
        if expanded_terms:
            search_query = refined_query + " " + " ".join(expanded_terms[:2])

        # ── Step 2.5: Decompose if broad query ────────────────────────────
        print("\n[Step 2/7] Checking if broad query...")
        sub_queries = self._decompose_query(refined_query)
        is_broad    = len(sub_queries) > 1

        # ── Step 3: Embed + Vector search ─────────────────────────────────
        print("\n[Step 3/7] Fetching candidates from Pinecone...")
        if is_broad:
            print(f"  [Pipeline] Broad query — decomposed into {len(sub_queries)} sub-queries")
            candidates = self._multi_retrieve(sub_queries)
        else:
            query_vector = embed_query(search_query)
            candidates   = similarity_search(self.index, query_vector, top_k=20)

        print(f"  [VectorStore] {len(candidates)} candidates retrieved")

        # ── Step 4: Rerank ─────────────────────────────────────────────────
        print("\n[Step 4/7] Reranking candidates...")
        # Use higher top_n for broad queries so LLM gets more coverage
        rerank_top_n = min(len(candidates), 20) if is_broad else RERANK_TOP_N
        top_chunks   = rerank_chunks(refined_query, candidates, top_n=rerank_top_n)

        # ── Step 5 & 6: Web fallback check ────────────────────────────────
        print("\n[Step 5/7] Checking relevance threshold...")
        web_context = ""
        if should_use_fallback(top_chunks):
            web_text    = fetch_web_context(refined_query)
            web_context = build_fallback_context(web_text)

        # ── Step 7: Build context block ────────────────────────────────────
        print("\n[Step 6/7] Building context block...")
        built_context = build_context(top_chunks, web_context)

        # ── Step 8: Generate response ──────────────────────────────────────
        print("\n[Step 7/7] Generating response...\n")
        history  = self.memory.get_history()
        response = generate_response(refined_query, built_context, history, is_broad)

        # ── Step 9: Update memory ──────────────────────────────────────────
        self.memory.add_user_message(raw_query)
        self.memory.add_assistant_message(response)

        return response, built_context.sources, built_context.has_web_content

    def reset_memory(self) -> None:
        self.memory.reset()

    # ── Private helpers ────────────────────────────────────────────────────

    def _decompose_query(self, refined_query: str) -> list[str]:
        """
        Ask Claude Haiku to break a broad query into focused sub-queries.
        Returns a single-item list if query is already specific.
        """
        response = _client.messages.create(
            model=CLAUDE_HAIKU_MODEL,
            max_tokens=300,
            temperature=0,
            system=_DECOMPOSE_PROMPT,
            messages=[{"role": "user", "content": refined_query}],
        )
        try:
            sub_queries = json.loads(response.content[0].text.strip())
            if not isinstance(sub_queries, list) or not sub_queries:
                return [refined_query]
            return sub_queries
        except (json.JSONDecodeError, IndexError):
            print("  [Pipeline] Decompose parse failed — treating as single query")
            return [refined_query]

    def _multi_retrieve(self, sub_queries: list[str]) -> list[dict]:
        """
        Embed and retrieve chunks for each sub-query independently.
        Deduplicates by chunk ID so the same chunk isn't passed twice.
        """
        seen_ids   = set()
        all_chunks = []

        for i, q in enumerate(sub_queries, 1):
            print(f"  [Pipeline] Sub-query {i}/{len(sub_queries)}: '{q}'")
            vector     = embed_query(q)
            candidates = similarity_search(self.index, vector, top_k=15)
            for c in candidates:
                if c["id"] not in seen_ids:
                    seen_ids.add(c["id"])
                    all_chunks.append(c)

        return all_chunks