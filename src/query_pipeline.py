"""
query_pipeline.py
─────────────────
Phase 2 orchestrator — ties together all query handling steps:

  1.  Receive raw user query
  2.  Grammar fix + query expansion     (query_refiner)
  3.  Embed refined query               (embedder)
  4.  Fetch top-20 candidates           (vector_store)
  5.  Rerank → top-5 chunks             (reranker)
  6.  Relevance threshold check         (web_fallback)
  7.  Web fallback if needed            (web_fallback)
  8.  Build context block               (context_builder)
  9.  Generate LLM response             (llm_generator)
  10. Update conversation memory        (conversation_memory)

Usage:
    from query_pipeline import QueryPipeline
    from vector_store import get_or_create_index

    index    = get_or_create_index()
    pipeline = QueryPipeline(index)

    response = pipeline.run("What is the TCP 3-way handshake?")
    response = pipeline.run("Can you simplify that?")   # follow-up works

    pipeline.reset_memory()   # call this on PDF swap
"""

from urllib import response

from vector_store import get_or_create_index, similarity_search
from embedder import embed_query
from query_refiner import refine_query
from reranker import rerank_chunks
from web_fallback import should_use_fallback, fetch_web_context, build_fallback_context
from context_builder import build_context
from llm_generator import generate_response
from conversation_memory import ConversationMemory


class QueryPipeline:
    """
    Stateful Phase 2 pipeline.
    Holds the Pinecone index and conversation memory across turns.
    """

    def __init__(self, index):
        """
        Args:
            index: Pinecone Index object from vector_store.get_or_create_index()
        """
        self.index  = index
        self.memory = ConversationMemory()

    def run(self, raw_query: str) -> str:
        """
        Process a user query end-to-end and return the LLM response.

        Args:
            raw_query: The raw question typed by the user.

        Returns:
            LLM response string (already printed via streaming during generation).
        """
        print(f"\n{'='*55}")
        print(f"  PHASE 2 — QUERY PIPELINE")
        print(f"  Raw query: {raw_query}")
        print(f"{'='*55}\n")

        # ── Step 1 & 2: Refine + expand query ─────────────────────────────
        print("[Step 1/7] Refining query...")
        refined_query, expanded_terms = refine_query(raw_query)

        # Build a combined search query using refined + top expanded terms
        # Joining gives the embedding more signal for retrieval
        search_query = refined_query
        if expanded_terms:
            search_query = refined_query + " " + " ".join(expanded_terms[:2])

        # ── Step 3: Embed ──────────────────────────────────────────────────
        print("\n[Step 2/7] Embedding query...")
        query_vector = embed_query(search_query)

        # ── Step 4: Vector search ──────────────────────────────────────────
        print("\n[Step 3/7] Fetching candidates from Pinecone...")
        candidates = similarity_search(self.index, query_vector, top_k=20)
        print(f"  [VectorStore] {len(candidates)} candidates retrieved")

        # ── Step 5: Rerank ─────────────────────────────────────────────────
        print("\n[Step 4/7] Reranking candidates...")
        top_chunks = rerank_chunks(refined_query, candidates)

        # ── Step 6 & 7: Web fallback check ────────────────────────────────
        print("\n[Step 5/7] Checking relevance threshold...")
        web_context = ""
        if should_use_fallback(top_chunks):
            web_text    = fetch_web_context(refined_query)
            web_context = build_fallback_context(web_text)

        # ── Step 8: Build context block ────────────────────────────────────
        print("\n[Step 6/7] Building context block...")
        built_context = build_context(top_chunks, web_context)

        # ── Step 9: Generate response ──────────────────────────────────────
        print("\n[Step 7/7] Generating response...\n")
        history  = self.memory.get_history()
        response = generate_response(refined_query, built_context, history)

        # ── Step 10: Update memory ─────────────────────────────────────────
        self.memory.add_user_message(raw_query)
        self.memory.add_assistant_message(response)

        return response, built_context.sources, built_context.has_web_content

    def reset_memory(self) -> None:
        """
        Clear conversation history.
        Call this whenever the user swaps PDFs for a new study session.
        """
        self.memory.reset()
