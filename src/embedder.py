"""
embedder.py
───────────
Handles embedding generation via OpenAI's text-embedding-3-small model.
Processes chunks in batches to stay within API rate limits.
"""

import time
from openai import OpenAI

from config import OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE
from chunker import TextChunk


# Initialise the OpenAI client once
_client = OpenAI(api_key=OPENAI_API_KEY)


def embed_chunks(chunks: list[TextChunk]) -> list[dict]:
    """
    Generate embeddings for all chunks in batches.

    Args:
        chunks: List of TextChunk objects from chunker.chunk_pages()

    Returns:
        List of dicts ready for Pinecone upsert, each containing:
        {
            "id":       chunk_id,
            "values":   [1536-dimensional float vector],
            "metadata": { pdf_name, page_number, section_heading, text }
        }
    """
    vectors = []
    total_batches = _total_batches(len(chunks))

    for batch_num, batch in enumerate(_batch(chunks, EMBEDDING_BATCH_SIZE), start=1):
        print(f"  [Embedder] Batch {batch_num}/{total_batches} — {len(batch)} chunks")

        texts = [chunk.text for chunk in batch]
        embeddings = _get_embeddings(texts)

        for chunk, embedding in zip(batch, embeddings):
            vectors.append({
                "id": chunk.chunk_id,
                "values": embedding,
                "metadata": {
                    "pdf_name": chunk.pdf_name,
                    "page_number": chunk.page_number,
                    "section_heading": chunk.section_heading or "Unknown",
                    "text": chunk.text,              # stored for retrieval context
                    "token_count": chunk.token_count,
                },
            })

        # Brief pause between batches to respect rate limits
        if batch_num < total_batches:
            time.sleep(0.3)

    print(f"  [Embedder] Done — {len(vectors)} vectors generated")
    return vectors


def embed_query(query: str) -> list[float]:
    """
    Embed a single user query using the same model as ingestion.
    Called during Phase 2 (query handling).

    Args:
        query: The refined user query string.

    Returns:
        1536-dimensional float vector.
    """
    result = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    return result.data[0].embedding


# ─── Private helpers ────────────────────────────────────────────────────────

def _get_embeddings(texts: list[str]) -> list[list[float]]:
    """Call OpenAI Embeddings API for a list of texts."""
    result = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    # API returns results sorted by index, so order is preserved
    return [item.embedding for item in sorted(result.data, key=lambda x: x.index)]


def _batch(items: list, size: int):
    """Yield successive slices of `items` of length `size`."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _total_batches(total_items: int) -> int:
    """Calculate total number of batches."""
    return max(1, -(-total_items // EMBEDDING_BATCH_SIZE))  # ceiling division
