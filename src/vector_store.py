"""
vector_store.py
───────────────
Manages all Pinecone interactions:
- Index creation (if not exists)
- Namespace clearing on PDF swap
- Batch upsert of vectors
- Similarity search (used in Phase 2)
"""

import time
from pinecone import Pinecone, ServerlessSpec

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_DIMENSION,
    PINECONE_METRIC,
    EMBEDDING_BATCH_SIZE,
)


# Initialise Pinecone client once
_pc = Pinecone(api_key=PINECONE_API_KEY)


def get_or_create_index():
    """
    Get existing Pinecone index or create it if it doesn't exist.
    Uses serverless spec (free tier compatible).

    Returns:
        Pinecone Index object.
    """
    existing_indexes = [idx.name for idx in _pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"  [VectorStore] Creating index '{PINECONE_INDEX_NAME}'...")
        _pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",  # Free tier supported region
            ),
        )
        # Wait until the index is ready
        _wait_for_index_ready()
        print(f"  [VectorStore] Index '{PINECONE_INDEX_NAME}' created")
    else:
        print(f"  [VectorStore] Using existing index '{PINECONE_INDEX_NAME}'")

    return _pc.Index(PINECONE_INDEX_NAME)


def clear_namespace(index) -> None:
    """
    Delete all vectors in the current namespace.
    Called at the start of every PDF swap to avoid stale data.

    Args:
        index: Pinecone Index object from get_or_create_index()
    """
    try:
        index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
        print(f"  [VectorStore] Namespace '{PINECONE_NAMESPACE}' cleared")
        time.sleep(1)  # brief pause to let deletion propagate
    except Exception as e:
        # Namespace may not exist on first run — that's fine
        print(f"  [VectorStore] Namespace clear skipped (likely first run): {e}")


def upsert_vectors(index, vectors: list[dict]) -> None:
    """
    Upload vectors to Pinecone in batches.

    Args:
        index:   Pinecone Index object.
        vectors: List of dicts from embedder.embed_chunks(), each containing
                 {"id", "values", "metadata"}.
    """
    total = len(vectors)
    total_batches = max(1, -(-total // EMBEDDING_BATCH_SIZE))

    for batch_num, batch in enumerate(_batch(vectors, EMBEDDING_BATCH_SIZE), start=1):
        print(f"  [VectorStore] Upserting batch {batch_num}/{total_batches} — {len(batch)} vectors")
        index.upsert(vectors=batch, namespace=PINECONE_NAMESPACE)

    print(f"  [VectorStore] Upsert complete — {total} vectors stored")


def similarity_search(index, query_vector: list[float], top_k: int = 20) -> list[dict]:
    """
    Fetch top-K most similar vectors for a query.
    Used in Phase 2 (query handling) before reranking.

    Args:
        index:        Pinecone Index object.
        query_vector: Embedded query from embedder.embed_query()
        top_k:        Number of candidates to retrieve (default 20 for reranking).

    Returns:
        List of match dicts with keys: id, score, metadata.
    """
    result = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=PINECONE_NAMESPACE,
        include_metadata=True,
    )
    return result.get("matches", [])


def get_index_stats(index) -> dict:
    """Return index stats — useful for debugging vector counts."""
    return index.describe_index_stats()


# ─── Private helpers ────────────────────────────────────────────────────────

def _batch(items: list, size: int):
    """Yield successive slices of `items` of length `size`."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _wait_for_index_ready(max_wait_seconds: int = 60) -> None:
    """Poll until the index is ready to accept requests."""
    start = time.time()
    while time.time() - start < max_wait_seconds:
        indexes = _pc.list_indexes()
        for idx in indexes:
            if idx.name == PINECONE_INDEX_NAME and idx.status.get("ready"):
                return
        time.sleep(2)
    raise TimeoutError(f"Index '{PINECONE_INDEX_NAME}' not ready after {max_wait_seconds}s")
