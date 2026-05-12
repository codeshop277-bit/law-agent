"""
embedder.py
───────────
Uses sentence-transformers locally — no API key, no cost, no rate limits.
Model downloads once (~90MB) and is cached at ~/.cache/torch/sentence_transformers/
"""

from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE
from .chunker import TextChunk

# Load once at import time — subsequent calls use the cached model
_model = SentenceTransformer(EMBEDDING_MODEL)


def embed_chunks(chunks: list[TextChunk]) -> list[dict]:
    """
    Embed all chunks locally in batches.
    sentence-transformers handles batching internally; we just pass all texts.
    """
    texts = [chunk.text for chunk in chunks]

    print(f"  [Embedder] Encoding {len(texts)} chunks locally...")
    # batch_size controls GPU/CPU memory usage — 64 is safe for most machines
    result = _model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
    )
    embeddings = result.tolist() 

    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id": chunk.chunk_id,
            "values": embedding,
            "metadata": {
                "pdf_name":        chunk.pdf_name,
                "page_number":     chunk.page_number,
                "section_heading": chunk.section_heading or "Unknown",
                "text":            chunk.text,
                "token_count":     chunk.token_count,
            },
        })

    print(f"  [Embedder] Done — {len(vectors)} vectors generated")
    return vectors


def embed_query(query: str) -> list[float]:
    """
    Embed a single query using the same local model.
    Must use the same model as ingestion — mixing models breaks similarity search.
    """
    result = _model.encode(query)
    embeddings = result.tolist() 
    return embeddings