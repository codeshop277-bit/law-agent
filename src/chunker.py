"""
chunker.py
──────────
Splits cleaned page text into overlapping token-aware chunks.
Respects section boundaries so each chunk stays contextually coherent.
"""

import re
from dataclasses import dataclass
from typing import Optional

from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH
from pdf_parser import PageContent


@dataclass
class TextChunk:
    """A single chunk ready for embedding."""
    chunk_id: str          # unique ID: "{pdf_name}_p{page}_c{index}"
    pdf_name: str
    page_number: int
    section_heading: Optional[str]
    text: str
    token_count: int


def chunk_pages(pages: list[PageContent]) -> list[TextChunk]:
    """
    Convert a list of PageContent objects into TextChunk objects.

    Strategy:
    - Each page's cleaned text is split into chunks of ~CHUNK_SIZE tokens.
    - Consecutive chunks share CHUNK_OVERLAP tokens for context continuity.
    - Chunks shorter than MIN_CHUNK_LENGTH characters are discarded.

    Args:
        pages: List of PageContent from pdf_parser.extract_pdf()

    Returns:
        List of TextChunk objects ready for embedding.
    """
    all_chunks: list[TextChunk] = []

    for page in pages:
        if not page.cleaned_text:
            continue

        raw_chunks = _split_into_chunks(page.cleaned_text)

        for idx, (chunk_text, token_count) in enumerate(raw_chunks):
            if len(chunk_text) < MIN_CHUNK_LENGTH:
                continue  # Discard very short / noise chunks

            chunk = TextChunk(
                chunk_id=f"{page.pdf_name}_p{page.page_number}_c{idx}",
                pdf_name=page.pdf_name,
                page_number=page.page_number,
                section_heading=page.section_heading,
                text=chunk_text.strip(),
                token_count=token_count,
            )
            all_chunks.append(chunk)

    print(f"  [Chunker] {len(pages)} pages → {len(all_chunks)} chunks")
    return all_chunks


def _split_into_chunks(text: str) -> list[tuple[str, int]]:
    """
    Split text into (chunk_text, token_count) pairs using a sliding window.

    Uses a simple whitespace tokeniser — close enough for chunk sizing
    without the overhead of a full tokeniser library.
    Overlap is applied by carrying forward the last CHUNK_OVERLAP tokens
    into the next chunk.
    """
    words = text.split()  # word-level approximation; ~1.3 words per token
    # Convert token targets to approximate word counts
    chunk_word_size = int(CHUNK_SIZE * 0.75)      # ~words in CHUNK_SIZE tokens
    overlap_word_size = int(CHUNK_OVERLAP * 0.75)

    chunks: list[tuple[str, int]] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_word_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        token_estimate = int(len(chunk_words) / 0.75)
        chunks.append((chunk_text, token_estimate))

        if end == len(words):
            break

        # Move forward by chunk size minus overlap
        start += chunk_word_size - overlap_word_size

    return chunks
