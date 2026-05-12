"""
ingestion.py
────────────
Phase 1 orchestrator — ties together parsing, chunking,
embedding, and vector store upsert.

Usage:
    python ingestion.py                            # uses PDFs in ./pdfs/
    python ingestion.py path/to/a.pdf path/to/b.pdf path/to/c.pdf
"""

import os
import sys
import time

from .pdf_parser import extract_pdf
from .chunker import chunk_pages
from .embedder import embed_chunks
from .vector_store import get_or_create_index, clear_namespace, upsert_vectors, get_index_stats

# Default folder to look for PDFs if no paths are provided
DEFAULT_PDF_DIR = "./pdfs"


def run_ingestion(pdf_paths: list[str]) -> None:
    """
    Full Phase 1 pipeline for a list of PDF file paths.

    Steps:
        1. Validate input files
        2. Get / create Pinecone index
        3. Clear existing namespace (clean slate for this session)
        4. For each PDF: extract → clean → chunk
        5. Embed all chunks in batches
        6. Upsert all vectors to Pinecone
        7. Print summary stats

    Args:
        pdf_paths: List of paths to PDF files (max 3 recommended).
    """
    start_time = time.time()

    # ── 1. Validate ────────────────────────────────────────────────────────
    pdf_paths = _validate_pdfs(pdf_paths)
    print(f"\n{'='*55}")
    print(f"  PHASE 1 — INGESTION")
    print(f"  PDFs to process: {len(pdf_paths)}")
    for p in pdf_paths:
        size_mb = os.path.getsize(p) / (1024 * 1024)
        print(f"    • {os.path.basename(p)}  ({size_mb:.1f} MB)")
    print(f"{'='*55}\n")

    # ── 2. Pinecone setup ──────────────────────────────────────────────────
    print("[Step 1/5] Connecting to Pinecone...")
    index = get_or_create_index()

    # ── 3. Clear stale vectors ─────────────────────────────────────────────
    print("\n[Step 2/5] Clearing previous session vectors...")
    clear_namespace(index)

    # ── 4. Parse and chunk all PDFs ────────────────────────────────────────
    print("\n[Step 3/5] Extracting and chunking PDFs...")
    all_chunks = []

    for pdf_path in pdf_paths:
        pdf_name = os.path.basename(pdf_path)
        print(f"\n  ► Processing: {pdf_name}")

        pages = extract_pdf(pdf_path)
        chunks = chunk_pages(pages)
        all_chunks.extend(chunks)

    print(f"\n  Total chunks across all PDFs: {len(all_chunks)}")

    # ── 5. Embed ───────────────────────────────────────────────────────────
    print("\n[Step 4/5] Generating embeddings...")
    vectors = embed_chunks(all_chunks)

    # ── 6. Upsert to Pinecone ──────────────────────────────────────────────
    print("\n[Step 5/5] Uploading vectors to Pinecone...")
    upsert_vectors(index, vectors)

    # ── 7. Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    stats = get_index_stats(index)

    print(f"\n{'='*55}")
    print(f"  INGESTION COMPLETE")
    print(f"  Time taken       : {elapsed:.1f}s")
    print(f"  Chunks created   : {len(all_chunks)}")
    print(f"  Vectors stored   : {len(vectors)}")
    print(f"  Pinecone stats   : {stats}")
    print(f"{'='*55}\n")


def _validate_pdfs(pdf_paths: list[str]) -> list[str]:
    """
    Check that each path exists and is a PDF.
    Raises FileNotFoundError if any path is invalid.
    """
    validated = []
    for path in pdf_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF not found: {path}")
        if not path.lower().endswith(".pdf"):
            raise ValueError(f"File is not a PDF: {path}")
        validated.append(path)
    return validated


def _get_pdfs_from_dir(directory: str) -> list[str]:
    """Find all PDFs in a directory (non-recursive)."""
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Directory not found: {directory}")
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".pdf")
    ]


# ─── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # PDF paths passed as command-line arguments
        paths = sys.argv[1:]
    else:
        # Fall back to default ./pdfs/ directory
        print(f"No paths given — scanning '{DEFAULT_PDF_DIR}/' for PDFs...")
        paths = _get_pdfs_from_dir(DEFAULT_PDF_DIR)

    if not paths:
        print("No PDF files found. Provide paths as arguments or place PDFs in ./pdfs/")
        sys.exit(1)

    run_ingestion(paths)
