"""
main.py
───────
FastAPI app entry point.
- On startup: ingests all PDFs from src/pdfs/ into Pinecone
- Mounts all routes from route.py
"""

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI

from .ingestion import run_ingestion, _get_pdfs_from_dir
from .vector_store import get_or_create_index
from .query_pipeline import QueryPipeline
from .route import router, init_router

PDF_DIR = "src/pdfs"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────
    print("\n[Startup] Initialising study revision agent...")

    pdf_paths = _get_pdfs_from_dir(PDF_DIR)
    if not pdf_paths:
        raise RuntimeError(f"No PDFs found in '{PDF_DIR}/'. Add PDFs and restart.")

    index = get_or_create_index()
    run_ingestion(pdf_paths)

    pipeline = QueryPipeline(index)
    init_router(pipeline)

    print("[Startup] Agent ready.\n")

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────
    print("[Shutdown] Cleaning up...")


app = FastAPI(
    title="Study Revision Agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)