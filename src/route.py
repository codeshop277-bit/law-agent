"""
route.py
────────
FastAPI routes for the study revision agent.
Two endpoints:
  POST /query   — runs the full Phase 2 pipeline and returns the answer
  POST /swap    — swaps PDFs and re-ingests (clears memory + Pinecone namespace)
  DELETE /clear — clears conversation memory only
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from .query_pipeline import QueryPipeline

router = APIRouter()


# ── Request / Response models ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    used_web_fallback: bool

class SwapRequest(BaseModel):
    pdf_paths: list[str]          # absolute or relative paths to new PDFs


# ── Pipeline instance (shared across requests) ─────────────────────────────
# Injected from main.py via init_router() after ingestion completes

_pipeline: QueryPipeline | None = None


def init_router(pipeline: QueryPipeline) -> None:
    """Called from main.py after Phase 1 ingestion to inject the pipeline."""
    global _pipeline
    _pipeline = pipeline


def _get_pipeline() -> QueryPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised. Ingestion may still be running.")
    return _pipeline


# ── Routes ─────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Run the full Phase 2 pipeline:
      refine query → embed → Pinecone fetch → rerank →
      web fallback (if needed) → build context → Claude Sonnet response
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    pipeline = _get_pipeline()

    answer, sources, used_web = pipeline.run(request.question)

    return QueryResponse(
        answer=answer,
        sources=sources,
        used_web_fallback=used_web,
    )


@router.post("/swap")
async def swap_pdfs(request: SwapRequest):
    """
    Swap study PDFs:
      - Re-runs Phase 1 ingestion on the new PDFs
      - Clears Pinecone namespace (removes old vectors)
      - Clears conversation memory
    """
    from ingestion import run_ingestion, _validate_pdfs

    pipeline = _get_pipeline()

    try:
        validated_paths = _validate_pdfs(request.pdf_paths)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    run_ingestion(validated_paths)
    pipeline.reset_memory()

    return {"message": f"Swapped successfully. {len(validated_paths)} PDFs ingested."}


@router.delete("/clear")
async def clear_memory():
    """Clear conversation history without re-ingesting PDFs."""
    _get_pipeline().reset_memory()
    return {"message": "Conversation memory cleared."}