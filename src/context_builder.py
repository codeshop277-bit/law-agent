"""
context_builder.py
──────────────────
Phase 2 — Step 8
Assembles the final context block from reranked PDF chunks
and optional web fallback content.

Each chunk is labelled with its source (PDF name, page, section)
so the LLM can cite them accurately in its response.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BuiltContext:
    """The assembled context ready to be inserted into the LLM prompt."""
    context_text: str           # full formatted context string
    sources: list[dict]         # list of source metadata dicts for citation display
    has_web_content: bool       # whether web fallback content is included


def build_context(top_chunks: list[dict], web_context: str = "") -> BuiltContext:
    """
    Assemble the LLM context block from PDF chunks and optional web content.

    Args:
        top_chunks:  Reranked chunks from reranker.rerank_chunks()
        web_context: Formatted web content from web_fallback.build_fallback_context()
                     Pass empty string if web fallback was not triggered.

    Returns:
        BuiltContext with the formatted context string and source metadata.
    """
    parts = []
    sources = []

    # ── PDF chunk context ──────────────────────────────────────────────────
    if top_chunks:
        parts.append("--- CONTEXT FROM YOUR STUDY MATERIALS ---\n")

        for idx, chunk in enumerate(top_chunks, start=1):
            meta = chunk.get("metadata", {})
            pdf_name       = meta.get("pdf_name", "Unknown PDF")
            page_number    = meta.get("page_number", "?")
            section        = meta.get("section_heading", "Unknown Section")
            text           = meta.get("text", "")
            rerank_score   = chunk.get("rerank_score", 0.0)

            chunk_label = (
                f"[Source {idx}] {pdf_name} | Page {page_number} | Section: {section}"
            )
            parts.append(f"{chunk_label}\n{text}\n")

            sources.append({
                "source_index": idx,
                "pdf_name":     pdf_name,
                "page_number":  page_number,
                "section":      section,
                "rerank_score": rerank_score,
            })

        parts.append("--- END STUDY MATERIAL CONTEXT ---")

    # ── Web fallback context ───────────────────────────────────────────────
    has_web = bool(web_context.strip())
    if has_web:
        parts.append(web_context)

    context_text = "\n".join(parts)

    print(f"  [ContextBuilder] Context assembled:")
    print(f"    PDF chunks : {len(top_chunks)}")
    print(f"    Web content: {'Yes' if has_web else 'No'}")
    print(f"    Total chars: {len(context_text)}")

    return BuiltContext(
        context_text=context_text,
        sources=sources,
        has_web_content=has_web,
    )
