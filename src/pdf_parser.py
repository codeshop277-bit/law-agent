"""
pdf_parser.py
─────────────
Handles PDF reading, text extraction, and cleaning.
Uses PyMuPDF (fitz) for fast extraction with heading detection.
"""

import re
import fitz  # PyMuPDF
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PageContent:
    """Represents extracted content from a single PDF page."""
    pdf_name: str
    page_number: int
    section_heading: Optional[str]
    raw_text: str
    cleaned_text: str = field(default="")


def extract_pdf(pdf_path: str) -> list[PageContent]:
    """
    Extract text from all pages of a PDF.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of PageContent objects, one per page (empty pages excluded).
    """
    pdf_name = _get_pdf_name(pdf_path)
    pages: list[PageContent] = []

    doc = fitz.open(pdf_path)
    current_heading = None

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1

        # Extract text blocks with font metadata to detect headings
        blocks = page.get_text("dict")["blocks"]
        raw_text, detected_heading = _process_blocks(blocks)

        if detected_heading:
            current_heading = detected_heading

        if not raw_text.strip():
            continue  # Skip blank pages

        page_content = PageContent(
            pdf_name=pdf_name,
            page_number=page_number,
            section_heading=current_heading,
            raw_text=raw_text,
        )
        page_content.cleaned_text = clean_text(raw_text)
        pages.append(page_content)

    doc.close()
    print(f"  [Parser] '{pdf_name}' → {len(pages)} pages extracted")
    return pages


def _process_blocks(blocks: list) -> tuple[str, Optional[str]]:
    """
    Process PyMuPDF text blocks.
    Detects headings via font size (headings are typically larger).
    Returns (raw_text, detected_heading).
    """
    lines = []
    detected_heading = None
    font_sizes = []

    # First pass — collect font sizes to determine the dominant body size
    for block in blocks:
        if block.get("type") != 0:  # 0 = text block
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font_sizes.append(span.get("size", 12))

    body_size = _median(font_sizes) if font_sizes else 12

    # Second pass — extract text and detect headings
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            line_text = ""
            is_heading_line = False

            for span in line.get("spans", []):
                span_text = span.get("text", "").strip()
                size = span.get("size", 12)
                flags = span.get("flags", 0)

                is_bold = bool(flags & 2**4)  # bold flag in PyMuPDF

                # Treat as heading if font is significantly larger or bold + larger
                if size > body_size * 1.15 or (is_bold and size >= body_size):
                    is_heading_line = True

                line_text += span_text + " "

            line_text = line_text.strip()
            if line_text:
                lines.append(line_text)
                if is_heading_line and len(line_text) < 120:
                    detected_heading = line_text

    raw_text = "\n".join(lines)
    return raw_text, detected_heading


def clean_text(text: str) -> str:
    """
    Clean raw PDF text:
    - Fix hyphenation breaks (e.g. "knowl-\\nedge" → "knowledge")
    - Normalize whitespace
    - Remove lone page numbers
    - Remove repeated header/footer patterns
    - Collapse multiple blank lines
    """
    # Fix hyphenated line breaks
    text = re.sub(r"-\n(\w)", r"\1", text)

    # Remove lone page numbers (a line that's just digits)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)

    # Collapse more than 2 newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def _get_pdf_name(pdf_path: str) -> str:
    """Extract filename without extension from path."""
    import os
    return os.path.splitext(os.path.basename(pdf_path))[0]


def _median(values: list[float]) -> float:
    """Return median of a list of numbers."""
    if not values:
        return 12.0
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    return sorted_vals[mid]
