"""
PDF and Document Loader for EduVision AI.
Extracts clean text page-by-page from uploaded documents with metadata tracking.
"""

from typing import Dict, Any, List, Optional
import io
import re
from pathlib import Path
from pypdf import PdfReader


def clean_extracted_text(text: str) -> str:
    """Normalize extracted document text, removing odd control characters while preserving structure."""
    if not text:
        return ""
    # Replace non-breaking spaces and unusual whitespace
    cleaned = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple consecutive blank lines into at most two
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    # Collapse excess inline spaces
    cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
    return cleaned.strip()


def load_pdf_from_bytes(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Extract text page-by-page from PDF byte content.
    Returns structured document dictionary.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    total_pages = len(reader.pages)
    pages_data: List[Dict[str, Any]] = []
    total_chars = 0

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        raw_text = page.extract_text() or ""
        cleaned = clean_extracted_text(raw_text)
        
        pages_data.append({
            "page_number": page_num,
            "text": cleaned,
            "char_count": len(cleaned),
        })
        total_chars += len(cleaned)

    return {
        "document_name": filename,
        "total_pages": total_pages,
        "char_count": total_chars,
        "pages": pages_data,
    }


def load_pdf_from_path(file_path: str) -> Dict[str, Any]:
    """Load and extract text from a PDF file path on disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found at: {file_path}")
    
    with open(path, "rb") as f:
        bytes_data = f.read()
    return load_pdf_from_bytes(bytes_data, path.name)
