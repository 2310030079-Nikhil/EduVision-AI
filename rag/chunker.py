"""
Document chunking engine for EduVision AI.
Splits document pages into semantically cohesive overlapping chunks
while preserving page numbers, chunk IDs, and source provenance.
"""

from typing import List, Dict, Any
from utils.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks using paragraph, sentence, and word boundaries.
    """
    if not text or len(text.strip()) == 0:
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        if end >= text_len:
            chunks.append(text[start:].strip())
            break

        # Try to break at a natural paragraph boundary
        break_point = text.rfind("\n\n", start, end)
        if break_point != -1 and break_point > start + (chunk_size // 3):
            actual_end = break_point + 2
        else:
            # Try to break at a sentence boundary (. ! ?)
            sentence_end = -1
            for punct in [". ", "? ", "! ", ".\n"]:
                pos = text.rfind(punct, start, end)
                if pos > sentence_end:
                    sentence_end = pos + len(punct)
            
            if sentence_end != -1 and sentence_end > start + (chunk_size // 3):
                actual_end = sentence_end
            else:
                # Try breaking at a space
                space_pos = text.rfind(" ", start, end)
                if space_pos != -1 and space_pos > start:
                    actual_end = space_pos + 1
                else:
                    actual_end = end

        chunk_str = text[start:actual_end].strip()
        if chunk_str:
            chunks.append(chunk_str)

        # Advance start position taking overlap into account
        start = actual_end - chunk_overlap
        if start >= actual_end:
            start = actual_end

    return chunks


def chunk_document(
    doc_data: Dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Chunk an entire document dictionary into a list of structured chunk dicts.
    Each chunk retains document name, page number, global chunk ID, and source text.
    """
    doc_name = doc_data.get("document_name", "Unknown")
    pages = doc_data.get("pages", [])
    all_chunks: List[Dict[str, Any]] = []
    chunk_counter = 0

    for page_info in pages:
        page_num = page_info.get("page_number", 1)
        page_text = page_info.get("text", "")

        raw_chunks = chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk_str in raw_chunks:
            chunk_counter += 1
            all_chunks.append({
                "chunk_id": chunk_counter,
                "document_name": doc_name,
                "page_number": page_num,
                "source_text": chunk_str,
                "token_count_est": max(1, len(chunk_str) // 4),
            })

    return all_chunks
