"""
General helper functions for session management, formatting, and data serialization.
"""

import base64
from io import BytesIO
from datetime import datetime
from typing import Dict, Any, List
from PIL import Image


def get_current_timestamp() -> str:
    """Return formatted timestamp string."""
    return datetime.now().strftime("%I:%M %p")


def encode_image_to_base64(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Encode binary image data into a base64 data URI string."""
    b64_data = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64_data}"


def pil_image_to_base64(pil_image: Image.Image, format: str = "JPEG") -> str:
    """Convert a PIL Image to a base64 data URI."""
    buffered = BytesIO()
    # If image has alpha channel and format is JPEG, convert to RGB
    if pil_image.mode in ("RGBA", "P") and format.upper() == "JPEG":
        pil_image = pil_image.convert("RGB")
    pil_image.save(buffered, format=format)
    mime = "image/png" if format.upper() == "PNG" else "image/jpeg"
    return encode_image_to_base64(buffered.getvalue(), mime_type=mime)


def truncate_text(text: str, max_chars: int = 120) -> str:
    """Truncate text with ellipsis if exceeding max characters."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def format_source_citation(meta: Dict[str, Any]) -> str:
    """Format source citation into a clean display title."""
    doc_name = meta.get("document_name", "Unknown Document")
    page_num = meta.get("page_number", 1)
    chunk_id = meta.get("chunk_id", 0)
    return f"📄 {doc_name} — Page {page_num} (Chunk #{chunk_id})"
