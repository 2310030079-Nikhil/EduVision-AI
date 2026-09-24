"""
Validation utilities for files, user inputs, and API configurations.
Ensures security, prevents arbitrary code injection, and protects runtime stability.
"""

from typing import Tuple, Optional
import os
from pathlib import Path
from utils.config import (
    MAX_FILE_SIZE_MB,
    ALLOWED_DOCUMENT_TYPES,
    ALLOWED_IMAGE_TYPES,
)


def validate_file_size(file_bytes: bytes, max_mb: int = MAX_FILE_SIZE_MB) -> Tuple[bool, str]:
    """Validate that the file size is within limits."""
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_mb:
        return False, f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit of {max_mb} MB."
    if len(file_bytes) == 0:
        return False, "Uploaded file is empty (0 bytes)."
    return True, "File size is valid."


def validate_document_file(filename: str, file_bytes: bytes) -> Tuple[bool, str]:
    """Validate that the uploaded document has an allowed extension and valid size."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_DOCUMENT_TYPES:
        return False, f"Unsupported document format '{ext}'. Only {', '.join(ALLOWED_DOCUMENT_TYPES)} files are supported."
    return validate_file_size(file_bytes)


def validate_image_file(filename: str, file_bytes: bytes) -> Tuple[bool, str]:
    """Validate that the uploaded image has an allowed extension and valid size."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_TYPES:
        return False, f"Unsupported image format '{ext}'. Supported formats: {', '.join(ALLOWED_IMAGE_TYPES)}."
    return validate_file_size(file_bytes, max_mb=10)


def sanitize_input_text(text: str, max_chars: int = 4000) -> str:
    """Sanitize user text input to avoid excessive payload or malicious formatting."""
    if not text:
        return ""
    cleaned = text.strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned


def validate_api_key(api_key: Optional[str]) -> Tuple[bool, str]:
    """Validate the format of a Groq API key."""
    if not api_key:
        return False, "Groq API key is missing. Please enter your key in the settings or set GROQ_API_KEY in .env."
    api_key_clean = api_key.strip()
    if len(api_key_clean) < 15:
        return False, "Groq API key is too short. Please provide a valid API key."
    return True, "API key format is valid."
