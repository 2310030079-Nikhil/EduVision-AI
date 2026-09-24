"""
Unit tests for input and file validation utilities.
"""

from utils.validators import (
    validate_file_size,
    validate_document_file,
    validate_image_file,
    validate_api_key,
    sanitize_input_text,
)


def test_file_size_validation():
    small_bytes = b"hello world"
    valid, _ = validate_file_size(small_bytes, max_mb=5)
    assert valid is True

    empty_bytes = b""
    valid, msg = validate_file_size(empty_bytes)
    assert valid is False
    assert "empty" in msg.lower()


def test_document_validation():
    pdf_bytes = b"%PDF-1.4 sample content"
    valid, _ = validate_document_file("lecture.pdf", pdf_bytes)
    assert valid is True

    valid, msg = validate_document_file("virus.exe", pdf_bytes)
    assert valid is False
    assert "unsupported" in msg.lower()


def test_image_validation():
    img_bytes = b"fake image bytes"
    valid, _ = validate_image_file("diagram.png", img_bytes)
    assert valid is True

    valid, _ = validate_image_file("diagram.jpg", img_bytes)
    assert valid is True

    valid, msg = validate_image_file("diagram.txt", img_bytes)
    assert valid is False


def test_api_key_validation():
    valid, _ = validate_api_key("gsk_1234567890abcdefghijklmnop")
    assert valid is True

    valid, msg = validate_api_key("")
    assert valid is False
    assert "missing" in msg.lower()

    valid, msg = validate_api_key("short")
    assert valid is False


def test_sanitize_input():
    clean = sanitize_input_text("  hello  ")
    assert clean == "hello"

    long_str = "a" * 5000
    truncated = sanitize_input_text(long_str, max_chars=100)
    assert len(truncated) == 100
