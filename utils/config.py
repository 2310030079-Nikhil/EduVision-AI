"""
Configuration settings for EduVision AI.
Manages environment variables, model identifiers, RAG defaults, and system constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_documents"
VECTOR_INDEX_DIR = DATA_DIR / "vector_store"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env if present
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Groq API Configuration
DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Supported Groq Models
TEXT_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]
DEFAULT_TEXT_MODEL = "qwen/qwen3.8-27b"

VISION_MODELS = [
    "qwen/qwen3.8-27b",
    "llama-3.2-11b-vision-preview",
]
DEFAULT_VISION_MODEL = "qwen/qwen3.8-27b"

# Embedding Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# RAG & Chunking Parameters
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 4
SIMILARITY_THRESHOLD = 0.32  # Minimum similarity score to qualify as relevant chunk

# File Upload Constraints
MAX_FILE_SIZE_MB = 25
ALLOWED_DOCUMENT_TYPES = [".pdf"]
ALLOWED_IMAGE_TYPES = [".png", ".jpg", ".jpeg", ".webp"]

# Application Metadata
APP_NAME = "EduVision AI"
APP_TAGLINE = "Learn smarter from your documents, images, and questions."
APP_VERSION = "1.0.0"
