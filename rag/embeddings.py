"""
Embedding Generator for EduVision AI.
Provides vector embeddings for document chunks and user queries using
SentenceTransformers (all-MiniLM-L6-v2) with normalization and memory caching.
"""

from typing import List, Union
import numpy as np
from utils.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION

_GLOBAL_MODEL = None


def get_embedding_model():
    """Lazy-load and cache the SentenceTransformer model in memory."""
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _GLOBAL_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception as exc:
            print(f"[Warning] Failed to load SentenceTransformer ({exc}). Using fallback vectorizer.")
            _GLOBAL_MODEL = "FALLBACK"
    return _GLOBAL_MODEL


class EmbeddingGenerator:
    """Handles text-to-vector embedding with L2-normalization."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.dimension = EMBEDDING_DIMENSION
        self._cache = {}

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors so inner product equals cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of text strings into an (N, dimension) float32 numpy array."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        model = get_embedding_model()

        if model != "FALLBACK":
            embeddings = model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings.astype(np.float32)
        else:
            # Fallback pseudo-dense hashing embedding for zero-dependency offline environments
            return self._fallback_embed(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string into a 1D (dimension,) float32 vector."""
        vectors = self.embed_texts([query])
        return vectors[0]

    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        """Lightweight deterministic hashing vectorizer fallback."""
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            words = text.lower().split()
            for word in words:
                idx = hash(word) % self.dimension
                vectors[i, idx] += 1.0
        return self._normalize(vectors)


# Global singleton instance
embedding_service = EmbeddingGenerator()
