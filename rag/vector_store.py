"""
Vector Store implementation for EduVision AI.
Provides fast similarity search using FAISS (IndexFlatIP) with a seamless NumPy fallback.
Manages chunk metadata, index persistence, and search statistics.
"""

from typing import List, Dict, Any, Tuple, Optional
import os
import pickle
from pathlib import Path
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from utils.config import EMBEDDING_DIMENSION, VECTOR_INDEX_DIR


class VectorStore:
    """
    Manages vector indexing and retrieval for RAG.
    Uses FAISS IndexFlatIP (cosine similarity on L2-normalized vectors).
    Maintains synchronized chunk metadata.
    """

    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        self.dimension = dimension
        self.chunks_metadata: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None
        self._init_index()

    def _init_index(self):
        """Initialize FAISS index or prepare for NumPy matrix operations."""
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = None

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> int:
        """
        Add document chunks and their corresponding embeddings to the vector store.
        Returns the count of added chunks.
        """
        if len(chunks) == 0 or len(embeddings) == 0:
            return 0

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        if HAS_FAISS and self.index is not None:
            self.index.add(embeddings)
        
        # Also maintain NumPy array for fallback or inspection
        if self.vectors is None:
            self.vectors = embeddings
        else:
            self.vectors = np.vstack([self.vectors, embeddings])

        self.chunks_metadata.extend(chunks)
        return len(chunks)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 4,
        score_threshold: float = 0.0,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform similarity search against the index.
        Returns list of (chunk_dict, similarity_score) sorted by highest score.
        """
        if self.total_chunks == 0:
            return []

        query_vec = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        k = min(top_k, self.total_chunks)
        results: List[Tuple[Dict[str, Any], float]] = []

        if HAS_FAISS and self.index is not None:
            scores, indices = self.index.search(query_vec, k)
            for idx, score in zip(indices[0], scores[0]):
                if idx != -1 and idx < len(self.chunks_metadata):
                    sc = float(score)
                    if sc >= score_threshold:
                        results.append((self.chunks_metadata[idx], sc))
        else:
            # NumPy cosine similarity fallback
            scores = np.dot(self.vectors, query_vec.T).flatten()
            top_indices = np.argsort(scores)[::-1][:k]
            for idx in top_indices:
                sc = float(scores[idx])
                if sc >= score_threshold:
                    results.append((self.chunks_metadata[idx], sc))

        return results

    @property
    def total_chunks(self) -> int:
        return len(self.chunks_metadata)

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics of indexed content."""
        unique_docs = set(c.get("document_name", "") for c in self.chunks_metadata)
        total_tokens = sum(c.get("token_count_est", 0) for c in self.chunks_metadata)
        return {
            "total_documents": len(unique_docs),
            "document_names": list(unique_docs),
            "total_chunks": self.total_chunks,
            "total_tokens_est": total_tokens,
            "backend": "FAISS (IndexFlatIP)" if (HAS_FAISS and self.index is not None) else "NumPy Cosine Store",
        }

    def clear(self):
        """Reset the vector store to empty state."""
        self.chunks_metadata.clear()
        self.vectors = None
        self._init_index()

    def save(self, directory: Optional[Path] = None) -> bool:
        """Save index and metadata to disk."""
        target_dir = directory or VECTOR_INDEX_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            meta_path = target_dir / "metadata.pkl"
            with open(meta_path, "wb") as f:
                pickle.dump(self.chunks_metadata, f)

            if HAS_FAISS and self.index is not None:
                faiss.write_index(self.index, str(target_dir / "index.faiss"))
            if self.vectors is not None:
                np.save(str(target_dir / "vectors.npy"), self.vectors)
            return True
        except Exception as exc:
            print(f"[Warning] Failed to save vector store: {exc}")
            return False

    def load(self, directory: Optional[Path] = None) -> bool:
        """Load index and metadata from disk."""
        target_dir = directory or VECTOR_INDEX_DIR
        meta_path = target_dir / "metadata.pkl"
        if not meta_path.exists():
            return False

        try:
            with open(meta_path, "rb") as f:
                self.chunks_metadata = pickle.load(f)

            faiss_path = target_dir / "index.faiss"
            if HAS_FAISS and faiss_path.exists():
                self.index = faiss.read_index(str(faiss_path))

            vec_path = target_dir / "vectors.npy"
            if vec_path.exists():
                self.vectors = np.load(str(vec_path))

            return True
        except Exception as exc:
            print(f"[Warning] Failed to load vector store: {exc}")
            return False
