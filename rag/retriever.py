"""
RAG Retriever for EduVision AI.
Coordinates vector similarity search, relevance score thresholding,
and context assembly for grounded LLM prompts.
"""

from typing import List, Dict, Any, Tuple
from rag.embeddings import embedding_service
from rag.vector_store import VectorStore
from utils.config import DEFAULT_TOP_K, SIMILARITY_THRESHOLD


class RAGRetriever:
    """Orchestrates query embedding, vector store retrieval, and context formatting."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """
        Embed query, perform similarity search, and return relevant chunk dicts with scores.
        """
        if self.vector_store.total_chunks == 0:
            return []

        query_vec = embedding_service.embed_query(query)
        raw_results = self.vector_store.search(
            query_vector=query_vec,
            top_k=top_k,
            score_threshold=threshold,
        )

        retrieved: List[Dict[str, Any]] = []
        for chunk, score in raw_results:
            item = dict(chunk)
            item["similarity_score"] = round(float(score), 4)
            retrieved.append(item)

        return retrieved

    def build_context_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build structured context string to inject into the LLM system/user prompt.
        """
        if not chunks:
            return ""

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            doc = chunk.get("document_name", "Unknown Document")
            page = chunk.get("page_number", 1)
            chunk_id = chunk.get("chunk_id", i)
            score = chunk.get("similarity_score", 0.0)
            text = chunk.get("source_text", "").strip()

            context_parts.append(
                f"--- [Document Excerpt {i}] ---\n"
                f"Document: {doc} | Page: {page} | Chunk ID: {chunk_id} | Relevance: {score:.2f}\n"
                f"Content:\n{text}\n"
            )

        return "\n".join(context_parts)
