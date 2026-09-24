"""
Unit tests for RAG pipeline components:
Document chunking, Vector store operations, and Retriever assembly.
"""

from rag.chunker import chunk_text, chunk_document
from rag.vector_store import VectorStore
from rag.retriever import RAGRetriever
import numpy as np


def test_chunk_text():
    sample_text = (
        "Artificial Intelligence is a wide-ranging branch of computer science. "
        "It aims to build smart machines capable of performing tasks that typically require human intelligence. "
        "Machine Learning is a subset of AI that allows systems to learn from data. "
        "Deep Learning is a subset of ML based on artificial neural networks."
    )
    chunks = chunk_text(sample_text, chunk_size=80, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 120 for c in chunks)


def test_chunk_document_metadata():
    doc_data = {
        "document_name": "test_lecture.pdf",
        "total_pages": 2,
        "char_count": 500,
        "pages": [
            {"page_number": 1, "text": "Page one text regarding gradient descent optimization."},
            {"page_number": 2, "text": "Page two text regarding regularization and overfitting."},
        ],
    }
    chunks = chunk_document(doc_data, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 2
    assert chunks[0]["document_name"] == "test_lecture.pdf"
    assert chunks[0]["page_number"] == 1
    assert chunks[-1]["page_number"] == 2
    assert "chunk_id" in chunks[0]
    assert "source_text" in chunks[0]


def test_vector_store_operations():
    vs = VectorStore(dimension=4)
    chunks = [
        {"chunk_id": 1, "document_name": "doc1.pdf", "page_number": 1, "source_text": "text A", "token_count_est": 2},
        {"chunk_id": 2, "document_name": "doc2.pdf", "page_number": 2, "source_text": "text B", "token_count_est": 2},
    ]
    # Unit vectors
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ], dtype=np.float32)

    added = vs.add_chunks(chunks, embeddings)
    assert added == 2
    assert vs.total_chunks == 2

    # Query matching vector 1
    query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = vs.search(query_vec, top_k=2)
    assert len(results) == 2
    top_chunk, score = results[0]
    assert top_chunk["chunk_id"] == 1
    assert score > 0.99


def test_retriever_context_building():
    vs = VectorStore(dimension=4)
    retriever = RAGRetriever(vs)
    sample_chunks = [
        {
            "chunk_id": 1,
            "document_name": "Operating_Systems.pdf",
            "page_number": 12,
            "similarity_score": 0.85,
            "source_text": "Virtual memory is a memory management capability of an OS.",
        }
    ]
    prompt_context = retriever.build_context_prompt(sample_chunks)
    assert "Operating_Systems.pdf" in prompt_context
    assert "Page: 12" in prompt_context
    assert "Virtual memory" in prompt_context
