"""
Knowledge Base and Document UI component for EduVision AI.
Provides document upload, index statistics, chunk inspection, and sample document loading.
"""

import streamlit as st
from typing import Dict, Any, List
from pathlib import Path
from rag.document_loader import load_pdf_from_bytes, load_pdf_from_path
from rag.chunker import chunk_document
from rag.embeddings import embedding_service
from rag.vector_store import VectorStore
from utils.validators import validate_document_file
from utils.config import SAMPLE_DOCS_DIR


def render_knowledge_base_ui(vector_store: VectorStore):
    """Render the Knowledge Base tab/page with upload controls and document analytics."""
    st.markdown("### 📚 Knowledge Base Management")
    st.markdown(
        "Upload course syllabi, lecture notes, textbook chapters, or reference PDFs. "
        "The RAG engine will extract text, generate chunk embeddings, and index them into FAISS for retrieval."
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### 📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files to index",
            type=["pdf"],
            accept_multiple_files=True,
            key="kb_pdf_uploader",
        )

        if uploaded_files:
            if st.button("🚀 Process & Index Uploaded Files", type="primary", use_container_width=True):
                for file_obj in uploaded_files:
                    file_bytes = file_obj.getvalue()
                    valid, msg = validate_document_file(file_obj.name, file_bytes)
                    if not valid:
                        st.error(f"❌ {file_obj.name}: {msg}")
                        continue

                    # Check if already processed in session
                    already_indexed = any(
                        d.get("document_name") == file_obj.name
                        for d in st.session_state.get("indexed_documents", [])
                    )
                    if already_indexed:
                        st.info(f"ℹ️ '{file_obj.name}' is already indexed.")
                        continue

                    with st.status(f"Indexing '{file_obj.name}'...", expanded=True) as status:
                        st.write("1️⃣ Extracting pages and cleaning text...")
                        doc_data = load_pdf_from_bytes(file_bytes, file_obj.name)

                        st.write(f"2️⃣ Splitting into semantically bounded chunks (Total pages: {doc_data['total_pages']})...")
                        chunks = chunk_document(doc_data)

                        st.write(f"3️⃣ Generating vector embeddings for {len(chunks)} chunks with SentenceTransformers...")
                        texts_to_embed = [c["source_text"] for c in chunks]
                        embeddings = embedding_service.embed_texts(texts_to_embed)

                        st.write("4️⃣ Committing vectors to FAISS index...")
                        vector_store.add_chunks(chunks, embeddings)

                        # Record in session state
                        if "indexed_documents" not in st.session_state:
                            st.session_state.indexed_documents = []
                        st.session_state.indexed_documents.append({
                            "document_name": file_obj.name,
                            "pages": doc_data["total_pages"],
                            "chunks": len(chunks),
                            "chars": doc_data["char_count"],
                            "status": "Ready",
                        })
                        status.update(label=f"✅ Successfully indexed '{file_obj.name}' ({len(chunks)} chunks)", state="complete")

                st.success("All documents indexed and ready for RAG!")
                st.rerun()

    with col2:
        st.markdown("#### ⚡ Quick Demo Preload")
        st.markdown(
            "Quickly test the system using our pre-built educational PDF: "
            "`Introduction to Artificial Intelligence and Machine Learning`."
        )

        sample_pdf_path = SAMPLE_DOCS_DIR / "introduction_to_ai_ml.pdf"
        sample_indexed = any(
            d.get("document_name") == "introduction_to_ai_ml.pdf"
            for d in st.session_state.get("indexed_documents", [])
        )

        if sample_indexed:
            st.success("✅ Sample AI/ML Document is active in Vector Store.")
        else:
            if st.button("📥 Load Sample AI/ML Document", use_container_width=True):
                if sample_pdf_path.exists():
                    with st.status("Loading sample document...", expanded=True) as status:
                        doc_data = load_pdf_from_path(str(sample_pdf_path))
                        chunks = chunk_document(doc_data)
                        embeddings = embedding_service.embed_texts([c["source_text"] for c in chunks])
                        vector_store.add_chunks(chunks, embeddings)

                        if "indexed_documents" not in st.session_state:
                            st.session_state.indexed_documents = []
                        st.session_state.indexed_documents.append({
                            "document_name": "introduction_to_ai_ml.pdf",
                            "pages": doc_data["total_pages"],
                            "chunks": len(chunks),
                            "chars": doc_data["char_count"],
                            "status": "Ready",
                        })
                        status.update(label="✅ Sample Document Ready!", state="complete")
                    st.success("Sample AI/ML document indexed successfully!")
                    st.rerun()
                else:
                    st.error("Sample document file not found on disk.")

        if vector_store.total_chunks > 0:
            if st.button("🗑️ Clear Vector Database", use_container_width=True):
                vector_store.clear()
                st.session_state.indexed_documents = []
                st.warning("Vector store cleared.")
                st.rerun()

    # Document Indexing Summary Table
    st.markdown("---")
    st.markdown("#### 📑 Indexed Documents")
    indexed_docs = st.session_state.get("indexed_documents", [])

    if not indexed_docs:
        st.markdown(
            """
            <div style="background-color: #F9FAFB; border: 2px dashed #E5E7EB; border-radius: 10px; padding: 24px; text-align: center; color: #6B7280;">
                <p style="font-size: 1.1rem; font-weight: 500; margin-bottom: 4px;">No documents indexed yet</p>
                <p style="font-size: 0.85rem;">Upload a PDF file or click 'Load Sample AI/ML Document' above to populate the RAG knowledge base.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for doc in indexed_docs:
            st.markdown(
                f"""
                <div style="background: white; border: 1px solid #E5E7EB; border-radius: 10px; padding: 12px 18px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 700; color: #111827; font-size: 0.95rem;">📄 {doc['document_name']}</div>
                        <div style="font-size: 0.8rem; color: #6B7280; margin-top: 2px;">
                            Pages: <b>{doc['pages']}</b> &nbsp;•&nbsp; Chunks: <b>{doc['chunks']}</b> &nbsp;•&nbsp; Characters: <b>{doc.get('chars', 'N/A'):,}</b>
                        </div>
                    </div>
                    <span style="background-color: #DEF7EC; color: #03543F; font-size: 0.75rem; font-weight: 700; padding: 4px 10px; border-radius: 9999px;">
                        ✅ {doc['status']}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Chunk Inspection Section
        with st.expander("🔍 Inspect Stored Chunks", expanded=False):
            st.markdown(f"Total Chunks in FAISS Vector Store: **{vector_store.total_chunks}**")
            all_chunks = vector_store.chunks_metadata
            for chunk in all_chunks[:15]:  # Preview first 15 chunks
                st.markdown(
                    f"**Chunk #{chunk.get('chunk_id')}** — *{chunk.get('document_name')} (Page {chunk.get('page_number')})*:"
                )
                st.code(chunk.get("source_text", "")[:250] + "...", language="text")
