"""
Knowledge Base Page for EduVision AI (Multipage support).
Allows document uploading, index management, and chunk inspection.
"""

import streamlit as st
import os
from rag.vector_store import VectorStore
from components.document_ui import render_knowledge_base_ui

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "indexed_documents" not in st.session_state:
    st.session_state.indexed_documents = []

render_knowledge_base_ui(st.session_state.vector_store)
