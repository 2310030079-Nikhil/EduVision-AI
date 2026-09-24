"""
Source citation and RAG retrieval display component for EduVision AI.
Renders expandable source citations with exact page numbers, similarity scores,
and chunk content previews.
"""

import streamlit as st
from typing import List, Dict, Any


def render_sources(sources: List[Dict[str, Any]], title: str = "📚 Sources"):
    """
    Render expandable source citations with clean visual design.
    """
    if not sources:
        return

    st.markdown(
        f"""
        <div style="margin-top: 14px; margin-bottom: 6px; font-weight: 700; color: #1E3A8A; font-size: 0.95rem; display: flex; align-items: center; gap: 6px;">
            <span>{title}</span>
            <span style="background-color: #DBEAFE; color: #1E40AF; font-size: 0.75rem; padding: 2px 8px; border-radius: 9999px; font-weight: 600;">
                {len(sources)} Evidence {'Segment' if len(sources) == 1 else 'Segments'}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for i, src in enumerate(sources, 1):
        doc_name = src.get("document_name", "Unknown Document")
        page_num = src.get("page_number", 1)
        chunk_id = src.get("chunk_id", i)
        score = src.get("similarity_score", 0.0)
        text = src.get("source_text", "").strip()

        score_percent = f"{int(score * 100)}%" if score > 0 else "N/A"
        expander_title = f"📄 {doc_name} — Page {page_num}  •  Match {score_percent}"

        with st.expander(expander_title, expanded=False):
            st.markdown(
                f"""
                <div style="display: flex; gap: 8px; margin-bottom: 8px;">
                    <span style="background: #F3F4F6; color: #374151; font-size: 0.78rem; padding: 2px 6px; border-radius: 4px; font-weight: 600;">
                        Chunk #{chunk_id}
                    </span>
                    <span style="background: #ECFDF5; color: #065F46; font-size: 0.78rem; padding: 2px 6px; border-radius: 4px; font-weight: 600;">
                        Similarity: {score:.3f}
                    </span>
                </div>
                <div style="background-color: #F8FAFC; border-left: 3px solid #3B82F6; padding: 10px 14px; border-radius: 0 6px 6px 0; font-size: 0.88rem; color: #1E293B; line-height: 1.5; font-family: sans-serif;">
                    {text}
                </div>
                """,
                unsafe_allow_html=True,
            )
