"""
Chat Page for EduVision AI (Multipage support).
Allows standalone execution or routing via Streamlit multipage architecture.
"""

import streamlit as st
import os
from PIL import Image

from utils.config import (
    APP_NAME,
    DEFAULT_TEXT_MODEL,
    DEFAULT_VISION_MODEL,
    DEFAULT_TOP_K,
    SIMILARITY_THRESHOLD,
    SAMPLE_DOCS_DIR,
)
from utils.helpers import get_current_timestamp
from tools.calculator import calculate_expression, is_math_query
from tools.web_search import search_web, is_search_query
from rag.vector_store import VectorStore
from rag.retriever import RAGRetriever
from rag.document_loader import load_pdf_from_path
from rag.chunker import chunk_document
from rag.embeddings import embedding_service
from ai.groq_client import GroqClientManager
from ai.vision import prepare_image_payload, validate_and_process_image
from ai.prompts import build_rag_prompt
from components.chat_ui import render_chat_messages

# Initialize state if accessed directly
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "groq_client" not in st.session_state:
    st.session_state.groq_client = GroqClientManager(os.getenv("GROQ_API_KEY", ""))
if "selected_text_model" not in st.session_state:
    st.session_state.selected_text_model = DEFAULT_TEXT_MODEL
if "selected_vision_model" not in st.session_state:
    st.session_state.selected_vision_model = DEFAULT_VISION_MODEL
if "top_k" not in st.session_state:
    st.session_state.top_k = DEFAULT_TOP_K
if "similarity_threshold" not in st.session_state:
    st.session_state.similarity_threshold = SIMILARITY_THRESHOLD
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.3
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

vector_store = st.session_state.vector_store
retriever = RAGRetriever(vector_store)
groq_mgr = st.session_state.groq_client

st.markdown("## 💬 Chat with EduVision AI")

if "image_uploader_key" not in st.session_state:
    st.session_state.image_uploader_key = 0

# Optional Image Attachment Drawer
with st.expander("🖼️ Attach Image for Multimodal Vision Analysis (Optional)", expanded=False):
    col_img_up, col_img_btn = st.columns([3, 1])
    with col_img_up:
        uploaded_image = st.file_uploader(
            "Upload mathematical problem, diagram, chart, or screenshot",
            type=["png", "jpg", "jpeg", "webp"],
            key=f"chat_page_image_uploader_{st.session_state.image_uploader_key}",
        )
    with col_img_btn:
        if uploaded_image:
            st.write("")
            if st.button("🗑️ Discard Image", key="discard_page_img_btn", use_container_width=True):
                st.session_state.image_uploader_key += 1
                st.rerun()

    if uploaded_image:
        img_bytes = uploaded_image.getvalue()
        v_info = validate_and_process_image(img_bytes)
        if v_info["valid"]:
            st.image(img_bytes, caption=f"Preview: {uploaded_image.name}", width=280)
        else:
            st.error(v_info["error"])

# Render Conversation Messages
render_chat_messages(st.session_state.messages)

user_query = st.chat_input("Ask a question, upload a problem, or ask about your documents...")
is_from_suggestion = False
is_visual_prompt = False
if st.session_state.pending_query:
    user_query = st.session_state.pending_query
    is_from_suggestion = True
    is_visual_prompt = "uploaded image" in user_query.lower() or "problem shown" in user_query.lower()
    st.session_state.pending_query = None

if user_query:
    is_doc_explicit = any(term in user_query.lower() for term in [
        "uploaded document", "uploaded machine learning", "according to the uploaded",
        "in the document", "from the document", "in the pdf", "from the pdf"
    ])

    has_image = bool(uploaded_image) and not is_doc_explicit and not (is_from_suggestion and not is_visual_prompt)
    raw_image_bytes = uploaded_image.getvalue() if has_image else None

    if has_image:
        st.session_state.image_uploader_key += 1

    st.session_state.messages.append({
        "role": "user",
        "content": user_query,
        "timestamp": get_current_timestamp(),
        "image_bytes": raw_image_bytes,
    })

    with st.chat_message("user", avatar="🎓"):
        if raw_image_bytes:
            st.image(raw_image_bytes, width=320)
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🤖"):
        # Vision Mode
        if has_image and raw_image_bytes:
            with st.spinner("👁️ Analyzing visual input..."):
                vision_messages = prepare_image_payload(raw_image_bytes, user_prompt=user_query)
                answer = groq_mgr.generate_vision_response(
                    vision_messages, model=st.session_state.selected_vision_model, temperature=st.session_state.temperature
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "mode": "Vision",
                    "timestamp": get_current_timestamp(),
                })
                st.rerun()

        # Calculator Tool
        is_math, math_expr = is_math_query(user_query)
        if is_math and not has_image:
            calc_res = calculate_expression(math_expr)
            tool_data = {"tool_type": "calculator", **calc_res}
            if calc_res["success"]:
                prompt_to_llm = [
                    {
                        "role": "user",
                        "content": f"User asked: '{user_query}'. Safe calculator result: {calc_res['result']} from '{calc_res['normalized_expression']}'. Please explain mathematically.",
                    }
                ]
                explanation = groq_mgr.generate_chat_response(
                    prompt_to_llm, model=st.session_state.selected_text_model, temperature=st.session_state.temperature
                )
            else:
                err_msg = str(calc_res.get("error", ""))
                if any(term in err_msg.lower() for term in ["division or modulo by zero", "overflow", "domain error"]):
                    explanation = f"Could not calculate: {err_msg}"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": explanation,
                        "mode": "Tool",
                        "tool_data": tool_data,
                        "timestamp": get_current_timestamp(),
                    })
                    st.rerun()
                # Otherwise fall through to Web Search, RAG, and General AI

        # Web Search Tool
        is_search, search_term = is_search_query(user_query)
        if is_search and not has_image:
            with st.spinner(f"🌐 Searching the web for: {search_term}..."):
                search_res = search_web(search_term, max_results=4)
                tool_data = {"tool_type": "web_search", **search_res}
                results_summary = ""
                if search_res["success"]:
                    for r in search_res["results"]:
                        results_summary += f"- {r['title']}: {r['snippet']} ({r['url']})\n"

                prompt_to_llm = [
                    {
                        "role": "user",
                        "content": f"User Query: '{user_query}'\nWeb Results:\n{results_summary}\nSummarize clearly with sources.",
                    }
                ]
                summary_resp = groq_mgr.generate_chat_response(
                    prompt_to_llm, model=st.session_state.selected_text_model, temperature=st.session_state.temperature
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": summary_resp,
                    "mode": "Tool",
                    "tool_data": tool_data,
                    "timestamp": get_current_timestamp(),
                })
                st.rerun()

        # RAG Mode
        relevant_chunks = retriever.retrieve(
            user_query, top_k=st.session_state.top_k, threshold=st.session_state.similarity_threshold
        )
        if relevant_chunks:
            context_str = retriever.build_context_prompt(relevant_chunks)
            rag_prompt_text = build_rag_prompt(user_query, context_str)
            llm_messages = [{"role": "user", "content": rag_prompt_text}]
            with st.spinner("📚 Retrieving document excerpts and generating response..."):
                answer = groq_mgr.generate_chat_response(
                    llm_messages, model=st.session_state.selected_text_model, temperature=st.session_state.temperature
                )
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "mode": "RAG",
                "sources": relevant_chunks,
                "timestamp": get_current_timestamp(),
            })
            st.rerun()

        # General AI Mode
        with st.spinner("🧠 Thinking..."):
            gen_messages = []
            for past_msg in st.session_state.messages[-6:-1]:
                gen_messages.append({"role": past_msg["role"], "content": past_msg["content"]})
            gen_messages.append({"role": "user", "content": user_query})
            answer = groq_mgr.generate_chat_response(
                gen_messages, model=st.session_state.selected_text_model, temperature=st.session_state.temperature
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "mode": "General AI",
                "timestamp": get_current_timestamp(),
            })
            st.rerun()
