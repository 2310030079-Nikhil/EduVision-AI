"""
EduVision AI — Multimodal RAG Learning Assistant.
Main Streamlit application entrypoint.
Combines Multimodal Vision, Genuine RAG, FAISS Vector Search,
Safe Tool Calling, and a modern AI SaaS user interface.
"""

import streamlit as st

# Streamlit Page Setup (Must be first Streamlit command)
st.set_page_config(
    page_title="EduVision AI — Multimodal RAG Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os
from pathlib import Path
from PIL import Image

# Import internal modules
from utils.config import (
    APP_NAME,
    APP_TAGLINE,
    APP_VERSION,
    DEFAULT_TEXT_MODEL,
    DEFAULT_VISION_MODEL,
    TEXT_MODELS,
    VISION_MODELS,
    DEFAULT_TOP_K,
    SIMILARITY_THRESHOLD,
    SAMPLE_DOCS_DIR,
)
from utils.helpers import get_current_timestamp
from utils.validators import validate_image_file, validate_api_key
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

# Components
from components.chat_ui import render_chat_messages
from components.document_ui import render_knowledge_base_ui
from components.source_display import render_sources

# Custom Styling for AI SaaS Polish
st.markdown(
    """
    <style>
        /* Modern Typography & Clean Font Weights */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Metric cards in sidebar */
        .metric-card {
            background: linear-gradient(135deg, rgba(243, 244, 246, 0.8), rgba(249, 250, 251, 0.9));
            border: 1px solid #E5E7EB;
            border-radius: 10px;
            padding: 10px 14px;
            margin-bottom: 8px;
        }
        .metric-label {
            font-size: 0.75rem;
            color: #6B7280;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-value {
            font-size: 1.05rem;
            font-weight: 700;
            color: #111827;
            margin-top: 2px;
        }

        /* Hide Streamlit default hamburger menu & footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Smooth Chat message containers */
        .stChatMessage {
            border-radius: 12px;
            padding: 8px 12px;
            margin-bottom: 12px;
        }

        /* Custom scrollbars */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-thumb {
            background: #CBD5E1;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #94A3B8;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_session_state():
    """Ensure all required session state variables are instantiated."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vector_store" not in st.session_state:
        vs = VectorStore()
        # Attempt to load sample document automatically if empty
        sample_path = SAMPLE_DOCS_DIR / "introduction_to_ai_ml.pdf"
        if sample_path.exists():
            try:
                doc_data = load_pdf_from_path(str(sample_path))
                chunks = chunk_document(doc_data)
                embeddings = embedding_service.embed_texts([c["source_text"] for c in chunks])
                vs.add_chunks(chunks, embeddings)
                st.session_state.indexed_documents = [{
                    "document_name": "introduction_to_ai_ml.pdf",
                    "pages": doc_data["total_pages"],
                    "chunks": len(chunks),
                    "chars": doc_data["char_count"],
                    "status": "Ready",
                }]
            except Exception as e:
                print(f"[Notice] Auto-load sample doc skipped: {e}")
                st.session_state.indexed_documents = []
        else:
            st.session_state.indexed_documents = []
        st.session_state.vector_store = vs

    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")

    if "groq_client" not in st.session_state:
        st.session_state.groq_client = GroqClientManager(st.session_state.groq_api_key)

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

    if "active_nav" not in st.session_state:
        st.session_state.active_nav = "💬 Chat"


initialize_session_state()

# References to session objects
vector_store: VectorStore = st.session_state.vector_store
retriever = RAGRetriever(vector_store)
groq_mgr: GroqClientManager = st.session_state.groq_client


# ==========================================
# SIDEBAR NAVIGATION & SYSTEM METRICS
# ==========================================
with st.sidebar:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
            <div style="font-size: 1.8rem;">🎓</div>
            <div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #1E3A8A; line-height: 1.1;">{APP_NAME}</div>
                <div style="font-size: 0.72rem; color: #6B7280; font-weight: 500;">v{APP_VERSION} • Multimodal RAG</div>
            </div>
        </div>
        <p style="font-size: 0.8rem; color: #4B5563; margin-top: 4px; line-height: 1.3;">
            <em>"{APP_TAGLINE}"</em>
        </p>
        <hr style="margin: 10px 0 14px 0; border: none; border-top: 1px solid #E5E7EB;" />
        """,
        unsafe_allow_html=True,
    )

    # Navigation Menu
    nav_options = ["💬 Chat", "📚 Knowledge Base", "📊 RAG Sources", "⚙️ Settings"]
    selected_nav = st.radio(
        "Navigation",
        options=nav_options,
        index=nav_options.index(st.session_state.active_nav) if st.session_state.active_nav in nav_options else 0,
        label_visibility="collapsed",
    )
    st.session_state.active_nav = selected_nav

    st.markdown("<hr style='margin: 14px 0 10px 0; border: none; border-top: 1px solid #E5E7EB;' />", unsafe_allow_html=True)
    st.markdown("#### 📊 System Status")

    # Display status metrics
    total_docs = len(st.session_state.get("indexed_documents", []))
    total_chunks = vector_store.total_chunks
    rag_status_label = f"🟢 Ready ({total_chunks} Chunks)" if total_chunks > 0 else "⚪ No Docs Indexed"
    api_ready = groq_mgr.is_configured()
    api_status_label = "🟢 Connected" if api_ready else "🟡 Key Not Set (Local Demo)"

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Documents Indexed</div>
            <div class="metric-value">📄 {total_docs} Files ({total_chunks} Chunks)</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Active Model</div>
            <div class="metric-value" style="font-size: 0.88rem; font-family: monospace;">{st.session_state.selected_text_model}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">RAG Engine</div>
            <div class="metric-value" style="font-size: 0.88rem;">{rag_status_label}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Groq API</div>
            <div class="metric-value" style="font-size: 0.88rem;">{api_status_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='margin: 12px 0; border: none; border-top: 1px solid #E5E7EB;' />", unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.rerun()
    with col_btn2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.rerun()


# ==========================================
# VIEW: 💬 CHAT
# ==========================================
if st.session_state.active_nav == "💬 Chat":
    # Header Banner
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div>
                <h2 style="margin: 0; font-weight: 800; color: #111827; font-size: 1.6rem;">Interactive Learning Assistant</h2>
                <p style="margin: 0; font-size: 0.88rem; color: #6B7280;">Multimodal Q&A • Document RAG • Vision Analysis • Safe Tools</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Ensure image uploader key exists in session state
    if "image_uploader_key" not in st.session_state:
        st.session_state.image_uploader_key = 0

    # Optional Image Attachment Drawer
    with st.expander("🖼️ Attach Image for Multimodal Vision Analysis (Optional)", expanded=False):
        col_img_up, col_img_btn = st.columns([3, 1])
        with col_img_up:
            uploaded_image = st.file_uploader(
                "Upload mathematical problem, diagram, chart, or screenshot (PNG, JPG, WEBP)",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"chat_image_uploader_{st.session_state.image_uploader_key}",
            )
        with col_img_btn:
            if uploaded_image:
                st.write("")
                if st.button("🗑️ Discard Image", key="discard_img_btn", use_container_width=True):
                    st.session_state.image_uploader_key += 1
                    st.rerun()

        if uploaded_image:
            img_bytes = uploaded_image.getvalue()
            v_info = validate_and_process_image(img_bytes)
            if v_info["valid"]:
                st.image(img_bytes, caption=f"Preview: {uploaded_image.name} ({v_info['width']}x{v_info['height']} px)", width=280)
            else:
                st.error(v_info["error"])

    # Render Conversation Messages
    render_chat_messages(st.session_state.messages)

    # Handle incoming query from either chat_input or a clicked suggestion prompt
    user_query = st.chat_input("Ask a question, upload a problem, or ask about your documents...")
    is_from_suggestion = False
    is_visual_prompt = False
    if st.session_state.pending_query:
        user_query = st.session_state.pending_query
        is_from_suggestion = True
        is_visual_prompt = "uploaded image" in user_query.lower() or "problem shown" in user_query.lower()
        st.session_state.pending_query = None

    if user_query:
        # Check if user query explicitly refers to documents
        is_doc_explicit = any(term in user_query.lower() for term in [
            "uploaded document", "uploaded machine learning", "according to the uploaded",
            "in the document", "from the document", "in the pdf", "from the pdf"
        ])

        # Attach image ONLY if:
        # 1. An image is uploaded
        # 2. Query is NOT an explicit document query
        # 3. If query is from suggestion prompts, it MUST be the visual prompt
        has_image = bool(uploaded_image) and not is_doc_explicit and not (is_from_suggestion and not is_visual_prompt)
        raw_image_bytes = uploaded_image.getvalue() if has_image else None

        # Reset uploader once an image is submitted so it doesn't linger for future turns
        if has_image:
            st.session_state.image_uploader_key += 1

        # 1. Append User Message
        user_msg = {
            "role": "user",
            "content": user_query,
            "timestamp": get_current_timestamp(),
            "image_bytes": raw_image_bytes,
        }
        st.session_state.messages.append(user_msg)

        # Rerender user message immediately in chat
        with st.chat_message("user", avatar="🎓"):
            if raw_image_bytes:
                st.image(raw_image_bytes, caption="Uploaded Visual Context", width=320)
            st.markdown(user_query)

        # 2. Determine Mode & Generate Assistant Response
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()

            # Mode A: Vision (Image provided)
            if has_image and raw_image_bytes:
                with st.spinner("👁️ Analyzing visual input with Groq Vision Model..."):
                    vision_messages = prepare_image_payload(
                        image_bytes=raw_image_bytes,
                        user_prompt=user_query,
                    )
                    answer = groq_mgr.generate_vision_response(
                        messages=vision_messages,
                        model=st.session_state.selected_vision_model,
                        temperature=st.session_state.temperature,
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "mode": "Vision",
                        "timestamp": get_current_timestamp(),
                    })
                    st.rerun()

            # Mode B: Safe Calculator Tool (Math problem detected)
            is_math, math_expr = is_math_query(user_query)
            if is_math and not has_image:
                calc_res = calculate_expression(math_expr)
                tool_data = {
                    "tool_type": "calculator",
                    **calc_res,
                }

                if calc_res["success"]:
                    prompt_to_llm = [
                        {
                            "role": "user",
                            "content": (
                                f"The user asked: '{user_query}'.\n"
                                f"The safe calculator computed the result: {calc_res['result']} "
                                f"from mathematical expression: '{calc_res['normalized_expression']}'.\n"
                                f"Please provide a structured educational response following the Mathematical style "
                                f"(Given, Formula, Calculation, Final Answer), incorporating this verified tool result."
                            ),
                        }
                    ]
                    explanation = groq_mgr.generate_chat_response(
                        messages=prompt_to_llm,
                        model=st.session_state.selected_text_model,
                        temperature=st.session_state.temperature,
                        stream=False,
                    )
                else:
                    err_msg = str(calc_res.get("error", ""))
                    if any(term in err_msg.lower() for term in ["division or modulo by zero", "overflow", "domain error"]):
                        explanation = f"Could not compute calculation: {err_msg}"
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": explanation,
                            "mode": "Tool",
                            "tool_data": tool_data,
                            "timestamp": get_current_timestamp(),
                        })
                        st.rerun()
                    # If not a math runtime error, fall through to Web Search, RAG, or General AI

            # Mode C: Web Search Tool (Search query detected)
            is_search, search_term = is_search_query(user_query)
            if is_search and not has_image:
                with st.spinner(f"🌐 Searching the web for: {search_term}..."):
                    search_res = search_web(search_term, max_results=4)
                    tool_data = {
                        "tool_type": "web_search",
                        **search_res,
                    }

                    results_summary = ""
                    if search_res["success"]:
                        for r in search_res["results"]:
                            results_summary += f"- {r['title']}: {r['snippet']} ({r['url']})\n"

                    prompt_to_llm = [
                        {
                            "role": "user",
                            "content": (
                                f"User Query: '{user_query}'\n\n"
                                f"Web Search Results:\n{results_summary}\n\n"
                                f"Please provide a well-structured summary answering the user's question, "
                                f"citing the sources from the web search."
                            ),
                        }
                    ]
                    summary_resp = groq_mgr.generate_chat_response(
                        messages=prompt_to_llm,
                        model=st.session_state.selected_text_model,
                        temperature=st.session_state.temperature,
                        stream=False,
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": summary_resp,
                        "mode": "Tool",
                        "tool_data": tool_data,
                        "timestamp": get_current_timestamp(),
                    })
                    st.rerun()

            # Mode D: Document RAG (Search Vector DB for matching context)
            relevant_chunks = retriever.retrieve(
                query=user_query,
                top_k=st.session_state.top_k,
                threshold=st.session_state.similarity_threshold,
            )

            if relevant_chunks:
                # Grounded RAG mode
                context_str = retriever.build_context_prompt(relevant_chunks)
                rag_prompt_text = build_rag_prompt(user_query, context_str)

                # Build conversation messages including RAG prompt
                llm_messages = [{"role": "user", "content": rag_prompt_text}]

                with st.spinner("📚 Retrieving document excerpts and generating grounded response..."):
                    answer = groq_mgr.generate_chat_response(
                        messages=llm_messages,
                        model=st.session_state.selected_text_model,
                        temperature=st.session_state.temperature,
                        stream=False,
                    )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "mode": "RAG",
                    "sources": relevant_chunks,
                    "timestamp": get_current_timestamp(),
                })
                st.rerun()

            # Mode E: General AI (No document match or vector store empty)
            with st.spinner("🧠 Generating educational response..."):
                gen_messages = []
                # Provide previous conversation context (last 6 messages)
                for past_msg in st.session_state.messages[-6:-1]:
                    gen_messages.append({"role": past_msg["role"], "content": past_msg["content"]})
                gen_messages.append({"role": "user", "content": user_query})

                answer = groq_mgr.generate_chat_response(
                    messages=gen_messages,
                    model=st.session_state.selected_text_model,
                    temperature=st.session_state.temperature,
                    stream=False,
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "mode": "General AI",
                    "timestamp": get_current_timestamp(),
                })
                st.rerun()


# ==========================================
# VIEW: 📚 KNOWLEDGE BASE
# ==========================================
elif st.session_state.active_nav == "📚 Knowledge Base":
    render_knowledge_base_ui(vector_store)


# ==========================================
# VIEW: 📊 RAG SOURCES & RETRIEVAL INSPECTOR
# ==========================================
elif st.session_state.active_nav == "📊 RAG Sources":
    st.markdown("### 📊 RAG Sources & Vector Search Inspector")
    st.markdown(
        "Directly test similarity retrieval queries against the FAISS vector database to evaluate "
        "matching chunks, cosine similarity scores, and metadata provenance."
    )

    test_query = st.text_input(
        "Enter a test query to inspect retrieved vector chunks:",
        placeholder="e.g., What causes overfitting in machine learning models?",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        top_k_test = st.slider("Top K chunks to retrieve", 1, 10, st.session_state.top_k)
    with col2:
        threshold_test = st.slider("Similarity score threshold", 0.0, 1.0, float(st.session_state.similarity_threshold), 0.05)

    if test_query:
        with st.spinner("Querying vector database..."):
            test_results = retriever.retrieve(
                query=test_query,
                top_k=top_k_test,
                threshold=threshold_test,
            )

        if test_results:
            st.success(f"Found {len(test_results)} relevant chunks meeting threshold >= {threshold_test:.2f}")
            render_sources(test_results, title="Retrieved Evidence Chunks")
        else:
            st.warning(f"No chunks exceeded the similarity threshold of {threshold_test:.2f}. Try lowering the threshold or asking about indexed topics.")

    st.markdown("---")
    st.markdown("#### 📈 Vector Store Telemetry")
    stats = vector_store.get_stats()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Indexed Chunks", stats["total_chunks"])
    m2.metric("Total Documents", stats["total_documents"])
    m3.metric("Estimated Tokens", f"{stats['total_tokens_est']:,}")
    m4.metric("Vector Backend", stats["backend"])


# ==========================================
# VIEW: ⚙️ SETTINGS
# ==========================================
elif st.session_state.active_nav == "⚙️ Settings":
    st.markdown("### ⚙️ System Configuration & Credentials")

    st.markdown("#### 🔑 Groq API Key")
    st.markdown(
        "EduVision AI connects directly to Groq's LPUs for ultra-fast text and multimodal vision inference. "
        "Get a free API key at [console.groq.com/keys](https://console.groq.com/keys)."
    )

    current_key = st.session_state.groq_api_key
    new_key = st.text_input(
        "Groq API Key",
        value=current_key,
        type="password",
        placeholder="gsk_...",
        help="Stored in session memory only. Never logged or exposed.",
    )

    c_save, c_test = st.columns([1, 1])
    with c_save:
        if st.button("💾 Save Key", use_container_width=True):
            st.session_state.groq_api_key = new_key
            groq_mgr.update_api_key(new_key)
            st.success("API key updated for current session!")
            st.rerun()

    with c_test:
        if st.button("🧪 Test API Connection", use_container_width=True):
            test_res = groq_mgr.test_connection()
            if test_res["success"]:
                st.success(f"✅ Connection successful! Model '{test_res['model']}' responded.")
            else:
                st.error(f"❌ Connection failed: {test_res['error']}")

    st.markdown("---")
    st.markdown("#### 🤖 Model Selection")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.session_state.selected_text_model = st.selectbox(
            "Text Model (Chat & RAG)",
            options=TEXT_MODELS,
            index=TEXT_MODELS.index(st.session_state.selected_text_model) if st.session_state.selected_text_model in TEXT_MODELS else 0,
            help="High-capacity Groq models for reasoning and educational responses.",
        )
    with col_m2:
        st.session_state.selected_vision_model = st.selectbox(
            "Vision Model (Multimodal Image Analysis)",
            options=VISION_MODELS,
            index=VISION_MODELS.index(st.session_state.selected_vision_model) if st.session_state.selected_vision_model in VISION_MODELS else 0,
            help="Groq multimodal vision model for diagrams, charts, and math screenshots.",
        )

    st.markdown("---")
    st.markdown("#### 🎛️ RAG & Generation Hyperparameters")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, float(st.session_state.temperature), 0.05)
    with col_p2:
        st.session_state.top_k = st.slider("RAG Top-K Chunks", 1, 10, int(st.session_state.top_k))
    with col_p3:
        st.session_state.similarity_threshold = st.slider(
            "Similarity Threshold", 0.0, 1.0, float(st.session_state.similarity_threshold), 0.05
        )
