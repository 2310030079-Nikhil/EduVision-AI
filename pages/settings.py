"""
Settings Page for EduVision AI (Multipage support).
Allows Groq API key configuration, model selection, and RAG parameter tuning.
"""

import streamlit as st
import os
from utils.config import (
    DEFAULT_TEXT_MODEL,
    DEFAULT_VISION_MODEL,
    TEXT_MODELS,
    VISION_MODELS,
)
from ai.groq_client import GroqClientManager

if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")
if "groq_client" not in st.session_state:
    st.session_state.groq_client = GroqClientManager(st.session_state.groq_api_key)
if "selected_text_model" not in st.session_state:
    st.session_state.selected_text_model = DEFAULT_TEXT_MODEL
if "selected_vision_model" not in st.session_state:
    st.session_state.selected_vision_model = DEFAULT_VISION_MODEL
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.3
if "top_k" not in st.session_state:
    st.session_state.top_k = 4
if "similarity_threshold" not in st.session_state:
    st.session_state.similarity_threshold = 0.32

groq_mgr: GroqClientManager = st.session_state.groq_client

st.markdown("### ⚙️ System Configuration")

st.markdown("#### 🔑 Groq API Key")
new_key = st.text_input(
    "Groq API Key",
    value=st.session_state.groq_api_key,
    type="password",
    placeholder="gsk_...",
)

c1, c2 = st.columns(2)
with c1:
    if st.button("💾 Save Key", use_container_width=True):
        st.session_state.groq_api_key = new_key
        groq_mgr.update_api_key(new_key)
        st.success("API key updated!")

with c2:
    if st.button("🧪 Test Connection", use_container_width=True):
        res = groq_mgr.test_connection()
        if res["success"]:
            st.success(f"Connection OK! Model '{res['model']}' responded.")
        else:
            st.error(f"Error: {res['error']}")

st.markdown("---")
st.markdown("#### 🤖 Model Selection")
cm1, cm2 = st.columns(2)
with cm1:
    st.session_state.selected_text_model = st.selectbox(
        "Text Model",
        options=TEXT_MODELS,
        index=TEXT_MODELS.index(st.session_state.selected_text_model) if st.session_state.selected_text_model in TEXT_MODELS else 0,
    )
with cm2:
    st.session_state.selected_vision_model = st.selectbox(
        "Vision Model",
        options=VISION_MODELS,
        index=VISION_MODELS.index(st.session_state.selected_vision_model) if st.session_state.selected_vision_model in VISION_MODELS else 0,
    )

st.markdown("---")
st.markdown("#### 🎛️ Parameters")
cp1, cp2, cp3 = st.columns(3)
with cp1:
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, float(st.session_state.temperature), 0.05)
with cp2:
    st.session_state.top_k = st.slider("RAG Top-K", 1, 10, int(st.session_state.top_k))
with cp3:
    st.session_state.similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, float(st.session_state.similarity_threshold), 0.05)
