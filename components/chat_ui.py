"""
Chat UI component for EduVision AI.
Renders message history, multimodal media previews, response mode badges,
interactive starter prompt cards, and conversation controls.
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from components.source_display import render_sources
from components.tool_ui import render_calculator_result, render_web_search_result


MODE_BADGES = {
    "General AI": ("🧠 General AI", "#3B82F6", "#EFF6FF", "#BFDBFE"),
    "RAG": ("📚 RAG", "#059669", "#ECFDF5", "#A7F3D0"),
    "Vision": ("👁️ Vision", "#7C3AED", "#F5F3FF", "#DDD6FE"),
    "Tool": ("🔧 Tool", "#D97706", "#FFFBEB", "#FDE68A"),
}


def render_mode_badge(mode_key: str):
    """Render a colored mode status pill."""
    label, text_color, bg_color, border_color = MODE_BADGES.get(
        mode_key, ("🧠 General AI", "#3B82F6", "#EFF6FF", "#BFDBFE")
    )
    st.markdown(
        f"""
        <span style="display: inline-block; background-color: {bg_color}; color: {text_color};
                     border: 1px solid {border_color}; border-radius: 9999px;
                     padding: 3px 10px; font-size: 0.78rem; font-weight: 700; margin-bottom: 8px;">
            {label}
        </span>
        """,
        unsafe_allow_html=True,
    )


def render_landing_state():
    """Render the welcome hero banner and suggested starter prompt cards."""
    st.markdown(
        """
        <div style="text-align: center; padding: 28px 20px; background: linear-gradient(135deg, rgba(239, 246, 255, 0.6), rgba(245, 243, 255, 0.6));
                    border-radius: 16px; border: 1px solid #E0E7FF; margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
            <div style="font-size: 2.2rem; font-weight: 800; color: #1E3A8A; margin-bottom: 6px;">
                👋 Welcome to <span style="background: linear-gradient(90deg, #2563EB, #7C3AED); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">EduVision AI</span>
            </div>
            <div style="font-size: 1.15rem; font-weight: 600; color: #4B5563; margin-bottom: 12px;">
                Your Multimodal AI Learning Companion
            </div>
            <div style="font-size: 0.95rem; color: #6B7280; max-width: 650px; margin: 0 auto; line-height: 1.6;">
                "Ask questions, upload documents, analyze images, and learn with context-aware AI."
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 💡 Suggested Exploration Prompts")
    c1, c2 = st.columns(2)

    with c1:
        if st.button("📚 What is the bias-variance tradeoff in Machine Learning?", use_container_width=True):
            st.session_state.pending_query = "What is the bias-variance tradeoff according to the uploaded Machine Learning document?"
            st.rerun()

        if st.button("🧠 Explain the difference between Supervised and Unsupervised Learning.", use_container_width=True):
            st.session_state.pending_query = "Explain the difference between Supervised and Unsupervised Learning with examples."
            st.rerun()

    with c2:
        if st.button("🔧 Calculate 15% of 87,500", use_container_width=True):
            st.session_state.pending_query = "Calculate 15% of 87,500."
            st.rerun()

        if st.button("🖼️ Analyze an attached mathematical problem step by step", use_container_width=True):
            st.session_state.pending_query = "Please explain and solve the problem shown in my uploaded image step by step."
            st.rerun()


def render_chat_messages(messages: List[Dict[str, Any]]):
    """Render conversation message history with avatars, mode badges, images, and tools."""
    if not messages:
        render_landing_state()
        return

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        mode = msg.get("mode")
        image_bytes = msg.get("image_bytes")
        tool_data = msg.get("tool_data")
        sources = msg.get("sources")
        timestamp = msg.get("timestamp", "")

        avatar = "🎓" if role == "user" else "🤖"

        with st.chat_message(role, avatar=avatar):
            if role == "assistant" and mode:
                render_mode_badge(mode)

            # Display uploaded image preview if attached to user message
            if image_bytes:
                st.image(image_bytes, caption="📸 Uploaded Visual Context", use_container_width=False, width=320)

            # Display tool results if applicable
            if tool_data:
                tool_type = tool_data.get("tool_type", "calculator")
                if tool_type == "calculator":
                    render_calculator_result(tool_data)
                elif tool_type == "web_search":
                    render_web_search_result(tool_data)

            # Main text content
            if content:
                st.markdown(content)

            # Display RAG citations if present
            if sources:
                render_sources(sources)
