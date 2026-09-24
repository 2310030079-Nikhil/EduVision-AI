"""
Tool UI rendering components for EduVision AI.
Provides clean, polished visual badges and result cards for Calculator and Web Search tools.
"""

import streamlit as st
from typing import Dict, Any, List


def render_calculator_result(tool_data: Dict[str, Any]):
    """Render a styled visual card for calculator execution."""
    raw_expr = tool_data.get("raw_expression", "")
    norm_expr = tool_data.get("normalized_expression", "")
    result = tool_data.get("result")
    error = tool_data.get("error")

    if tool_data.get("success", False):
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, rgba(238, 242, 255, 0.9), rgba(224, 231, 255, 0.7));
                        border: 1px solid #C7D2FE; border-radius: 12px; padding: 14px 18px; margin: 10px 0;
                        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-weight: 700; color: #4338CA; font-size: 0.9rem; display: flex; align-items: center; gap: 6px;">
                        🔧 Tool Used: <b>Safe Calculator</b>
                    </span>
                    <span style="background-color: #E0E7FF; color: #3730A3; font-size: 0.75rem; padding: 2px 8px; border-radius: 9999px; font-weight: 600;">
                        AST Evaluated
                    </span>
                </div>
                <div style="font-family: monospace; font-size: 0.95rem; color: #1E1B4B; margin-bottom: 4px;">
                    Expression: <span style="background: #FFFFFF; padding: 2px 6px; border-radius: 4px; border: 1px solid #E5E7EB;">{raw_expr}</span>
                </div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #1E3A8A; margin-top: 6px;">
                    Result: <span style="color: #059669;">{result}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background-color: #FEF2F2; border: 1px solid #FECACA; border-radius: 10px; padding: 12px 16px; margin: 10px 0;">
                <div style="font-weight: 700; color: #DC2626; font-size: 0.88rem; margin-bottom: 4px;">
                    🔧 Tool Failed: Calculator Error
                </div>
                <div style="font-size: 0.85rem; color: #991B1B;">
                    {error}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_web_search_result(tool_data: Dict[str, Any]):
    """Render a styled visual card for web search results."""
    query = tool_data.get("query", "")
    results: List[Dict[str, Any]] = tool_data.get("results", [])
    error = tool_data.get("error")

    if tool_data.get("success", False) and results:
        with st.expander(f"🌐 Web Search Result: {query} ({len(results)} sources found)", expanded=False):
            for i, item in enumerate(results, 1):
                title = item.get("title", "Untitled")
                snippet = item.get("snippet", "")
                url = item.get("url", "#")
                st.markdown(
                    f"""
                    <div style="margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #E5E7EB;">
                        <a href="{url}" target="_blank" style="font-weight: 600; color: #2563EB; text-decoration: none; font-size: 0.95rem;">
                            {i}. {title} ↗
                        </a>
                        <p style="font-size: 0.85rem; color: #4B5563; margin-top: 3px; line-height: 1.4;">
                            {snippet}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    elif error:
        st.markdown(
            f"""
            <div style="background-color: #FFFBEB; border: 1px solid #FDE68A; border-radius: 8px; padding: 10px 14px; margin: 6px 0; font-size: 0.85rem; color: #92400E;">
                🌐 Web Search Notice: {error}
            </div>
            """,
            unsafe_allow_html=True,
        )
