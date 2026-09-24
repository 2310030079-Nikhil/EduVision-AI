"""
Web Search Tool for EduVision AI.
Provides real-time web search capabilities using DuckDuckGo.
Safely extracts query results, formats citations, and returns structured data.
"""

from typing import List, Dict, Any, Tuple
import re

try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False


def search_web(query: str, max_results: int = 4) -> Dict[str, Any]:
    """
    Search the web using DuckDuckGo.
    Returns a dictionary with search status, results, and formatted summary.
    """
    clean_query = query.strip()
    if not clean_query:
        return {"success": False, "query": "", "results": [], "error": "Query cannot be empty."}

    if not HAS_DDGS:
        return {
            "success": False,
            "query": clean_query,
            "results": [],
            "error": "duckduckgo_search package is not available in the current environment.",
        }

    try:
        results = []
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(clean_query, max_results=max_results))
            for item in raw_results:
                results.append({
                    "title": item.get("title", "No Title"),
                    "snippet": item.get("body", item.get("snippet", "")),
                    "url": item.get("href", item.get("link", "")),
                })

        return {
            "success": True,
            "query": clean_query,
            "results": results,
            "error": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "query": clean_query,
            "results": [],
            "error": f"Search request failed: {str(exc)}",
        }


def is_search_query(query: str) -> Tuple[bool, str]:
    """
    Determine if a user query is asking for current web search info.
    Returns (is_search, clean_search_term).
    """
    q = query.strip()
    search_triggers = [
        r'^(?:search(?:\s+the\s+web)?(?:\s+for)?|google|lookup|find\s+online)\s+(.+)$',
        r'^(?:what\s+are\s+the\s+latest|who\s+won|recent\s+news\s+about)\s+(.+)$',
    ]
    for pattern in search_triggers:
        match = re.match(pattern, q, re.IGNORECASE)
        if match:
            return True, match.group(1).rstrip("?.").strip()

    # Keywords signaling web search needs
    if any(k in q.lower() for k in ["latest developments in", "current news on", "recent updates on"]):
        return True, q

    return False, ""
