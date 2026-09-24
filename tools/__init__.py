"""Tools package initialization."""
from tools.calculator import calculate_expression, is_math_query
from tools.web_search import search_web, is_search_query

__all__ = ["calculate_expression", "is_math_query", "search_web", "is_search_query"]
