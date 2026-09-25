"""
Safe Mathematical Calculator Tool for EduVision AI.
Parses and evaluates mathematical expressions using Python's Abstract Syntax Tree (AST).
Strictly forbids eval() and blocks arbitrary code execution, system calls, and attributes.
Supports arithmetic, percentages, powers, trigonometry, logarithms, and constants.
"""

import ast
import math
import re
from typing import Dict, Any, Union, Tuple


class SafeMathEvaluator:
    """
    Evaluates math expressions safely using AST node visitor.
    Allowed operators: +, -, *, /, //, %, **
    Allowed functions: sqrt, sin, cos, tan, asin, acos, atan, log, log10, log2, ln, exp, abs, round, floor, ceil, factorial
    Allowed constants: pi, e, tau
    """

    ALLOWED_OPERATORS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b,
        ast.USub: lambda a: -a,
        ast.UAdd: lambda a: a,
    }

    ALLOWED_FUNCTIONS = {
        "sqrt": math.sqrt,
        "cbrt": lambda x: math.pow(x, 1 / 3) if x >= 0 else -math.pow(-x, 1 / 3),
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "log": math.log,
        "ln": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "abs": abs,
        "round": round,
        "floor": math.floor,
        "ceil": math.ceil,
        "factorial": math.factorial,
    }

    ALLOWED_CONSTANTS = {
        "pi": math.pi,
        "PI": math.pi,
        "e": math.e,
        "E": math.e,
        "tau": math.tau,
    }

    def _eval_node(self, node: ast.AST) -> Union[int, float]:
        """Recursively evaluate an AST node safely."""
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body)

        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Disallowed constant type: {type(node.value).__name__}")

        elif isinstance(node, ast.Name):
            if node.id in self.ALLOWED_CONSTANTS:
                return self.ALLOWED_CONSTANTS[node.id]
            raise ValueError(f"Unknown or unauthorized variable '{node.id}'")

        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type in self.ALLOWED_OPERATORS:
                operand = self._eval_node(node.operand)
                return self.ALLOWED_OPERATORS[op_type](operand)
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")

        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type in self.ALLOWED_OPERATORS:
                left = self._eval_node(node.left)
                right = self._eval_node(node.right)
                # Protect against excessive exponentiation DoS
                if op_type == ast.Pow and (abs(right) > 1000 or abs(left) > 1e10):
                    raise OverflowError("Exponent values too large to safely compute.")
                if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                    raise ZeroDivisionError("Division or modulo by zero is undefined.")
                return self.ALLOWED_OPERATORS[op_type](left, right)
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")

        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Dynamic or nested function calls are forbidden.")
            func_name = node.func.id
            if func_name not in self.ALLOWED_FUNCTIONS:
                raise ValueError(f"Function '{func_name}' is not in the safe mathematical whitelist.")

            args = [self._eval_node(arg) for arg in node.args]
            func = self.ALLOWED_FUNCTIONS[func_name]
            return func(*args)

        else:
            raise ValueError(f"Disallowed syntax structure: {type(node).__name__}")

    def evaluate(self, expression: str) -> Union[int, float]:
        """
        Parse and evaluate an arithmetic/mathematical expression string.
        """
        cleaned = expression.strip()
        # Replace caret '^' with exponent operator '**'
        cleaned = cleaned.replace("^", "**")
        
        # Remove commas in numbers (e.g. 87,500 -> 87500)
        cleaned = re.sub(r'(\d+),(\d+)', r'\1\2', cleaned)

        # Parse with ast
        try:
            tree = ast.parse(cleaned, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid math syntax: {str(e)}")

        result = self._eval_node(tree)
        # Simplify integer floats (e.g. 5.0 -> 5)
        if isinstance(result, float) and result.is_integer():
            return int(result)
        return round(result, 6) if isinstance(result, float) else result


_evaluator = SafeMathEvaluator()


def calculate_expression(expression: str) -> Dict[str, Any]:
    """
    Main entry point for the safe calculator tool.
    Normalizes natural language percentage phrases (e.g. '15% of 87,500').
    Returns structured result dict.
    """
    raw_expr = expression.strip()
    norm_expr = raw_expr

    # Handle percentage patterns like "X% of Y" or "X percent of Y"
    percent_match = re.search(r'([\d\.]+)\s*(?:%|percent)\s+of\s+([\d,]+(?:\.\d+)?)', norm_expr, re.IGNORECASE)
    if percent_match:
        pct = float(percent_match.group(1))
        val_str = percent_match.group(2).replace(",", "")
        val = float(val_str)
        norm_expr = f"({pct} / 100) * {val}"

    # Handle standalone percentage like "50% * 200"
    norm_expr = re.sub(r'([\d\.]+)%', r'(\1/100)', norm_expr)

    try:
        result = _evaluator.evaluate(norm_expr)
        return {
            "success": True,
            "raw_expression": raw_expr,
            "normalized_expression": norm_expr,
            "result": result,
            "error": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "raw_expression": raw_expr,
            "normalized_expression": norm_expr,
            "result": None,
            "error": str(exc),
        }


def is_math_query(query: str) -> Tuple[bool, str]:
    """
    Detect if the user query contains an explicit calculation or math expression.
    Returns (is_math, extracted_expression).
    Prevents false positives on natural language questions (e.g. definitions, document queries).
    """
    q = query.strip()
    if not q:
        return False, ""

    candidate = ""

    # 1. Check for calculation prefixes
    calc_prefixes = [
        r'^(?:can you\s+)?(?:please\s+)?(?:calculate|compute|solve|evaluate)\s+(.+)$',
        r'^(?:how much is)\s+(.+)$',
        r'^(?:what is)\s+(.+)$',
    ]
    for prefix in calc_prefixes:
        match = re.match(prefix, q, re.IGNORECASE)
        if match:
            candidate = match.group(1).rstrip("?.").strip()
            break

    # 2. Check for percentage pattern if no prefix matched: e.g. "15% of 87,500"
    if not candidate:
        pct_match = re.search(r'[\d\.]+\s*(?:%|percent)\s+of\s+[\d,]+(?:\.\d+)?', q, re.IGNORECASE)
        if pct_match and len(q.split()) <= 6:
            candidate = pct_match.group(0)

    # 3. Check if raw query looks like a standalone math expression (e.g. "(10 - 4) * 2", "2^3")
    if not candidate:
        candidate = q.rstrip("?.").strip()

    if not candidate:
        return False, ""

    # Whitelist of allowed word tokens in mathematical expressions
    allowed_math_words = {
        k.lower() for k in list(SafeMathEvaluator.ALLOWED_FUNCTIONS.keys()) + list(SafeMathEvaluator.ALLOWED_CONSTANTS.keys())
    } | {"of", "percent", "percentage"}

    # Extract all alphabetic word tokens
    words = re.findall(r'[a-zA-Z_]+', candidate)

    # If there are no words and no digits, it cannot be a calculation
    if not words and not re.search(r'\d', candidate):
        return False, ""

    # If ANY word in the candidate is NOT in the allowed math words whitelist,
    # it is natural language (e.g. "bias", "variance", "tradeoff", "machine", "learning", "document", "logistic")
    if any(w.lower() not in allowed_math_words for w in words):
        return False, ""

    # Must contain at least one digit or recognized math function
    has_digit = bool(re.search(r'\d', candidate))
    has_func = any(w.lower() in SafeMathEvaluator.ALLOWED_FUNCTIONS for w in words)
    if not (has_digit or has_func):
        return False, ""

    # Must contain at least one math operator, function, or percentage phrase
    has_operator = any(op in candidate for op in ['+', '-', '*', '/', '^', '%'])
    has_pct = bool(re.search(r'(?:%|percent)\s+of', candidate, re.IGNORECASE))
    if not (has_operator or has_func or has_pct):
        return False, ""

    # Normalize expression for AST syntax verification
    norm = candidate.replace("^", "**")
    norm = re.sub(r'(\d+),(\d+)', r'\1\2', norm)
    pct_m = re.search(r'([\d\.]+)\s*(?:%|percent)\s+of\s+([\d,]+(?:\.\d+)?)', norm, re.IGNORECASE)
    if pct_m:
        pct = float(pct_m.group(1))
        val = float(pct_m.group(2).replace(",", ""))
        norm = f"({pct}/100)*{val}"
    norm = re.sub(r'([\d\.]+)%', r'(\1/100)', norm)

    try:
        tree = ast.parse(norm, mode="eval")
        # Ensure only allowed node types and valid identifiers exist
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id not in SafeMathEvaluator.ALLOWED_CONSTANTS and node.id not in SafeMathEvaluator.ALLOWED_FUNCTIONS:
                    return False, ""
            elif isinstance(node, (ast.Import, ast.ImportFrom, ast.Attribute, ast.Lambda)):
                return False, ""
        return True, candidate
    except Exception:
        return False, ""
