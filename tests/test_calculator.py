"""
Unit tests for the safe mathematical calculator tool.
Verifies arithmetic correctness, percentage evaluation, function execution,
error handling, and robust prevention of code injection.
"""

import pytest
from tools.calculator import calculate_expression, is_math_query, SafeMathEvaluator


def test_basic_arithmetic():
    assert calculate_expression("2 + 2")["result"] == 4
    assert calculate_expression("10 - 4 * 2")["result"] == 2
    assert calculate_expression("(10 - 4) * 2")["result"] == 12
    assert calculate_expression("100 / 4")["result"] == 25
    assert calculate_expression("2^3")["result"] == 8
    assert calculate_expression("2 ** 4")["result"] == 16


def test_percentage_calculations():
    # Primary user requirement example
    res1 = calculate_expression("15% of 87,500")
    assert res1["success"] is True
    assert res1["result"] == 13125

    res2 = calculate_expression("20% of 500")
    assert res2["success"] is True
    assert res2["result"] == 100

    res3 = calculate_expression("50% * 200")
    assert res3["success"] is True
    assert res3["result"] == 100


def test_math_functions_and_constants():
    assert calculate_expression("sqrt(144)")["result"] == 12
    assert calculate_expression("round(sin(pi / 2))")["result"] == 1
    assert calculate_expression("log10(1000)")["result"] == 3
    assert calculate_expression("factorial(5)")["result"] == 120
    assert calculate_expression("abs(-42)")["result"] == 42


def test_division_by_zero():
    res = calculate_expression("10 / 0")
    assert res["success"] is False
    assert "division or modulo by zero" in res["error"].lower()


def test_injection_prevention():
    # Attempting to call unauthorized functions
    res1 = calculate_expression("__import__('os').system('dir')")
    assert res1["success"] is False

    res2 = calculate_expression("open('/etc/passwd')")
    assert res2["success"] is False

    res3 = calculate_expression("eval('2+2')")
    assert res3["success"] is False


def test_math_query_detection():
    is_m, expr = is_math_query("Calculate 15% of 87,500")
    assert is_m is True
    assert "15% of 87,500" in expr

    is_m, expr = is_math_query("What is 45 * 12?")
    assert is_m is True

    is_m, _ = is_math_query("What is the capital of France?")
    assert is_m is False

    # Prevent false positives on document queries and natural language
    is_m, _ = is_math_query("What is the bias-variance tradeoff according to the uploaded Machine Learning document?")
    assert is_m is False

    is_m, _ = is_math_query("Explain the difference between Supervised and Unsupervised Learning with examples.")
    assert is_m is False

    is_m, _ = is_math_query("What is logistic regression?")
    assert is_m is False

    is_m, _ = is_math_query("What is cosine similarity?")
    assert is_m is False

    is_m, _ = is_math_query("What is 3-tier architecture?")
    assert is_m is False

    is_m, _ = is_math_query("What is machine-learning?")
    assert is_m is False

    is_m, _ = is_math_query("What is a trade-off in software design?")
    assert is_m is False

    # Valid math variations
    assert is_math_query("Compute sqrt(144) + 10")[0] is True
    assert is_math_query("What is 2 + 2?")[0] is True
    assert is_math_query("15% of 87,500")[0] is True
    assert is_math_query("(10 - 4) * 2")[0] is True
    assert is_math_query("sin(pi / 2)")[0] is True
    assert is_math_query("What is 10 / 0?")[0] is True
    assert is_math_query("What is 42?")[0] is False
