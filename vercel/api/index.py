"""
Vercel Serverless Python Function for EduVision AI.
Provides REST API endpoints for AI Chat, Multimodal Vision, Web Search, Safe Calculator, and Health Status.
Runs natively on Vercel Edge / Serverless infrastructure.
"""

from http.server import BaseHTTPRequestHandler
import json
import urllib.parse
import os
import sys
from pathlib import Path

# Add project root to sys.path so project modules can be imported
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Attempt imports for modules
try:
    from tools.calculator import calculate_expression, is_math_query
except ImportError:
    calculate_expression = None
    is_math_query = lambda q: (False, "")

try:
    from tools.web_search import search_web, is_search_query
except ImportError:
    search_web = None
    is_search_query = lambda q: (False, "")

try:
    from ai.groq_client import GroqClientManager
except ImportError:
    GroqClientManager = None


class handler(BaseHTTPRequestHandler):
    """Vercel Python Serverless HTTP Request Handler."""

    def _send_json_response(self, data: dict, status_code: int = 200):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))

    def do_OPTIONS(self):
        """Handle CORS pre-flight."""
        self._send_json_response({"status": "ok"})

    def do_GET(self):
        """Handle GET requests."""
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path.endswith("/health") or path == "/api/health" or path == "/api":
            api_key = os.getenv("GROQ_API_KEY", "")
            self._send_json_response({
                "status": "healthy",
                "service": "EduVision AI Serverless API",
                "version": "1.0.0",
                "groq_configured": bool(api_key and len(api_key.strip()) > 0),
                "endpoints": [
                    {"path": "/api/health", "method": "GET"},
                    {"path": "/api/calculate?expr=15%25+of+87500", "method": "GET"},
                    {"path": "/api/search?q=quantum+computing", "method": "GET"},
                    {"path": "/api/chat", "method": "POST"},
                    {"path": "/api/vision", "method": "POST"},
                ],
            })
            return

        if "/calculate" in path:
            params = urllib.parse.parse_qs(parsed_url.query)
            expr = params.get("expr", [""])[0]
            if not expr:
                self._send_json_response({"error": "Missing 'expr' query parameter."}, 400)
                return

            if calculate_expression:
                res = calculate_expression(expr)
                self._send_json_response(res)
            else:
                self._send_json_response({"error": "Calculator module not loaded."}, 500)
            return

        if "/search" in path:
            params = urllib.parse.parse_qs(parsed_url.query)
            query = params.get("q", [""])[0]
            if not query:
                self._send_json_response({"error": "Missing 'q' query parameter."}, 400)
                return

            if search_web:
                res = search_web(query)
                self._send_json_response(res)
            else:
                self._send_json_response({"error": "Web search module not loaded."}, 500)
            return

        # Default info
        self._send_json_response({
            "message": "EduVision AI Serverless API Gateway is operational.",
            "documentation": "Full interactive web workspace running directly on Vercel.",
        })

    def do_POST(self):
        """Handle POST requests for chat, vision, calculation, or search."""
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}

            # Handle /api/calculate
            if "/calculate" in path or ("expression" in body and "/chat" not in path):
                expr = body.get("expression") or body.get("expr", "")
                if calculate_expression and expr:
                    res = calculate_expression(expr)
                    self._send_json_response(res)
                else:
                    self._send_json_response({"error": "Invalid calculation payload."}, 400)
                return

            # Handle /api/search
            if "/search" in path:
                query = body.get("query") or body.get("q", "")
                if search_web and query:
                    res = search_web(query)
                    self._send_json_response(res)
                else:
                    self._send_json_response({"error": "Invalid search payload."}, 400)
                return

            # Handle /api/vision
            if "/vision" in path or "image_b64" in body or "image_url" in body:
                user_api_key = body.get("api_key") or os.getenv("GROQ_API_KEY", "")
                prompt = body.get("prompt", "Analyze this educational image, diagram, or problem in detail.")
                model = body.get("model", "llama-3.2-11b-vision-preview")

                if not GroqClientManager:
                    self._send_json_response({"error": "Groq client library not available."}, 500)
                    return

                groq_mgr = GroqClientManager(api_key=user_api_key)
                if not groq_mgr.is_configured():
                    self._send_json_response({"error": "Groq API key is not configured. Please enter a valid key in Settings."}, 400)
                    return

                image_url = body.get("image_url", "")
                image_b64 = body.get("image_b64", "")

                if not image_url and image_b64:
                    if not image_b64.startswith("data:"):
                        image_url = f"data:image/jpeg;base64,{image_b64}"
                    else:
                        image_url = image_b64

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ]

                answer = groq_mgr.generate_vision_response(messages=messages, model=model)
                self._send_json_response({"success": True, "answer": answer, "model": model})
                return

            # Handle /api/chat
            if "/chat" in path or "messages" in body or "prompt" in body:
                user_api_key = body.get("api_key") or os.getenv("GROQ_API_KEY", "")
                model = body.get("model", "llama-3.3-70b-versatile")
                style = body.get("style", "Standard Explanation")
                messages = body.get("messages", [])

                if not messages and "prompt" in body:
                    messages = [{"role": "user", "content": body["prompt"]}]

                if not messages:
                    self._send_json_response({"error": "No messages provided."}, 400)
                    return

                # Check latest user message for automatic tool calling
                last_user_msg = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
                tool_used = "none"
                tool_data = None
                tool_context = ""

                # Tool 1: Math Calculator Check
                if is_math_query and calculate_expression:
                    is_math, math_expr = is_math_query(last_user_msg)
                    if is_math and math_expr:
                        calc_res = calculate_expression(math_expr)
                        if calc_res.get("success"):
                            tool_used = "calculator"
                            tool_data = calc_res
                            tool_context = f"\n\n[TOOL EXECUTION - Safe AST Calculator]:\nExpression: {math_expr}\nResult: {calc_res.get('result')}\nExplanation: {calc_res.get('explanation')}\n"

                # Tool 2: Web Search Check (if math wasn't triggered)
                if tool_used == "none" and is_search_query and search_web:
                    is_srch, srch_query = is_search_query(last_user_msg)
                    if is_srch and srch_query:
                        srch_res = search_web(srch_query)
                        if srch_res.get("success"):
                            tool_used = "web_search"
                            tool_data = srch_res
                            results_text = "\n".join([f"- [{item['title']}]({item['url']}): {item['snippet']}" for item in srch_res.get("results", [])])
                            tool_context = f"\n\n[TOOL EXECUTION - Web Search Results]:\nQuery: {srch_query}\nResults:\n{results_text}\n"

                # Prepare Groq messages
                groq_messages = []
                system_instruction = (
                    "You are EduVision AI, an expert educational assistant.\n"
                    f"Selected Response Format Style: {style}.\n"
                    "Structure your answer clearly with Markdown headings, bullet points, and clean syntax."
                )

                if tool_context:
                    system_instruction += tool_context

                groq_messages.append({"role": "system", "content": system_instruction})

                for m in messages:
                    groq_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

                if not GroqClientManager:
                    self._send_json_response({"error": "Groq client library not available."}, 500)
                    return

                groq_mgr = GroqClientManager(api_key=user_api_key)
                if not groq_mgr.is_configured():
                    self._send_json_response({
                        "success": False,
                        "answer": (
                            "⚠️ **Groq API Key Not Configured**\n\n"
                            "Please enter your free Groq API key in the **Settings** panel above.\n"
                            "You can get a free key instantly at [console.groq.com](https://console.groq.com/keys).\n\n"
                            "*(Local endpoints like the Safe AST Calculator and Web Search are fully working!)*"
                        ),
                        "tool_used": tool_used,
                        "tool_data": tool_data,
                    })
                    return

                answer = groq_mgr.generate_chat_response(messages=groq_messages, model=model, stream=False)
                self._send_json_response({
                    "success": True,
                    "answer": answer,
                    "tool_used": tool_used,
                    "tool_data": tool_data,
                    "model": model,
                })
                return

            self._send_json_response({"status": "received", "body": body})
        except Exception as exc:
            self._send_json_response({"error": str(exc)}, 500)

