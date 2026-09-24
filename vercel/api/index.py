"""
Vercel Serverless Python Function for EduVision AI.
Provides REST API endpoints for health status, safe calculation, and info queries
when deployed to Vercel's serverless edge infrastructure.
"""

from http.server import BaseHTTPRequestHandler
import json
import urllib.parse
import os
import sys
from pathlib import Path

# Add project root to sys.path so tools can be imported
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    from tools.calculator import calculate_expression
except ImportError:
    calculate_expression = None


class handler(BaseHTTPRequestHandler):
    """Vercel Python Serverless HTTP Request Handler."""

    def _send_json_response(self, data: dict, status_code: int = 200):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
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
            self._send_json_response({
                "status": "healthy",
                "service": "EduVision AI Serverless API",
                "version": "1.0.0",
                "groq_configured": bool(os.getenv("GROQ_API_KEY")),
                "endpoints": [
                    {"path": "/api/health", "method": "GET"},
                    {"path": "/api/calculate?expr=15%25+of+87500", "method": "GET"},
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

        # Default info
        self._send_json_response({
            "message": "EduVision AI Serverless API Gateway is operational.",
            "documentation": "See README.md for full Streamlit architecture and deployment details.",
        })

    def do_POST(self):
        """Handle POST requests for calculations or queries."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}

            if "expression" in body and calculate_expression:
                res = calculate_expression(body["expression"])
                self._send_json_response(res)
                return

            self._send_json_response({"status": "received", "body": body})
        except Exception as exc:
            self._send_json_response({"error": str(exc)}, 500)
