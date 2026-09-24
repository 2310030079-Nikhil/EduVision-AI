"""
Groq API Client Integration for EduVision AI.
Directly communicates with Groq's high-speed inference engine for LLM text and Vision models.
Includes streaming generation, error handling, rate-limit recovery, and fallback simulation.
"""

from typing import List, Dict, Any, Optional, Generator, Union
import os
import groq
from groq import Groq
from utils.config import (
    DEFAULT_TEXT_MODEL,
    DEFAULT_VISION_MODEL,
    TEXT_MODELS,
    VISION_MODELS,
)
from ai.prompts import SYSTEM_PROMPT, EDUCATIONAL_STYLE_GUIDE


class GroqClientManager:
    """Wrapper for Groq API client with error handling and model fallback."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.client: Optional[Groq] = None
        self._init_client()

    def _init_client(self):
        """Initialize Groq client if key is available."""
        if self.api_key and len(self.api_key.strip()) > 0:
            try:
                self.client = Groq(api_key=self.api_key.strip())
            except Exception as exc:
                print(f"[Warning] Failed to initialize Groq client: {exc}")
                self.client = None
        else:
            self.client = None

    def update_api_key(self, new_key: str):
        """Update client with a new API key."""
        self.api_key = new_key.strip()
        self._init_client()

    def is_configured(self) -> bool:
        """Check if Groq API client is ready to call."""
        return self.client is not None and bool(self.api_key)

    def test_connection(self) -> Dict[str, Any]:
        """Test API connection with a minimal prompt."""
        if not self.is_configured():
            return {"success": False, "error": "Groq API key is not configured."}
        try:
            resp = self.client.chat.completions.create(
                model=DEFAULT_TEXT_MODEL,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return {"success": True, "model": DEFAULT_TEXT_MODEL, "response": resp.choices[0].message.content}
        except groq.AuthenticationError:
            return {"success": False, "error": "Invalid Groq API key. Please check your credentials."}
        except groq.RateLimitError:
            return {"success": False, "error": "Groq rate limit exceeded. Please wait a moment or switch models."}
        except Exception as exc:
            return {"success": False, "error": f"API connection failed: {str(exc)}"}

    def generate_chat_response(
        self,
        messages: List[Dict[str, Any]],
        model: str = DEFAULT_TEXT_MODEL,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Send conversational messages to Groq text model.
        Supports both streaming and full string return.
        """
        if not self.is_configured():
            mock_text = (
                "⚠️ **Groq API Key Not Configured**\n\n"
                "Please enter your free Groq API key in the **Settings** tab in the sidebar.\n"
                "You can get a free key instantly at [console.groq.com](https://console.groq.com/keys).\n\n"
                "*(In the meantime, local features like Document Processing, FAISS Vector Search, "
                "and the Safe Calculator Tool are fully functional!)*"
            )
            if stream:
                def mock_gen():
                    yield mock_text
                return mock_gen()
            return mock_text

        # Ensure system prompt is present
        full_messages = []
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            full_messages.append({
                "role": "system",
                "content": f"{SYSTEM_PROMPT}\n\n{EDUCATIONAL_STYLE_GUIDE}",
            })
        full_messages.extend(messages)

        try:
            if stream:
                stream_resp = self.client.chat.completions.create(
                    model=model,
                    messages=full_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                def token_generator():
                    for chunk in stream_resp:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            yield delta
                return token_generator()
            else:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=full_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
                return resp.choices[0].message.content or ""

        except groq.AuthenticationError:
            err = "Authentication Error: Invalid Groq API key. Please verify your key in Settings."
            if stream:
                def err_gen(): yield err
                return err_gen()
            return err
        except groq.RateLimitError:
            err = "Rate Limit Reached: Groq API limit reached. Please wait a few seconds or try a lighter model (e.g., llama-3.1-8b-instant)."
            if stream:
                def err_gen(): yield err
                return err_gen()
            return err
        except Exception as exc:
            err = f"AI Generation Error: {str(exc)}"
            if stream:
                def err_gen(): yield err
                return err_gen()
            return err

    def generate_vision_response(
        self,
        messages: List[Dict[str, Any]],
        model: str = DEFAULT_VISION_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """
        Send multimodal vision messages (text + image) to Groq vision model.
        """
        if not self.is_configured():
            return (
                "⚠️ **Groq API Key Required for Vision**\n\n"
                "Please configure your Groq API key in Settings to analyze images with `llama-3.2-11b-vision-preview`."
            )

        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or "No response received from vision model."

        except groq.AuthenticationError:
            return "Authentication Error: Invalid Groq API key. Please check your credentials in Settings."
        except groq.RateLimitError:
            return "Rate Limit Error: Vision model request limit reached. Please retry in a few moments."
        except Exception as exc:
            return f"Vision Model Error: {str(exc)}"
