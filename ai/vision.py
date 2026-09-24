"""
Vision processing and multimodal message formatting for EduVision AI.
Handles image encoding and payload preparation for Groq vision models.
"""

from typing import List, Dict, Any, Optional
import base64
from PIL import Image
import io
from utils.config import DEFAULT_VISION_MODEL
from ai.prompts import SYSTEM_PROMPT, build_vision_prompt


def prepare_image_payload(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    user_prompt: str = "",
) -> List[Dict[str, Any]]:
    """
    Format image and text prompt into a Groq multimodal message payload.
    """
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:{mime_type};base64,{b64_image}"
    prompt_text = build_vision_prompt(user_prompt)

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_text,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_uri,
                    },
                },
            ],
        },
    ]


def validate_and_process_image(image_bytes: bytes) -> Dict[str, Any]:
    """
    Verify image integrity, extract dimensions, format, and prepare clean bytes.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        format_name = (img.format or "JPEG").lower()
        mime_type = f"image/{format_name if format_name != 'jpg' else 'jpeg'}"

        return {
            "valid": True,
            "width": width,
            "height": height,
            "format": format_name,
            "mime_type": mime_type,
            "error": None,
        }
    except Exception as exc:
        return {
            "valid": False,
            "width": 0,
            "height": 0,
            "format": None,
            "mime_type": None,
            "error": f"Invalid or unreadable image file: {str(exc)}",
        }
