"""
Vision processing and multimodal message formatting for EduVision AI.
Handles image encoding and payload preparation for Groq vision models.
"""

from typing import List, Dict, Any, Optional, Tuple
import base64
from PIL import Image
import io
from utils.config import DEFAULT_VISION_MODEL
from ai.prompts import SYSTEM_PROMPT, build_vision_prompt


def optimize_image_for_vision(
    image_bytes: bytes,
    max_dimension: int = 1024,
    quality: int = 85,
) -> Tuple[bytes, str]:
    """
    Downscale and compress images to prevent Groq API rate limits and reduce token overhead.
    Converts RGBA/PNG to clean optimized JPEG while preserving visual clarity.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        w, h = img.size
        if max(w, h) > max_dimension:
            scale = max_dimension / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        out_buf = io.BytesIO()
        img.save(out_buf, format="JPEG", quality=quality, optimize=True)
        return out_buf.getvalue(), "image/jpeg"
    except Exception:
        return image_bytes, "image/jpeg"


def prepare_image_payload(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    user_prompt: str = "",
) -> List[Dict[str, Any]]:
    """
    Format image and text prompt into a Groq multimodal message payload.
    Automatically compresses and optimizes visual content to avoid rate limit spikes.
    """
    opt_bytes, opt_mime = optimize_image_for_vision(image_bytes)
    b64_image = base64.b64encode(opt_bytes).decode("utf-8")
    data_uri = f"data:{opt_mime};base64,{b64_image}"
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
