"""OpenRouter multimodal example (image + text).

Usage:
    python openrouter_multimodal_example.py --image ./screenshot.png
    python openrouter_multimodal_example.py --image ./screenshot.png --model qwen/qwen3-vl-30b-a3b-instruct

Requires:
    - OPENROUTER_API_KEY environment variable
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import time
from pathlib import Path

import httpx

PROMPT = """You are a chat screenshot structure parser.

Output strict JSON with:
{
  "nickname": ["..."],
  "bubbles": [
    {
      "bbox": [x1, y1, x2, y2],
      "sender": "U" | "T"
    }
  ]
}

Rules:
- All coordinates in original image pixels
- One item per chat bubble
- nickname list contains all visible participant names
- JSON only, no extra text
- U: user, T: talker
"""

# DEFAULT_MODEL = "google/gemini-2.5-flash-image"
DEFAULT_MODEL = "qwen/qwen-2.5-vl-7b-instruct"
# DEFAULT_MODEL = "qwen/qwen3-vl-30b-a3b-instruct"
# DEFAULT_MODEL = "nvidia/nemotron-nano-12b-v2-vl"
# DEFAULT_MODEL = "mistralai/ministral-3b-2512"
# DEFAULT_MODEL = "bytedance-seed/seed-1.6-flash"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _image_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type:
        mime_type = "image/png"
    data = image_path.read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


async def main() -> None:
    parser = argparse.ArgumentParser(description="OpenRouter multimodal example.")
    parser.add_argument("--image", required=True, help="Path to the screenshot image")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model name")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenRouter base URL")
    parser.add_argument(
        "--api-key",
        help="OpenRouter API key (overrides OPENROUTER_API_KEY)",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY environment variable.")

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    image_url = _image_to_data_url(image_path)

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    start_time = time.monotonic()
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{args.base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
    elapsed = time.monotonic() - start_time

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise SystemExit(f"Unexpected response format: {exc}")

    try:
        parsed = json.loads(content)
        formatted = json.dumps(parsed, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        formatted = content

    print(formatted)

    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    cost = usage.get("total_cost") or usage.get("cost")
    header_cost = response.headers.get("x-openrouter-cost")
    cost = cost or header_cost

    print("\n--- Stats ---")
    print(f"Latency: {elapsed:.2f}s")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Completion tokens: {completion_tokens}")
    print(f"Total tokens: {total_tokens}")
    print(f"Cost: {cost}")


if __name__ == "__main__":
    asyncio.run(main())
