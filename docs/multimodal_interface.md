# Multimodal Interface Documentation

The LLM Adapter provides a unified interface for multimodal (text + image) generation across different providers.

## Overview

The multimodal interface allows you to send both text and images to vision-capable LLM models. It supports two modes for providing images:

1. **URL Mode**: Provide images via public URLs
2. **Base64 Mode**: Provide images as base64-encoded strings

## Core Classes

### ImageInput

Represents an image input for multimodal requests.

```python
from llm_adapter.adapters.base import ImageInput, ImageInputType

# Create from URL
image = ImageInput.from_url("https://example.com/image.jpg")

# Create from base64
image = ImageInput.from_base64(
    base64_data="iVBORw0KGgoAAAANS...",
    mime_type="image/jpeg"  # or "image/png", "image/webp", etc.
)

# Manual creation
image = ImageInput(
    type=ImageInputType.URL,
    data="https://example.com/image.jpg"
)
```

### MultimodalContent

Container for text and image content.

```python
from llm_adapter.adapters.base import MultimodalContent, ImageInput

# Text + single image
content = MultimodalContent(
    text="What's in this image?",
    images=[ImageInput.from_url("https://example.com/image.jpg")]
)

# Text + multiple images
content = MultimodalContent(
    text="Compare these images",
    images=[
        ImageInput.from_url("https://example.com/image1.jpg"),
        ImageInput.from_url("https://example.com/image2.jpg"),
    ]
)

# Image only (no text)
content = MultimodalContent(
    images=[ImageInput.from_url("https://example.com/image.jpg")]
)
```

## Usage Examples

### Basic Usage with URL

```python
import asyncio
from llm_adapter.adapters.openrouter_adapter import OpenRouterAdapter
from llm_adapter.adapters.base import ImageInput, MultimodalContent

async def main():
    adapter = OpenRouterAdapter(
        api_key="your-api-key",
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Create multimodal content
    content = MultimodalContent(
        text="Describe this image in detail",
        images=[ImageInput.from_url("https://example.com/photo.jpg")]
    )
    
    # Generate response
    result = await adapter.generate_multimodal(
        content=content,
        model="google/gemini-2.0-flash-exp:free"
    )
    
    print(result.text)
    await adapter.aclose()

asyncio.run(main())
```

### Using Base64 Images

```python
import base64
from pathlib import Path

# Read local image file
image_path = Path("local_image.jpg")
with open(image_path, "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

# Create content with base64 image
content = MultimodalContent(
    text="What do you see?",
    images=[ImageInput.from_base64(image_data, "image/jpeg")]
)

result = await adapter.generate_multimodal(content, model)
```

### Multiple Images

```python
# Analyze multiple images
content = MultimodalContent(
    text="Compare these three images and identify the differences",
    images=[
        ImageInput.from_url("https://example.com/img1.jpg"),
        ImageInput.from_url("https://example.com/img2.jpg"),
        ImageInput.from_url("https://example.com/img3.jpg"),
    ]
)

result = await adapter.generate_multimodal(content, "anthropic/claude-3-opus")
```

## Provider Support

### OpenRouter

Supports various vision models:
- `google/gemini-2.0-flash-exp:free`
- `anthropic/claude-3-opus`
- `anthropic/claude-3-sonnet`
- `openai/gpt-4-vision-preview`

```python
from llm_adapter.adapters.openrouter_adapter import OpenRouterAdapter

adapter = OpenRouterAdapter(api_key="sk-or-v1-...")
result = await adapter.generate_multimodal(content, "google/gemini-2.0-flash-exp:free")
```

### Gemini

Supports Gemini vision models:
- `gemini-2.0-flash`
- `gemini-2.0-flash-lite`
- `gemini-pro-vision`

```python
from llm_adapter.adapters.gemini_adapter import GeminiAdapter

adapter = GeminiAdapter(api_key="AIza...", mode="http")
result = await adapter.generate_multimodal(content, "gemini-2.0-flash")
```

### DashScope (Qwen-VL)

Supports Qwen vision models:
- `qwen-vl-plus`
- `qwen-vl-max`
- `qwen3-vl-30b-a3b-instruct`

```python
from llm_adapter.adapters.dashscope_adapter import DashScopeAdapter

adapter = DashScopeAdapter(api_key="sk-...", mode="dashscope")
result = await adapter.generate_multimodal(content, "qwen-vl-plus")
```

## Image Format Support

### Supported MIME Types

- `image/jpeg` (JPEG/JPG)
- `image/png` (PNG)
- `image/webp` (WebP)
- `image/gif` (GIF)

### URL Requirements

- Must be publicly accessible
- Should use HTTPS for security
- Image size limits vary by provider (typically 4-20MB)

### Base64 Requirements

- Must include MIME type
- Image data should be properly encoded
- Consider size limits (base64 increases size by ~33%)

## Error Handling

```python
try:
    result = await adapter.generate_multimodal(content, model)
except NotImplementedError:
    print("This adapter doesn't support multimodal generation")
except ProviderError as e:
    print(f"API error: {e.message}")
    print(f"Status code: {e.status_code}")
```

## Best Practices

1. **Use URL mode when possible**: It's more efficient and doesn't increase request size
2. **Optimize image size**: Resize images before sending to reduce latency and costs
3. **Check model capabilities**: Not all models support multiple images
4. **Handle errors gracefully**: Some providers may have different limitations
5. **Consider costs**: Vision models typically cost more than text-only models

## Implementation Status

| Provider | Status | Notes |
|----------|--------|-------|
| OpenRouter | ✅ Implemented | Full support: URL, Base64, Streaming |
| DashScope | ✅ Implemented | Full support: URL, Base64, Streaming (SDK & HTTP modes) |
| Gemini | 🚧 Planned | Implementation needed |
| OpenAI | 🚧 Planned | Implementation needed |

## Example: Complete Workflow

```python
import asyncio
import os
from llm_adapter.adapters.openrouter_adapter import OpenRouterAdapter
from llm_adapter.adapters.base import ImageInput, MultimodalContent

async def analyze_image(image_url: str, question: str):
    """Analyze an image and answer a question about it."""
    
    adapter = OpenRouterAdapter(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL")
    )
    
    try:
        content = MultimodalContent(
            text=question,
            images=[ImageInput.from_url(image_url)]
        )
        
        result = await adapter.generate_multimodal(
            content=content,
            model="google/gemini-2.0-flash-exp:free"
        )
        
        return {
            "answer": result.text,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": result.cost_usd
        }
        
    finally:
        await adapter.aclose()

# Usage
result = asyncio.run(analyze_image(
    image_url="https://example.com/chart.png",
    question="What are the key trends shown in this chart?"
))

print(f"Answer: {result['answer']}")
print(f"Cost: ${result['cost_usd']:.6f}")
```

## See Also

- [OpenRouter Multimodal Example](../examples/openrouter_multimodal_example.py)
- [Multimodal Example](../examples/multimodal_example.py)
- [Provider Adapters Documentation](./adapters.md)
