"""
Example demonstrating multimodal (text + image) generation.

This example shows how to use the multimodal interface with:
- URL mode: Provide image via URL
- Base64 mode: Provide image as base64-encoded string

Supported providers:
- OpenRouter (with vision models like gpt-4-vision, claude-3-opus, etc.)
- Gemini (gemini-pro-vision, gemini-2.0-flash, etc.)
- DashScope (qwen-vl-plus, qwen-vl-max, etc.)
"""

import asyncio
import os
import base64
from pathlib import Path

from llm_adapter.adapters.base import ImageInput, MultimodalContent
from llm_adapter.adapters.openrouter_adapter import OpenRouterAdapter
from llm_adapter.adapters.gemini_adapter import GeminiAdapter
from llm_adapter.adapters.dashscope_adapter import DashScopeAdapter


async def test_url_mode():
    """Test multimodal generation with image URL."""
    
    print("=" * 70)
    print("Multimodal Generation - URL Mode")
    print("=" * 70)
    
    # Example with OpenRouter
    adapter = OpenRouterAdapter(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL")
    )
    
    # Create multimodal content with image URL
    content = MultimodalContent(
        text="What's in this image? Please describe it briefly.",
        images=[
            ImageInput.from_url(
                "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241108/ctdzex/biaozhun.jpg"
            )
        ]
    )
    
    model = "google/gemini-2.0-flash-exp:free"
    
    print(f"\nModel: {model}")
    print(f"Text: {content.text}")
    print(f"Image URL: {content.images[0].data}")
    
    try:
        result = await adapter.generate_multimodal(content, model)
        print(f"\n✓ Generation successful!")
        print(f"  Response: {result.text}")
        print(f"  Input tokens: {result.input_tokens}")
        print(f"  Output tokens: {result.output_tokens}")
    except NotImplementedError as e:
        print(f"\n⚠ {e}")
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
    
    await adapter.aclose()


async def test_base64_mode():
    """Test multimodal generation with base64-encoded image."""
    
    print("\n" + "=" * 70)
    print("Multimodal Generation - Base64 Mode")
    print("=" * 70)
    
    # Read a local image file and convert to base64
    # For this example, we'll create a simple test
    # In real usage, you would read an actual image file
    
    adapter = OpenRouterAdapter(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL")
    )
    
    # Example: Read image file and encode to base64
    # image_path = Path("path/to/your/image.jpg")
    # if image_path.exists():
    #     with open(image_path, "rb") as f:
    #         image_data = base64.b64encode(f.read()).decode("utf-8")
    # else:
    #     print("Image file not found, skipping base64 test")
    #     return
    
    # For demo purposes, we'll use a URL instead
    print("Note: Base64 mode requires a local image file.")
    print("Skipping this test. See code comments for implementation.")
    
    await adapter.aclose()


async def test_dashscope_multimodal():
    """Test multimodal with DashScope (Qwen-VL models)."""
    
    print("\n" + "=" * 70)
    print("DashScope Multimodal Test")
    print("=" * 70)
    
    adapter = DashScopeAdapter(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        mode="dashscope"
    )
    
    # Create multimodal content
    content = MultimodalContent(
        text="请仅输出图像中的文本内容。",
        images=[
            ImageInput.from_url(
                "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241108/ctdzex/biaozhun.jpg"
            )
        ]
    )
    
    model = "qwen-vl-plus"
    
    print(f"\nModel: {model}")
    print(f"Text: {content.text}")
    print(f"Image: {content.images[0].data[:50]}...")
    
    try:
        result = await adapter.generate_multimodal(content, model)
        print(f"\n✓ Generation successful!")
        print(f"  Response: {result.text}")
        print(f"  Input tokens: {result.input_tokens}")
        print(f"  Output tokens: {result.output_tokens}")
    except NotImplementedError as e:
        print(f"\n⚠ {e}")
        print("  DashScope multimodal support needs to be implemented")
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
    
    await adapter.aclose()


async def test_gemini_multimodal():
    """Test multimodal with Gemini."""
    
    print("\n" + "=" * 70)
    print("Gemini Multimodal Test")
    print("=" * 70)
    
    adapter = GeminiAdapter(
        api_key=os.getenv("GEMINI_API_KEY"),
        mode="http"
    )
    
    # Create multimodal content
    content = MultimodalContent(
        text="What do you see in this image?",
        images=[
            ImageInput.from_url(
                "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241108/ctdzex/biaozhun.jpg"
            )
        ]
    )
    
    model = "gemini-2.0-flash"
    
    print(f"\nModel: {model}")
    print(f"Text: {content.text}")
    
    try:
        result = await adapter.generate_multimodal(content, model)
        print(f"\n✓ Generation successful!")
        print(f"  Response: {result.text}")
    except NotImplementedError as e:
        print(f"\n⚠ {e}")
        print("  Gemini multimodal support needs to be implemented")
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
    
    await adapter.aclose()


async def test_multiple_images():
    """Test with multiple images in one request."""
    
    print("\n" + "=" * 70)
    print("Multiple Images Test")
    print("=" * 70)
    
    adapter = OpenRouterAdapter(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL")
    )
    
    # Create content with multiple images
    content = MultimodalContent(
        text="Compare these images and describe the differences.",
        images=[
            ImageInput.from_url("https://example.com/image1.jpg"),
            ImageInput.from_url("https://example.com/image2.jpg"),
        ]
    )
    
    model = "anthropic/claude-3-opus"
    
    print(f"\nModel: {model}")
    print(f"Text: {content.text}")
    print(f"Number of images: {len(content.images)}")
    
    try:
        result = await adapter.generate_multimodal(content, model)
        print(f"\n✓ Generation successful!")
        print(f"  Response: {result.text[:200]}...")
    except NotImplementedError as e:
        print(f"\n⚠ {e}")
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
    
    await adapter.aclose()


async def main():
    """Run all multimodal tests."""
    
    print("\n" + "=" * 70)
    print("Multimodal Interface Examples")
    print("=" * 70)
    print("\nThis example demonstrates the multimodal interface for")
    print("text + image generation across different providers.")
    
    # Check for API keys
    if not os.getenv("OPENROUTER_API_KEY"):
        print("\n⚠ Warning: OPENROUTER_API_KEY not set")
        print("Some tests will be skipped")
    
    await test_url_mode()
    await test_base64_mode()
    await test_dashscope_multimodal()
    await test_gemini_multimodal()
    await test_multiple_images()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nMultimodal interface usage:")
    print("\n1. URL mode (recommended for remote images):")
    print("   content = MultimodalContent(")
    print("       text='Describe this image',")
    print("       images=[ImageInput.from_url('https://...')']")
    print("   )")
    print("\n2. Base64 mode (for local images):")
    print("   content = MultimodalContent(")
    print("       text='What is this?',")
    print("       images=[ImageInput.from_base64(base64_str, 'image/jpeg')]")
    print("   )")
    print("\n3. Multiple images:")
    print("   content = MultimodalContent(")
    print("       text='Compare these',")
    print("       images=[img1, img2, img3]")
    print("   )")
    print("\nSupported providers:")
    print("  • OpenRouter (various vision models)")
    print("  • Gemini (gemini-pro-vision, gemini-2.0-flash)")
    print("  • DashScope (qwen-vl-plus, qwen-vl-max)")


if __name__ == "__main__":
    asyncio.run(main())
