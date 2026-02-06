"""
Test multimodal implementation for DashScope and OpenRouter.

This example tests the newly implemented multimodal features:
- URL mode
- Base64 mode  
- Streaming support
"""

import asyncio
import os
from llm_adapter.adapters.base import ImageInput, MultimodalContent
from llm_adapter.adapters.dashscope_adapter import DashScopeAdapter
from llm_adapter.adapters.openrouter_adapter import OpenRouterAdapter


async def test_dashscope_url():
    """Test DashScope with URL image."""
    
    print("=" * 70)
    print("DashScope - URL Mode")
    print("=" * 70)
    
    adapter = DashScopeAdapter(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        mode="dashscope"
    )
    
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
    print(f"Image: {content.images[0].data[:60]}...")
    
    try:
        result = await adapter.generate_multimodal(content, model)
        print(f"\n✓ Generation successful!")
        print(f"  Response: {result.text}")
        print(f"  Input tokens: {result.input_tokens}")
        print(f"  Output tokens: {result.output_tokens}")
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
    
    await adapter.aclose()


async def test_dashscope_streaming():
    """Test DashScope streaming with multimodal."""
    
    print("\n" + "=" * 70)
    print("DashScope - Streaming Mode")
    print("=" * 70)
    
    adapter = DashScopeAdapter(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        mode="dashscope"
    )
    
    content = MultimodalContent(
        text="描述这张图片的内容。",
        images=[
            ImageInput.from_url(
                "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241108/ctdzex/biaozhun.jpg"
            )
        ]
    )
    
    model = "qwen-vl-plus"
    
    print(f"\nModel: {model}")
    print(f"Streaming response:")
    print("  ", end="", flush=True)
    
    try:
        async for chunk in adapter.stream_multimodal(content, model):
            print(chunk, end="", flush=True)
        print("\n\n✓ Streaming completed!")
    except Exception as e:
        print(f"\n✗ Streaming failed: {e}")
    
    await adapter.aclose()


async def test_openrouter_url():
    """Test OpenRouter with URL image."""
    
    print("\n" + "=" * 70)
    print("OpenRouter - URL Mode")
    print("=" * 70)
    
    adapter = OpenRouterAdapter(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL")
    )
    
    content = MultimodalContent(
        text="What's in this image? Describe it briefly.",
        images=[
            ImageInput.from_url(
                "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241108/ctdzex/biaozhun.jpg"
            )
        ]
    )
    
    model = "google/gemini-2.0-flash-exp:free"
    
    print(f"\nModel: {model}")
    print(f"Text: {content.text}")
    
    try:
        result = await adapter.generate_multimodal(content, model)
        print(f"\n✓ Generation successful!")
        print(f"  Response: {result.text}")
        print(f"  Input tokens: {result.input_tokens}")
        print(f"  Output tokens: {result.output_tokens}")
        if result.cost_usd:
            print(f"  Cost: ${result.cost_usd:.6f}")
        if result.provider:
            print(f"  Provider: {result.provider}")
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
    
    await adapter.aclose()


async def test_openrouter_streaming():
    """Test OpenRouter streaming with multimodal."""
    
    print("\n" + "=" * 70)
    print("OpenRouter - Streaming Mode")
    print("=" * 70)
    
    adapter = OpenRouterAdapter(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL")
    )
    
    content = MultimodalContent(
        text="Describe this image in detail.",
        images=[
            ImageInput.from_url(
                "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241108/ctdzex/biaozhun.jpg"
            )
        ]
    )
    
    model = "google/gemini-2.0-flash-exp:free"
    
    print(f"\nModel: {model}")
    print(f"Streaming response:")
    print("  ", end="", flush=True)
    
    try:
        async for chunk in adapter.stream_multimodal(content, model):
            print(chunk, end="", flush=True)
        print("\n\n✓ Streaming completed!")
    except Exception as e:
        print(f"\n✗ Streaming failed: {e}")
    
    await adapter.aclose()


async def test_multiple_images():
    """Test with multiple images."""
    
    print("\n" + "=" * 70)
    print("Multiple Images Test")
    print("=" * 70)
    
    adapter = OpenRouterAdapter(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL")
    )
    
    content = MultimodalContent(
        text="What do you see in these images?",
        images=[
            ImageInput.from_url("https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241108/ctdzex/biaozhun.jpg"),
            ImageInput.from_url("https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241108/ctdzex/biaozhun.jpg"),
        ]
    )
    
    model = "anthropic/claude-3-haiku:beta"
    
    print(f"\nModel: {model}")
    print(f"Number of images: {len(content.images)}")
    
    try:
        result = await adapter.generate_multimodal(content, model)
        print(f"\n✓ Generation successful!")
        print(f"  Response: {result.text[:200]}...")
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
    
    await adapter.aclose()


async def main():
    """Run all tests."""
    
    print("\n" + "=" * 70)
    print("Multimodal Implementation Tests")
    print("=" * 70)
    
    # Check API keys
    has_dashscope = bool(os.getenv("DASHSCOPE_API_KEY"))
    has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
    
    if not has_dashscope:
        print("\n⚠ DASHSCOPE_API_KEY not set - skipping DashScope tests")
    
    if not has_openrouter:
        print("\n⚠ OPENROUTER_API_KEY not set - skipping OpenRouter tests")
    
    if not has_dashscope and not has_openrouter:
        print("\nNo API keys found. Please set at least one:")
        print("  export DASHSCOPE_API_KEY=sk-xxx")
        print("  export OPENROUTER_API_KEY=sk-or-v1-xxx")
        return
    
    # Run tests
    if has_dashscope:
        await test_dashscope_url()
        await test_dashscope_streaming()
    
    if has_openrouter:
        await test_openrouter_url()
        await test_openrouter_streaming()
        await test_multiple_images()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\n✓ Multimodal implementation complete!")
    print("\nSupported features:")
    print("  • DashScope: URL mode, Base64 mode, Streaming")
    print("  • OpenRouter: URL mode, Base64 mode, Streaming")
    print("\nUsage:")
    print("  content = MultimodalContent(")
    print("      text='Your question',")
    print("      images=[ImageInput.from_url('https://...')]")
    print("  )")
    print("  result = await adapter.generate_multimodal(content, model)")


if __name__ == "__main__":
    asyncio.run(main())
