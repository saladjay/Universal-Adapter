"""
Example demonstrating DashScope adapter modes.

This example shows the difference between:
- "dashscope" mode: Using official dashscope SDK (default, better stability)
- "http" mode: Direct HTTP API calls (fewer dependencies)
"""

import asyncio
import os
from llm_adapter.adapters.dashscope_adapter import DashScopeAdapter


async def test_dashscope_sdk_mode():
    """Test using official DashScope SDK (default mode)."""
    
    print("=" * 70)
    print("DashScope SDK Mode Test")
    print("=" * 70)
    
    adapter = DashScopeAdapter(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        mode="dashscope"  # Use official SDK (default)
    )
    
    print(f"\nConfiguration:")
    print(f"  Mode: {adapter.mode}")
    print(f"  Base URL: {adapter.base_url}")
    
    test_model = "qwen-turbo"
    test_prompt = "你好，请用一句话介绍你自己。"
    
    print(f"\nTesting with model: {test_model}")
    print(f"Prompt: {test_prompt}")
    
    try:
        result = await adapter.generate(test_prompt, test_model)
        print(f"\n✓ Generation successful!")
        print(f"  Response: {result.text}")
        print(f"  Input tokens: {result.input_tokens}")
        print(f"  Output tokens: {result.output_tokens}")
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
    
    await adapter.aclose()


async def test_http_mode():
    """Test using direct HTTP API calls."""
    
    print("\n" + "=" * 70)
    print("HTTP Mode Test")
    print("=" * 70)
    
    adapter = DashScopeAdapter(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        mode="http"  # Use direct HTTP calls
    )
    
    print(f"\nConfiguration:")
    print(f"  Mode: {adapter.mode}")
    print(f"  Base URL: {adapter.base_url}")
    
    test_model = "qwen-turbo"
    test_prompt = "你好，请用一句话介绍你自己。"
    
    print(f"\nTesting with model: {test_model}")
    print(f"Prompt: {test_prompt}")
    
    try:
        result = await adapter.generate(test_prompt, test_model)
        print(f"\n✓ Generation successful!")
        print(f"  Response: {result.text}")
        print(f"  Input tokens: {result.input_tokens}")
        print(f"  Output tokens: {result.output_tokens}")
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
    
    await adapter.aclose()


async def test_international_endpoint():
    """Test using international endpoint."""
    
    print("\n" + "=" * 70)
    print("International Endpoint Test")
    print("=" * 70)
    
    adapter = DashScopeAdapter(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        mode="dashscope",
        use_international=True  # Use dashscope-intl.aliyuncs.com
    )
    
    print(f"\nConfiguration:")
    print(f"  Mode: {adapter.mode}")
    print(f"  Base URL: {adapter.base_url}")
    print(f"  International: {adapter.use_international}")
    
    test_model = "qwen-turbo"
    test_prompt = "Hello, introduce yourself in one sentence."
    
    print(f"\nTesting with model: {test_model}")
    print(f"Prompt: {test_prompt}")
    
    try:
        result = await adapter.generate(test_prompt, test_model)
        print(f"\n✓ Generation successful!")
        print(f"  Response: {result.text}")
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
    
    await adapter.aclose()


async def test_streaming_sdk():
    """Test streaming with DashScope SDK."""
    
    print("\n" + "=" * 70)
    print("Streaming Test (SDK Mode)")
    print("=" * 70)
    
    adapter = DashScopeAdapter(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        mode="dashscope"
    )
    
    test_model = "qwen-turbo"
    test_prompt = "请数从1到5，每个数字一行。"
    
    print(f"\nStreaming with model: {test_model}")
    print(f"Prompt: {test_prompt}")
    print("\nStreaming response:")
    
    try:
        print("  ", end="", flush=True)
        async for chunk in adapter.stream(test_prompt, test_model):
            print(chunk, end="", flush=True)
        print("\n\n✓ Streaming completed successfully!")
    except Exception as e:
        print(f"\n✗ Streaming failed: {e}")
    
    await adapter.aclose()


async def test_streaming_http():
    """Test streaming with HTTP mode."""
    
    print("\n" + "=" * 70)
    print("Streaming Test (HTTP Mode)")
    print("=" * 70)
    
    adapter = DashScopeAdapter(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        mode="http"
    )
    
    test_model = "qwen-turbo"
    test_prompt = "请数从1到5，每个数字一行。"
    
    print(f"\nStreaming with model: {test_model}")
    print(f"Prompt: {test_prompt}")
    print("\nStreaming response:")
    
    try:
        print("  ", end="", flush=True)
        async for chunk in adapter.stream(test_prompt, test_model):
            print(chunk, end="", flush=True)
        print("\n\n✓ Streaming completed successfully!")
    except Exception as e:
        print(f"\n✗ Streaming failed: {e}")
    
    await adapter.aclose()


async def compare_modes():
    """Compare performance between SDK and HTTP modes."""
    
    print("\n" + "=" * 70)
    print("Mode Comparison")
    print("=" * 70)
    
    import time
    
    test_model = "qwen-turbo"
    test_prompt = "你好"
    
    # Test SDK mode
    print("\n1. SDK Mode:")
    adapter_sdk = DashScopeAdapter(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        mode="dashscope"
    )
    
    start = time.time()
    try:
        result = await adapter_sdk.generate(test_prompt, test_model)
        duration = (time.time() - start) * 1000
        print(f"  ✓ Success in {duration:.2f}ms")
        print(f"  Response: {result.text[:50]}...")
    except Exception as e:
        duration = (time.time() - start) * 1000
        print(f"  ✗ Failed in {duration:.2f}ms: {e}")
    
    await adapter_sdk.aclose()
    
    # Test HTTP mode
    print("\n2. HTTP Mode:")
    adapter_http = DashScopeAdapter(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        mode="http"
    )
    
    start = time.time()
    try:
        result = await adapter_http.generate(test_prompt, test_model)
        duration = (time.time() - start) * 1000
        print(f"  ✓ Success in {duration:.2f}ms")
        print(f"  Response: {result.text[:50]}...")
    except Exception as e:
        duration = (time.time() - start) * 1000
        print(f"  ✗ Failed in {duration:.2f}ms: {e}")
    
    await adapter_http.aclose()


async def main():
    """Run all mode tests."""
    
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("⚠ Warning: DASHSCOPE_API_KEY not set")
        print("Please set it to your DashScope API key")
        print("Example: export DASHSCOPE_API_KEY=sk-xxx")
        return
    
    await test_dashscope_sdk_mode()
    await test_http_mode()
    await test_international_endpoint()
    await test_streaming_sdk()
    await test_streaming_http()
    await compare_modes()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nMode comparison:")
    print("\nDashScope SDK mode (default):")
    print("  ✓ Better stability and error handling")
    print("  ✓ Official support from Alibaba")
    print("  ✓ Automatic retries and rate limiting")
    print("  ✗ Requires dashscope package")
    print("\nHTTP mode:")
    print("  ✓ Fewer dependencies (only httpx)")
    print("  ✓ More control over requests")
    print("  ✗ Manual error handling")
    print("\nRecommendation:")
    print("  Use 'dashscope' mode (default) for production")
    print("  Use 'http' mode if you want minimal dependencies")


if __name__ == "__main__":
    asyncio.run(main())
