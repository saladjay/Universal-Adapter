"""
Example demonstrating different Gemini adapter modes.

This example shows how to use Gemini with:
1. HTTP mode (default, direct API calls)
2. SDK mode (official google-generativeai SDK)
3. Vertex AI mode (for GCP projects)
"""

import asyncio
from llm_adapter.adapters import GeminiAdapter, ProviderError


async def test_http_mode():
    """Test Gemini with HTTP mode (default)."""
    print("\n=== HTTP Mode (Direct API) ===")
    api_key = "your-gemini-api-key"
    
    adapter = GeminiAdapter(api_key=api_key, mode="http")
    
    try:
        result = await adapter.generate(
            prompt="Say hello in 3 languages",
            model="gemini-2.5-flash"
        )
        print(f"Response: {result.text}")
        print(f"Tokens: {result.input_tokens} in, {result.output_tokens} out")
    except ProviderError as e:
        print(f"Error: {e}")
    finally:
        await adapter.aclose()


async def test_sdk_mode():
    """Test Gemini with SDK mode."""
    print("\n=== SDK Mode (google-generativeai) ===")
    api_key = "your-gemini-api-key"
    
    try:
        adapter = GeminiAdapter(api_key=api_key, mode="sdk")
    except ProviderError as e:
        print(f"SDK not installed: {e}")
        print("Install with: pip install google-generativeai")
        return
    
    try:
        result = await adapter.generate(
            prompt="Say hello in 3 languages",
            model="gemini-2.5-flash"
        )
        print(f"Response: {result.text}")
        print(f"Tokens: {result.input_tokens} in, {result.output_tokens} out")
    except ProviderError as e:
        print(f"Error: {e}")
    finally:
        await adapter.aclose()


async def test_vertex_mode():
    """Test Gemini with Vertex AI mode."""
    print("\n=== Vertex AI Mode (GCP) ===")
    
    # Vertex AI configuration
    project_id = "your-gcp-project-id"
    location = "asia-southeast1"
    
    try:
        adapter = GeminiAdapter(
            api_key="",  # Not used in vertex mode
            mode="vertex",
            project_id=project_id,
            location=location
        )
    except ProviderError as e:
        print(f"Vertex AI SDK not installed: {e}")
        print("Install with: pip install google-cloud-aiplatform")
        return
    
    try:
        result = await adapter.generate(
            prompt="你好 Gemini！现在新加坡天气如何？",
            model="gemini-2.0-flash"
        )
        print(f"Response: {result.text}")
        print(f"Tokens: {result.input_tokens} in, {result.output_tokens} out")
    except ProviderError as e:
        print(f"Error: {e}")
    finally:
        await adapter.aclose()


async def test_streaming():
    """Test streaming with different modes."""
    print("\n=== Streaming Example (SDK Mode) ===")
    api_key = "your-gemini-api-key"
    
    try:
        adapter = GeminiAdapter(api_key=api_key, mode="sdk")
    except ProviderError as e:
        print(f"SDK not installed: {e}")
        return
    
    try:
        print("Streaming response: ", end="", flush=True)
        async for chunk in adapter.stream(
            prompt="Count from 1 to 5 slowly",
            model="gemini-2.5-flash"
        ):
            print(chunk, end="", flush=True)
        print()  # New line after streaming
    except ProviderError as e:
        print(f"\nError: {e}")
    finally:
        await adapter.aclose()


async def main():
    """Run all examples."""
    print("Gemini Adapter Modes Example")
    print("=" * 50)
    
    # Test HTTP mode (default)
    await test_http_mode()
    
    # Test SDK mode
    await test_sdk_mode()
    
    # Test Vertex AI mode
    await test_vertex_mode()
    
    # Test streaming
    await test_streaming()


if __name__ == "__main__":
    asyncio.run(main())
