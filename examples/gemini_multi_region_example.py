"""
Example demonstrating Vertex AI global endpoint.

The "global" endpoint lets Google automatically select the best available zone
worldwide, providing better availability and automatic failover without needing
manual region fallback logic.

Supported locations:
- "global" (recommended): Google auto-selects best zone worldwide
- Specific regions: "us-central1", "us-east4", "europe-west1", "asia-southeast1", etc.
"""

import asyncio
import os
from llm_adapter.adapters.gemini_adapter import GeminiAdapter


async def test_global_endpoint():
    """Test using global endpoint where Google auto-selects the best zone."""
    
    print("=" * 70)
    print("Global Endpoint Test")
    print("=" * 70)
    
    # Use "global" - Google will automatically select the best available zone
    adapter = GeminiAdapter(
        api_key="dummy_key",
        mode="vertex",
        project_id=os.getenv("GCP_PROJECT_ID", "your-project-id"),
        location="global",  # Global: Google auto-selects best zone worldwide
        enable_region_fallback=False  # Not needed with global endpoint
    )
    
    print(f"\nConfiguration:")
    print(f"  Mode: {adapter.mode}")
    print(f"  Project ID: {adapter.project_id}")
    print(f"  Location: {adapter.location}")
    print(f"  Note: Google will automatically select the best available zone")
    
    test_model = "gemini-2.0-flash-lite-001"
    test_prompt = "Say hello in 5 words."
    
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


async def test_specific_region():
    """Test using a specific region endpoint."""
    
    print("\n" + "=" * 70)
    print("Specific Region Endpoint Test")
    print("=" * 70)
    
    adapter = GeminiAdapter(
        api_key="dummy_key",
        mode="vertex",
        project_id=os.getenv("GCP_PROJECT_ID", "your-project-id"),
        location="us-central1",  # Specific region
        enable_region_fallback=False
    )
    
    print(f"\nConfiguration:")
    print(f"  Location: {adapter.location} (specific region)")
    
    test_model = "gemini-2.0-flash-lite-001"
    test_prompt = "Count from 1 to 3."
    
    print(f"\nTesting with model: {test_model}")
    
    try:
        result = await adapter.generate(test_prompt, test_model)
        print(f"\n✓ Generation successful!")
        print(f"  Response: {result.text}")
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
    
    await adapter.aclose()


async def compare_global_vs_specific_region():
    """Compare global endpoint vs specific region performance."""
    
    print("\n" + "=" * 70)
    print("Global vs Specific Region Comparison")
    print("=" * 70)
    
    import time
    
    test_model = "gemini-2.0-flash-lite-001"
    test_prompt = "Hello"
    
    # Test 1: Specific region
    print("\n1. Specific Region (us-central1):")
    adapter_specific = GeminiAdapter(
        api_key="dummy_key",
        mode="vertex",
        project_id=os.getenv("GCP_PROJECT_ID"),
        location="us-central1",
        enable_region_fallback=False
    )
    
    start = time.time()
    try:
        result = await adapter_specific.generate(test_prompt, test_model)
        duration = (time.time() - start) * 1000
        print(f"  ✓ Success in {duration:.2f}ms")
        print(f"  Response: {result.text}")
    except Exception as e:
        duration = (time.time() - start) * 1000
        print(f"  ✗ Failed in {duration:.2f}ms: {e}")
    
    await adapter_specific.aclose()
    
    # Test 2: Global endpoint
    print("\n2. Global Endpoint:")
    adapter_global = GeminiAdapter(
        api_key="dummy_key",
        mode="vertex",
        project_id=os.getenv("GCP_PROJECT_ID"),
        location="global",
        enable_region_fallback=False
    )
    
    start = time.time()
    try:
        result = await adapter_global.generate(test_prompt, test_model)
        duration = (time.time() - start) * 1000
        print(f"  ✓ Success in {duration:.2f}ms")
        print(f"  Response: {result.text}")
    except Exception as e:
        duration = (time.time() - start) * 1000
        print(f"  ✗ Failed in {duration:.2f}ms: {e}")
    
    await adapter_global.aclose()


async def main():
    """Run all multi-region tests."""
    
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("⚠ Warning: GOOGLE_APPLICATION_CREDENTIALS not set")
        print("Please set it to your service account JSON key file path")
        return
    
    if not os.getenv("GCP_PROJECT_ID"):
        print("⚠ Warning: GCP_PROJECT_ID not set")
        print("Please set it to your Google Cloud project ID")
        return
    
    await test_global_endpoint()
    await test_specific_region()
    await compare_global_vs_specific_region()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nGlobal endpoint benefits:")
    print("  • Google automatically selects the best available zone worldwide")
    print("  • Better availability and automatic failover")
    print("  • No need for manual region fallback logic")
    print("  • Simpler configuration")
    print("  • Works with all Gemini models")
    print("\nRecommended configuration:")
    print("  location: 'global' - Let Google choose the best zone")
    print("\nAlternative (for compliance/data residency):")
    print("  location: 'us-central1' - Specific zone for data residency requirements")


if __name__ == "__main__":
    asyncio.run(main())
