"""
Example demonstrating Gemini region fallback functionality.

This example shows how the Gemini adapter automatically falls back to a global
region (us-central1) when the specified regional endpoint fails or is unavailable.

The fallback tracker records timing and success metrics for each fallback event.
"""

import asyncio
import os
from llm_adapter.adapters.gemini_adapter import GeminiAdapter
from llm_adapter.fallback_tracker import get_fallback_tracker


async def test_region_fallback():
    """Test region fallback with an unavailable model in a specific region."""
    
    print("=" * 60)
    print("Gemini Region Fallback Test")
    print("=" * 60)
    
    # Get fallback tracker
    tracker = get_fallback_tracker()
    
    # Initialize adapter with a regional endpoint
    # Note: Some models may not be available in all regions
    adapter = GeminiAdapter(
        api_key="dummy_key",  # Not used in vertex mode
        mode="vertex",
        project_id=os.getenv("GCP_PROJECT_ID", "your-project-id"),
        location="asia-southeast1",  # Regional endpoint
        enable_region_fallback=True,
        fallback_location="us-central1"  # Global fallback
    )
    
    print(f"\nInitial Configuration:")
    print(f"  Mode: {adapter.mode}")
    print(f"  Project ID: {adapter.project_id}")
    print(f"  Primary Location: {adapter.location}")
    print(f"  Fallback Location: {adapter.fallback_location}")
    print(f"  Fallback Enabled: {adapter.enable_region_fallback}")
    
    # Test with a model that might not be available in the regional endpoint
    test_model = "gemini-2.0-flash-lite-001"
    test_prompt = "Say hello in 5 words or less."
    
    print(f"\nTesting with model: {test_model}")
    print(f"Prompt: {test_prompt}")
    print("\nAttempting generation...")
    
    try:
        result = await adapter.generate(test_prompt, test_model)
        print(f"\n✓ Generation successful!")
        print(f"  Response: {result.text[:100]}...")
        print(f"  Input tokens: {result.input_tokens}")
        print(f"  Output tokens: {result.output_tokens}")
        
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
    
    # Display fallback statistics
    print("\n" + "=" * 60)
    print("Fallback Statistics")
    print("=" * 60)
    
    stats = tracker.get_stats()
    summary = stats.get_summary()
    
    print(f"\nSummary:")
    print(f"  Total fallbacks: {summary['total_fallbacks']}")
    print(f"  Successful: {summary['successful_fallbacks']}")
    print(f"  Failed: {summary['failed_fallbacks']}")
    print(f"  Success rate: {summary['success_rate']:.1%}")
    print(f"  Total duration: {summary['total_duration_ms']:.2f}ms")
    print(f"  Average duration: {summary['average_duration_ms']:.2f}ms")
    
    # Display recent fallback events
    recent_events = tracker.get_recent_events(limit=5)
    if recent_events:
        print(f"\nRecent Fallback Events:")
        for i, event in enumerate(recent_events, 1):
            print(f"\n  Event {i}:")
            print(f"    Timestamp: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"    Provider: {event.provider}")
            print(f"    {event.original_location} → {event.fallback_location}")
            print(f"    Model: {event.original_model}")
            print(f"    Success: {'✓' if event.success else '✗'}")
            print(f"    Duration: {event.fallback_duration_ms:.2f}ms")
            print(f"    Error: {event.error_message[:80]}...")
    
    await adapter.aclose()


async def test_streaming_with_fallback():
    """Test streaming with region fallback."""
    
    print("\n" + "=" * 60)
    print("Streaming with Region Fallback Test")
    print("=" * 60)
    
    adapter = GeminiAdapter(
        api_key="dummy_key",
        mode="vertex",
        project_id=os.getenv("GCP_PROJECT_ID", "your-project-id"),
        location="asia-southeast1",
        enable_region_fallback=True,
        fallback_location="us-central1"
    )
    
    test_model = "gemini-2.0-flash-lite-001"
    test_prompt = "Count from 1 to 5, one number per line."
    
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


async def test_fallback_disabled():
    """Test behavior when fallback is disabled."""
    
    print("\n" + "=" * 60)
    print("Fallback Disabled Test")
    print("=" * 60)
    
    adapter = GeminiAdapter(
        api_key="dummy_key",
        mode="vertex",
        project_id=os.getenv("GCP_PROJECT_ID", "your-project-id"),
        location="asia-southeast1",
        enable_region_fallback=False  # Disable fallback
    )
    
    print(f"\nFallback Enabled: {adapter.enable_region_fallback}")
    print("When fallback is disabled, errors will be raised immediately.")
    
    test_model = "gemini-2.0-flash-lite-001"
    test_prompt = "Hello"
    
    try:
        result = await adapter.generate(test_prompt, test_model)
        print(f"\n✓ Generation successful: {result.text[:50]}...")
    except Exception as e:
        print(f"\n✗ Generation failed (as expected): {e}")
    
    await adapter.aclose()


async def main():
    """Run all fallback tests."""
    
    # Check for required environment variable
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("⚠ Warning: GOOGLE_APPLICATION_CREDENTIALS not set")
        print("Please set it to your service account JSON key file path")
        print("Example: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json")
        return
    
    if not os.getenv("GCP_PROJECT_ID"):
        print("⚠ Warning: GCP_PROJECT_ID not set")
        print("Please set it to your Google Cloud project ID")
        print("Example: export GCP_PROJECT_ID=your-project-id")
        return
    
    # Run tests
    await test_region_fallback()
    await test_streaming_with_fallback()
    await test_fallback_disabled()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
