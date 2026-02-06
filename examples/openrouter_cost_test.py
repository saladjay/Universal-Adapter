"""
Test OpenRouter cost and metadata extraction from HTTP headers.

This example demonstrates how OpenRouter provides cost information
through HTTP response headers, not just in the JSON body.
"""

import asyncio
import os
from llm_adapter.adapters.openrouter_adapter import OpenRouterAdapter


async def test_cost_extraction():
    """Test extracting cost and metadata from OpenRouter response."""
    
    print("=" * 70)
    print("OpenRouter Cost & Metadata Extraction Test")
    print("=" * 70)
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("\n⚠ Error: OPENROUTER_API_KEY environment variable not set")
        return
    
    adapter = OpenRouterAdapter(
        api_key=api_key,
        site_name="LLM Adapter Test",
        site_url="https://github.com/your-repo"
    )
    
    # Test with a cheap model
    model = "google/gemini-2.0-flash-lite-001"
    prompt = "Say hello in 5 words."
    
    print(f"\nTesting with model: {model}")
    print(f"Prompt: {prompt}")
    print("\nGenerating response...")
    
    try:
        result = await adapter.generate(prompt, model)
        
        print("\n" + "=" * 70)
        print("Response Details")
        print("=" * 70)
        
        print(f"\n📝 Generated Text:")
        print(f"  {result.text}")
        
        print(f"\n🔢 Token Usage:")
        print(f"  Input tokens:  {result.input_tokens}")
        print(f"  Output tokens: {result.output_tokens}")
        print(f"  Total tokens:  {(result.input_tokens or 0) + (result.output_tokens or 0)}")
        
        print(f"\n💰 Cost Information (from response body):")
        if result.cost_usd is not None:
            print(f"  ✓ Cost: ${result.cost_usd:.6f} USD")
            print(f"    (≈ ${result.cost_usd * 1000:.4f} per 1K requests)")
        else:
            print(f"  ✗ Cost not available in response")
        
        print(f"\n🔍 Provider Metadata (from response body):")
        if result.provider:
            print(f"  ✓ Provider: {result.provider}")
        else:
            print(f"  ✗ Provider not available")
            
        if result.actual_model:
            print(f"  ✓ Actual Model: {result.actual_model}")
        else:
            print(f"  ✗ Actual model not available")
            
        if result.latency_ms is not None:
            print(f"  ✓ Processing Time: {result.latency_ms}ms")
        else:
            print(f"  ✗ Processing time not available")
        
        # Calculate cost per token if available
        if result.cost_usd and result.input_tokens and result.output_tokens:
            total_tokens = result.input_tokens + result.output_tokens
            cost_per_1m_tokens = (result.cost_usd / total_tokens) * 1_000_000
            print(f"\n📊 Cost Analysis:")
            print(f"  Cost per 1M tokens: ${cost_per_1m_tokens:.2f}")
            print(f"  Cost per token: ${result.cost_usd / total_tokens:.8f}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    await adapter.aclose()


async def test_multiple_models():
    """Test cost extraction across different models."""
    
    print("\n" + "=" * 70)
    print("Multi-Model Cost Comparison")
    print("=" * 70)
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return
    
    adapter = OpenRouterAdapter(api_key=api_key)
    
    models = [
        "google/gemini-2.0-flash-lite-001",
        "meta-llama/llama-3.1-8b-instruct",
        "anthropic/claude-3.5-sonnet",
    ]
    
    prompt = "Hi"
    results = []
    
    print(f"\nTesting {len(models)} models with prompt: '{prompt}'")
    print("\nGenerating responses...\n")
    
    for model in models:
        try:
            result = await adapter.generate(prompt, model)
            results.append({
                "model": model,
                "cost": result.cost_usd,
                "tokens": (result.input_tokens or 0) + (result.output_tokens or 0),
                "provider": result.provider,
                "latency": result.latency_ms
            })
            print(f"✓ {model}")
        except Exception as e:
            print(f"✗ {model}: {e}")
            results.append({
                "model": model,
                "cost": None,
                "tokens": 0,
                "provider": None,
                "latency": None
            })
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("Cost Comparison Table")
    print("=" * 70)
    print(f"\n{'Model':<40} | {'Cost (USD)':<12} | {'Tokens':<8} | {'Latency':<10}")
    print("-" * 70)
    
    for r in results:
        cost_str = f"${r['cost']:.6f}" if r['cost'] is not None else "N/A"
        latency_str = f"{r['latency']}ms" if r['latency'] is not None else "N/A"
        print(f"{r['model']:<40} | {cost_str:<12} | {r['tokens']:<8} | {latency_str:<10}")
    
    # Find cheapest
    valid_results = [r for r in results if r['cost'] is not None]
    if valid_results:
        cheapest = min(valid_results, key=lambda x: x['cost'])
        print(f"\n💡 Cheapest model: {cheapest['model']} (${cheapest['cost']:.6f})")
    
    await adapter.aclose()


async def main():
    """Run all tests."""
    await test_cost_extraction()
    await test_multiple_models()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\n✅ OpenRouter provides cost information in the JSON response body:")
    print("  • usage.cost: Actual USD cost for the request")
    print("  • provider: Which provider handled the request (top-level)")
    print("  • model: Actual model used (top-level)")
    print("  • usage.cost_details: Breakdown of prompt/completion costs")
    print("\n✅ Token usage is also in the response body:")
    print("  • usage.prompt_tokens: Input tokens")
    print("  • usage.completion_tokens: Output tokens")
    print("  • usage.total_tokens: Total tokens")
    print("\n💡 This enables real-time cost tracking and provider analytics!")
    print("\n📝 Note: The chat2.md document mentioned HTTP headers, but")
    print("   OpenRouter actually returns cost in the JSON body instead.")


if __name__ == "__main__":
    asyncio.run(main())
