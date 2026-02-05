"""
Example: Using generation parameters with LLM Adapter

Demonstrates the three-layer configuration system:
1. Global defaults (config.yaml: llm.default_generation_params)
2. Provider-level (config.yaml: providers.{provider}.generation_params)
3. Model-specific (config.yaml: providers.{provider}.model_params.{model})
4. Runtime override (code)
"""

import asyncio

from llm_adapter.adapter import LLMAdapter
from llm_adapter.config import ConfigManager, GenerationParams


async def example_basic_usage():
    """Basic usage: Use configured parameters"""
    print("=" * 60)
    print("Example 1: Basic Usage (Using Configured Parameters)")
    print("=" * 60)
    
    adapter = LLMAdapter()
    
    # This will use parameters from config.yaml
    # Priority: model_params > generation_params > default_generation_params
    result = await adapter.generate(
        prompt="Write a haiku about programming",
        provider="openai",
        model="gpt-4o-mini"
    )
    
    print(f"Prompt: Write a haiku about programming")
    print(f"Response: {result.text}")
    print(f"Tokens: {result.input_tokens} in, {result.output_tokens} out")
    print()


async def example_runtime_override():
    """Runtime override: Override parameters in code"""
    print("=" * 60)
    print("Example 2: Runtime Override")
    print("=" * 60)
    
    adapter = LLMAdapter()
    
    # Override temperature and max_tokens at runtime
    result = await adapter.generate(
        prompt="Explain quantum computing in simple terms",
        provider="openai",
        model="gpt-4o",
        temperature=0.3,      # Lower temperature for more focused output
        max_tokens=200        # Limit output length
    )
    
    print(f"Prompt: Explain quantum computing in simple terms")
    print(f"Response: {result.text}")
    print(f"Tokens: {result.input_tokens} in, {result.output_tokens} out")
    print()


async def example_creative_vs_deterministic():
    """Compare creative vs deterministic outputs"""
    print("=" * 60)
    print("Example 3: Creative vs Deterministic")
    print("=" * 60)
    
    adapter = LLMAdapter()
    prompt = "Write a creative story opening about a robot"
    
    # Deterministic output (temperature = 0)
    print("Deterministic (temperature=0.0):")
    result1 = await adapter.generate(
        prompt=prompt,
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=100
    )
    print(f"Response: {result1.text}\n")
    
    # Creative output (temperature = 1.2)
    print("Creative (temperature=1.2):")
    result2 = await adapter.generate(
        prompt=prompt,
        provider="openai",
        model="gpt-4o-mini",
        temperature=1.2,
        max_tokens=100
    )
    print(f"Response: {result2.text}\n")


async def example_with_config_manager():
    """Use ConfigManager to get merged parameters"""
    print("=" * 60)
    print("Example 4: Using ConfigManager")
    print("=" * 60)
    
    config_manager = ConfigManager()
    
    # Get merged parameters for a specific provider/model
    params = config_manager.get_generation_params(
        provider="openai",
        model="gpt-4o-mini",
        override_params=GenerationParams(
            temperature=0.5,
            max_tokens=500
        )
    )
    
    print(f"Merged parameters for openai/gpt-4o-mini:")
    print(f"  {params.to_dict()}")
    print()
    
    # Use these parameters
    adapter = LLMAdapter()
    result = await adapter.generate(
        prompt="List 3 benefits of async programming",
        provider="openai",
        model="gpt-4o-mini",
        **params.to_dict()
    )
    
    print(f"Response: {result.text}")
    print()


async def example_provider_specific_params():
    """Use provider-specific parameters"""
    print("=" * 60)
    print("Example 5: Provider-Specific Parameters")
    print("=" * 60)
    
    adapter = LLMAdapter()
    
    # OpenAI with presence_penalty and frequency_penalty
    print("OpenAI with penalties:")
    result1 = await adapter.generate(
        prompt="Write about the importance of code review",
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.7,
        presence_penalty=0.6,    # Encourage new topics
        frequency_penalty=0.3,   # Reduce repetition
        max_tokens=150
    )
    print(f"Response: {result1.text}\n")
    
    # Gemini with top_k
    print("Gemini with top_k:")
    result2 = await adapter.generate(
        prompt="Write about the importance of code review",
        provider="gemini",
        model="gemini-2.5-flash",
        temperature=0.7,
        top_k=40,               # Gemini-specific
        top_p=0.95,
        max_tokens=150
    )
    print(f"Response: {result2.text}\n")


async def example_reproducible_output():
    """Generate reproducible output using seed"""
    print("=" * 60)
    print("Example 6: Reproducible Output (with seed)")
    print("=" * 60)
    
    adapter = LLMAdapter()
    prompt = "Generate a random number between 1 and 100"
    
    # First run with seed
    print("First run (seed=12345):")
    result1 = await adapter.generate(
        prompt=prompt,
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
        seed=12345
    )
    print(f"Response: {result1.text}\n")
    
    # Second run with same seed (should be similar)
    print("Second run (seed=12345):")
    result2 = await adapter.generate(
        prompt=prompt,
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
        seed=12345
    )
    print(f"Response: {result2.text}\n")
    
    # Third run with different seed
    print("Third run (seed=54321):")
    result3 = await adapter.generate(
        prompt=prompt,
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
        seed=54321
    )
    print(f"Response: {result3.text}\n")


async def example_stop_sequences():
    """Use stop sequences to control output"""
    print("=" * 60)
    print("Example 7: Stop Sequences")
    print("=" * 60)
    
    adapter = LLMAdapter()
    
    # Stop at double newline
    result = await adapter.generate(
        prompt="List programming languages:\n1.",
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=200,
        stop=["\n\n", "10."]  # Stop at double newline or item 10
    )
    
    print(f"Response (stopped at '\\n\\n' or '10.'):")
    print(f"{result.text}")
    print()


async def main():
    """Run all examples"""
    try:
        await example_basic_usage()
        await example_runtime_override()
        await example_creative_vs_deterministic()
        await example_with_config_manager()
        await example_provider_specific_params()
        await example_reproducible_output()
        await example_stop_sequences()
        
        print("=" * 60)
        print("All examples completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
