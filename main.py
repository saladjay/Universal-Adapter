"""
LLM Adapter Usage Example

This script demonstrates the complete call flow of the LLM Adapter system:
1. Initialize the adapter with configuration
2. Make LLM requests with different quality levels
3. View usage statistics and costs

Requirements: 1.1, 1.2
"""

import asyncio
import os
from datetime import datetime, timedelta

from llm_adapter import (
    LLMAdapter,
    LLMRequest,
    LLMResponse,
    ConfigManager,
    ConfigError,
    UsageLogger,
    ValidationError,
    LLMAdapterError,
    BillingEngine,
    Router,
    TokenUsage,
    PricingRule,
)


def check_config_available() -> bool:
    """Check if configuration with API keys is available."""
    try:
        config = ConfigManager(config_path="config.yaml")
        _ = config.get_default_provider()
        return True
    except ConfigError:
        return False


async def basic_usage_example():
    """
    Basic usage example: Make a simple LLM call.
    
    Demonstrates:
    - Creating an LLMAdapter instance
    - Making a generate() call
    - Accessing response fields
    """
    print("=" * 60)
    print("Basic Usage Example")
    print("=" * 60)
    
    # Initialize adapter with config file
    adapter = LLMAdapter(config_path="config.yaml")
    
    try:
        # Make a simple LLM call
        response: LLMResponse = await adapter.generate(
            user_id="user_001",
            prompt="What is the capital of France?",
            scene="chat",
            quality="medium",
        )
        
        print(f"Response: {response.text[:200]}...")
        print(f"Provider: {response.provider}")
        print(f"Model: {response.model}")
        print(f"Input Tokens: {response.input_tokens}")
        print(f"Output Tokens: {response.output_tokens}")
        print(f"Cost: ${response.cost_usd:.6f}")
        
    except ValidationError as e:
        print(f"Validation Error: {e.errors}")
    except LLMAdapterError as e:
        print(f"LLM Error: {e}")


def request_validation_example():
    """
    Request validation example.
    
    Demonstrates:
    - Automatic validation of request parameters
    - Clear error messages for invalid inputs
    """
    print("\n" + "=" * 60)
    print("Request Validation Example")
    print("=" * 60)
    
    # Test with invalid parameters
    invalid_requests = [
        {"user_id": "", "prompt": "Hello", "scene": "chat", "quality": "medium"},
        {"user_id": "user1", "prompt": "", "scene": "chat", "quality": "medium"},
        {"user_id": "user1", "prompt": "Hello", "scene": "invalid", "quality": "medium"},
        {"user_id": "user1", "prompt": "Hello", "scene": "chat", "quality": "invalid"},
    ]
    
    for params in invalid_requests:
        request = LLMRequest(**params)
        errors = request.validate()
        if errors:
            print(f"\nInvalid request: {params}")
            print(f"  Errors: {errors}")
    
    # Test with valid parameters
    valid_request = LLMRequest(
        user_id="user_001",
        prompt="Hello, world!",
        scene="chat",
        quality="medium",
    )
    errors = valid_request.validate()
    print(f"\nValid request: {valid_request}")
    print(f"  Errors: {errors if errors else 'None (valid)'}")


def token_usage_example():
    """
    Token usage and billing example.
    
    Demonstrates:
    - TokenUsage data class
    - PricingRule and cost calculation
    """
    print("\n" + "=" * 60)
    print("Token Usage & Billing Example")
    print("=" * 60)
    
    # Create token usage
    usage = TokenUsage(input_tokens=1000, output_tokens=500)
    print(f"\nToken Usage:")
    print(f"  Input Tokens: {usage.input_tokens}")
    print(f"  Output Tokens: {usage.output_tokens}")
    print(f"  Total Tokens: {usage.total_tokens}")
    
    # Create pricing rule and calculate cost
    pricing = PricingRule(
        provider="openai",
        model="gpt-4o-mini",
        input_cost_per_1m=0.15,
        output_cost_per_1m=0.60,
    )
    
    cost = pricing.calculate_cost(usage.input_tokens, usage.output_tokens)
    print(f"\nPricing Rule ({pricing.provider}/{pricing.model}):")
    print(f"  Input Cost: ${pricing.input_cost_per_1m}/1M tokens")
    print(f"  Output Cost: ${pricing.output_cost_per_1m}/1M tokens")
    print(f"  Calculated Cost: ${cost:.6f}")


def usage_logging_example():
    """
    Usage logging and statistics example.
    
    Demonstrates:
    - Logging is automatic after each call
    - Query usage by user
    - Get total costs and token usage
    """
    print("\n" + "=" * 60)
    print("Usage Logging Example")
    print("=" * 60)
    
    logger = UsageLogger()
    
    # Log some sample usage
    logger.log(
        user_id="user_001",
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=100,
        output_tokens=50,
        cost=0.000045,
    )
    
    logger.log(
        user_id="user_001",
        provider="gemini",
        model="gemini-1.5-flash",
        input_tokens=200,
        output_tokens=100,
        cost=0.000045,
    )
    
    logger.log(
        user_id="user_002",
        provider="openai",
        model="gpt-4o",
        input_tokens=500,
        output_tokens=200,
        cost=0.003250,
    )
    
    # Query usage by user
    print("\nUser 001 Usage:")
    user_logs = logger.get_logs_by_user("user_001")
    total_input, total_output = logger.get_user_total_tokens("user_001")
    total_cost = logger.get_user_total_cost("user_001")
    print(f"  Total Calls: {len(user_logs)}")
    print(f"  Total Input Tokens: {total_input}")
    print(f"  Total Output Tokens: {total_output}")
    print(f"  Total Cost: ${total_cost:.6f}")
    
    print("\nUser 002 Usage:")
    user_logs = logger.get_logs_by_user("user_002")
    total_input, total_output = logger.get_user_total_tokens("user_002")
    total_cost = logger.get_user_total_cost("user_002")
    print(f"  Total Calls: {len(user_logs)}")
    print(f"  Total Input Tokens: {total_input}")
    print(f"  Total Output Tokens: {total_output}")
    print(f"  Total Cost: ${total_cost:.6f}")
    
    # Query by time range
    print("\nLogs from last hour:")
    one_hour_ago = datetime.now() - timedelta(hours=1)
    recent_logs = logger.get_logs_by_time_range(start_time=one_hour_ago)
    for log in recent_logs:
        print(f"  {log.timestamp.strftime('%H:%M:%S')}: {log.provider}/{log.model} - ${log.cost:.6f}")


def llm_response_example():
    """
    LLM Response structure example.
    
    Demonstrates:
    - LLMResponse data class structure
    - All required fields for a complete response
    """
    print("\n" + "=" * 60)
    print("LLM Response Structure Example")
    print("=" * 60)
    
    # Create a sample response
    response = LLMResponse(
        text="The capital of France is Paris.",
        model="gpt-4o-mini",
        provider="openai",
        input_tokens=15,
        output_tokens=8,
        cost_usd=0.000007,
    )
    
    print(f"\nLLM Response:")
    print(f"  Text: {response.text}")
    print(f"  Provider: {response.provider}")
    print(f"  Model: {response.model}")
    print(f"  Input Tokens: {response.input_tokens}")
    print(f"  Output Tokens: {response.output_tokens}")
    print(f"  Cost (USD): ${response.cost_usd:.6f}")


async def full_integration_example():
    """
    Full integration example with actual API calls.
    
    Requires API keys to be configured in environment variables or config.yaml.
    """
    print("\n" + "=" * 60)
    print("Full Integration Example (requires API keys)")
    print("=" * 60)
    
    if not check_config_available():
        print("\nSkipping: API keys not configured.")
        print("To run this example, set the following environment variables:")
        print("  - OPENAI_API_KEY")
        print("  - GEMINI_API_KEY")
        print("  - CF_API_KEY and CF_ACCOUNT_ID")
        print("  - HF_TOKEN")
        return
    
    adapter = LLMAdapter(config_path="config.yaml")
    
    # Show available providers
    print("\nAvailable providers by quality level:")
    for quality in ["low", "medium", "high"]:
        providers = adapter.get_available_providers(quality)
        print(f"  {quality}: {providers}")
    
    # Make actual LLM calls
    await basic_usage_example()


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("LLM Adapter - Usage Examples")
    print("=" * 60)
    print("\nThis demo shows the LLM Adapter system capabilities.")
    print("Some examples work without API keys, others require configuration.")
    
    # Examples that work without API keys
    request_validation_example()
    token_usage_example()
    usage_logging_example()
    llm_response_example()
    
    # Full integration (requires API keys)
    await full_integration_example()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
