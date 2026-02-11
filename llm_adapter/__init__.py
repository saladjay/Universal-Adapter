"""
LLM Adapter - 多海外LLM统一接入与计费监控系统

A unified interface for multiple LLM providers with intelligent routing,
cost calculation, and usage logging.

Example usage:
    from llm_adapter import LLMAdapter, LLMRequest
    
    adapter = LLMAdapter(config_path="config.yaml")
    response = await adapter.generate(
        user_id="user123",
        prompt="Hello, world!",
        scene="chat",
        quality="medium",
    )
    print(response.text)
    print(f"Cost: ${response.cost_usd:.6f}")
"""

__version__ = "0.1.0"

# Core data models
from .models import LLMRequest, LLMResponse, TokenUsage, PricingRule, UsageLog

# Configuration management
from .config import ConfigManager, ConfigError

# Billing engine
from .billing import BillingEngine, BillingError

# Router
from .router import Router, RouteResult, RouterError

# Usage logger
from .logger import UsageLogger

# Fallback tracker
from .fallback_tracker import FallbackTracker, FallbackEvent, FallbackStats, get_fallback_tracker

# Main adapter (unified entry point)
from .adapter import LLMAdapter, LLMAdapterError, ValidationError

# Provider adapters (for advanced usage)
from .adapters import (
    ProviderAdapter,
    ProviderError,
    RawLLMResult,
    ImageInput,
    ImageInputType,
    MultimodalContent,
    OpenAIAdapter,
    GeminiAdapter,
    CloudflareAdapter,
    HuggingFaceAdapter,
)

__all__ = [
    # Version
    "__version__",
    # Main entry point
    "LLMAdapter",
    "LLMAdapterError",
    "ValidationError",
    # Data models
    "LLMRequest",
    "LLMResponse",
    "TokenUsage",
    "PricingRule",
    "UsageLog",
    # Configuration
    "ConfigManager",
    "ConfigError",
    # Billing
    "BillingEngine",
    "BillingError",
    # Router
    "Router",
    "RouteResult",
    "RouterError",
    # Logger
    "UsageLogger",
    # Fallback tracker
    "FallbackTracker",
    "FallbackEvent",
    "FallbackStats",
    "get_fallback_tracker",
    # Provider adapters
    "ProviderAdapter",
    "ProviderError",
    "RawLLMResult",
    "ImageInput",
    "ImageInputType",
    "MultimodalContent",
    "OpenAIAdapter",
    "GeminiAdapter",
    "CloudflareAdapter",
    "HuggingFaceAdapter",
]
