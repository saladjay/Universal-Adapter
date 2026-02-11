"""
Provider adapters for various LLM platforms.
"""

from .base import (
    ProviderAdapter, 
    ProviderError, 
    RawLLMResult,
    ImageInput,
    ImageInputType,
    MultimodalContent,
)
from .openai_adapter import OpenAIAdapter
from .gemini_adapter import GeminiAdapter
from .cloudflare_adapter import CloudflareAdapter
from .huggingface_adapter import HuggingFaceAdapter
from .dashscope_adapter import DashScopeAdapter
from .openrouter_adapter import OpenRouterAdapter

__all__ = [
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
    "DashScopeAdapter",
    "OpenRouterAdapter",
]
