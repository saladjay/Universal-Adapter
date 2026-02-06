"""
Abstract base class for LLM provider adapters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

from ..models import TokenUsage


@dataclass
class RawLLMResult:
    """Raw result from LLM API call before processing."""
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw_response: dict | None = None
    
    # OpenRouter-specific fields (from HTTP headers)
    cost_usd: float | None = None
    provider: str | None = None
    actual_model: str | None = None
    latency_ms: int | None = None


class ProviderAdapter(ABC):
    """
    Abstract base class for LLM provider adapters.
    
    All provider adapters must implement:
    - generate(): Async method to call the LLM API
    - estimate_tokens(): Method to estimate/extract token usage
    """
    
    name: str = "base"
    
    def __init__(self, api_key: str, **kwargs):
        """
        Initialize the adapter.
        
        Args:
            api_key: API key for the provider
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.config = kwargs
    
    @abstractmethod
    async def generate(self, prompt: str, model: str) -> RawLLMResult:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The input prompt to send to the LLM
            model: The model identifier to use
            
        Returns:
            RawLLMResult containing the generated text and token information
            
        Raises:
            ProviderError: If the API call fails
        """
        pass

    async def stream(self, prompt: str, model: str) -> AsyncIterator[str]:
        """
        Stream response text from the LLM.

        Default implementation falls back to generate() and yields a single chunk.
        Adapters can override for true streaming support.
        """
        result = await self.generate(prompt, model)
        yield result.text

    async def aclose(self) -> None:
        """Optional async cleanup hook for adapters."""
        return None
    
    @abstractmethod
    def estimate_tokens(self, prompt: str, output: str) -> TokenUsage:
        """
        Estimate or extract token usage for a prompt/output pair.
        
        Args:
            prompt: The input prompt
            output: The generated output text
            
        Returns:
            TokenUsage with input_tokens and output_tokens
        """
        pass


class ProviderError(Exception):
    """Exception raised when a provider API call fails."""
    
    def __init__(self, provider: str, message: str, status_code: int | None = None):
        self.provider = provider
        self.message = message
        self.status_code = status_code
        super().__init__(f"[{provider}] {message}")
