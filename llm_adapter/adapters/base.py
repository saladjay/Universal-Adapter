"""
Abstract base class for LLM provider adapters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Literal, Union, List
from enum import Enum


class ImageInputType(str, Enum):
    """Type of image input."""
    URL = "url"
    BASE64 = "base64"


@dataclass
class ImageInput:
    """
    Image input for multimodal requests.
    
    Supports two modes:
    - URL mode: Provide image via URL
    - Base64 mode: Provide image as base64-encoded string
    """
    type: ImageInputType
    data: str  # URL or base64 string
    mime_type: str | None = None  # e.g., "image/jpeg", "image/png" (required for base64)
    
    @classmethod
    def from_url(cls, url: str) -> "ImageInput":
        """Create ImageInput from URL."""
        return cls(type=ImageInputType.URL, data=url)
    
    @classmethod
    def from_base64(cls, base64_data: str, mime_type: str = "image/jpeg") -> "ImageInput":
        """Create ImageInput from base64 string."""
        return cls(type=ImageInputType.BASE64, data=base64_data, mime_type=mime_type)


@dataclass
class MultimodalContent:
    """
    Content for multimodal requests.
    
    Can contain text and/or images.
    """
    text: str | None = None
    images: List[ImageInput] | None = None
    
    def __post_init__(self):
        if not self.text and not self.images:
            raise ValueError("MultimodalContent must have at least text or images")


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
    
    async def generate_multimodal(
        self, 
        content: MultimodalContent, 
        model: str
    ) -> RawLLMResult:
        """
        Generate a response from a multimodal LLM (text + images).
        
        This is an optional method. Adapters that support multimodal input
        should override this method. By default, it raises NotImplementedError.
        
        Args:
            content: MultimodalContent with text and/or images
            model: The model identifier to use (must support multimodal)
            
        Returns:
            RawLLMResult containing the generated text and token information
            
        Raises:
            NotImplementedError: If the adapter doesn't support multimodal
            ProviderError: If the API call fails
            
        Example:
            # URL mode
            content = MultimodalContent(
                text="What's in this image?",
                images=[ImageInput.from_url("https://example.com/image.jpg")]
            )
            result = await adapter.generate_multimodal(content, "gpt-4-vision")
            
            # Base64 mode
            content = MultimodalContent(
                text="Describe this image",
                images=[ImageInput.from_base64(base64_str, "image/jpeg")]
            )
            result = await adapter.generate_multimodal(content, "gemini-pro-vision")
        """
        raise NotImplementedError(
            f"{self.name} adapter does not support multimodal generation. "
            f"Please use a multimodal-capable model and adapter."
        )
    
    async def stream_multimodal(
        self,
        content: MultimodalContent,
        model: str
    ) -> AsyncIterator[str]:
        """
        Stream response from a multimodal LLM (text + images).
        
        This is an optional method. Adapters that support multimodal streaming
        should override this method. By default, it raises NotImplementedError.
        
        Args:
            content: MultimodalContent with text and/or images
            model: The model identifier to use (must support multimodal)
            
        Yields:
            Text chunks as they are generated
            
        Raises:
            NotImplementedError: If the adapter doesn't support multimodal streaming
            ProviderError: If the API call fails
            
        Example:
            content = MultimodalContent(
                text="Describe this image",
                images=[ImageInput.from_url("https://example.com/image.jpg")]
            )
            async for chunk in adapter.stream_multimodal(content, model):
                print(chunk, end="", flush=True)
        """
        raise NotImplementedError(
            f"{self.name} adapter does not support multimodal streaming. "
            f"Please use a multimodal-capable model and adapter."
        )

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
