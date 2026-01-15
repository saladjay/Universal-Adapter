"""
OpenRouter provider adapter implementation.

OpenRouter is a unified API that provides access to multiple LLM providers
including OpenAI, Anthropic, Google, Meta, and many others.
"""

import httpx

from ..models import TokenUsage
from .base import ProviderAdapter, ProviderError, RawLLMResult


class OpenRouterAdapter(ProviderAdapter):
    """
    Adapter for OpenRouter API.
    
    OpenRouter provides a unified OpenAI-compatible API for accessing
    multiple LLM providers. Supports models from OpenAI, Anthropic,
    Google, Meta, Mistral, and more.
    
    API Documentation: https://openrouter.ai/docs
    """
    
    name: str = "openrouter"
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(
        self, 
        api_key: str, 
        base_url: str | None = None,
        site_url: str | None = None,
        site_name: str | None = None,
        **kwargs
    ):
        """
        Initialize OpenRouter adapter.
        
        Args:
            api_key: OpenRouter API key
            base_url: Optional custom base URL
            site_url: Optional URL of your site (for rankings/analytics)
            site_name: Optional name of your site
            **kwargs: Additional configuration
        """
        super().__init__(api_key, **kwargs)
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.site_url = site_url
        self.site_name = site_name
    
    async def generate(self, prompt: str, model: str) -> RawLLMResult:
        """
        Generate a response using OpenRouter API.
        
        Args:
            prompt: The input prompt
            model: Model identifier (e.g., 'openai/gpt-4o', 'anthropic/claude-3-opus')
            
        Returns:
            RawLLMResult with generated text and token counts
            
        Raises:
            ProviderError: If API call fails
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Add optional headers for OpenRouter analytics
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
            except httpx.TimeoutException:
                raise ProviderError(self.name, "Request timed out")
            except httpx.HTTPStatusError as e:
                error_detail = ""
                try:
                    error_data = e.response.json()
                    error_detail = error_data.get("error", {}).get("message", e.response.text)
                except Exception:
                    error_detail = e.response.text
                raise ProviderError(
                    self.name,
                    f"API error: {error_detail}",
                    status_code=e.response.status_code
                )
            except Exception as e:
                raise ProviderError(self.name, f"Request failed: {str(e)}")
        
        # Check for API-level errors
        if "error" in data:
            raise ProviderError(
                self.name,
                f"API error: {data['error'].get('message', 'Unknown error')}"
            )
        
        # Extract response text
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ProviderError(self.name, f"Invalid response format: {e}")
        
        # Extract token usage from response
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        
        return RawLLMResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_response=data
        )
    
    def estimate_tokens(self, prompt: str, output: str) -> TokenUsage:
        """
        Estimate token usage for OpenRouter.
        
        OpenRouter returns actual token counts from the underlying provider,
        so this is a fallback estimation.
        
        Args:
            prompt: The input prompt
            output: The generated output
            
        Returns:
            TokenUsage with estimated token counts
        """
        # Rough estimation: ~4 characters per token for English text
        input_tokens = max(1, len(prompt) // 4)
        output_tokens = max(1, len(output) // 4)
        
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
