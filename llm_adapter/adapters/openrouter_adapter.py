"""
OpenRouter provider adapter implementation.

OpenRouter is a unified API that provides access to multiple LLM providers
including OpenAI, Anthropic, Google, Meta, and many others.
"""

import json

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
        
        # Build client kwargs with proxy support for different httpx versions
        client_kwargs = {
            "timeout": 120.0,
            "limits": httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
            ),
        }
        
        # Handle proxy configuration for different httpx versions
        proxy_url = self.config.get("proxy_url")
        if proxy_url:
            try:
                # Try newer httpx versions (0.24+) that use 'proxy' parameter
                self._client = httpx.AsyncClient(proxy=proxy_url, **client_kwargs)
            except TypeError:
                # Fall back to older httpx versions that use 'proxies' parameter
                self._client = httpx.AsyncClient(proxies=proxy_url, **client_kwargs)
        else:
            self._client = httpx.AsyncClient(**client_kwargs)
    
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
            "messages": [{"role": "user", "content": prompt}],
        }
        
        try:
            response = await self._client.post(url, headers=headers, json=payload)
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

    async def stream(self, prompt: str, model: str):
        """Stream response text using OpenRouter's OpenAI-compatible API."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }

        async with self._client.stream("POST", url, headers=headers, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                    delta = payload["choices"][0].get("delta", {})
                    content = delta.get("content")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if content:
                    yield content

    async def aclose(self) -> None:
        if not self._client.is_closed:
            await self._client.aclose()
    
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
