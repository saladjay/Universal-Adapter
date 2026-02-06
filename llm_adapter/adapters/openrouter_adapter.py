"""
OpenRouter provider adapter implementation.

OpenRouter is a unified API that provides access to multiple LLM providers
including OpenAI, Anthropic, Google, Meta, and many others.
"""

import json
import time

import httpx

from ..models import TokenUsage
from ..request_logger import get_logger
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
        
        # 初始化日志记录器
        self._logger = get_logger(self.name)
        
        # Get HTTP client config from config manager
        http_config = self.config.get("http_client", {})
        max_connections = http_config.get("max_connections", 100)
        max_keepalive = http_config.get("max_keepalive_connections", 20)
        timeout = http_config.get("timeout", 120.0)
        
        # Build client kwargs with proxy support for different httpx versions
        client_kwargs = {
            "timeout": timeout,
            "limits": httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive,
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
    
    async def generate(self, prompt: str, model: str, **kwargs) -> RawLLMResult:
        """
        Generate a response using OpenRouter API.
        
        Args:
            prompt: The input prompt
            model: Model identifier (e.g., 'openai/gpt-4o', 'anthropic/claude-3-opus')
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            RawLLMResult with generated text and token counts
            
        Raises:
            ProviderError: If API call fails
        """
        start_time = time.time()
        error_message = None
        result = None
        
        try:
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
            
            # Build payload with generation parameters
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }
            
            # Add generation parameters from kwargs
            # Filter out None values and non-generation params
            generation_keys = {
                'temperature', 'top_p', 'top_k', 'max_tokens', 
                'presence_penalty', 'frequency_penalty', 'stop', 'seed'
            }
            for key, value in kwargs.items():
                if key in generation_keys and value is not None:
                    payload[key] = value
            
            response = await self._client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Extract OpenRouter-specific metadata from response body
            # OpenRouter returns cost in the usage object, not in headers
            usage = data.get("usage", {})
            cost_usd = usage.get("cost")  # Direct cost in USD
            
            # Provider and model info are in the top-level response
            provider = data.get("provider")
            actual_model = data.get("model")
            
            # Latency is not provided by OpenRouter in the response
            # Could calculate from request time if needed
            latency_ms = None
                    
            # Check for API-level errors
            if "error" in data:
                raise ProviderError(
                    self.name,
                    f"API error: {data['error'].get('message', 'Unknown error')}"
                )
            
            # Extract response text
            text = data["choices"][0]["message"]["content"]
            
            # Extract token usage from response
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
            
            result = RawLLMResult(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                raw_response=data,
                cost_usd=cost_usd,
                provider=provider,
                actual_model=actual_model,
                latency_ms=latency_ms
            )
            
            return result
            
        except httpx.TimeoutException:
            error_message = "Request timed out"
            raise ProviderError(self.name, error_message)
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", e.response.text)
            except Exception:
                error_detail = e.response.text
            error_message = f"API error: {error_detail}"
            raise ProviderError(
                self.name,
                error_message,
                status_code=e.response.status_code
            )
        except Exception as e:
            error_message = f"Request failed: {str(e)}"
            raise ProviderError(self.name, error_message)
        
        finally:
            # 记录请求日志
            duration_ms = (time.time() - start_time) * 1000
            self._logger.log_request(
                model=model,
                prompt=prompt,
                response_text=result.text if result else None,
                input_tokens=result.input_tokens if result else None,
                output_tokens=result.output_tokens if result else None,
                duration_ms=duration_ms,
                success=result is not None,
                error_message=error_message,
                cost_usd=result.cost_usd if result else None,
                provider=result.provider if result else None,
                actual_model=result.actual_model if result else None,
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
