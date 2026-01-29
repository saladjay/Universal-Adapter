"""
OpenAI provider adapter implementation.
"""

import httpx

from ..models import TokenUsage
from .base import ProviderAdapter, ProviderError, RawLLMResult


class OpenAIAdapter(ProviderAdapter):
    """
    Adapter for OpenAI API.
    
    Implements the unified generate interface for OpenAI models.
    Extracts token usage from API response (prompt_tokens, completion_tokens).
    """
    
    name: str = "openai"
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    
    def __init__(self, api_key: str, base_url: str | None = None, **kwargs):
        """
        Initialize OpenAI adapter.
        
        Args:
            api_key: OpenAI API key
            base_url: Optional custom base URL (for Azure OpenAI or proxies)
            **kwargs: Additional configuration
        """
        super().__init__(api_key, **kwargs)
        self.base_url = base_url or self.DEFAULT_BASE_URL
    
    async def generate(self, prompt: str, model: str) -> RawLLMResult:
        """
        Generate a response using OpenAI API.
        
        Args:
            prompt: The input prompt
            model: Model identifier (e.g., 'gpt-4o', 'gpt-4o-mini')
            
        Returns:
            RawLLMResult with generated text and token counts
            
        Raises:
            ProviderError: If API call fails
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        client_kwargs = {"timeout": 60.0}
        proxy_url = self.config.get("proxy_url")
        if proxy_url:
            client_kwargs["proxies"] = proxy_url
        async with httpx.AsyncClient(**client_kwargs) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
            except httpx.TimeoutException:
                raise ProviderError(self.name, "Request timed out")
            except httpx.HTTPStatusError as e:
                raise ProviderError(
                    self.name,
                    f"API error: {e.response.text}",
                    status_code=e.response.status_code
                )
            except Exception as e:
                raise ProviderError(self.name, f"Request failed: {str(e)}")
        
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
        Estimate token usage for OpenAI.
        
        For OpenAI, we prefer to use the actual token counts from the API response.
        This method provides a fallback estimation using character-based heuristics.
        
        Args:
            prompt: The input prompt
            output: The generated output
            
        Returns:
            TokenUsage with estimated token counts
        """
        # Rough estimation: ~4 characters per token for English text
        # This is a fallback when actual counts aren't available
        input_tokens = max(1, len(prompt) // 4)
        output_tokens = max(1, len(output) // 4)
        
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
