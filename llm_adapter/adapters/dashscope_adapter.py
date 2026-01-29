"""
阿里百炼(DashScope) provider adapter implementation.
"""

import httpx

from ..models import TokenUsage
from .base import ProviderAdapter, ProviderError, RawLLMResult


class DashScopeAdapter(ProviderAdapter):
    """
    Adapter for Alibaba DashScope (阿里百炼) API.
    
    Implements the unified generate interface for DashScope models.
    Supports Qwen series models (通义千问).
    """
    
    name: str = "dashscope"
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
    
    def __init__(self, api_key: str, base_url: str | None = None, **kwargs):
        """
        Initialize DashScope adapter.
        
        Args:
            api_key: DashScope API key (阿里云百炼API Key)
            base_url: Optional custom base URL
            **kwargs: Additional configuration
        """
        super().__init__(api_key, **kwargs)
        self.base_url = base_url or self.DEFAULT_BASE_URL
    
    async def generate(self, prompt: str, model: str) -> RawLLMResult:
        """
        Generate a response using DashScope API.
        
        Args:
            prompt: The input prompt
            model: Model identifier (e.g., 'qwen-turbo', 'qwen-plus', 'qwen-max')
            
        Returns:
            RawLLMResult with generated text and token counts
            
        Raises:
            ProviderError: If API call fails
        """
        url = f"{self.base_url}/services/aigc/text-generation/generation"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "input": {
                "messages": [{"role": "user", "content": prompt}]
            },
            "parameters": {
                "result_format": "message"
            }
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
        
        # Check for API-level errors
        if "code" in data and data["code"] != "":
            raise ProviderError(
                self.name,
                f"API error: {data.get('message', 'Unknown error')} (code: {data.get('code')})"
            )
        
        # Extract response text
        try:
            output = data["output"]
            text = output["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ProviderError(self.name, f"Invalid response format: {e}")
        
        # Extract token usage from response
        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        
        return RawLLMResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_response=data
        )
    
    def estimate_tokens(self, prompt: str, output: str) -> TokenUsage:
        """
        Estimate token usage for DashScope.
        
        For DashScope, we prefer to use the actual token counts from the API response.
        This method provides a fallback estimation.
        
        Chinese text typically has ~1.5-2 tokens per character.
        
        Args:
            prompt: The input prompt
            output: The generated output
            
        Returns:
            TokenUsage with estimated token counts
        """
        # Rough estimation for Chinese text: ~1.5 tokens per character
        # For mixed Chinese/English, use ~2 characters per token
        input_tokens = max(1, len(prompt) // 2)
        output_tokens = max(1, len(output) // 2)
        
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
