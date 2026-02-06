"""
阿里百炼(DashScope) provider adapter implementation.
"""

import json
import time

import httpx

from ..models import TokenUsage
from ..request_logger import get_logger
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
        
        # 初始化日志记录器
        self._logger = get_logger(self.name)
        
        # Get HTTP client config from config manager
        http_config = self.config.get("http_client", {})
        max_connections = http_config.get("max_connections", 100)
        max_keepalive = http_config.get("max_keepalive_connections", 20)
        timeout = http_config.get("timeout", 60.0)
        
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
        start_time = time.time()
        error_message = None
        result = None
        
        try:
            url = f"{self.base_url}/services/aigc/text-generation/generation"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "input": {
                    "messages": [{"role": "user", "content": prompt}]
                },
                "parameters": {
                    "result_format": "message",
                },
            }
            
            response = await self._client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Check for API-level errors
            if "code" in data and data["code"] != "":
                error_message = f"API error: {data.get('message', 'Unknown error')} (code: {data.get('code')})"
                raise ProviderError(self.name, error_message)
            
            # Extract response text
            output = data["output"]
            text = output["choices"][0]["message"]["content"]
            
            # Extract token usage from response
            usage = data.get("usage", {})
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            
            result = RawLLMResult(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                raw_response=data
            )
            
            return result
            
        except httpx.TimeoutException:
            error_message = "Request timed out"
            raise ProviderError(self.name, error_message)
        except httpx.HTTPStatusError as e:
            error_message = f"API error: {e.response.text}"
            raise ProviderError(
                self.name,
                error_message,
                status_code=e.response.status_code
            )
        except (KeyError, IndexError) as e:
            error_message = f"Invalid response format: {e}"
            raise ProviderError(self.name, error_message)
        except Exception as e:
            if not error_message:
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
            )

    async def stream(self, prompt: str, model: str):
        """Stream response text from DashScope."""
        url = f"{self.base_url}/services/aigc/text-generation/generation"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        payload = {
            "model": model,
            "input": {
                "messages": [{"role": "user", "content": prompt}]
            },
            "parameters": {
                "result_format": "message",
                "incremental_output": True,
            },
        }

        async with self._client.stream("POST", url, headers=headers, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    data = line[5:].strip()
                else:
                    data = line.strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    payload = json.loads(data)
                    output = payload.get("output", {})
                    choices = output.get("choices", [])
                    if not choices:
                        continue
                    text = choices[0].get("message", {}).get("content")
                except (json.JSONDecodeError, AttributeError):
                    continue
                if text:
                    yield text

    async def aclose(self) -> None:
        await self._client.aclose()
    
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
