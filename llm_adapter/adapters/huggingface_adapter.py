"""
HuggingFace Inference API provider adapter implementation.
"""

import httpx

from ..models import TokenUsage
from .base import ProviderAdapter, ProviderError, RawLLMResult


class HuggingFaceAdapter(ProviderAdapter):
    """
    Adapter for HuggingFace Inference API.
    
    Implements the unified generate interface for HuggingFace models.
    Uses local tokenizer estimation for token counting.
    """
    
    name: str = "huggingface"
    BASE_URL = "https://api-inference.huggingface.co/models"
    
    def __init__(self, api_key: str, default_model: str | None = None, **kwargs):
        """
        Initialize HuggingFace adapter.
        
        Args:
            api_key: HuggingFace API token (HF_TOKEN)
            default_model: Default model to use if none specified
            **kwargs: Additional configuration
        """
        super().__init__(api_key, **kwargs)
        self.default_model = default_model or "meta-llama/Llama-3.1-8B-Instruct"
        
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
    
    async def generate(self, prompt: str, model: str) -> RawLLMResult:
        """
        Generate a response using HuggingFace Inference API.
        
        Args:
            prompt: The input prompt
            model: Model identifier (e.g., 'meta-llama/Llama-3.1-8B-Instruct')
            
        Returns:
            RawLLMResult with generated text and estimated token counts
            
        Raises:
            ProviderError: If API call fails
        """
        model_to_use = model or self.default_model
        url = f"{self.BASE_URL}/{model_to_use}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1024,
                "return_full_text": False,
            },
        }
        
        try:
            response = await self._client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException:
            raise ProviderError(self.name, "Request timed out")
        except httpx.HTTPStatusError as e:
            # Handle specific HuggingFace errors
            if e.response.status_code == 503:
                raise ProviderError(
                    self.name,
                    "Model is loading, please retry in a few seconds",
                    status_code=503
                )
            raise ProviderError(
                self.name,
                f"API error: {e.response.text}",
                status_code=e.response.status_code
            )
        except Exception as e:
            raise ProviderError(self.name, f"Request failed: {str(e)}")
        
        # Extract response text
        # HuggingFace returns a list of generated texts
        try:
            if isinstance(data, list) and len(data) > 0:
                text = data[0].get("generated_text", "")
            elif isinstance(data, dict):
                text = data.get("generated_text", "")
            else:
                text = str(data)
        except Exception as e:
            raise ProviderError(self.name, f"Invalid response format: {e}")
        
        if not text:
            raise ProviderError(self.name, "Empty response from API")
        
        # HuggingFace doesn't return token counts directly
        # We'll estimate them in estimate_tokens()
        return RawLLMResult(
            text=text,
            input_tokens=None,
            output_tokens=None,
            raw_response=data
        )
    
    def estimate_tokens(self, prompt: str, output: str) -> TokenUsage:
        """
        Estimate token usage for HuggingFace models.
        
        Uses local estimation based on character count heuristics.
        For more accurate counting, a local tokenizer could be used.
        
        Args:
            prompt: The input prompt
            output: The generated output
            
        Returns:
            TokenUsage with estimated token counts
        """
        # Estimation based on character count
        # Llama-style tokenizers typically use ~4 chars per token for English
        # This is a simplified estimation; for production, consider using
        # the actual tokenizer from transformers library
        input_tokens = self._estimate_token_count(prompt)
        output_tokens = self._estimate_token_count(output)
        
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

    async def aclose(self) -> None:
        await self._client.aclose()
    
    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for a given text.
        
        Uses a simple heuristic based on:
        - Word count (most words = 1 token)
        - Special characters and punctuation
        - Numbers (often split into multiple tokens)
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        # Count words (rough approximation)
        words = text.split()
        word_count = len(words)
        
        # Add extra tokens for special characters and numbers
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        
        # Estimate: words + some overhead for special chars
        estimated = word_count + (special_chars // 4)
        
        return max(1, estimated)
