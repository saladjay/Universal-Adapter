"""
Cloudflare Workers AI provider adapter implementation.
"""

import httpx

from ..models import TokenUsage
from .base import ProviderAdapter, ProviderError, RawLLMResult


class CloudflareAdapter(ProviderAdapter):
    """
    Adapter for Cloudflare Workers AI API.
    
    Implements the unified generate interface for Cloudflare AI models.
    Uses neurons-to-token estimation since Cloudflare doesn't return token counts.
    """
    
    name: str = "cloudflare"
    BASE_URL = "https://api.cloudflare.com/client/v4/accounts"
    
    # Neurons to tokens estimation ratio
    # Based on Cloudflare's billing model where neurons roughly correlate to compute
    NEURONS_TO_TOKENS_RATIO = 0.1  # Approximate: 10 neurons ≈ 1 token
    
    def __init__(self, api_key: str, account_id: str | None = None, **kwargs):
        """
        Initialize Cloudflare adapter.
        
        Args:
            api_key: Cloudflare API token
            account_id: Cloudflare account ID
            **kwargs: Additional configuration
        """
        super().__init__(api_key, **kwargs)
        self.account_id = account_id or kwargs.get('account_id', '')
    
    async def generate(self, prompt: str, model: str) -> RawLLMResult:
        """
        Generate a response using Cloudflare Workers AI API.
        
        Args:
            prompt: The input prompt
            model: Model identifier (e.g., '@cf/meta/llama-3-8b-instruct')
            
        Returns:
            RawLLMResult with generated text and estimated token counts
            
        Raises:
            ProviderError: If API call fails
        """
        if not self.account_id:
            raise ProviderError(self.name, "account_id is required for Cloudflare")
        
        url = f"{self.BASE_URL}/{self.account_id}/ai/run/{model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "messages": [{"role": "user", "content": prompt}]
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
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
        
        # Check for API errors
        if not data.get("success", False):
            errors = data.get("errors", [])
            error_msg = errors[0].get("message", "Unknown error") if errors else "Unknown error"
            raise ProviderError(self.name, f"API error: {error_msg}")
        
        # Extract response text
        result = data.get("result", {})
        text = result.get("response", "")
        
        if not text:
            raise ProviderError(self.name, "Empty response from API")
        
        # Cloudflare doesn't return token counts directly
        # We'll estimate them in estimate_tokens()
        return RawLLMResult(
            text=text,
            input_tokens=None,
            output_tokens=None,
            raw_response=data
        )
    
    def estimate_tokens(self, prompt: str, output: str) -> TokenUsage:
        """
        Estimate token usage for Cloudflare Workers AI.
        
        Cloudflare uses neurons for billing, not tokens directly.
        This method estimates tokens based on character count heuristics.
        
        Args:
            prompt: The input prompt
            output: The generated output
            
        Returns:
            TokenUsage with estimated token counts
        """
        # Estimation based on character count
        # Cloudflare models (like Llama) typically use ~4 chars per token
        input_tokens = max(1, len(prompt) // 4)
        output_tokens = max(1, len(output) // 4)
        
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
    
    def estimate_tokens_from_neurons(self, neurons: int) -> int:
        """
        Convert Cloudflare neurons to approximate token count.
        
        Args:
            neurons: Number of neurons used
            
        Returns:
            Estimated token count
        """
        return max(1, int(neurons * self.NEURONS_TO_TOKENS_RATIO))
