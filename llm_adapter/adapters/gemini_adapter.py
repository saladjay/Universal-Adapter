"""
Google Gemini provider adapter implementation.
Supports both HTTP API and official SDK modes.
"""

from typing import Literal
import httpx

from ..models import TokenUsage
from .base import ProviderAdapter, ProviderError, RawLLMResult


class GeminiAdapter(ProviderAdapter):
    """
    Adapter for Google Gemini API.
    
    Supports two modes:
    - "http": Direct HTTP calls (default, fewer dependencies)
    - "sdk": Official google-generativeai SDK (more features, better stability)
    
    Example:
        # HTTP mode (default)
        adapter = GeminiAdapter(api_key="xxx")
        
        # SDK mode
        adapter = GeminiAdapter(api_key="xxx", mode="sdk")
    """
    
    name: str = "gemini"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    
    def __init__(
        self, 
        api_key: str, 
        mode: Literal["http", "sdk"] = "http",
        **kwargs
    ):
        """
        Initialize Gemini adapter.
        
        Args:
            api_key: Google API key
            mode: "http" for direct API calls, "sdk" for official SDK
            **kwargs: Additional configuration
        """
        super().__init__(api_key, **kwargs)
        self.mode = mode
        self._sdk_client = None
        
        if mode == "sdk":
            self._init_sdk()
    
    def _init_sdk(self):
        """Initialize the official SDK client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._genai = genai
        except ImportError:
            raise ProviderError(
                self.name,
                "SDK mode requires 'google-generativeai' package. "
                "Install with: pip install google-generativeai"
            )
    
    async def generate(self, prompt: str, model: str) -> RawLLMResult:
        """
        Generate a response using Gemini API.
        
        Args:
            prompt: The input prompt
            model: Model identifier (e.g., 'gemini-2.0-flash', 'gemini-2.5-pro-preview')
            
        Returns:
            RawLLMResult with generated text and token counts
        """
        if self.mode == "sdk":
            return await self._generate_sdk(prompt, model)
        return await self._generate_http(prompt, model)
    
    async def _generate_http(self, prompt: str, model: str) -> RawLLMResult:
        """Generate using direct HTTP API calls."""
        url = f"{self.BASE_URL}/models/{model}:generateContent"
        params = {"key": self.api_key}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        
        client_kwargs = {"timeout": 60.0}
        proxy_url = self.config.get("proxy_url")
        if proxy_url:
            client_kwargs["proxies"] = proxy_url
        async with httpx.AsyncClient(**client_kwargs) as client:
            try:
                response = await client.post(url, params=params, json=payload)
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
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise ProviderError(self.name, f"Invalid response format: {e}")
        
        # Extract token usage from usageMetadata
        usage_metadata = data.get("usageMetadata", {})
        input_tokens = usage_metadata.get("promptTokenCount")
        output_tokens = usage_metadata.get("candidatesTokenCount")
        
        return RawLLMResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_response=data
        )
    
    async def _generate_sdk(self, prompt: str, model: str) -> RawLLMResult:
        """Generate using official google-generativeai SDK."""
        import asyncio
        
        try:
            # SDK is sync, run in executor
            def _sync_generate():
                gen_model = self._genai.GenerativeModel(model)
                return gen_model.generate_content(prompt)
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, _sync_generate)
            
            text = response.text
            
            # Extract token usage
            input_tokens = None
            output_tokens = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                input_tokens = getattr(response.usage_metadata, 'prompt_token_count', None)
                output_tokens = getattr(response.usage_metadata, 'candidates_token_count', None)
            
            return RawLLMResult(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                raw_response=response
            )
            
        except Exception as e:
            error_msg = str(e)
            # Handle common SDK errors
            if "quota" in error_msg.lower() or "429" in error_msg:
                raise ProviderError(self.name, f"Rate limit exceeded: {error_msg}", status_code=429)
            if "not found" in error_msg.lower() or "404" in error_msg:
                raise ProviderError(self.name, f"Model not found: {error_msg}", status_code=404)
            raise ProviderError(self.name, f"SDK error: {error_msg}")
    
    def estimate_tokens(self, prompt: str, output: str) -> TokenUsage:
        """
        Estimate token usage for Gemini.
        
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
