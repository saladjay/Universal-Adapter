"""
Google Gemini provider adapter implementation.
Supports HTTP API, official SDK, and Vertex AI SDK modes.
"""

import json
import time

from typing import Literal, Optional

import httpx

from ..models import TokenUsage
from ..fallback_tracker import get_fallback_tracker
from .base import ProviderAdapter, ProviderError, RawLLMResult


class GeminiAdapter(ProviderAdapter):
    """
    Adapter for Google Gemini API.
    
    Supports three modes:
    - "http": Direct HTTP calls (default, fewer dependencies)
    - "sdk": Official google-generativeai SDK (more features, better stability)
    - "vertex": Vertex AI SDK (for GCP projects with regional deployment)
    
    Example:
        # HTTP mode (default)
        adapter = GeminiAdapter(api_key="xxx")
        
        # SDK mode
        adapter = GeminiAdapter(api_key="xxx", mode="sdk")
        
        # Vertex AI mode
        adapter = GeminiAdapter(
            api_key="xxx",  # Not used in vertex mode
            mode="vertex",
            project_id="your-project-id",
            location="asia-southeast1"
        )
    """
    
    name: str = "gemini"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    
    def __init__(
        self, 
        api_key: str, 
        mode: Literal["http", "sdk", "vertex"] = "http",
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        enable_region_fallback: bool = True,
        fallback_location: str = "us-central1",
        **kwargs
    ):
        """
        Initialize Gemini adapter.
        
        Args:
            api_key: Google API key (not used in vertex mode)
            mode: "http" for direct API calls, "sdk" for official SDK, "vertex" for Vertex AI
            project_id: GCP project ID (required for vertex mode)
            location: GCP region (required for vertex mode, e.g., "asia-southeast1")
            enable_region_fallback: Enable automatic fallback to global region on failure
            fallback_location: Fallback region to use (default: "us-central1")
            **kwargs: Additional configuration
        """
        super().__init__(api_key, **kwargs)
        self.mode = mode
        self.project_id = project_id
        self.location = location
        self.enable_region_fallback = enable_region_fallback
        self.fallback_location = fallback_location
        self._sdk_client = None
        self._vertex_model_cache = {}  # Cache for Vertex AI models
        self._client: httpx.AsyncClient | None = None
        self._fallback_tracker = get_fallback_tracker()
        
        if mode == "http":
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
        
        if mode == "sdk":
            self._init_sdk()
        
        if mode == "vertex":
            self._init_vertex()
    
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
    
    def _init_vertex(self):
        """Initialize Vertex AI SDK with environment variable checks."""
        import os
        
        if not self.project_id:
            raise ProviderError(
                self.name,
                "Vertex AI mode requires 'project_id' parameter"
            )
        if not self.location:
            raise ProviderError(
                self.name,
                "Vertex AI mode requires 'location' parameter (e.g., 'asia-southeast1')"
            )
        
        # Check for required environment variables
        required_env_vars = [
            "GOOGLE_APPLICATION_CREDENTIALS",
        ]
        
        missing_vars = []
        for var in required_env_vars:
            if not os.environ.get(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ProviderError(
                self.name,
                f"Vertex AI mode requires environment variable(s): {', '.join(missing_vars)}. "
                f"Set GOOGLE_APPLICATION_CREDENTIALS to the path of your service account JSON key file. "
                f"Example: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json"
            )
        
        # Verify the credentials file exists
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path and not os.path.exists(creds_path):
            raise ProviderError(
                self.name,
                f"GOOGLE_APPLICATION_CREDENTIALS points to non-existent file: {creds_path}"
            )
        
        try:
            import vertexai
            vertexai.init(project=self.project_id, location=self.location)
            self._vertexai = vertexai
        except ImportError:
            raise ProviderError(
                self.name,
                "Vertex AI mode requires 'google-cloud-aiplatform' package. "
                "Install with: pip install google-cloud-aiplatform"
            )
        except Exception as e:
            error_msg = str(e)
            if "credentials" in error_msg.lower() or "authentication" in error_msg.lower():
                raise ProviderError(
                    self.name,
                    f"Vertex AI authentication failed: {error_msg}. "
                    f"Please check your GOOGLE_APPLICATION_CREDENTIALS environment variable."
                )
            raise ProviderError(
                self.name,
                f"Failed to initialize Vertex AI: {error_msg}"
            )
    
    def _get_vertex_model(self, model: str):
        """Get or create a cached Vertex AI GenerativeModel instance."""
        if model not in self._vertex_model_cache:
            from vertexai.generative_models import GenerativeModel
            self._vertex_model_cache[model] = GenerativeModel(model)
        return self._vertex_model_cache[model]
    
    async def generate(self, prompt: str, model: str) -> RawLLMResult:
        """
        Generate a response using Gemini API.
        
        Args:
            prompt: The input prompt
            model: Model identifier (e.g., 'gemini-2.0-flash', 'gemini-2.5-pro-preview')
            
        Returns:
            RawLLMResult with generated text and token counts
        """
        if self.mode == "vertex":
            return await self._generate_vertex(prompt, model)
        if self.mode == "sdk":
            return await self._generate_sdk(prompt, model)
        return await self._generate_http(prompt, model)

    async def stream(self, prompt: str, model: str):
        """Stream response text from Gemini."""
        if self.mode == "vertex":
            async for chunk in self._stream_vertex(prompt, model):
                yield chunk
            return
        if self.mode == "sdk":
            async for chunk in self._stream_sdk(prompt, model):
                yield chunk
            return
        async for chunk in self._stream_http(prompt, model):
            yield chunk
    
    async def _generate_http(self, prompt: str, model: str) -> RawLLMResult:
        """Generate using direct HTTP API calls."""
        url = f"{self.BASE_URL}/models/{model}:generateContent"
        params = {"key": self.api_key}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
        }
        
        if not self._client:
            raise ProviderError(self.name, "HTTP client not initialized")
        try:
            response = await self._client.post(url, params=params, json=payload)
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

    async def _stream_http(self, prompt: str, model: str):
        url = f"{self.BASE_URL}/models/{model}:streamGenerateContent"
        params = {"key": self.api_key}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        headers = {"Accept": "text/event-stream"}

        if not self._client:
            raise ProviderError(self.name, "HTTP client not initialized")
        async with self._client.stream(
            "POST",
            url,
            params=params,
            headers=headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                    text = payload["candidates"][0]["content"]["parts"][0].get("text")
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue
                if text:
                    yield text
    
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

    async def _stream_sdk(self, prompt: str, model: str):
        import asyncio

        if not self._genai:
            self._init_sdk()

        def _sync_stream():
            gen_model = self._genai.GenerativeModel(model)
            return gen_model.generate_content(prompt, stream=True)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _sync_stream)
        for chunk in response:
            text = getattr(chunk, "text", None)
            if text:
                yield text

    async def _generate_vertex(self, prompt: str, model: str) -> RawLLMResult:
        """Generate using Vertex AI SDK with region fallback support."""
        import asyncio
        
        try:
            # Vertex AI SDK is sync, run in executor
            def _sync_generate():
                vertex_model = self._get_vertex_model(model)
                return vertex_model.generate_content(prompt)
            
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
            
            # Check if we should attempt region fallback
            should_fallback = (
                self.enable_region_fallback and
                self.location != self.fallback_location and
                ("not found" in error_msg.lower() or 
                 "404" in error_msg or
                 "unavailable" in error_msg.lower() or
                 "permission" in error_msg.lower() or
                 "403" in error_msg)
            )
            
            if should_fallback:
                return await self._fallback_to_global_region(prompt, model, error_msg)
            
            # Handle common Vertex AI errors without fallback
            if "quota" in error_msg.lower() or "429" in error_msg:
                raise ProviderError(self.name, f"Rate limit exceeded: {error_msg}", status_code=429)
            if "not found" in error_msg.lower() or "404" in error_msg:
                raise ProviderError(self.name, f"Model not found: {error_msg}", status_code=404)
            if "permission" in error_msg.lower() or "403" in error_msg:
                raise ProviderError(self.name, f"Permission denied: {error_msg}", status_code=403)
            raise ProviderError(self.name, f"Vertex AI error: {error_msg}")

    async def _fallback_to_global_region(
        self, 
        prompt: str, 
        model: str, 
        original_error: str
    ) -> RawLLMResult:
        """
        Fallback to global region when regional endpoint fails.
        
        Args:
            prompt: The input prompt
            model: Model identifier
            original_error: Error message from original attempt
            
        Returns:
            RawLLMResult from fallback region
            
        Raises:
            ProviderError: If fallback also fails
        """
        import asyncio
        
        fallback_start = time.time()
        original_location = self.location
        
        try:
            # Reinitialize Vertex AI with fallback location
            self._vertexai.init(project=self.project_id, location=self.fallback_location)
            self.location = self.fallback_location
            
            # Clear model cache to force new model with fallback region
            self._vertex_model_cache.clear()
            
            # Attempt generation with fallback region
            def _sync_generate():
                vertex_model = self._get_vertex_model(model)
                return vertex_model.generate_content(prompt)
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, _sync_generate)
            
            text = response.text
            
            # Extract token usage
            input_tokens = None
            output_tokens = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                input_tokens = getattr(response.usage_metadata, 'prompt_token_count', None)
                output_tokens = getattr(response.usage_metadata, 'candidates_token_count', None)
            
            # Record successful fallback
            fallback_duration = (time.time() - fallback_start) * 1000
            self._fallback_tracker.record_fallback(
                provider=self.name,
                original_location=original_location,
                fallback_location=self.fallback_location,
                original_model=model,
                fallback_model=model,
                error_message=original_error,
                fallback_duration_ms=fallback_duration,
                success=True
            )
            
            return RawLLMResult(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                raw_response=response
            )
            
        except Exception as fallback_error:
            # Record failed fallback
            fallback_duration = (time.time() - fallback_start) * 1000
            self._fallback_tracker.record_fallback(
                provider=self.name,
                original_location=original_location,
                fallback_location=self.fallback_location,
                original_model=model,
                fallback_model=model,
                error_message=original_error,
                fallback_duration_ms=fallback_duration,
                success=False
            )
            
            # Restore original location
            self.location = original_location
            self._vertexai.init(project=self.project_id, location=self.location)
            
            raise ProviderError(
                self.name,
                f"Region fallback failed. Original error: {original_error}. "
                f"Fallback error: {str(fallback_error)}"
            )

    async def _stream_vertex(self, prompt: str, model: str):
        """Stream using Vertex AI SDK with region fallback support."""
        import asyncio

        def _sync_stream():
            vertex_model = self._get_vertex_model(model)
            return vertex_model.generate_content(prompt, stream=True)

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, _sync_stream)
            for chunk in response:
                text = getattr(chunk, "text", None)
                if text:
                    yield text
        except Exception as e:
            error_msg = str(e)
            
            # Check if we should attempt region fallback for streaming
            should_fallback = (
                self.enable_region_fallback and
                self.location != self.fallback_location and
                ("not found" in error_msg.lower() or 
                 "404" in error_msg or
                 "unavailable" in error_msg.lower() or
                 "permission" in error_msg.lower() or
                 "403" in error_msg)
            )
            
            if should_fallback:
                # For streaming, we need to retry with fallback region
                async for chunk in self._stream_vertex_with_fallback(prompt, model, error_msg):
                    yield chunk
                return
            
            if "quota" in error_msg.lower() or "429" in error_msg:
                raise ProviderError(self.name, f"Rate limit exceeded: {error_msg}", status_code=429)
            if "not found" in error_msg.lower() or "404" in error_msg:
                raise ProviderError(self.name, f"Model not found: {error_msg}", status_code=404)
            if "permission" in error_msg.lower() or "403" in error_msg:
                raise ProviderError(self.name, f"Permission denied: {error_msg}", status_code=403)
            raise ProviderError(self.name, f"Vertex AI streaming error: {error_msg}")
    
    async def _stream_vertex_with_fallback(self, prompt: str, model: str, original_error: str):
        """Stream with fallback to global region."""
        import asyncio
        
        fallback_start = time.time()
        original_location = self.location
        
        try:
            # Reinitialize with fallback location
            self._vertexai.init(project=self.project_id, location=self.fallback_location)
            self.location = self.fallback_location
            self._vertex_model_cache.clear()
            
            def _sync_stream():
                vertex_model = self._get_vertex_model(model)
                return vertex_model.generate_content(prompt, stream=True)
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, _sync_stream)
            
            # Record successful fallback
            fallback_duration = (time.time() - fallback_start) * 1000
            self._fallback_tracker.record_fallback(
                provider=self.name,
                original_location=original_location,
                fallback_location=self.fallback_location,
                original_model=model,
                fallback_model=model,
                error_message=original_error,
                fallback_duration_ms=fallback_duration,
                success=True
            )
            
            for chunk in response:
                text = getattr(chunk, "text", None)
                if text:
                    yield text
                    
        except Exception as fallback_error:
            # Record failed fallback
            fallback_duration = (time.time() - fallback_start) * 1000
            self._fallback_tracker.record_fallback(
                provider=self.name,
                original_location=original_location,
                fallback_location=self.fallback_location,
                original_model=model,
                fallback_model=model,
                error_message=original_error,
                fallback_duration_ms=fallback_duration,
                success=False
            )
            
            # Restore original location
            self.location = original_location
            self._vertexai.init(project=self.project_id, location=self.location)
            
            raise ProviderError(
                self.name,
                f"Region fallback failed for streaming. Original error: {original_error}. "
                f"Fallback error: {str(fallback_error)}"
            )

    async def aclose(self) -> None:
        if self._client:
            await self._client.aclose()
    
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
