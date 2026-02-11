"""
阿里百炼(DashScope) provider adapter implementation.
Supports HTTP API and official SDK modes.
"""

import json
import time

from typing import Literal, Optional

import httpx

from ..models import TokenUsage
from ..request_logger import get_logger
from .base import ProviderAdapter, ProviderError, RawLLMResult, MultimodalContent, ImageInput, ImageInputType


class DashScopeAdapter(ProviderAdapter):
    """
    Adapter for Alibaba DashScope (阿里百炼) API.
    
    Supports two modes:
    - "dashscope": Official dashscope SDK (default, better stability)
    - "http": Direct HTTP calls (fewer dependencies)
    
    Example:
        # DashScope SDK mode (default)
        adapter = DashScopeAdapter(api_key="sk-xxx")
        
        # HTTP mode
        adapter = DashScopeAdapter(api_key="sk-xxx", mode="http")
    """
    
    name: str = "dashscope"
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
    DEFAULT_INTL_BASE_URL = "https://dashscope-intl.aliyuncs.com/api/v1"
    
    def __init__(
        self, 
        api_key: str, 
        base_url: str | None = None,
        mode: Literal["dashscope", "http"] | None = None,
        use_international: bool | None = None,
        **kwargs
    ):
        """
        Initialize DashScope adapter.
        
        Args:
            api_key: DashScope API key (阿里云百炼API Key)
            base_url: Optional custom base URL
            mode: "dashscope" for official SDK (default), "http" for direct API calls
                  If not provided, reads from config (default: "dashscope")
            use_international: Use international endpoint (dashscope-intl.aliyuncs.com)
                              If not provided, reads from config (default: False)
            **kwargs: Additional configuration
        """
        super().__init__(api_key, **kwargs)
        
        # Read mode from config if not explicitly provided
        if mode is None:
            mode = self.config.get("mode", "dashscope")
        self.mode = mode
        
        # Read use_international from config if not explicitly provided
        if use_international is None:
            use_international = self.config.get("use_international", False)
        self.use_international = use_international
        
        # Set base URL
        if base_url:
            self.base_url = base_url
        elif use_international:
            self.base_url = self.DEFAULT_INTL_BASE_URL
        else:
            self.base_url = self.DEFAULT_BASE_URL
        
        # 初始化日志记录器
        self._logger = get_logger(self.name)
        
        # Initialize based on mode
        self._client: httpx.AsyncClient | None = None
        self._dashscope = None
        
        if mode == "http":
            self._init_http_client()
        elif mode == "dashscope":
            self._init_dashscope_sdk()
    
    def _init_http_client(self):
        """Initialize HTTP client for direct API calls."""
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
    
    def _init_dashscope_sdk(self):
        """Initialize the official DashScope SDK."""
        try:
            import dashscope
            
            # Configure API key
            dashscope.api_key = self.api_key
            
            # Configure base URL if using international endpoint
            if self.use_international:
                dashscope.base_http_api_url = self.DEFAULT_INTL_BASE_URL
            elif self.base_url != self.DEFAULT_BASE_URL:
                dashscope.base_http_api_url = self.base_url
            
            self._dashscope = dashscope
            
        except ImportError:
            raise ProviderError(
                self.name,
                "DashScope SDK mode requires 'dashscope' package. "
                "Install with: pip install dashscope"
            )
    
    async def generate(self, prompt: str, model: str) -> RawLLMResult:
        """
        Generate a response using DashScope API.
        
        Args:
            prompt: The input prompt
            model: Model identifier (e.g., 'qwen-turbo', 'qwen-plus', 'qwen-max')
            
        Returns:
            RawLLMResult with generated text and token counts
        """
        if self.mode == "dashscope":
            return await self._generate_sdk(prompt, model)
        return await self._generate_http(prompt, model)

    async def stream(self, prompt: str, model: str):
        """Stream response text from DashScope."""
        if self.mode == "dashscope":
            async for chunk in self._stream_sdk(prompt, model):
                yield chunk
            return
        async for chunk in self._stream_http(prompt, model):
            yield chunk
    
    async def _generate_http(self, prompt: str, model: str) -> RawLLMResult:
        """Generate using direct HTTP API calls."""
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
    
    async def _generate_sdk(self, prompt: str, model: str) -> RawLLMResult:
        """Generate using official DashScope SDK."""
        import asyncio
        
        start_time = time.time()
        error_message = None
        result = None
        
        try:
            # SDK is sync, run in executor
            def _sync_generate():
                response = self._dashscope.Generation.call(
                    api_key=self.api_key,
                    model=model,
                    prompt=prompt,
                    result_format='message'
                )
                return response
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, _sync_generate)
            
            # Check for errors
            if response.status_code != 200:
                error_message = f"API error: {response.message} (code: {response.code})"
                raise ProviderError(
                    self.name, 
                    error_message,
                    status_code=response.status_code
                )
            
            # Extract response text
            text = response.output.choices[0].message.content
            
            # Extract token usage
            input_tokens = response.usage.input_tokens if hasattr(response.usage, 'input_tokens') else None
            output_tokens = response.usage.output_tokens if hasattr(response.usage, 'output_tokens') else None
            
            result = RawLLMResult(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                raw_response=response
            )
            
            return result
            
        except Exception as e:
            if not error_message:
                error_message = f"SDK error: {str(e)}"
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

    async def _stream_http(self, prompt: str, model: str):
        """Stream using direct HTTP API calls."""
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
    
    async def _stream_sdk(self, prompt: str, model: str):
        """Stream using official DashScope SDK."""
        import asyncio
        
        def _sync_stream():
            responses = self._dashscope.Generation.call(
                api_key=self.api_key,
                model=model,
                prompt=prompt,
                result_format='message',
                stream=True,
                incremental_output=True
            )
            return responses
        
        loop = asyncio.get_event_loop()
        responses = await loop.run_in_executor(None, _sync_stream)
        
        for response in responses:
            if response.status_code == 200:
                text = response.output.choices[0].message.content
                if text:
                    yield text
            else:
                raise ProviderError(
                    self.name,
                    f"Streaming error: {response.message} (code: {response.code})",
                    status_code=response.status_code
                )
    
    async def generate_multimodal(
        self,
        content: MultimodalContent,
        model: str
    ) -> RawLLMResult:
        """Generate multimodal response (text + images)."""
        if self.mode == "dashscope":
            return await self._generate_multimodal_sdk(content, model)
        return await self._generate_multimodal_http(content, model)
    
    async def stream_multimodal(
        self,
        content: MultimodalContent,
        model: str
    ):
        """Stream multimodal response (text + images)."""
        if self.mode == "dashscope":
            async for chunk in self._stream_multimodal_sdk(content, model):
                yield chunk
            return
        async for chunk in self._stream_multimodal_http(content, model):
            yield chunk
    
    def _build_multimodal_messages(self, content: MultimodalContent) -> list:
        """Build messages array for multimodal request."""
        message_content = []
        
        # Add images first
        if content.images:
            for image in content.images:
                if image.type == ImageInputType.URL:
                    message_content.append({"image": image.data})
                elif image.type == ImageInputType.BASE64:
                    # DashScope base64 format
                    data_url = f"data:{image.mime_type};base64,{image.data}"
                    message_content.append({"image": data_url})
        
        # Add text
        if content.text:
            message_content.append({"text": content.text})
        
        return [{"role": "user", "content": message_content}]
    
    async def _generate_multimodal_http(
        self,
        content: MultimodalContent,
        model: str
    ) -> RawLLMResult:
        """Generate multimodal response using HTTP API."""
        start_time = time.time()
        error_message = None
        result = None
        
        try:
            url = f"{self.base_url}/services/aigc/multimodal-generation/generation"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            messages = self._build_multimodal_messages(content)
            
            payload = {
                "model": model,
                "input": {
                    "messages": messages
                },
            }
            
            if not self._client:
                raise ProviderError(self.name, "HTTP client not initialized")
            
            response = await self._client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Check for API-level errors
            if "code" in data and data["code"] != "":
                error_message = f"API error: {data.get('message', 'Unknown error')} (code: {data.get('code')})"
                raise ProviderError(self.name, error_message)
            
            # Extract response text
            output = data["output"]
            message_content = output["choices"][0]["message"]["content"]
            
            # Handle different response formats
            if isinstance(message_content, list):
                # Extract text from content array
                text_parts = [item.get("text", "") for item in message_content if isinstance(item, dict) and "text" in item]
                text = "".join(text_parts)
            else:
                text = message_content
            
            # Extract token usage
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
            # Log request
            duration_ms = (time.time() - start_time) * 1000
            prompt_text = content.text or "[multimodal content]"
            self._logger.log_request(
                model=model,
                prompt=prompt_text,
                response_text=result.text if result else None,
                input_tokens=result.input_tokens if result else None,
                output_tokens=result.output_tokens if result else None,
                duration_ms=duration_ms,
                success=result is not None,
                error_message=error_message,
            )
    
    async def _generate_multimodal_sdk(
        self,
        content: MultimodalContent,
        model: str
    ) -> RawLLMResult:
        """Generate multimodal response using DashScope SDK."""
        import asyncio
        
        start_time = time.time()
        error_message = None
        result = None
        
        try:
            messages = self._build_multimodal_messages(content)
            
            def _sync_generate():
                response = self._dashscope.MultiModalConversation.call(
                    api_key=self.api_key,
                    model=model,
                    messages=messages
                )
                return response
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, _sync_generate)
            
            # Check for errors
            if response.status_code != 200:
                error_message = f"API error: {response.message} (code: {response.code})"
                raise ProviderError(
                    self.name,
                    error_message,
                    status_code=response.status_code
                )
            
            # Extract response text
            message_content = response.output.choices[0].message.content
            if isinstance(message_content, list):
                text_parts = [item.get("text", "") for item in message_content if isinstance(item, dict) and "text" in item]
                text = "".join(text_parts)
            else:
                text = message_content
            
            # Extract token usage
            input_tokens = response.usage.input_tokens if hasattr(response.usage, 'input_tokens') else None
            output_tokens = response.usage.output_tokens if hasattr(response.usage, 'output_tokens') else None
            
            result = RawLLMResult(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                raw_response=response
            )
            
            return result
            
        except Exception as e:
            if not error_message:
                error_message = f"SDK error: {str(e)}"
            raise ProviderError(self.name, error_message)
        
        finally:
            # Log request
            duration_ms = (time.time() - start_time) * 1000
            prompt_text = content.text or "[multimodal content]"
            self._logger.log_request(
                model=model,
                prompt=prompt_text,
                response_text=result.text if result else None,
                input_tokens=result.input_tokens if result else None,
                output_tokens=result.output_tokens if result else None,
                duration_ms=duration_ms,
                success=result is not None,
                error_message=error_message,
            )
    
    async def _stream_multimodal_http(
        self,
        content: MultimodalContent,
        model: str
    ):
        """Stream multimodal response using HTTP API."""
        url = f"{self.base_url}/services/aigc/multimodal-generation/generation"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        
        messages = self._build_multimodal_messages(content)
        
        payload = {
            "model": model,
            "input": {
                "messages": messages
            },
            "parameters": {
                "incremental_output": True,
            },
        }
        
        if not self._client:
            raise ProviderError(self.name, "HTTP client not initialized")
        
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
                    message = choices[0].get("message", {})
                    content_list = message.get("content", [])
                    if content_list and isinstance(content_list, list):
                        for item in content_list:
                            if isinstance(item, dict) and "text" in item:
                                text = item["text"]
                                if text:
                                    yield text
                except (json.JSONDecodeError, AttributeError):
                    continue
    
    async def _stream_multimodal_sdk(
        self,
        content: MultimodalContent,
        model: str
    ):
        """Stream multimodal response using DashScope SDK."""
        import asyncio
        
        messages = self._build_multimodal_messages(content)
        
        def _sync_stream():
            responses = self._dashscope.MultiModalConversation.call(
                api_key=self.api_key,
                model=model,
                messages=messages,
                stream=True,
                incremental_output=True
            )
            return responses
        
        loop = asyncio.get_event_loop()
        responses = await loop.run_in_executor(None, _sync_stream)
        
        for response in responses:
            if response.status_code == 200:
                content_list = response.output.choices[0].message.content
                if content_list and isinstance(content_list, list):
                    for item in content_list:
                        if isinstance(item, dict) and "text" in item:
                            text = item["text"]
                            if text:
                                yield text
            else:
                raise ProviderError(
                    self.name,
                    f"Streaming error: {response.message} (code: {response.code})",
                    status_code=response.status_code
                )

    async def aclose(self) -> None:
        if self._client:
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
