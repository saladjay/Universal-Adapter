"""
LLMAdapter - Unified entry point for multi-provider LLM access.

Integrates Router, Provider adapters, Billing, and Logger to provide
a single API for LLM calls with automatic routing, cost calculation,
and usage logging.
"""

from typing import Literal

from .adapters import (
    ProviderAdapter,
    ProviderError,
    OpenAIAdapter,
    GeminiAdapter,
    CloudflareAdapter,
    HuggingFaceAdapter,
    DashScopeAdapter,
    OpenRouterAdapter,
)
from .billing import BillingEngine, BillingError
from .config import ConfigManager, ConfigError
from .logger import UsageLogger
from .models import LLMRequest, LLMResponse, TokenUsage
from .router import Router, RouterError


class LLMAdapterError(Exception):
    """Base exception for LLMAdapter errors."""
    pass


class ValidationError(LLMAdapterError):
    """Exception raised when request validation fails."""
    
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"Validation failed: {'; '.join(errors)}")


class LLMAdapter:
    """
    统一LLM接入层 - 对外提供单一API，对内调度多个海外大模型平台。
    
    Features:
    - Unified API for multiple LLM providers (OpenAI, Gemini, Cloudflare, HuggingFace)
    - Intelligent routing based on quality level
    - Automatic fallback when primary provider fails
    - Real-time cost calculation
    - Usage logging for analytics
    
    Requirements: 1.1, 1.2, 1.3
    """
    
    # Maximum retry attempts for fallback
    MAX_RETRIES = 3
    
    # Provider adapter class mapping
    PROVIDER_ADAPTERS: dict[str, type[ProviderAdapter]] = {
        "openai": OpenAIAdapter,
        "gemini": GeminiAdapter,
        "cloudflare": CloudflareAdapter,
        "huggingface": HuggingFaceAdapter,
        "dashscope": DashScopeAdapter,
        "openrouter": OpenRouterAdapter,
    }
    
    def __init__(
        self,
        config_manager: ConfigManager | None = None,
        config_path: str | None = None,
    ):
        """
        Initialize LLMAdapter.
        
        Args:
            config_manager: Optional ConfigManager instance. If None, creates one.
            config_path: Optional path to config file (used if config_manager is None)
        """
        self._config_manager = config_manager or ConfigManager(config_path)
        self._router = Router(self._config_manager)
        self._billing = BillingEngine(self._config_manager)
        self._logger = UsageLogger()
        self._adapters: dict[str, ProviderAdapter] = {}

    @property
    def config_manager(self) -> ConfigManager:
        """Get the configuration manager."""
        return self._config_manager
    
    @property
    def router(self) -> Router:
        """Get the router instance."""
        return self._router
    
    @property
    def billing(self) -> BillingEngine:
        """Get the billing engine."""
        return self._billing
    
    @property
    def logger(self) -> UsageLogger:
        """Get the usage logger."""
        return self._logger
    
    def _get_adapter(self, provider: str) -> ProviderAdapter:
        """
        Get or create an adapter for the specified provider.
        
        Args:
            provider: Provider name
            
        Returns:
            ProviderAdapter instance
            
        Raises:
            LLMAdapterError: If provider is not supported or not configured
        """
        if provider in self._adapters:
            return self._adapters[provider]
        
        if provider not in self.PROVIDER_ADAPTERS:
            raise LLMAdapterError(f"Unsupported provider: {provider}")
        
        try:
            provider_config = self._config_manager.get_provider_config(provider)
        except ConfigError as e:
            raise LLMAdapterError(f"Provider not configured: {provider}. {e}")
        
        adapter_class = self.PROVIDER_ADAPTERS[provider]
        
        # Build adapter kwargs from provider config
        kwargs = {}
        if provider_config.base_url:
            kwargs["base_url"] = provider_config.base_url
        if provider_config.account_id:
            kwargs["account_id"] = provider_config.account_id
        
        adapter = adapter_class(api_key=provider_config.api_key, **kwargs)
        self._adapters[provider] = adapter
        return adapter
    
    def validate_request(self, request: LLMRequest) -> list[str]:
        """
        Validate an LLM request.
        
        Args:
            request: LLMRequest to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        return request.validate()
    
    async def generate(
        self,
        user_id: str,
        prompt: str,
        scene: Literal["chat", "coach", "persona", "system"],
        quality: Literal["low", "medium", "high"],
    ) -> LLMResponse:
        """
        Generate a response from an LLM.
        
        This is the main entry point for LLM calls. It:
        1. Validates the request parameters
        2. Routes to the appropriate provider based on quality
        3. Calls the LLM API
        4. Calculates the cost
        5. Logs the usage
        6. Returns a unified response
        
        Args:
            user_id: User identifier for logging and billing
            prompt: The input prompt to send to the LLM
            scene: Usage scene (chat, coach, persona, system)
            quality: Quality level (low, medium, high)
            
        Returns:
            LLMResponse with generated text, token counts, and cost
            
        Raises:
            ValidationError: If request parameters are invalid
            LLMAdapterError: If all providers fail
        """
        # Create and validate request
        request = LLMRequest(
            user_id=user_id,
            prompt=prompt,
            scene=scene,
            quality=quality,
        )
        
        errors = self.validate_request(request)
        if errors:
            raise ValidationError(errors)
        
        return await self._generate_with_fallback(request)

    async def generate_from_request(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response from an LLMRequest object.
        
        Args:
            request: LLMRequest object with all parameters
            
        Returns:
            LLMResponse with generated text, token counts, and cost
            
        Raises:
            ValidationError: If request parameters are invalid
            LLMAdapterError: If all providers fail
        """
        errors = self.validate_request(request)
        if errors:
            raise ValidationError(errors)
        
        return await self._generate_with_fallback(request)
    
    async def _generate_with_fallback(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response with automatic fallback on failure.
        
        Args:
            request: Validated LLMRequest
            
        Returns:
            LLMResponse from successful provider
            
        Raises:
            LLMAdapterError: If all providers fail
        """
        failed_providers: set[str] = set()
        last_error: Exception | None = None
        
        for attempt in range(self.MAX_RETRIES):
            try:
                # Get route (excluding failed providers)
                if attempt == 0:
                    route = self._router.route(request.quality)
                else:
                    route = self._router.get_fallback(
                        request.quality, 
                        failed_provider=list(failed_providers)[-1]
                    )
                
                provider = route.provider
                model = route.model
                
                # Skip if already failed
                if provider in failed_providers:
                    continue
                
                # Get adapter and generate
                adapter = self._get_adapter(provider)
                result = await adapter.generate(request.prompt, model)
                
                # Get token usage
                if result.input_tokens is not None and result.output_tokens is not None:
                    input_tokens = result.input_tokens
                    output_tokens = result.output_tokens
                else:
                    # Fallback to estimation
                    token_usage = adapter.estimate_tokens(request.prompt, result.text)
                    input_tokens = token_usage.input_tokens
                    output_tokens = token_usage.output_tokens
                
                # Calculate cost
                try:
                    cost_usd = self._billing.calculate_cost(
                        provider=provider,
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                except BillingError:
                    # If no pricing rule, set cost to 0
                    cost_usd = 0.0
                
                # Log usage
                self._logger.log(
                    user_id=request.user_id,
                    provider=provider,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost_usd,
                )
                
                # Return response
                return LLMResponse(
                    text=result.text,
                    model=model,
                    provider=provider,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                )
                
            except ProviderError as e:
                failed_providers.add(e.provider)
                last_error = e
                continue
            except RouterError as e:
                last_error = e
                break
            except Exception as e:
                last_error = e
                break
        
        # All attempts failed
        error_msg = f"All providers failed for quality '{request.quality}'"
        if last_error:
            error_msg += f". Last error: {last_error}"
        raise LLMAdapterError(error_msg)

    def get_user_usage(self, user_id: str) -> dict:
        """
        Get usage statistics for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with usage statistics
        """
        logs = self._logger.get_logs_by_user(user_id)
        total_input, total_output = self._logger.get_user_total_tokens(user_id)
        total_cost = self._logger.get_user_total_cost(user_id)
        
        return {
            "user_id": user_id,
            "total_calls": len(logs),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost_usd": total_cost,
        }
    
    def reset_router_availability(self) -> None:
        """Reset all providers to available status in the router."""
        self._router.reset_availability()
    
    def mark_provider_unavailable(self, provider: str) -> None:
        """
        Mark a provider as temporarily unavailable.
        
        Args:
            provider: Provider name to mark as unavailable
        """
        self._router.mark_provider_unavailable(provider)
    
    def mark_provider_available(self, provider: str) -> None:
        """
        Mark a provider as available again.
        
        Args:
            provider: Provider name to mark as available
        """
        self._router.mark_provider_available(provider)
    
    def get_available_providers(
        self, 
        quality: Literal["low", "medium", "high"] | None = None
    ) -> list[tuple[str, str]] | list[str]:
        """
        Get list of available providers.
        
        Args:
            quality: Optional quality level. If provided, returns (provider, model) tuples
                    for that quality. If None, returns all configured provider names.
            
        Returns:
            If quality is provided: List of (provider, model) tuples
            If quality is None: List of provider names
        """
        if quality:
            return self._router.get_available_providers(quality)
        return self._config_manager.get_available_providers()
    
    def get_provider_models(self, provider: str) -> dict[str, str]:
        """
        Get all available models for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Dictionary mapping tier names (cheap/normal/premium) to model names
        """
        return self._config_manager.get_provider_models(provider)
    
    async def generate_with_provider(
        self,
        user_id: str,
        prompt: str,
        provider: str,
        model: str,
        scene: Literal["chat", "coach", "persona", "system"] = "chat",
    ) -> LLMResponse:
        """
        Generate a response using a specific provider and model.
        
        This bypasses the router and directly calls the specified provider/model.
        
        Args:
            user_id: User identifier for logging and billing
            prompt: The input prompt to send to the LLM
            provider: Provider name (e.g., 'openai', 'gemini', 'dashscope')
            model: Model name (e.g., 'gpt-4o', 'gemini-1.5-flash', 'qwen-plus')
            scene: Usage scene (default: 'chat')
            
        Returns:
            LLMResponse with generated text, token counts, and cost
            
        Raises:
            ValidationError: If request parameters are invalid
            LLMAdapterError: If the provider call fails
        """
        # Validate basic parameters
        if not user_id or not user_id.strip():
            raise ValidationError(["user_id is required and cannot be empty"])
        if not prompt or not prompt.strip():
            raise ValidationError(["prompt is required and cannot be empty"])
        
        # Get adapter and generate
        try:
            adapter = self._get_adapter(provider)
            result = await adapter.generate(prompt, model)
        except ProviderError as e:
            raise LLMAdapterError(f"Provider error: {e}")
        
        # Get token usage
        if result.input_tokens is not None and result.output_tokens is not None:
            input_tokens = result.input_tokens
            output_tokens = result.output_tokens
        else:
            token_usage = adapter.estimate_tokens(prompt, result.text)
            input_tokens = token_usage.input_tokens
            output_tokens = token_usage.output_tokens
        
        # Calculate cost
        try:
            cost_usd = self._billing.calculate_cost(
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except BillingError:
            cost_usd = 0.0
        
        # Log usage
        self._logger.log(
            user_id=user_id,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost_usd,
        )
        
        return LLMResponse(
            text=result.text,
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
