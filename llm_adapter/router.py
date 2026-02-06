"""
Router module for intelligent LLM provider selection.
Implements quality-based routing strategy with automatic fallback.
"""

import os
from dataclasses import dataclass
from typing import Literal

from .config import ConfigManager, ConfigError


@dataclass
class RouteResult:
    """Result of routing decision."""
    provider: str
    model: str
    is_fallback: bool = False


class RouterError(Exception):
    """Exception raised when routing fails."""
    pass


class Router:
    """
    Intelligent router for LLM provider selection.
    
    Routes requests based on quality level:
    - low: cloudflare or huggingface (cost-effective)
    - medium: openai (mini) or gemini (flash) or dashscope (normal)
    - high: openai (premium) or gemini (pro) or dashscope (premium)
    - medium: openai (mini) or gemini (flash)
    - high: openai (premium) or gemini (pro)
    
    Supports automatic fallback when primary provider is unavailable.
    """
    
    # Quality to provider mapping with fallback order
    QUALITY_ROUTES: dict[str, list[tuple[str, str]]] = {
        # (provider, model_tier)
        "low": [
            ("cloudflare", "cheap"),
            ("huggingface", "cheap"),
            ("dashscope", "normal"),  # qwen-* (configured per config.yaml)
        ],
        "medium": [
            ("dashscope", "normal"),  # qwen-* (configured per config.yaml)
            ("openrouter", "normal"),
            ("openai", "cheap"),      # gpt-4o-mini
            ("gemini", "cheap"),      # gemini-1.5-flash
        ],
        "high": [
            ("openai", "premium"),    # gpt-4-turbo
            ("gemini", "premium"),    # gemini-1.5-pro
            ("dashscope", "premium"), # qwen-* (configured per config.yaml)
        ],
    }
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize Router with configuration manager.
        
        Args:
            config_manager: ConfigManager instance for accessing provider configs
        """
        self.config_manager = config_manager
        self._unavailable_providers: set[str] = set()
    
    def route(
        self, 
        quality: Literal["low", "medium", "high"],
        excluded_providers: set[str] | None = None
    ) -> RouteResult:
        """
        Select the best provider and model for the given quality level.
        
        Args:
            quality: Quality level ('low', 'medium', 'high')
            excluded_providers: Set of provider names to exclude from selection
            
        Returns:
            RouteResult with selected provider and model
            
        Raises:
            RouterError: If no suitable provider is available
            
        Environment Variables:
            LLM_DISABLE_QUALITY_ROUTING: If set to 'true', '1', or 'yes', 
                                         always use default_provider from config
        """
        if quality not in self.QUALITY_ROUTES:
            raise RouterError(f"Invalid quality level: {quality}")
        
        excluded = excluded_providers or set()
        excluded = excluded.union(self._unavailable_providers)

        # Check if quality routing is disabled via environment variable
        disable_routing = os.getenv("LLM_DISABLE_QUALITY_ROUTING", "").lower() in ("true", "1", "yes")
        
        # Prefer default_provider when configured and available
        tier_by_quality: dict[str, str] = {
            "low": "cheap",
            "medium": "normal",
            "high": "premium",
        }
        default_provider = self.config_manager.config.llm.default_provider
        
        # If routing is disabled, ONLY use default_provider
        if disable_routing:
            if not default_provider:
                raise RouterError(
                    "LLM_DISABLE_QUALITY_ROUTING is enabled but no default_provider "
                    "is configured in config.yaml"
                )
            if default_provider in excluded:
                raise RouterError(
                    f"LLM_DISABLE_QUALITY_ROUTING is enabled but default_provider "
                    f"'{default_provider}' is excluded"
                )
            
            preferred_tier = tier_by_quality.get(quality)
            if preferred_tier:
                model = self._get_model_for_tier(default_provider, preferred_tier)
                if model:
                    return RouteResult(
                        provider=default_provider,
                        model=model,
                        is_fallback=False,
                    )
            raise RouterError(
                f"Default provider '{default_provider}' does not have a model "
                f"configured for tier '{preferred_tier}'"
            )
        
        # Normal routing: prefer default_provider first, then fallback to QUALITY_ROUTES
        if default_provider and default_provider not in excluded:
            preferred_tier = tier_by_quality.get(quality)
            if preferred_tier:
                model = self._get_model_for_tier(default_provider, preferred_tier)
                if model:
                    return RouteResult(
                        provider=default_provider,
                        model=model,
                        is_fallback=False,
                    )
        
        routes = self.QUALITY_ROUTES[quality]
        is_fallback = False
        
        for idx, (provider, model_tier) in enumerate(routes):
            if provider in excluded:
                continue
            
            try:
                model = self._get_model_for_tier(provider, model_tier)
                if model:
                    return RouteResult(
                        provider=provider,
                        model=model,
                        is_fallback=idx > 0 or is_fallback
                    )
            except ConfigError:
                # Provider not configured, try next
                continue
        
        raise RouterError(
            f"No available provider for quality '{quality}'. "
            f"Excluded providers: {excluded}"
        )
    
    def _get_model_for_tier(self, provider: str, tier: str) -> str | None:
        """
        Get the model name for a provider at a specific tier.
        
        Args:
            provider: Provider name
            tier: Model tier ('cheap', 'normal', 'premium')
            
        Returns:
            Model name or None if not available
        """
        try:
            provider_config = self.config_manager.get_provider_config(provider)
            model = getattr(provider_config.models, tier, None)
            
            # Fallback to default_model if tier not available
            if not model and provider_config.default_model:
                return provider_config.default_model
            
            return model
        except ConfigError:
            return None
    
    def mark_provider_unavailable(self, provider: str) -> None:
        """
        Mark a provider as temporarily unavailable.
        
        Args:
            provider: Provider name to mark as unavailable
        """
        self._unavailable_providers.add(provider)
    
    def mark_provider_available(self, provider: str) -> None:
        """
        Mark a provider as available again.
        
        Args:
            provider: Provider name to mark as available
        """
        self._unavailable_providers.discard(provider)
    
    def reset_availability(self) -> None:
        """Reset all providers to available status."""
        self._unavailable_providers.clear()
    
    def get_fallback(
        self,
        quality: Literal["low", "medium", "high"],
        failed_provider: str
    ) -> RouteResult:
        """
        Get a fallback provider after a failure.
        
        Args:
            quality: Quality level
            failed_provider: Provider that failed
            
        Returns:
            RouteResult with fallback provider and model
            
        Raises:
            RouterError: If no fallback is available
        """
        # Mark the failed provider as unavailable for this request
        excluded = {failed_provider}
        
        try:
            result = self.route(quality, excluded_providers=excluded)
            result.is_fallback = True
            return result
        except RouterError:
            raise RouterError(
                f"No fallback available for quality '{quality}' "
                f"after '{failed_provider}' failed"
            )
    
    def get_available_providers(
        self, 
        quality: Literal["low", "medium", "high"]
    ) -> list[tuple[str, str]]:
        """
        Get list of available providers for a quality level.
        
        Args:
            quality: Quality level
            
        Returns:
            List of (provider, model) tuples that are available
        """
        if quality not in self.QUALITY_ROUTES:
            return []
        
        available = []
        routes = self.QUALITY_ROUTES[quality]
        
        for provider, model_tier in routes:
            if provider in self._unavailable_providers:
                continue
            
            model = self._get_model_for_tier(provider, model_tier)
            if model:
                available.append((provider, model))
        
        return available
    
    def validate_route(
        self,
        provider: str,
        model: str,
        quality: Literal["low", "medium", "high"]
    ) -> bool:
        """
        Validate that a provider/model combination is appropriate for a quality level.
        
        Args:
            provider: Provider name
            model: Model name
            quality: Quality level
            
        Returns:
            True if the combination is valid for the quality level
        """
        if quality not in self.QUALITY_ROUTES:
            return False
        
        routes = self.QUALITY_ROUTES[quality]
        
        for route_provider, model_tier in routes:
            if route_provider != provider:
                continue
            
            expected_model = self._get_model_for_tier(provider, model_tier)
            if expected_model == model:
                return True
        
        return False
