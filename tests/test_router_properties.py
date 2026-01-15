"""
Property-based tests for router strategy.

Feature: llm-adapter
Property 5: 路由策略正确性
Validates: Requirements 5.1, 5.2, 5.3
"""

import os
import tempfile
from typing import Literal

import yaml
from hypothesis import given, strategies as st, settings, assume

from llm_adapter.config import ConfigManager
from llm_adapter.router import Router, RouterError


def create_test_config_file() -> str:
    """Create a temporary config file with all providers configured."""
    config = {
        "llm": {"default_provider": "openai"},
        "providers": {
            "openai": {
                "api_key": "test-key",
                "models": {
                    "cheap": "gpt-4o-mini",
                    "normal": "gpt-4o",
                    "premium": "gpt-4-turbo",
                },
            },
            "gemini": {
                "api_key": "test-key",
                "models": {
                    "cheap": "gemini-1.5-flash",
                    "premium": "gemini-1.5-pro",
                },
            },
            "cloudflare": {
                "api_key": "test-key",
                "account_id": "test-account",
                "models": {
                    "cheap": "@cf/meta/llama-3-8b-instruct",
                },
            },
            "huggingface": {
                "api_key": "test-key",
                "default_model": "meta-llama/Llama-3.1-8B-Instruct",
                "models": {
                    "cheap": "meta-llama/Llama-3.1-8B-Instruct",
                },
            },
        },
        "pricing": {
            "openai": {
                "gpt-4o-mini": {"input_cost_per_1m": 0.15, "output_cost_per_1m": 0.60},
                "gpt-4o": {"input_cost_per_1m": 2.50, "output_cost_per_1m": 10.00},
                "gpt-4-turbo": {"input_cost_per_1m": 10.00, "output_cost_per_1m": 30.00},
            },
            "gemini": {
                "gemini-1.5-flash": {"input_cost_per_1m": 0.075, "output_cost_per_1m": 0.30},
                "gemini-1.5-pro": {"input_cost_per_1m": 1.25, "output_cost_per_1m": 5.00},
            },
        },
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        return f.name


class TestRouterStrategyCorrectness:
    """
    Property 5: 路由策略正确性
    
    For any LLMRequest:
    - When quality="low", the selected provider must be "cloudflare" or "huggingface"
    - When quality="medium", the selected model must be cheap or normal level
    - When quality="high", the selected model must be premium level
    
    Validates: Requirements 5.1, 5.2, 5.3
    """

    @settings(max_examples=100)
    @given(
        quality=st.sampled_from(["low", "medium", "high"]),
        excluded_providers=st.lists(
            st.sampled_from(["openai", "gemini", "cloudflare", "huggingface"]),
            max_size=2,
            unique=True,
        ),
    )
    def test_routing_strategy_correctness(
        self,
        quality: Literal["low", "medium", "high"],
        excluded_providers: list[str],
    ):
        """
        Property 5: 路由策略正确性
        
        For any quality level and set of excluded providers, the router
        must select a provider/model combination that matches the quality
        requirements.
        
        Validates: Requirements 5.1, 5.2, 5.3
        """
        config_path = create_test_config_file()
        
        try:
            config_manager = ConfigManager(config_path)
            config_manager.load()
            router = Router(config_manager)
            
            # Determine valid providers for this quality level
            valid_providers_for_quality = {
                "low": {"cloudflare", "huggingface"},
                "medium": {"openai", "gemini"},
                "high": {"openai", "gemini"},
            }
            
            valid_providers = valid_providers_for_quality[quality]
            available_providers = valid_providers - set(excluded_providers)
            
            # Skip if no providers are available for this quality
            if not available_providers:
                assume(False)
            
            try:
                result = router.route(quality, excluded_providers=set(excluded_providers))
                
                # Verify the selected provider is valid for the quality level
                assert result.provider in valid_providers, (
                    f"For quality '{quality}', provider must be one of {valid_providers}, "
                    f"but got '{result.provider}'"
                )
                
                # Verify the provider is not in the excluded list
                assert result.provider not in excluded_providers, (
                    f"Provider '{result.provider}' should have been excluded"
                )
                
                # Verify model tier matches quality level
                if quality == "low":
                    # For low quality, provider must be cloudflare or huggingface
                    assert result.provider in {"cloudflare", "huggingface"}, (
                        f"For quality 'low', provider must be 'cloudflare' or 'huggingface', "
                        f"but got '{result.provider}'"
                    )
                
                elif quality == "medium":
                    # For medium quality, model should be cheap tier (mini/flash)
                    provider_config = config_manager.get_provider_config(result.provider)
                    expected_model = provider_config.models.cheap
                    assert result.model == expected_model, (
                        f"For quality 'medium', model should be cheap tier '{expected_model}', "
                        f"but got '{result.model}'"
                    )
                
                elif quality == "high":
                    # For high quality, model should be premium tier
                    provider_config = config_manager.get_provider_config(result.provider)
                    expected_model = provider_config.models.premium
                    assert result.model == expected_model, (
                        f"For quality 'high', model should be premium tier '{expected_model}', "
                        f"but got '{result.model}'"
                    )
                    
            except RouterError:
                # RouterError is acceptable if no providers are available
                # This can happen if all valid providers are excluded
                pass
                
        finally:
            os.unlink(config_path)

    @settings(max_examples=100)
    @given(quality=st.sampled_from(["low", "medium", "high"]))
    def test_low_quality_selects_cost_effective_providers(
        self,
        quality: Literal["low", "medium", "high"],
    ):
        """
        Property 5.1: When quality="low", router selects cloudflare or huggingface.
        
        Validates: Requirements 5.1
        """
        if quality != "low":
            assume(False)
        
        config_path = create_test_config_file()
        
        try:
            config_manager = ConfigManager(config_path)
            config_manager.load()
            router = Router(config_manager)
            
            result = router.route(quality)
            
            # For low quality, must select cloudflare or huggingface
            assert result.provider in {"cloudflare", "huggingface"}, (
                f"For quality 'low', provider must be 'cloudflare' or 'huggingface', "
                f"but got '{result.provider}'"
            )
        finally:
            os.unlink(config_path)

    @settings(max_examples=100)
    @given(quality=st.sampled_from(["low", "medium", "high"]))
    def test_medium_quality_selects_mini_or_flash_models(
        self,
        quality: Literal["low", "medium", "high"],
    ):
        """
        Property 5.2: When quality="medium", router selects OpenAI-mini or Gemini-flash.
        
        Validates: Requirements 5.2
        """
        if quality != "medium":
            assume(False)
        
        config_path = create_test_config_file()
        
        try:
            config_manager = ConfigManager(config_path)
            config_manager.load()
            router = Router(config_manager)
            
            result = router.route(quality)
            
            # For medium quality, must select cheap tier model
            provider_config = config_manager.get_provider_config(result.provider)
            expected_model = provider_config.models.cheap
            
            assert result.model == expected_model, (
                f"For quality 'medium', model should be cheap tier '{expected_model}', "
                f"but got '{result.model}'"
            )
            
            # Provider should be openai or gemini
            assert result.provider in {"openai", "gemini"}, (
                f"For quality 'medium', provider must be 'openai' or 'gemini', "
                f"but got '{result.provider}'"
            )
        finally:
            os.unlink(config_path)

    @settings(max_examples=100)
    @given(quality=st.sampled_from(["low", "medium", "high"]))
    def test_high_quality_selects_premium_models(
        self,
        quality: Literal["low", "medium", "high"],
    ):
        """
        Property 5.3: When quality="high", router selects OpenAI or Gemini-pro.
        
        Validates: Requirements 5.3
        """
        if quality != "high":
            assume(False)
        
        config_path = create_test_config_file()
        
        try:
            config_manager = ConfigManager(config_path)
            config_manager.load()
            router = Router(config_manager)
            
            result = router.route(quality)
            
            # For high quality, must select premium tier model
            provider_config = config_manager.get_provider_config(result.provider)
            expected_model = provider_config.models.premium
            
            assert result.model == expected_model, (
                f"For quality 'high', model should be premium tier '{expected_model}', "
                f"but got '{result.model}'"
            )
            
            # Provider should be openai or gemini
            assert result.provider in {"openai", "gemini"}, (
                f"For quality 'high', provider must be 'openai' or 'gemini', "
                f"but got '{result.provider}'"
            )
        finally:
            os.unlink(config_path)
