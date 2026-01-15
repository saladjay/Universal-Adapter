"""
Configuration management for LLM Adapter system.
Supports YAML configuration loading with environment variable substitution.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .models import PricingRule


class ConfigError(Exception):
    """Configuration related errors"""
    pass


@dataclass
class ModelConfig:
    """Model configuration for a provider"""
    cheap: str | None = None
    normal: str | None = None
    premium: str | None = None


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider"""
    api_key: str
    models: ModelConfig
    base_url: str | None = None
    account_id: str | None = None
    default_model: str | None = None


@dataclass
class LLMConfig:
    """Top-level LLM configuration"""
    default_provider: str = "openai"


@dataclass
class Config:
    """Complete system configuration"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    pricing: dict[str, dict[str, PricingRule]] = field(default_factory=dict)


class ConfigManager:
    """
    Configuration manager for LLM Adapter system.
    
    Supports:
    - Loading configuration from YAML files
    - Environment variable substitution (${VAR_NAME} syntax)
    - Default provider configuration
    - Multiple models per provider (cheap, normal, premium)
    """
    
    ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')
    
    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize ConfigManager.
        
        Args:
            config_path: Path to YAML configuration file. If None, uses default config.yaml
        """
        self._config: Config | None = None
        self._config_path = Path(config_path) if config_path else Path("config.yaml")
    
    @property
    def config(self) -> Config:
        """Get the loaded configuration, loading it if necessary."""
        if self._config is None:
            self._config = self.load()
        return self._config
    
    def load(self, config_path: str | Path | None = None) -> Config:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Optional path to override the default config path
            
        Returns:
            Loaded Config object
            
        Raises:
            ConfigError: If configuration file is invalid or cannot be loaded
        """
        path = Path(config_path) if config_path else self._config_path
        
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML format: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to read configuration file: {e}")
        
        if raw_config is None:
            raise ConfigError("Configuration file is empty")
        
        if not isinstance(raw_config, dict):
            raise ConfigError("Configuration must be a YAML mapping")
        
        # Substitute environment variables
        raw_config = self._substitute_env_vars(raw_config)
        
        return self._parse_config(raw_config)
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """
        Recursively substitute environment variables in configuration.
        
        Supports ${VAR_NAME} syntax. If environment variable is not set,
        raises ConfigError.
        """
        if isinstance(obj, str):
            def replace_env_var(match: re.Match) -> str:
                var_name = match.group(1)
                value = os.environ.get(var_name)
                if value is None:
                    raise ConfigError(f"Environment variable not set: {var_name}")
                return value
            return self.ENV_VAR_PATTERN.sub(replace_env_var, obj)
        elif isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        return obj
    
    def _parse_config(self, raw: dict) -> Config:
        """Parse raw configuration dictionary into Config object."""
        config = Config()
        
        # Parse LLM config
        if 'llm' in raw:
            llm_raw = raw['llm']
            if not isinstance(llm_raw, dict):
                raise ConfigError("'llm' section must be a mapping")
            config.llm = LLMConfig(
                default_provider=llm_raw.get('default_provider', 'openai')
            )
        
        # Parse providers
        if 'providers' in raw:
            providers_raw = raw['providers']
            if not isinstance(providers_raw, dict):
                raise ConfigError("'providers' section must be a mapping")
            
            for provider_name, provider_data in providers_raw.items():
                if not isinstance(provider_data, dict):
                    raise ConfigError(f"Provider '{provider_name}' configuration must be a mapping")
                
                # Parse models
                models_raw = provider_data.get('models', {})
                if isinstance(models_raw, str):
                    # Single model specified
                    models = ModelConfig(normal=models_raw)
                elif isinstance(models_raw, dict):
                    models = ModelConfig(
                        cheap=models_raw.get('cheap'),
                        normal=models_raw.get('normal'),
                        premium=models_raw.get('premium')
                    )
                else:
                    models = ModelConfig()
                
                config.providers[provider_name] = ProviderConfig(
                    api_key=provider_data.get('api_key', ''),
                    models=models,
                    base_url=provider_data.get('base_url'),
                    account_id=provider_data.get('account_id'),
                    default_model=provider_data.get('default_model')
                )
        
        # Parse pricing rules
        if 'pricing' in raw:
            pricing_raw = raw['pricing']
            if not isinstance(pricing_raw, dict):
                raise ConfigError("'pricing' section must be a mapping")
            
            for provider_name, models_pricing in pricing_raw.items():
                if not isinstance(models_pricing, dict):
                    raise ConfigError(f"Pricing for '{provider_name}' must be a mapping")
                
                config.pricing[provider_name] = {}
                for model_name, pricing_data in models_pricing.items():
                    if not isinstance(pricing_data, dict):
                        raise ConfigError(f"Pricing for '{provider_name}/{model_name}' must be a mapping")
                    
                    config.pricing[provider_name][model_name] = PricingRule(
                        provider=provider_name,
                        model=model_name,
                        input_cost_per_1m=float(pricing_data.get('input_cost_per_1m', 0)),
                        output_cost_per_1m=float(pricing_data.get('output_cost_per_1m', 0))
                    )
        
        return config
    
    def get_provider_config(self, provider: str) -> ProviderConfig:
        """
        Get configuration for a specific provider.
        
        Args:
            provider: Provider name (e.g., 'openai', 'gemini')
            
        Returns:
            ProviderConfig for the specified provider
            
        Raises:
            ConfigError: If provider is not configured
        """
        if provider not in self.config.providers:
            raise ConfigError(f"Provider not configured: {provider}")
        return self.config.providers[provider]
    
    def get_pricing_rule(self, provider: str, model: str) -> PricingRule:
        """
        Get pricing rule for a specific provider and model.
        
        Args:
            provider: Provider name
            model: Model name
            
        Returns:
            PricingRule for the specified provider/model
            
        Raises:
            ConfigError: If pricing rule is not found
        """
        if provider not in self.config.pricing:
            raise ConfigError(f"No pricing rules for provider: {provider}")
        if model not in self.config.pricing[provider]:
            raise ConfigError(f"No pricing rule for model: {provider}/{model}")
        return self.config.pricing[provider][model]
    
    def get_default_provider(self) -> str:
        """Get the default provider name."""
        return self.config.llm.default_provider
    
    def get_model_for_quality(self, provider: str, quality: str) -> str:
        """
        Get the appropriate model for a given quality level.
        
        Args:
            provider: Provider name
            quality: Quality level ('low', 'medium', 'high')
            
        Returns:
            Model name for the specified quality level
            
        Raises:
            ConfigError: If no suitable model is found
        """
        provider_config = self.get_provider_config(provider)
        models = provider_config.models
        
        # Map quality to model tier
        quality_to_tier = {
            'low': ['cheap', 'normal', 'premium'],
            'medium': ['normal', 'cheap', 'premium'],
            'high': ['premium', 'normal', 'cheap']
        }
        
        tiers = quality_to_tier.get(quality, ['normal', 'cheap', 'premium'])
        
        for tier in tiers:
            model = getattr(models, tier, None)
            if model:
                return model
        
        # Fallback to default_model if available
        if provider_config.default_model:
            return provider_config.default_model
        
        raise ConfigError(f"No model available for provider '{provider}' at quality '{quality}'")
