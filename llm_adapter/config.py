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
    multimodal: str | None = None


@dataclass
class GenerationParams:
    """Generation parameters for LLM output control"""
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stop: list[str] | None = None
    seed: int | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    def merge(self, other: 'GenerationParams | None') -> 'GenerationParams':
        """Merge with another GenerationParams, other takes precedence"""
        if other is None:
            return self
        
        return GenerationParams(
            temperature=other.temperature if other.temperature is not None else self.temperature,
            top_p=other.top_p if other.top_p is not None else self.top_p,
            top_k=other.top_k if other.top_k is not None else self.top_k,
            max_tokens=other.max_tokens if other.max_tokens is not None else self.max_tokens,
            presence_penalty=other.presence_penalty if other.presence_penalty is not None else self.presence_penalty,
            frequency_penalty=other.frequency_penalty if other.frequency_penalty is not None else self.frequency_penalty,
            stop=other.stop if other.stop is not None else self.stop,
            seed=other.seed if other.seed is not None else self.seed,
        )


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider"""
    api_key: str
    models: ModelConfig
    base_url: str | None = None
    account_id: str | None = None
    default_model: str | None = None
    generation_params: GenerationParams = field(default_factory=GenerationParams)
    model_params: dict[str, GenerationParams] = field(default_factory=dict)
    mode: str | None = None  # For Gemini: "http", "sdk", "vertex"
    project_id: str | None = None  # For Vertex AI
    location: str | None = None  # For Vertex AI


@dataclass
class LLMConfig:
    """Top-level LLM configuration"""
    default_provider: str = "openai"
    default_generation_params: GenerationParams = field(default_factory=GenerationParams)


@dataclass
class ProxyConfig:
    """Proxy configuration for outbound HTTP requests"""
    enable: bool = False
    host: str | None = None
    port: int | None = None


@dataclass
class HttpClientConfig:
    """HTTP client connection pool configuration"""
    max_connections: int = 100
    max_keepalive_connections: int = 20
    timeout: float = 60.0


@dataclass
class Config:
    """Complete system configuration"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    pricing: dict[str, dict[str, PricingRule]] = field(default_factory=dict)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    http_client: HttpClientConfig = field(default_factory=HttpClientConfig)


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

        self._load_env_file_if_present(path)
        
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

    def load_env_file(self, config_path: str | Path | None = None) -> None:
        """Load .env file into environment variables if present."""
        path = Path(config_path) if config_path else self._config_path
        self._load_env_file_if_present(path)

    def _load_env_file_if_present(self, config_path: Path) -> None:
        search_root = config_path if config_path.is_dir() else config_path.parent
        env_path: Path | None = None
        for candidate_dir in [search_root, *search_root.parents]:
            candidate = candidate_dir / ".env"
            if candidate.exists():
                env_path = candidate
                break

        if env_path is None:
            return

        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if not key:
                    continue
                if key not in os.environ or os.environ.get(key, "") == "":
                    os.environ[key] = value
        except Exception:
            return

    def get_env_vars_used(self, config_path: str | Path | None = None) -> set[str]:
        """
        Scan the config file and return all referenced environment variables.

        Args:
            config_path: Optional path to override the default config path

        Returns:
            Set of environment variable names referenced via ${VAR_NAME}
        """
        path = Path(config_path) if config_path else self._config_path
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {path}")
        raw_text = path.read_text(encoding="utf-8")
        return {match.group(1) for match in self.ENV_VAR_PATTERN.finditer(raw_text)}

    def print_env_var_status(self, config_path: str | Path | None = None) -> None:
        """
        Print whether referenced environment variables are set (without values).

        Args:
            config_path: Optional path to override the default config path
        """
        env_vars = sorted(self.get_env_vars_used(config_path))
        if not env_vars:
            print("No environment variables referenced in config.")
            return
        print("Environment variables referenced in config:")
        for var_name in env_vars:
            status = "SET" if os.environ.get(var_name) else "MISSING"
            print(f"- {var_name}: {status}")
    
    def _substitute_env_vars(self, obj: Any, skip_missing: bool = False) -> Any:
        """
        Recursively substitute environment variables in configuration.
        
        Supports ${VAR_NAME} syntax. If environment variable is not set,
        returns None when skip_missing=True, otherwise raises ConfigError.
        
        Args:
            obj: Object to process
            skip_missing: If True, return None for missing env vars instead of raising
        """
        if isinstance(obj, str):
            def replace_env_var(match: re.Match) -> str:
                var_name = match.group(1)
                value = os.environ.get(var_name)
                if value is None:
                    if skip_missing:
                        return ""  # Return empty string as marker
                    raise ConfigError(f"Environment variable not set: {var_name}")
                return value
            
            result = self.ENV_VAR_PATTERN.sub(replace_env_var, obj)
            # If the entire string was an env var that wasn't set, return None
            if skip_missing and result == "" and self.ENV_VAR_PATTERN.search(obj):
                return None
            return result
        elif isinstance(obj, dict):
            return {k: self._substitute_env_vars(v, skip_missing) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item, skip_missing) for item in obj]
        return obj
    
    def _parse_generation_params(self, params_raw: dict) -> GenerationParams:
        """
        Parse generation parameters from raw configuration.
        
        Args:
            params_raw: Raw parameters dictionary
            
        Returns:
            GenerationParams object
        """
        if not isinstance(params_raw, dict):
            return GenerationParams()
        
        return GenerationParams(
            temperature=float(params_raw['temperature']) if 'temperature' in params_raw else None,
            top_p=float(params_raw['top_p']) if 'top_p' in params_raw else None,
            top_k=int(params_raw['top_k']) if 'top_k' in params_raw else None,
            max_tokens=int(params_raw['max_tokens']) if 'max_tokens' in params_raw else None,
            presence_penalty=float(params_raw['presence_penalty']) if 'presence_penalty' in params_raw else None,
            frequency_penalty=float(params_raw['frequency_penalty']) if 'frequency_penalty' in params_raw else None,
            stop=params_raw.get('stop') if 'stop' in params_raw else None,
            seed=int(params_raw['seed']) if 'seed' in params_raw else None,
        )
    
    def _parse_config(self, raw: dict) -> Config:
        """Parse raw configuration dictionary into Config object."""
        config = Config()
        
        # Parse LLM config
        if 'llm' in raw:
            llm_raw = raw['llm']
            if not isinstance(llm_raw, dict):
                raise ConfigError("'llm' section must be a mapping")
            
            # Parse default generation params
            default_gen_params = GenerationParams()
            if 'default_generation_params' in llm_raw:
                default_gen_params = self._parse_generation_params(
                    llm_raw['default_generation_params']
                )
            
            config.llm = LLMConfig(
                default_provider=llm_raw.get('default_provider', 'openai'),
                default_generation_params=default_gen_params
            )

        # Parse proxy config
        if 'proxy' in raw:
            proxy_raw = raw['proxy']
            if not isinstance(proxy_raw, dict):
                raise ConfigError("'proxy' section must be a mapping")
            port = proxy_raw.get('port')
            try:
                port_value = int(port) if port is not None else None
            except (TypeError, ValueError):
                raise ConfigError("'proxy.port' must be an integer")
            config.proxy = ProxyConfig(
                enable=bool(proxy_raw.get('enable', False)),
                host=proxy_raw.get('host'),
                port=port_value,
            )
        
        # Parse HTTP client config
        if 'http_client' in raw:
            http_client_raw = raw['http_client']
            if not isinstance(http_client_raw, dict):
                raise ConfigError("'http_client' section must be a mapping")
            config.http_client = HttpClientConfig(
                max_connections=int(http_client_raw.get('max_connections', 100)),
                max_keepalive_connections=int(http_client_raw.get('max_keepalive_connections', 20)),
                timeout=float(http_client_raw.get('timeout', 60.0)),
            )
        
        # Parse providers (skip those without valid API keys)
        if 'providers' in raw:
            providers_raw = raw['providers']
            if not isinstance(providers_raw, dict):
                raise ConfigError("'providers' section must be a mapping")
            
            for provider_name, provider_data in providers_raw.items():
                if not isinstance(provider_data, dict):
                    raise ConfigError(f"Provider '{provider_name}' configuration must be a mapping")
                
                # Substitute env vars for this provider, skipping missing ones
                try:
                    provider_data = self._substitute_env_vars(provider_data, skip_missing=True)
                except ConfigError:
                    # Skip provider if env var substitution fails
                    continue
                
                # Skip provider if api_key is missing or empty
                api_key = provider_data.get('api_key')
                if not api_key or api_key.strip() == '':
                    continue
                
                # Parse models
                models_raw = provider_data.get('models', {})
                if isinstance(models_raw, str):
                    # Single model specified
                    models = ModelConfig(normal=models_raw)
                elif isinstance(models_raw, dict):
                    models = ModelConfig(
                        cheap=models_raw.get('cheap'),
                        normal=models_raw.get('normal'),
                        premium=models_raw.get('premium'),
                        multimodal=models_raw.get('multimodal'),
                    )
                else:
                    models = ModelConfig()
                
                # Parse provider-level generation params
                provider_gen_params = GenerationParams()
                if 'generation_params' in provider_data:
                    provider_gen_params = self._parse_generation_params(
                        provider_data['generation_params']
                    )
                
                # Parse model-specific generation params
                model_params = {}
                if 'model_params' in provider_data:
                    model_params_raw = provider_data['model_params']
                    if isinstance(model_params_raw, dict):
                        for model_name, params_raw in model_params_raw.items():
                            if isinstance(params_raw, dict):
                                model_params[model_name] = self._parse_generation_params(params_raw)
                
                config.providers[provider_name] = ProviderConfig(
                    api_key=api_key,
                    models=models,
                    base_url=provider_data.get('base_url'),
                    account_id=provider_data.get('account_id'),
                    default_model=provider_data.get('default_model'),
                    generation_params=provider_gen_params,
                    model_params=model_params,
                    mode=provider_data.get('mode'),
                    project_id=provider_data.get('project_id'),
                    location=provider_data.get('location'),
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

    def get_proxy_url(self) -> str | None:
        """Get proxy URL if proxy is enabled, otherwise None."""
        proxy = self.config.proxy
        if not proxy.enable or not proxy.host:
            return None
        host = proxy.host.rstrip("/")
        if proxy.port:
            return f"{host}:{proxy.port}"
        return host
    
    def get_available_providers(self) -> list[str]:
        """
        Get list of all configured and available providers.
        
        Returns:
            List of provider names that have valid API keys configured
        """
        return list(self.config.providers.keys())
    
    def get_provider_models(self, provider: str) -> dict[str, str]:
        """
        Get all available models for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Dictionary mapping tier names to model names
            
        Raises:
            ConfigError: If provider is not configured
        """
        provider_config = self.get_provider_config(provider)
        models = {}
        
        if provider_config.models.cheap:
            models['cheap'] = provider_config.models.cheap
        if provider_config.models.normal:
            models['normal'] = provider_config.models.normal
        if provider_config.models.premium:
            models['premium'] = provider_config.models.premium
        if provider_config.models.multimodal:
            models['multimodal'] = provider_config.models.multimodal
        if provider_config.default_model and not models:
            models['default'] = provider_config.default_model
            
        return models
    
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
    
    def get_generation_params(
        self, 
        provider: str, 
        model: str,
        override_params: GenerationParams | None = None
    ) -> GenerationParams:
        """
        Get merged generation parameters with proper precedence.
        
        Precedence (lowest to highest):
        1. Global default (llm.default_generation_params)
        2. Provider-level (providers.{provider}.generation_params)
        3. Model-specific (providers.{provider}.model_params.{model})
        4. Runtime override (override_params parameter)
        
        Args:
            provider: Provider name
            model: Model name
            override_params: Optional runtime override parameters
            
        Returns:
            Merged GenerationParams with all applicable settings
        """
        # Start with global defaults
        params = self.config.llm.default_generation_params
        
        # Merge provider-level params
        if provider in self.config.providers:
            provider_config = self.config.providers[provider]
            params = params.merge(provider_config.generation_params)
            
            # Merge model-specific params
            if model in provider_config.model_params:
                params = params.merge(provider_config.model_params[model])
        
        # Merge runtime overrides
        if override_params:
            params = params.merge(override_params)
        
        return params
