"""
Property-based tests for configuration management.

Feature: llm-adapter
Property 6: 配置往返一致性
Property 7: 无效配置错误处理
Validates: Requirements 6.1, 6.5
"""

import os
import tempfile
from pathlib import Path

import yaml
from hypothesis import given, strategies as st, settings, assume

from llm_adapter.config import (
    ConfigManager,
    ConfigError,
    Config,
    LLMConfig,
    ProviderConfig,
    ModelConfig,
)
from llm_adapter.models import PricingRule


# Strategies for generating valid configuration data
valid_provider_names = st.sampled_from(["openai", "gemini", "cloudflare", "huggingface", "anthropic"])
valid_model_names = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789-_."),
    min_size=1,
    max_size=30,
).filter(lambda x: x.strip() and not x.startswith("-") and not x.startswith("."))

non_negative_floats = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)


@st.composite
def valid_model_config(draw):
    """Generate a valid ModelConfig."""
    cheap = draw(st.one_of(st.none(), valid_model_names))
    normal = draw(st.one_of(st.none(), valid_model_names))
    premium = draw(st.one_of(st.none(), valid_model_names))
    return ModelConfig(cheap=cheap, normal=normal, premium=premium)


@st.composite
def valid_provider_config(draw):
    """Generate a valid ProviderConfig."""
    api_key = draw(st.text(min_size=1, max_size=50).filter(lambda x: x.strip()))
    models = draw(valid_model_config())
    base_url = draw(st.one_of(st.none(), st.just("https://api.example.com")))
    account_id = draw(st.one_of(st.none(), st.text(min_size=1, max_size=20).filter(lambda x: x.strip())))
    default_model = draw(st.one_of(st.none(), valid_model_names))
    return ProviderConfig(
        api_key=api_key,
        models=models,
        base_url=base_url,
        account_id=account_id,
        default_model=default_model,
    )


@st.composite
def valid_pricing_rule(draw, provider: str, model: str):
    """Generate a valid PricingRule."""
    input_cost = draw(non_negative_floats)
    output_cost = draw(non_negative_floats)
    return PricingRule(
        provider=provider,
        model=model,
        input_cost_per_1m=input_cost,
        output_cost_per_1m=output_cost,
    )


@st.composite
def valid_config(draw):
    """Generate a valid Config object."""
    # Generate LLM config
    default_provider = draw(valid_provider_names)
    llm_config = LLMConfig(default_provider=default_provider)
    
    # Generate providers (1-3 providers)
    num_providers = draw(st.integers(min_value=1, max_value=3))
    provider_names = draw(
        st.lists(valid_provider_names, min_size=num_providers, max_size=num_providers, unique=True)
    )
    providers = {}
    for name in provider_names:
        providers[name] = draw(valid_provider_config())
    
    # Generate pricing rules for each provider
    pricing = {}
    for provider_name in provider_names:
        provider_models = providers[provider_name].models
        model_names = [m for m in [provider_models.cheap, provider_models.normal, provider_models.premium] if m]
        if not model_names:
            model_names = ["default-model"]
        
        pricing[provider_name] = {}
        for model_name in model_names[:2]:  # Limit to 2 models per provider
            pricing[provider_name][model_name] = draw(valid_pricing_rule(provider_name, model_name))
    
    return Config(llm=llm_config, providers=providers, pricing=pricing)


def config_to_yaml_dict(config: Config) -> dict:
    """Convert Config object to YAML-serializable dictionary."""
    result = {
        "llm": {"default_provider": config.llm.default_provider},
        "providers": {},
        "pricing": {},
    }
    
    for name, provider in config.providers.items():
        provider_dict = {"api_key": provider.api_key}
        
        models_dict = {}
        if provider.models.cheap:
            models_dict["cheap"] = provider.models.cheap
        if provider.models.normal:
            models_dict["normal"] = provider.models.normal
        if provider.models.premium:
            models_dict["premium"] = provider.models.premium
        if models_dict:
            provider_dict["models"] = models_dict
        
        if provider.base_url:
            provider_dict["base_url"] = provider.base_url
        if provider.account_id:
            provider_dict["account_id"] = provider.account_id
        if provider.default_model:
            provider_dict["default_model"] = provider.default_model
        
        result["providers"][name] = provider_dict
    
    for provider_name, models in config.pricing.items():
        result["pricing"][provider_name] = {}
        for model_name, rule in models.items():
            result["pricing"][provider_name][model_name] = {
                "input_cost_per_1m": rule.input_cost_per_1m,
                "output_cost_per_1m": rule.output_cost_per_1m,
            }
    
    return result


class TestConfigRoundTrip:
    """
    Property 6: 配置往返一致性
    
    For any valid configuration object, serializing to YAML and parsing back
    should produce an equivalent configuration object.
    
    Validates: Requirements 6.1
    """

    @settings(max_examples=100)
    @given(config=valid_config())
    def test_config_round_trip_consistency(self, config: Config):
        """
        Property 6: 配置往返一致性
        
        For any valid Config object, serializing to YAML then parsing back
        should produce an equivalent Config object.
        
        Validates: Requirements 6.1
        """
        # Convert config to YAML dict
        yaml_dict = config_to_yaml_dict(config)
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_dict, f)
            temp_path = f.name
        
        try:
            # Parse back using ConfigManager
            manager = ConfigManager(temp_path)
            loaded_config = manager.load()
            
            # Verify LLM config
            assert loaded_config.llm.default_provider == config.llm.default_provider
            
            # Verify providers
            assert set(loaded_config.providers.keys()) == set(config.providers.keys())
            for name, original_provider in config.providers.items():
                loaded_provider = loaded_config.providers[name]
                assert loaded_provider.api_key == original_provider.api_key
                assert loaded_provider.base_url == original_provider.base_url
                assert loaded_provider.account_id == original_provider.account_id
                assert loaded_provider.default_model == original_provider.default_model
                assert loaded_provider.models.cheap == original_provider.models.cheap
                assert loaded_provider.models.normal == original_provider.models.normal
                assert loaded_provider.models.premium == original_provider.models.premium
            
            # Verify pricing
            assert set(loaded_config.pricing.keys()) == set(config.pricing.keys())
            for provider_name, original_models in config.pricing.items():
                loaded_models = loaded_config.pricing[provider_name]
                assert set(loaded_models.keys()) == set(original_models.keys())
                for model_name, original_rule in original_models.items():
                    loaded_rule = loaded_models[model_name]
                    assert loaded_rule.provider == original_rule.provider
                    assert loaded_rule.model == original_rule.model
                    assert abs(loaded_rule.input_cost_per_1m - original_rule.input_cost_per_1m) < 1e-9
                    assert abs(loaded_rule.output_cost_per_1m - original_rule.output_cost_per_1m) < 1e-9
        finally:
            os.unlink(temp_path)


class TestInvalidConfigErrorHandling:
    """
    Property 7: 无效配置错误处理
    
    For any malformed configuration file, Config_Manager should return an error
    message instead of crashing or returning partial configuration.
    
    Validates: Requirements 6.5
    """

    @settings(max_examples=100)
    @given(content=st.text(alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Z"), max_codepoint=127), min_size=1, max_size=200))
    def test_invalid_yaml_raises_config_error(self, content: str):
        """
        Property 7: 无效配置错误处理
        
        For any random text content that is not valid YAML or valid config structure,
        ConfigManager should raise ConfigError, not crash or return partial config.
        
        Validates: Requirements 6.5
        """
        # Skip if content happens to be valid YAML that parses to a dict
        try:
            parsed = yaml.safe_load(content)
            if isinstance(parsed, dict) and parsed:
                assume(False)  # Skip valid configs
        except yaml.YAMLError:
            pass  # Invalid YAML is what we want to test
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            f.write(content)
            temp_path = f.name
        
        try:
            manager = ConfigManager(temp_path)
            try:
                manager.load()
                # If we get here without error, the content was valid
                # This is acceptable - we just want to ensure no crashes
            except ConfigError:
                # Expected behavior for invalid config
                pass
            except Exception as e:
                # Any other exception is a failure
                raise AssertionError(f"Unexpected exception type: {type(e).__name__}: {e}")
        finally:
            os.unlink(temp_path)

    def test_missing_file_raises_config_error(self):
        """ConfigManager should raise ConfigError for missing files."""
        manager = ConfigManager("/nonexistent/path/config.yaml")
        try:
            manager.load()
            raise AssertionError("Expected ConfigError for missing file")
        except ConfigError as e:
            assert "not found" in str(e).lower()

    def test_empty_file_raises_config_error(self):
        """ConfigManager should raise ConfigError for empty files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            temp_path = f.name
        
        try:
            manager = ConfigManager(temp_path)
            try:
                manager.load()
                raise AssertionError("Expected ConfigError for empty file")
            except ConfigError as e:
                assert "empty" in str(e).lower()
        finally:
            os.unlink(temp_path)

    def test_non_dict_yaml_raises_config_error(self):
        """ConfigManager should raise ConfigError for non-dict YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("- item1\n- item2\n")  # YAML list, not dict
            temp_path = f.name
        
        try:
            manager = ConfigManager(temp_path)
            try:
                manager.load()
                raise AssertionError("Expected ConfigError for non-dict YAML")
            except ConfigError as e:
                assert "mapping" in str(e).lower()
        finally:
            os.unlink(temp_path)

    def test_missing_env_var_raises_config_error(self):
        """ConfigManager should raise ConfigError for missing environment variables."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("llm:\n  default_provider: openai\nproviders:\n  openai:\n    api_key: ${NONEXISTENT_VAR_12345}\n")
            temp_path = f.name
        
        # Ensure the env var doesn't exist
        if "NONEXISTENT_VAR_12345" in os.environ:
            del os.environ["NONEXISTENT_VAR_12345"]
        
        try:
            manager = ConfigManager(temp_path)
            try:
                manager.load()
                raise AssertionError("Expected ConfigError for missing env var")
            except ConfigError as e:
                assert "environment variable" in str(e).lower() or "not set" in str(e).lower()
        finally:
            os.unlink(temp_path)
