"""
Billing engine for LLM Adapter system.
Calculates costs based on token usage and pricing rules.
"""

from .config import ConfigManager, ConfigError
from .models import PricingRule, TokenUsage


class BillingError(Exception):
    """Billing related errors"""
    pass


class BillingEngine:
    """
    计费引擎 - 根据Token使用量实时计算成本。
    
    Supports:
    - Loading pricing rules from configuration
    - Calculating costs using the formula:
      cost = (input_tokens / 1_000_000) * input_cost_per_1m + 
             (output_tokens / 1_000_000) * output_cost_per_1m
    - Returns costs in USD
    
    Requirements: 4.1, 4.2, 4.3
    """
    
    def __init__(self, config_manager: ConfigManager | None = None):
        """
        Initialize BillingEngine.
        
        Args:
            config_manager: ConfigManager instance. If None, creates a new one.
        """
        self._config_manager = config_manager or ConfigManager()
    
    @property
    def config_manager(self) -> ConfigManager:
        """Get the configuration manager."""
        return self._config_manager
    
    def get_pricing_rule(self, provider: str, model: str) -> PricingRule:
        """
        Get pricing rule for a specific provider and model.
        
        Args:
            provider: Provider name (e.g., 'openai', 'gemini')
            model: Model name (e.g., 'gpt-4o-mini', 'gemini-1.5-flash')
            
        Returns:
            PricingRule for the specified provider/model
            
        Raises:
            BillingError: If pricing rule is not found
        """
        try:
            return self._config_manager.get_pricing_rule(provider, model)
        except ConfigError as e:
            raise BillingError(f"Failed to get pricing rule: {e}")
    
    def calculate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate cost for a given token usage.
        
        Formula: cost = (input_tokens / 1_000_000) * input_cost_per_1m + 
                        (output_tokens / 1_000_000) * output_cost_per_1m
        
        Args:
            provider: Provider name
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Cost in USD (non-negative float)
            
        Raises:
            BillingError: If pricing rule is not found or tokens are negative
        """
        if input_tokens < 0:
            raise BillingError("input_tokens cannot be negative")
        if output_tokens < 0:
            raise BillingError("output_tokens cannot be negative")
        
        pricing_rule = self.get_pricing_rule(provider, model)
        return pricing_rule.calculate_cost(input_tokens, output_tokens)
    
    def calculate_cost_from_usage(
        self,
        provider: str,
        model: str,
        usage: TokenUsage
    ) -> float:
        """
        Calculate cost from a TokenUsage object.
        
        Args:
            provider: Provider name
            model: Model name
            usage: TokenUsage object containing input and output token counts
            
        Returns:
            Cost in USD (non-negative float)
            
        Raises:
            BillingError: If pricing rule is not found
        """
        return self.calculate_cost(
            provider=provider,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens
        )
    
    def estimate_cost(
        self,
        provider: str,
        model: str,
        estimated_input_tokens: int,
        estimated_output_tokens: int
    ) -> float:
        """
        Estimate cost before making an API call.
        
        Useful for pre-flight cost estimation and budget checks.
        
        Args:
            provider: Provider name
            model: Model name
            estimated_input_tokens: Estimated number of input tokens
            estimated_output_tokens: Estimated number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        return self.calculate_cost(
            provider=provider,
            model=model,
            input_tokens=estimated_input_tokens,
            output_tokens=estimated_output_tokens
        )
    
    def get_all_pricing_rules(self) -> dict[str, dict[str, PricingRule]]:
        """
        Get all pricing rules from configuration.
        
        Returns:
            Dictionary mapping provider -> model -> PricingRule
        """
        return self._config_manager.config.pricing
    
    def list_providers_with_pricing(self) -> list[str]:
        """
        List all providers that have pricing rules configured.
        
        Returns:
            List of provider names
        """
        return list(self._config_manager.config.pricing.keys())
    
    def list_models_for_provider(self, provider: str) -> list[str]:
        """
        List all models with pricing rules for a given provider.
        
        Args:
            provider: Provider name
            
        Returns:
            List of model names
            
        Raises:
            BillingError: If provider has no pricing rules
        """
        pricing = self._config_manager.config.pricing
        if provider not in pricing:
            raise BillingError(f"No pricing rules for provider: {provider}")
        return list(pricing[provider].keys())
