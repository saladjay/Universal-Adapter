"""
Property-based tests for billing engine.

Feature: llm-adapter
Property 4: 计费公式正确性
Validates: Requirements 4.2, 4.3
"""

from hypothesis import given, strategies as st, settings

from llm_adapter.models import PricingRule, TokenUsage


class TestBillingFormulaCorrectness:
    """
    Property 4: 计费公式正确性
    
    For any Token usage (input_tokens, output_tokens) and pricing rules
    (input_cost_per_1m, output_cost_per_1m), the calculated cost must equal:
    cost = (input_tokens / 1_000_000) * input_cost_per_1m + 
           (output_tokens / 1_000_000) * output_cost_per_1m
    
    And the result must be non-negative.
    
    Validates: Requirements 4.2, 4.3
    """

    @settings(max_examples=100)
    @given(
        input_tokens=st.integers(min_value=0, max_value=100_000_000),
        output_tokens=st.integers(min_value=0, max_value=100_000_000),
        input_cost_per_1m=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        output_cost_per_1m=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    def test_billing_formula_correctness(
        self,
        input_tokens: int,
        output_tokens: int,
        input_cost_per_1m: float,
        output_cost_per_1m: float,
    ):
        """
        Property 4: 计费公式正确性
        
        For any valid token counts and pricing rules, the calculated cost
        must match the expected formula and be non-negative.
        
        Validates: Requirements 4.2, 4.3
        """
        # Create a pricing rule with the generated values
        pricing_rule = PricingRule(
            provider="test_provider",
            model="test_model",
            input_cost_per_1m=input_cost_per_1m,
            output_cost_per_1m=output_cost_per_1m,
        )
        
        # Calculate cost using the PricingRule method
        calculated_cost = pricing_rule.calculate_cost(input_tokens, output_tokens)
        
        # Calculate expected cost using the formula from requirements
        expected_cost = (
            (input_tokens / 1_000_000) * input_cost_per_1m +
            (output_tokens / 1_000_000) * output_cost_per_1m
        )
        
        # Verify the formula is correct (with floating point tolerance)
        assert abs(calculated_cost - expected_cost) < 1e-9, (
            f"Cost mismatch: calculated={calculated_cost}, expected={expected_cost}"
        )
        
        # Verify the result is non-negative (Requirement 4.3)
        assert calculated_cost >= 0, f"Cost must be non-negative, got {calculated_cost}"

    @settings(max_examples=100)
    @given(
        input_tokens=st.integers(min_value=0, max_value=100_000_000),
        output_tokens=st.integers(min_value=0, max_value=100_000_000),
        input_cost_per_1m=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        output_cost_per_1m=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    def test_billing_cost_is_usd_non_negative(
        self,
        input_tokens: int,
        output_tokens: int,
        input_cost_per_1m: float,
        output_cost_per_1m: float,
    ):
        """
        Property 4 (continued): Cost must be returned in USD as non-negative value.
        
        Validates: Requirements 4.3
        """
        pricing_rule = PricingRule(
            provider="test_provider",
            model="test_model",
            input_cost_per_1m=input_cost_per_1m,
            output_cost_per_1m=output_cost_per_1m,
        )
        
        cost = pricing_rule.calculate_cost(input_tokens, output_tokens)
        
        # Cost must be a float (USD value)
        assert isinstance(cost, float), f"Cost must be a float, got {type(cost)}"
        
        # Cost must be non-negative
        assert cost >= 0, f"Cost must be non-negative, got {cost}"

