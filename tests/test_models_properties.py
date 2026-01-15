"""
Property-based tests for data models.

Feature: llm-adapter, Property 3: Token统计一致性
Validates: Requirements 3.5
"""

from hypothesis import given, strategies as st, settings

from llm_adapter.models import TokenUsage


class TestTokenUsageProperties:
    """Property tests for TokenUsage data class."""

    @settings(max_examples=100)
    @given(
        input_tokens=st.integers(min_value=0, max_value=10_000_000),
        output_tokens=st.integers(min_value=0, max_value=10_000_000),
    )
    def test_total_tokens_equals_sum_of_input_and_output(
        self, input_tokens: int, output_tokens: int
    ):
        """
        Property 3: Token统计一致性
        
        For any TokenUsage result, total_tokens must equal input_tokens + output_tokens.
        
        Validates: Requirements 3.5
        """
        usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
        
        assert usage.total_tokens == input_tokens + output_tokens
        assert usage.total_tokens == usage.input_tokens + usage.output_tokens
