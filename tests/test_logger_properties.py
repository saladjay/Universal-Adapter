"""
Property-based tests for usage logger.

Feature: llm-adapter
Property 8: 使用日志完整性
Validates: Requirements 7.1
"""

from datetime import datetime, timedelta

from hypothesis import given, strategies as st, settings

from llm_adapter.logger import UsageLogger
from llm_adapter.models import UsageLog


# Custom strategies for generating valid test data
user_id_strategy = st.text(min_size=1, max_size=50).filter(lambda x: x.strip())
provider_strategy = st.sampled_from(["openai", "gemini", "cloudflare", "huggingface"])
model_strategy = st.text(min_size=1, max_size=100).filter(lambda x: x.strip())
non_negative_int_strategy = st.integers(min_value=0, max_value=10_000_000)
non_negative_float_strategy = st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False)
timestamp_strategy = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31)
)


class TestUsageLogIntegrity:
    """
    Property 8: 使用日志完整性
    
    For any LLM call completion, the recorded UsageLog must contain:
    - Valid user_id (non-empty string)
    - Valid provider (non-empty string)
    - Valid model (non-empty string)
    - Non-negative input_tokens
    - Non-negative output_tokens
    - Non-negative cost
    - Valid timestamp
    
    Validates: Requirements 7.1
    """

    @settings(max_examples=100)
    @given(
        user_id=user_id_strategy,
        provider=provider_strategy,
        model=model_strategy,
        input_tokens=non_negative_int_strategy,
        output_tokens=non_negative_int_strategy,
        cost=non_negative_float_strategy,
        timestamp=timestamp_strategy,
    )
    def test_usage_log_contains_all_required_fields(
        self,
        user_id: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        timestamp: datetime,
    ):
        """
        Property 8: 使用日志完整性
        
        For any LLM call, the logged UsageLog must contain valid user_id,
        provider, model, non-negative input_tokens, output_tokens, cost,
        and a valid timestamp.
        
        Validates: Requirements 7.1
        """
        logger = UsageLogger()
        
        # Log the usage
        log_entry = logger.log(
            user_id=user_id,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            timestamp=timestamp,
        )
        
        # Verify the log entry is a UsageLog instance
        assert isinstance(log_entry, UsageLog), "Log entry must be a UsageLog instance"
        
        # Verify user_id is valid (non-empty)
        assert log_entry.user_id == user_id, "user_id must match input"
        assert log_entry.user_id.strip(), "user_id must be non-empty"
        
        # Verify provider is valid (non-empty)
        assert log_entry.provider == provider, "provider must match input"
        assert log_entry.provider.strip(), "provider must be non-empty"
        
        # Verify model is valid (non-empty)
        assert log_entry.model == model, "model must match input"
        assert log_entry.model.strip(), "model must be non-empty"
        
        # Verify input_tokens is non-negative
        assert log_entry.input_tokens == input_tokens, "input_tokens must match input"
        assert log_entry.input_tokens >= 0, "input_tokens must be non-negative"
        
        # Verify output_tokens is non-negative
        assert log_entry.output_tokens == output_tokens, "output_tokens must match input"
        assert log_entry.output_tokens >= 0, "output_tokens must be non-negative"
        
        # Verify cost is non-negative
        assert log_entry.cost == cost, "cost must match input"
        assert log_entry.cost >= 0, "cost must be non-negative"
        
        # Verify timestamp is valid
        assert log_entry.timestamp == timestamp, "timestamp must match input"
        assert isinstance(log_entry.timestamp, datetime), "timestamp must be a datetime"

    @settings(max_examples=100)
    @given(
        user_id=user_id_strategy,
        provider=provider_strategy,
        model=model_strategy,
        input_tokens=non_negative_int_strategy,
        output_tokens=non_negative_int_strategy,
        cost=non_negative_float_strategy,
    )
    def test_usage_log_default_timestamp_is_valid(
        self,
        user_id: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
    ):
        """
        Property 8 (continued): When timestamp is not provided, a valid
        default timestamp should be assigned.
        
        Validates: Requirements 7.1
        """
        logger = UsageLogger()
        before_log = datetime.now()
        
        # Log without explicit timestamp
        log_entry = logger.log(
            user_id=user_id,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )
        
        after_log = datetime.now()
        
        # Verify timestamp is a valid datetime
        assert isinstance(log_entry.timestamp, datetime), "timestamp must be a datetime"
        
        # Verify timestamp is within reasonable bounds (between before and after log call)
        assert before_log <= log_entry.timestamp <= after_log, (
            f"timestamp {log_entry.timestamp} should be between {before_log} and {after_log}"
        )

    @settings(max_examples=100)
    @given(
        user_id=user_id_strategy,
        provider=provider_strategy,
        model=model_strategy,
        input_tokens=non_negative_int_strategy,
        output_tokens=non_negative_int_strategy,
        cost=non_negative_float_strategy,
        timestamp=timestamp_strategy,
    )
    def test_logged_entry_is_retrievable(
        self,
        user_id: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        timestamp: datetime,
    ):
        """
        Property 8 (continued): Any logged UsageLog must be retrievable
        from the logger.
        
        Validates: Requirements 7.1
        """
        logger = UsageLogger()
        
        # Log the usage
        log_entry = logger.log(
            user_id=user_id,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            timestamp=timestamp,
        )
        
        # Verify the log entry can be retrieved
        all_logs = logger.get_all_logs()
        assert log_entry in all_logs, "Logged entry must be retrievable from all logs"
        
        # Verify the log entry can be retrieved by user
        user_logs = logger.get_logs_by_user(user_id)
        assert log_entry in user_logs, "Logged entry must be retrievable by user_id"
