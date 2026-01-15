"""
Property-based tests for LLMAdapter request validation and response structure.

Feature: llm-adapter
Property 1: 请求参数验证
Property 2: 响应结构完整性
Validates: Requirements 1.1, 1.2, 1.3
"""

from hypothesis import given, strategies as st, settings, assume

from llm_adapter.models import LLMRequest, LLMResponse


# Valid values for scene and quality fields
VALID_SCENES = ["chat", "coach", "persona", "system"]
VALID_QUALITIES = ["low", "medium", "high"]


class TestRequestValidationProperty:
    """
    Property 1: 请求参数验证
    
    For any LLMRequest, if it contains valid user_id, prompt, scene, and quality fields,
    THE LLM_Adapter SHALL successfully receive and process the request;
    if any required field is missing or invalid, SHALL return a clear error message
    instead of crashing.
    
    Validates: Requirements 1.1, 1.3
    """

    @settings(max_examples=100)
    @given(
        user_id=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        prompt=st.text(min_size=1, max_size=1000).filter(lambda x: x.strip()),
        scene=st.sampled_from(VALID_SCENES),
        quality=st.sampled_from(VALID_QUALITIES),
    )
    def test_valid_request_passes_validation(
        self, user_id: str, prompt: str, scene: str, quality: str
    ):
        """
        Property 1: Valid requests should pass validation.
        
        For any LLMRequest with valid user_id, prompt, scene, and quality,
        validation should return an empty error list.
        
        Validates: Requirements 1.1
        """
        request = LLMRequest(
            user_id=user_id,
            prompt=prompt,
            scene=scene,
            quality=quality,
        )
        
        errors = request.validate()
        
        assert errors == [], f"Valid request should have no errors, got: {errors}"

    @settings(max_examples=100)
    @given(
        user_id=st.one_of(
            st.just(""),
            st.just("   "),
            st.text(max_size=10).filter(lambda x: not x.strip()),
        ),
        prompt=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        scene=st.sampled_from(VALID_SCENES),
        quality=st.sampled_from(VALID_QUALITIES),
    )
    def test_empty_user_id_fails_validation(
        self, user_id: str, prompt: str, scene: str, quality: str
    ):
        """
        Property 1: Empty or whitespace-only user_id should fail validation.
        
        Validates: Requirements 1.3
        """
        request = LLMRequest(
            user_id=user_id,
            prompt=prompt,
            scene=scene,
            quality=quality,
        )
        
        errors = request.validate()
        
        assert len(errors) > 0, "Empty user_id should produce validation errors"
        assert any("user_id" in e for e in errors), "Error should mention user_id"

    @settings(max_examples=100)
    @given(
        user_id=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        prompt=st.one_of(
            st.just(""),
            st.just("   "),
            st.text(max_size=10).filter(lambda x: not x.strip()),
        ),
        scene=st.sampled_from(VALID_SCENES),
        quality=st.sampled_from(VALID_QUALITIES),
    )
    def test_empty_prompt_fails_validation(
        self, user_id: str, prompt: str, scene: str, quality: str
    ):
        """
        Property 1: Empty or whitespace-only prompt should fail validation.
        
        Validates: Requirements 1.3
        """
        request = LLMRequest(
            user_id=user_id,
            prompt=prompt,
            scene=scene,
            quality=quality,
        )
        
        errors = request.validate()
        
        assert len(errors) > 0, "Empty prompt should produce validation errors"
        assert any("prompt" in e for e in errors), "Error should mention prompt"

    @settings(max_examples=100)
    @given(
        user_id=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        prompt=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        scene=st.text(min_size=1, max_size=20).filter(lambda x: x not in VALID_SCENES),
        quality=st.sampled_from(VALID_QUALITIES),
    )
    def test_invalid_scene_fails_validation(
        self, user_id: str, prompt: str, scene: str, quality: str
    ):
        """
        Property 1: Invalid scene value should fail validation.
        
        Validates: Requirements 1.3
        """
        request = LLMRequest(
            user_id=user_id,
            prompt=prompt,
            scene=scene,
            quality=quality,
        )
        
        errors = request.validate()
        
        assert len(errors) > 0, "Invalid scene should produce validation errors"
        assert any("scene" in e for e in errors), "Error should mention scene"

    @settings(max_examples=100)
    @given(
        user_id=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        prompt=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        scene=st.sampled_from(VALID_SCENES),
        quality=st.text(min_size=1, max_size=20).filter(lambda x: x not in VALID_QUALITIES),
    )
    def test_invalid_quality_fails_validation(
        self, user_id: str, prompt: str, scene: str, quality: str
    ):
        """
        Property 1: Invalid quality value should fail validation.
        
        Validates: Requirements 1.3
        """
        request = LLMRequest(
            user_id=user_id,
            prompt=prompt,
            scene=scene,
            quality=quality,
        )
        
        errors = request.validate()
        
        assert len(errors) > 0, "Invalid quality should produce validation errors"
        assert any("quality" in e for e in errors), "Error should mention quality"


class TestResponseStructureProperty:
    """
    Property 2: 响应结构完整性
    
    For any successful LLM call, the returned LLMResponse must contain:
    - Non-empty text, model, provider fields
    - Non-negative input_tokens, output_tokens, and cost_usd values
    
    Validates: Requirements 1.2
    """

    @settings(max_examples=100)
    @given(
        text=st.text(min_size=1, max_size=1000).filter(lambda x: x.strip()),
        model=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        provider=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        input_tokens=st.integers(min_value=0, max_value=10_000_000),
        output_tokens=st.integers(min_value=0, max_value=10_000_000),
        cost_usd=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    )
    def test_valid_response_structure(
        self,
        text: str,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ):
        """
        Property 2: Valid LLMResponse should have complete structure.
        
        For any successful LLM call, the response must contain non-empty text,
        model, provider fields, and non-negative token counts and cost.
        
        Validates: Requirements 1.2
        """
        response = LLMResponse(
            text=text,
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
        
        # Verify non-empty string fields
        assert response.text and response.text.strip(), "text must be non-empty"
        assert response.model and response.model.strip(), "model must be non-empty"
        assert response.provider and response.provider.strip(), "provider must be non-empty"
        
        # Verify non-negative numeric fields
        assert response.input_tokens >= 0, f"input_tokens must be non-negative, got {response.input_tokens}"
        assert response.output_tokens >= 0, f"output_tokens must be non-negative, got {response.output_tokens}"
        assert response.cost_usd >= 0, f"cost_usd must be non-negative, got {response.cost_usd}"

    @settings(max_examples=100)
    @given(
        text=st.text(min_size=1, max_size=1000).filter(lambda x: x.strip()),
        model=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        provider=st.sampled_from(["openai", "gemini", "cloudflare", "huggingface"]),
        input_tokens=st.integers(min_value=0, max_value=10_000_000),
        output_tokens=st.integers(min_value=0, max_value=10_000_000),
        cost_usd=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    )
    def test_response_fields_are_correct_types(
        self,
        text: str,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ):
        """
        Property 2: LLMResponse fields should have correct types.
        
        Validates: Requirements 1.2
        """
        response = LLMResponse(
            text=text,
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
        
        # Verify field types
        assert isinstance(response.text, str), f"text must be str, got {type(response.text)}"
        assert isinstance(response.model, str), f"model must be str, got {type(response.model)}"
        assert isinstance(response.provider, str), f"provider must be str, got {type(response.provider)}"
        assert isinstance(response.input_tokens, int), f"input_tokens must be int, got {type(response.input_tokens)}"
        assert isinstance(response.output_tokens, int), f"output_tokens must be int, got {type(response.output_tokens)}"
        assert isinstance(response.cost_usd, float), f"cost_usd must be float, got {type(response.cost_usd)}"
