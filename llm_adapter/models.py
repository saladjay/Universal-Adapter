"""
Core data models for LLM Adapter system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class LLMRequest:
    """统一LLM请求参数"""
    user_id: str
    prompt: str
    scene: Literal["chat", "coach", "persona", "system"]
    quality: Literal["low", "medium", "high"]

    def validate(self) -> list[str]:
        """验证请求参数，返回错误列表"""
        errors = []
        if not self.user_id or not self.user_id.strip():
            errors.append("user_id is required and cannot be empty")
        if not self.prompt or not self.prompt.strip():
            errors.append("prompt is required and cannot be empty")
        if self.scene not in ("chat", "coach", "persona", "system"):
            errors.append(f"scene must be one of: chat, coach, persona, system")
        if self.quality not in ("low", "medium", "high"):
            errors.append(f"quality must be one of: low, medium, high")
        return errors


@dataclass
class TokenUsage:
    """Token使用统计"""
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        """总Token数 = 输入Token + 输出Token"""
        return self.input_tokens + self.output_tokens


@dataclass
class LLMResponse:
    """统一LLM响应结构"""
    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class PricingRule:
    """定价规则"""
    provider: str
    model: str
    input_cost_per_1m: float
    output_cost_per_1m: float

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """根据Token使用量计算成本(USD)"""
        input_cost = (input_tokens / 1_000_000) * self.input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * self.output_cost_per_1m
        return input_cost + output_cost


@dataclass
class UsageLog:
    """使用日志记录"""
    user_id: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    timestamp: datetime = field(default_factory=datetime.now)
