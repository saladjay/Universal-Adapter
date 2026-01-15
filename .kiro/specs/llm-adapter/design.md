# Design Document

## Overview

本设计文档描述多海外LLM统一接入与计费监控系统的技术架构和实现方案。系统采用适配器模式，通过统一接口封装多个LLM平台的差异，实现智能路由、Token统计和实时计费功能。

## Architecture

```
                ┌────────────────┐
                │  Application   │
                └───────┬────────┘
                        │
                ┌───────▼────────┐
                │   LLMAdapter   │
                │  (统一入口API)  │
                └───────┬────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼───────┐ ┌─────▼─────┐ ┌───────▼───────┐
│    Router     │ │  Config   │ │ UsageLogger   │
│  (路由策略)    │ │  Manager  │ │  (日志记录)    │
└───────┬───────┘ └───────────┘ └───────────────┘
        │
        ├─────────────┬─────────────┬─────────────┐
        │             │             │             │
┌───────▼───────┐ ┌───▼───┐ ┌──────▼──────┐ ┌────▼────┐
│ OpenAIAdapter │ │Gemini │ │ Cloudflare  │ │HuggingFace│
│               │ │Adapter│ │   Adapter   │ │ Adapter │
└───────┬───────┘ └───┬───┘ └──────┬──────┘ └────┬────┘
        │             │            │              │
        └─────────────┴─────┬──────┴──────────────┘
                            │
                    ┌───────▼───────┐
                    │ Token Counter │
                    │ & Billing     │
                    └───────────────┘
```

## Components and Interfaces

### LLMRequest 接口

```python
@dataclass
class LLMRequest:
    user_id: str
    prompt: str
    scene: Literal["chat", "coach", "persona", "system"]
    quality: Literal["low", "medium", "high"]
```

### LLMResponse 接口

```python
@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
```

### ProviderAdapter 抽象基类

```python
class ProviderAdapter(ABC):
    name: str
    
    @abstractmethod
    async def generate(self, prompt: str, model: str) -> RawLLMResult:
        pass
    
    @abstractmethod
    def estimate_tokens(self, prompt: str, output: str) -> TokenUsage:
        pass
```

### TokenUsage 数据类

```python
@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
```

### PricingRule 数据类

```python
@dataclass
class PricingRule:
    provider: str
    model: str
    input_cost_per_1m: float
    output_cost_per_1m: float
```

### UsageLog 数据类

```python
@dataclass
class UsageLog:
    user_id: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    timestamp: datetime
```

## Data Models

### 配置文件结构 (config.yaml)

```yaml
llm:
  default_provider: openai

providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    base_url: https://api.openai.com
    models:
      cheap: gpt-4o-mini
      normal: gpt-4o
      premium: gpt-4-turbo

  gemini:
    api_key: ${GEMINI_API_KEY}
    models:
      cheap: gemini-1.5-flash
      premium: gemini-1.5-pro

  cloudflare:
    api_key: ${CF_API_KEY}
    account_id: ${CF_ACCOUNT_ID}
    models:
      cheap: "@cf/meta/llama-3-8b-instruct"

  huggingface:
    api_key: ${HF_TOKEN}
    default_model: meta-llama/Llama-3.1-8B-Instruct

pricing:
  openai:
    gpt-4o-mini:
      input_cost_per_1m: 0.15
      output_cost_per_1m: 0.60
    gpt-4o:
      input_cost_per_1m: 2.50
      output_cost_per_1m: 10.00
  gemini:
    gemini-1.5-flash:
      input_cost_per_1m: 0.075
      output_cost_per_1m: 0.30
    gemini-1.5-pro:
      input_cost_per_1m: 1.25
      output_cost_per_1m: 5.00
```

### 路由策略映射

| Quality Level | Primary Provider | Fallback Provider |
|---------------|------------------|-------------------|
| low           | cloudflare       | huggingface       |
| medium        | openai (mini)    | gemini (flash)    |
| high          | openai (premium) | gemini (pro)      |



## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: 请求参数验证

*For any* LLMRequest，如果包含有效的user_id、prompt、scene和quality字段，THE LLM_Adapter SHALL 成功接收并处理该请求；如果任何必需字段缺失或无效，SHALL 返回明确的错误信息而不是崩溃。

**Validates: Requirements 1.1, 1.3**

### Property 2: 响应结构完整性

*For any* 成功的LLM调用，返回的LLMResponse必须包含非空的text、model、provider字段，以及非负的input_tokens、output_tokens和cost_usd值。

**Validates: Requirements 1.2**

### Property 3: Token统计一致性

*For any* TokenUsage结果，total_tokens必须等于input_tokens + output_tokens。

**Validates: Requirements 3.5**

### Property 4: 计费公式正确性

*For any* Token使用量(input_tokens, output_tokens)和定价规则(input_cost_per_1m, output_cost_per_1m)，计算的成本必须等于：
`(input_tokens / 1_000_000) * input_cost_per_1m + (output_tokens / 1_000_000) * output_cost_per_1m`
且结果必须是非负数。

**Validates: Requirements 4.2, 4.3**

### Property 5: 路由策略正确性

*For any* LLMRequest：
- 当quality="low"时，选择的provider必须是"cloudflare"或"huggingface"
- 当quality="medium"时，选择的model必须是cheap或normal级别
- 当quality="high"时，选择的model必须是premium级别

**Validates: Requirements 5.1, 5.2, 5.3**

### Property 6: 配置往返一致性

*For any* 有效的配置对象，序列化为YAML后再解析回来，应该得到等价的配置对象。

**Validates: Requirements 6.1**

### Property 7: 无效配置错误处理

*For any* 格式错误的配置文件，Config_Manager应该返回错误信息而不是崩溃或返回部分配置。

**Validates: Requirements 6.5**

### Property 8: 使用日志完整性

*For any* LLM调用完成后记录的UsageLog，必须包含有效的user_id、provider、model、非负的input_tokens、output_tokens、cost，以及有效的timestamp。

**Validates: Requirements 7.1**

## Error Handling

### 网络错误处理

- 当LLM平台API调用超时时，系统应重试最多3次
- 当所有重试失败后，系统应尝试降级到备选平台
- 如果所有平台都不可用，返回明确的错误信息

### 配置错误处理

- 配置文件不存在时，使用默认配置或返回错误
- 环境变量未设置时，返回明确的错误信息
- API密钥无效时，返回认证错误

### 输入验证错误

- 请求参数类型错误时，返回400错误
- prompt为空时，返回验证错误
- quality值不在允许范围内时，返回验证错误

## Testing Strategy

### 单元测试

使用pytest框架进行单元测试：

- **ConfigManager测试**: 测试配置加载、环境变量替换、错误处理
- **BillingEngine测试**: 测试成本计算公式、边界值
- **Router测试**: 测试各quality级别的路由选择
- **TokenCounter测试**: 测试各平台的Token统计逻辑

### 属性测试

使用hypothesis库进行属性测试，每个属性测试运行最少100次迭代：

- **Property 1**: 生成随机请求参数，验证参数验证逻辑
- **Property 3**: 生成随机Token数量，验证total_tokens = input_tokens + output_tokens
- **Property 4**: 生成随机Token和定价，验证计费公式
- **Property 5**: 生成随机quality值，验证路由选择
- **Property 6**: 生成随机配置，验证YAML序列化往返一致性

### 集成测试

- 使用mock模拟各LLM平台API响应
- 测试完整的调用流程：请求 → 路由 → 调用 → Token统计 → 计费 → 日志记录
- 测试降级逻辑：模拟平台不可用时的自动切换
