# 大模型输出参数配置指南

## 概述

LLM Adapter 支持三层配置系统来管理大模型的输出参数（如 temperature、max_tokens 等），提供灵活的参数控制和覆盖机制。

## 配置层级

配置优先级从低到高：

```
1. 全局默认配置 (llm.default_generation_params)
   ↓
2. Provider 级别配置 (providers.{provider}.generation_params)
   ↓
3. 模型级别配置 (providers.{provider}.model_params.{model})
   ↓
4. 运行时覆盖 (代码中传入的参数)
```

高优先级的配置会覆盖低优先级的配置。

## 支持的参数

### 通用参数（所有 Provider 支持）

- **temperature** (float): 控制输出的随机性，范围 0.0-2.0
  - 0.0: 确定性输出，每次结果相同
  - 0.7: 平衡创造性和一致性（推荐默认值）
  - 1.0+: 更有创造性，输出更随机

- **max_tokens** (int): 最大输出 token 数
  - 控制响应长度
  - 不同模型有不同的上限

- **top_p** (float): 核采样参数，范围 0.0-1.0
  - 控制输出的多样性
  - 0.9: 推荐值，平衡质量和多样性

- **stop** (list[str]): 停止序列
  - 遇到这些字符串时停止生成
  - 例如: ["\\n\\n", "END"]

### OpenAI/OpenRouter 特有参数

- **presence_penalty** (float): 存在惩罚，范围 -2.0 到 2.0
  - 正值鼓励谈论新话题
  - 负值鼓励重复已有话题

- **frequency_penalty** (float): 频率惩罚，范围 -2.0 到 2.0
  - 正值减少重复词语
  - 负值允许更多重复

- **seed** (int): 随机种子
  - 用于可重现的输出
  - 相同的 seed 和参数会产生相似的结果

### Gemini/DashScope 特有参数

- **top_k** (int): Top-K 采样
  - 限制每步只考虑前 K 个最可能的 token
  - Gemini 推荐值: 40
  - DashScope 支持但不常用

## 配置示例

### 1. 全局默认配置

适用于所有 provider 和模型的基础配置：

```yaml
llm:
  default_provider: openai
  default_generation_params:
    temperature: 0.7        # 平衡的随机性
    max_tokens: 2048        # 适中的输出长度
    top_p: 0.9             # 核采样
```

### 2. Provider 级别配置

为特定 provider 设置默认参数：

```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    models:
      cheap: gpt-4o-mini
      normal: gpt-4o
    # OpenAI 的默认参数
    generation_params:
      temperature: 0.8
      max_tokens: 2048
      presence_penalty: 0.1
      frequency_penalty: 0.1
  
  gemini:
    api_key: ${GEMINI_API_KEY}
    models:
      normal: gemini-2.5-flash
    # Gemini 的默认参数
    generation_params:
      temperature: 0.7
      top_p: 0.95
      top_k: 40              # Gemini 特有
      max_tokens: 2048
```

### 3. 模型级别配置

为特定模型设置专属参数：

```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    models:
      cheap: gpt-4o-mini
      normal: gpt-4o
      premium: gpt-4-turbo
    generation_params:
      temperature: 0.8      # Provider 默认
    model_params:
      # gpt-4o-mini 使用更低的 temperature 和更短的输出
      gpt-4o-mini:
        temperature: 0.5
        max_tokens: 1024
      # gpt-4-turbo 使用更高的 temperature 和更长的输出
      gpt-4-turbo:
        temperature: 0.9
        max_tokens: 4096
        presence_penalty: 0.2
```

### 4. 运行时覆盖

在代码中动态覆盖参数：

```python
from llm_adapter.config import ConfigManager, GenerationParams

config_manager = ConfigManager()

# 获取合并后的参数
params = config_manager.get_generation_params(
    provider="openai",
    model="gpt-4o",
    override_params=GenerationParams(
        temperature=0.3,      # 运行时覆盖
        max_tokens=500        # 运行时覆盖
    )
)

# params 现在包含了所有层级合并后的参数
print(params.to_dict())
# 输出: {'temperature': 0.3, 'max_tokens': 500, 'presence_penalty': 0.1, ...}
```

## 实际使用场景

### 场景 1: 代码生成（需要确定性）

```yaml
providers:
  openai:
    model_params:
      gpt-4o:
        temperature: 0.2      # 低随机性
        max_tokens: 2048
        top_p: 0.95
```

### 场景 2: 创意写作（需要多样性）

```yaml
providers:
  openai:
    model_params:
      gpt-4o:
        temperature: 1.2      # 高随机性
        max_tokens: 4096
        presence_penalty: 0.6  # 鼓励新话题
```

### 场景 3: 客服对话（平衡质量）

```yaml
providers:
  openai:
    model_params:
      gpt-4o-mini:
        temperature: 0.7      # 平衡
        max_tokens: 1024
        frequency_penalty: 0.3  # 减少重复
```

### 场景 4: 可重现的测试

```yaml
providers:
  openai:
    generation_params:
      temperature: 0.0      # 完全确定性
      seed: 12345          # 固定种子
      max_tokens: 1024
```

## 在 Adapter 中使用

### 更新 Adapter 以支持 generation_params

```python
from llm_adapter.config import GenerationParams

class OpenAIAdapter(ProviderAdapter):
    async def generate(
        self, 
        prompt: str, 
        model: str,
        generation_params: GenerationParams | None = None
    ) -> RawLLMResult:
        # 获取合并后的参数
        from llm_adapter.config import ConfigManager
        config_manager = ConfigManager()
        params = config_manager.get_generation_params(
            provider=self.name,
            model=model,
            override_params=generation_params
        )
        
        # 构建请求 payload
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            **params.to_dict()  # 添加所有生成参数
        }
        
        response = await self._client.post(url, json=payload)
        # ... 处理响应
```

## 参数验证

不同的 provider 和模型对参数有不同的限制：

### OpenAI
- temperature: 0.0 - 2.0
- max_tokens: 1 - 模型上限（如 gpt-4o: 16384）
- top_p: 0.0 - 1.0

### Gemini
- temperature: 0.0 - 2.0
- max_tokens: 1 - 32768
- top_p: 0.0 - 1.0
- top_k: 1 - 100

### DashScope (Qwen)
- temperature: 0.0 - 2.0
- max_tokens: 1 - 模型上限
- top_p: 0.0 - 1.0

建议在配置中使用合理的默认值，避免超出限制。

## 最佳实践

1. **设置合理的全局默认值**
   - temperature: 0.7（平衡）
   - max_tokens: 2048（适中）
   - top_p: 0.9（推荐）

2. **为不同用途的模型设置不同参数**
   - 便宜模型（cheap）: 较低的 max_tokens
   - 高级模型（premium）: 较高的 max_tokens 和 temperature

3. **使用 Provider 特有参数**
   - OpenAI: 使用 presence_penalty 和 frequency_penalty
   - Gemini: 使用 top_k
   - DashScope: 使用 seed 保证可重现性

4. **运行时覆盖用于特殊场景**
   - 不要在配置文件中设置所有可能的参数
   - 只在需要时通过代码覆盖

5. **测试和监控**
   - 记录使用的参数和输出质量
   - 根据实际效果调整配置

## 故障排查

### 问题: 参数没有生效

检查配置优先级，确保没有被更高优先级的配置覆盖。

```python
# 调试: 打印最终使用的参数
params = config_manager.get_generation_params("openai", "gpt-4o")
print(params.to_dict())
```

### 问题: API 返回参数错误

某些参数可能不被特定模型支持，检查 provider 文档。

### 问题: 输出质量不理想

尝试调整 temperature 和 top_p：
- 输出太随机 → 降低 temperature
- 输出太死板 → 提高 temperature
- 输出重复 → 增加 frequency_penalty

## 总结

三层配置系统提供了灵活的参数管理：
- ✅ 全局默认保证一致性
- ✅ Provider 配置适应不同 API 特性
- ✅ 模型配置针对特定模型优化
- ✅ 运行时覆盖满足动态需求
