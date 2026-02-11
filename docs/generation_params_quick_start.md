# Generation Parameters 快速开始

## 什么是 Generation Parameters？

Generation Parameters（生成参数）控制大模型的输出行为，例如：
- **temperature**: 控制输出的随机性（0.0 = 确定性，1.0+ = 创造性）
- **max_tokens**: 限制输出长度
- **top_p**: 核采样，控制输出多样性
- **presence_penalty**: 鼓励谈论新话题（OpenAI）
- **frequency_penalty**: 减少重复词语（OpenAI）
- **top_k**: Top-K 采样（Gemini/DashScope）
- **seed**: 随机种子，用于可重现输出

## 三层配置系统

```
全局默认 → Provider 默认 → 模型特定 → 运行时覆盖
(最低优先级)                        (最高优先级)
```

## 快速配置

### 1. 在 config.yaml 中设置全局默认

```yaml
llm:
  default_provider: openai
  default_generation_params:
    temperature: 0.7
    max_tokens: 2048
    top_p: 0.9
```

### 2. 为 Provider 设置默认参数

```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    generation_params:
      temperature: 0.8
      presence_penalty: 0.1
      frequency_penalty: 0.1
```

### 3. 为特定模型设置参数

```yaml
providers:
  openai:
    model_params:
      gpt-4o-mini:
        temperature: 0.5
        max_tokens: 1024
      gpt-4-turbo:
        temperature: 0.9
        max_tokens: 4096
```

### 4. 在代码中运行时覆盖

```python
result = await adapter.generate(
    prompt="Your prompt here",
    provider="openai",
    model="gpt-4o",
    temperature=0.3,      # 运行时覆盖
    max_tokens=500
)
```

## 常见使用场景

### 代码生成（需要确定性）
```python
result = await adapter.generate(
    prompt="Write a Python function to sort a list",
    provider="openai",
    model="gpt-4o",
    temperature=0.2,      # 低随机性
    max_tokens=1024
)
```

### 创意写作（需要多样性）
```python
result = await adapter.generate(
    prompt="Write a creative story about AI",
    provider="openai",
    model="gpt-4o",
    temperature=1.2,      # 高随机性
    presence_penalty=0.6  # 鼓励新话题
)
```

### 客服对话（平衡质量）
```python
result = await adapter.generate(
    prompt="How can I help you today?",
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.7,      # 平衡
    frequency_penalty=0.3  # 减少重复
)
```

### 可重现的测试
```python
result = await adapter.generate(
    prompt="Generate test data",
    provider="openai",
    model="gpt-4o",
    temperature=0.0,      # 完全确定性
    seed=12345           # 固定种子
)
```

## 参数推荐值

### Temperature
- **0.0-0.3**: 代码生成、数据提取、事实性回答
- **0.4-0.7**: 通用对话、客服、问答
- **0.8-1.2**: 创意写作、头脑风暴、故事生成
- **1.3-2.0**: 极度创造性（可能不连贯）

### Max Tokens
- **256-512**: 简短回答、摘要
- **1024-2048**: 标准回答、文章段落
- **4096+**: 长文章、详细分析

### Top P
- **0.9-0.95**: 推荐值，平衡质量和多样性
- **0.5-0.8**: 更保守的输出
- **0.95-1.0**: 更多样化的输出

## 完整示例

查看 `examples/generation_params_example.py` 获取更多示例。

运行示例：
```bash
python examples/generation_params_example.py
```

## 更多信息

详细文档请参考：
- [完整配置指南](./generation_params_config.md)
- [HTTP 客户端并发配置](./http_client_concurrency.md)
