# LLMAdapter

多海外 LLM 统一接入与计费监控系统（Python 包）。提供统一的异步调用入口、质量分级路由、成本计费和使用日志记录。

## 框架与模块

- **定位**：非 Web 服务框架，而是一个 Python 库，提供统一的 LLM Adapter 接口。
- **统一入口**：`LLMAdapter`（包含路由、计费、日志）
- **配置管理**：`ConfigManager` 负责读取 `config.yaml` 并进行环境变量替换
- **智能路由**：`Router` 根据质量等级（low/medium/high）选择 provider + model，并支持 fallback
- **计费引擎**：`BillingEngine` 根据定价规则计算 token 成本
- **日志记录**：`UsageLogger` 内存级日志记录，支持按用户/时间查询

核心模块路径：`llm_adapter/adapter.py`, `llm_adapter/router.py`, `llm_adapter/billing.py`, `llm_adapter/logger.py`

## 环境要求

- Python >= 3.13
- 依赖：`httpx`, `pyyaml`（详见 `pyproject.toml`）

## 安装

```bash
python -m pip install -e .
```

## 配置

默认读取根目录 `config.yaml`，支持 `${ENV_VAR}` 形式引用环境变量。

### 配置字段说明

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `llm.default_provider` | string | 否 | 默认 provider（如 `openai`） |
| `providers.<name>.api_key` | string | 是 | Provider 的 API Key（建议用环境变量） |
| `providers.<name>.models.cheap` | string | 否 | 低成本模型标识 |
| `providers.<name>.models.normal` | string | 否 | 常规模型标识 |
| `providers.<name>.models.premium` | string | 否 | 高质量模型标识 |
| `providers.<name>.default_model` | string | 否 | 无分级模型时使用的默认模型 |
| `providers.<name>.base_url` | string | 否 | 自定义 API Base URL |
| `providers.<name>.account_id` | string | 否 | Cloudflare Workers AI 需要的账户 ID |
| `pricing.<provider>.<model>.input_cost_per_1m` | float | 否 | 输入 Token 费用（每 1M token） |
| `pricing.<provider>.<model>.output_cost_per_1m` | float | 否 | 输出 Token 费用（每 1M token） |

### Provider 参数细节

- **OpenAI**：`api_key`, `base_url`（默认 `https://api.openai.com`）
- **Gemini**：`api_key`
- **Cloudflare Workers AI**：`api_key`, `account_id`, `base_url`
- **HuggingFace**：`api_key`, `default_model`
- **DashScope (Qwen)**：`api_key`, `base_url`
- **OpenRouter**：`api_key`, `base_url`
  - 兼容 OpenAI Chat Completions 接口，模型格式为 `provider/model`（如 `anthropic/claude-3.5-sonnet`）
  - 额外可选参数：`site_url`, `site_name`（用于 OpenRouter 统计/排名）

常用配置项：

- `llm.default_provider`
- `providers.<provider>.api_key`
- `providers.<provider>.models.{cheap,normal,premium}`
- `providers.<provider>.base_url` / `account_id`

建议将 API Key 写入环境变量，并在 `config.yaml` 中引用，例如：

```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    models:
      cheap: gpt-4o-mini
      normal: gpt-4o
      premium: gpt-4-turbo
```

## 调用方式（核心 API）

### 1) 统一异步调用

```python
from llm_adapter import LLMAdapter

adapter = LLMAdapter(config_path="config.yaml")
response = await adapter.generate(
    user_id="user_001",
    prompt="What is the capital of France?",
    scene="chat",         # chat/coach/persona/system
    quality="medium",     # low/medium/high
)

print(response.text)
print(response.provider, response.model)
print(response.input_tokens, response.output_tokens)
print(response.cost_usd)
```

### 2) 路由与计费说明

- 路由基于质量等级选择 provider + model
- 成本计算：`(input_tokens / 1_000_000) * input_cost_per_1m + (output_tokens / 1_000_000) * output_cost_per_1m`
- `UsageLogger` 默认记录每次调用的 token 与成本

### 3) 运行内置示例

`main.py` 提供完整示例（包含基础调用、计费、日志、完整集成）：

```bash
python main.py
```

## 运行测试

使用 `pytest` 与 `hypothesis` 的性质测试：

```bash
python -m pytest
```

可单独跑某个模块：

```bash
python -m pytest tests/test_router_properties.py
```

## 支持的 Provider

当前支持的 provider 入口（以配置为准）：

- OpenAI
- Gemini
- Cloudflare Workers AI
- HuggingFace Inference
- DashScope（Qwen）
- OpenRouter

## 目录结构

```
llm_adapter/
  adapter.py      # 统一入口
  config.py       # 配置加载与校验
  router.py       # 质量路由
  billing.py      # 计费引擎
  logger.py       # 使用日志
  models.py       # 请求/响应/定价模型
examples/
  select_provider_example.py
main.py           # 端到端演示
config.yaml       # 默认配置模板
```
