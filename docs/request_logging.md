# LLM Adapter 请求日志功能

## 概述

所有 LLM adapter 都会自动记录每次请求的详细信息，包括：
- 触发时间（ISO 8601 格式）
- 请求耗时（毫秒）
- Token 使用量（输入/输出）
- 模型信息
- 成功/失败状态
- 成本信息（如果可用）
- 错误信息（如果失败）

## 日志文件格式

### 文件位置
```
logs/
├── openrouter/
│   ├── 2026-02-06.jsonl
│   └── 2026-02-07.jsonl
├── dashscope/
│   └── 2026-02-06.jsonl
├── gemini/
│   └── 2026-02-06.jsonl
└── ...
```

### 文件格式
每个日志文件是 JSONL 格式（每行一个 JSON 对象），便于流式处理和分析。

### 日志字段

```json
{
  "timestamp": "2026-02-06T19:15:42.686960",
  "adapter": "openrouter",
  "model": "google/gemini-2.0-flash-lite-001",
  "prompt_length": 8,
  "prompt_preview": "Say hi",
  "response_length": 35,
  "response_preview": "Hi there! How can I help you today?",
  "input_tokens": 2,
  "output_tokens": 11,
  "total_tokens": 13,
  "duration_ms": 1433.98,
  "success": true,
  "error_message": null,
  "cost_usd": 0.00000345,
  "provider": "Google",
  "actual_model": "google/gemini-2.0-flash-lite-001"
}
```

## 配置

### 环境变量

#### `LLM_ADAPTER_LOGGING`
控制是否启用日志记录。

- **默认值**: `true`
- **可选值**: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`

```bash
# 禁用日志
export LLM_ADAPTER_LOGGING=false

# 启用日志（默认）
export LLM_ADAPTER_LOGGING=true
```

#### `LLM_ADAPTER_LOG_DIR`
自定义日志文件存储目录。

- **默认值**: `logs`
- **示例**:

```bash
# 使用自定义目录
export LLM_ADAPTER_LOG_DIR=/var/log/llm_adapter

# 使用相对路径
export LLM_ADAPTER_LOG_DIR=./my_logs
```

## 使用示例

### Python 代码中使用

```python
import os
from llm_adapter.adapters.openrouter_adapter import OpenRouterAdapter

# 方式 1: 使用默认配置（日志已启用）
adapter = OpenRouterAdapter(api_key="your-key")

# 方式 2: 通过环境变量禁用日志
os.environ["LLM_ADAPTER_LOGGING"] = "false"
adapter = OpenRouterAdapter(api_key="your-key")

# 方式 3: 自定义日志目录
os.environ["LLM_ADAPTER_LOG_DIR"] = "./custom_logs"
adapter = OpenRouterAdapter(api_key="your-key")
```

### 命令行使用

```bash
# 运行时禁用日志
LLM_ADAPTER_LOGGING=false python your_script.py

# 使用自定义日志目录
LLM_ADAPTER_LOG_DIR=/tmp/logs python your_script.py
```

## 日志分析

### 读取日志

```python
import json
from pathlib import Path

# 读取今天的 OpenRouter 日志
log_file = Path("logs/openrouter/2026-02-06.jsonl")

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        log_entry = json.loads(line)
        print(f"{log_entry['timestamp']}: {log_entry['model']} - {log_entry['duration_ms']}ms")
```

### 统计分析

```python
import json
from pathlib import Path
from collections import defaultdict

def analyze_logs(log_file):
    """分析日志文件"""
    
    stats = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "total_tokens": 0,
        "total_cost": 0.0,
        "total_duration": 0.0,
        "models": defaultdict(int),
    }
    
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            log = json.loads(line)
            
            stats["total_requests"] += 1
            if log["success"]:
                stats["successful_requests"] += 1
            else:
                stats["failed_requests"] += 1
            
            stats["total_tokens"] += log.get("total_tokens", 0)
            stats["total_cost"] += log.get("cost_usd", 0) or 0
            stats["total_duration"] += log["duration_ms"]
            stats["models"][log["model"]] += 1
    
    return stats

# 使用示例
stats = analyze_logs("logs/openrouter/2026-02-06.jsonl")
print(f"总请求数: {stats['total_requests']}")
print(f"成功率: {stats['successful_requests'] / stats['total_requests'] * 100:.1f}%")
print(f"总成本: ${stats['total_cost']:.6f}")
print(f"平均耗时: {stats['total_duration'] / stats['total_requests']:.2f}ms")
```

### 使用 jq 命令行工具

```bash
# 统计今天的请求数
cat logs/openrouter/2026-02-06.jsonl | wc -l

# 查看失败的请求
cat logs/openrouter/2026-02-06.jsonl | jq 'select(.success == false)'

# 计算总成本
cat logs/openrouter/2026-02-06.jsonl | jq -s 'map(.cost_usd // 0) | add'

# 按模型统计请求数
cat logs/openrouter/2026-02-06.jsonl | jq -r '.model' | sort | uniq -c

# 查看最慢的 10 个请求
cat logs/openrouter/2026-02-06.jsonl | jq -s 'sort_by(.duration_ms) | reverse | .[0:10]'
```

## 性能影响

日志记录是异步的，对主流程的性能影响极小：
- 日志写入使用追加模式，不会阻塞
- 失败的日志写入不会影响主流程
- 可以随时通过环境变量禁用

## 隐私和安全

### 默认行为
- **Prompt 和 Response**: 只记录前 100 个字符的预览
- **完整内容**: 不会记录到日志文件中
- **API Key**: 不会记录

### 自定义隐私设置

如果需要完全不记录内容，可以修改 `llm_adapter/request_logger.py`:

```python
# 完全不记录 prompt 和 response
log_entry = {
    # ...
    "prompt_preview": None,  # 改为 None
    "response_preview": None,  # 改为 None
    # ...
}
```

## 日志轮转

日志文件按日期自动分割，建议定期清理旧日志：

```bash
# 删除 30 天前的日志
find logs -name "*.jsonl" -mtime +30 -delete

# 压缩 7 天前的日志
find logs -name "*.jsonl" -mtime +7 -exec gzip {} \;
```

## 已支持的 Adapter

- ✅ OpenRouter
- ✅ DashScope
- ✅ Gemini (部分支持)
- ⏳ OpenAI (待更新)
- ⏳ Cloudflare (待更新)
- ⏳ HuggingFace (待更新)

## 故障排查

### 日志文件未创建

1. 检查环境变量:
   ```bash
   echo $LLM_ADAPTER_LOGGING
   ```

2. 检查目录权限:
   ```bash
   ls -la logs/
   ```

3. 检查是否有写入错误（会打印警告）

### 日志文件过大

使用日志轮转或定期清理：
```bash
# 每天运行的 cron 任务
0 0 * * * find /path/to/logs -name "*.jsonl" -mtime +7 -delete
```

## 最佳实践

1. **生产环境**: 保持日志启用，用于监控和调试
2. **开发环境**: 可以禁用日志以减少磁盘占用
3. **定期分析**: 使用日志分析成本、性能和错误模式
4. **日志轮转**: 定期清理或归档旧日志
5. **监控告警**: 基于日志设置错误率和延迟告警
