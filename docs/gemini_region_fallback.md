# Gemini Region Fallback

## 概述

Gemini adapter 现在支持区域故障转移（Region Fallback）功能。当指定的区域端点不可用或模型在该区域不存在时，系统会自动切换到全局区域（默认为 `us-central1`）重试请求。

## 功能特性

- **自动故障转移**: 当区域端点失败时自动切换到备用区域
- **性能追踪**: 记录每次故障转移的耗时和成功率
- **灵活配置**: 可以启用/禁用故障转移，自定义备用区域
- **流式支持**: 支持普通生成和流式生成的故障转移

## 使用方法

### 基本配置

```python
from llm_adapter.adapters.gemini_adapter import GeminiAdapter

adapter = GeminiAdapter(
    api_key="your-api-key",
    mode="vertex",
    project_id="your-project-id",
    location="asia-southeast1",           # 主要区域
    enable_region_fallback=True,          # 启用故障转移（默认）
    fallback_location="us-central1"       # 备用区域（默认）
)
```

### 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_region_fallback` | bool | `True` | 是否启用区域故障转移 |
| `fallback_location` | str | `"us-central1"` | 备用区域位置 |

### 触发条件

故障转移会在以下情况下触发：

- 模型在指定区域不存在（404 错误）
- 区域端点不可用（unavailable 错误）
- 权限问题（403 错误）
- 其他区域特定的错误

**注意**: 配额限制（429 错误）不会触发故障转移。

## 故障转移追踪

### 获取统计信息

```python
from llm_adapter.fallback_tracker import get_fallback_tracker

# 获取全局追踪器
tracker = get_fallback_tracker()

# 获取统计摘要
stats = tracker.get_stats()
summary = stats.get_summary()

print(f"总故障转移次数: {summary['total_fallbacks']}")
print(f"成功次数: {summary['successful_fallbacks']}")
print(f"失败次数: {summary['failed_fallbacks']}")
print(f"成功率: {summary['success_rate']:.1%}")
print(f"平均耗时: {summary['average_duration_ms']:.2f}ms")
```

### 查看故障转移事件

```python
# 获取最近的故障转移事件
recent_events = tracker.get_recent_events(limit=10)

for event in recent_events:
    print(f"时间: {event.timestamp}")
    print(f"区域: {event.original_location} → {event.fallback_location}")
    print(f"模型: {event.original_model}")
    print(f"成功: {event.success}")
    print(f"耗时: {event.fallback_duration_ms:.2f}ms")
    print(f"错误: {event.error_message}")
```

### 清除统计数据

```python
# 清除所有记录的事件和统计信息
tracker.clear()
```

## 完整示例

参见 `examples/gemini_region_fallback_example.py` 获取完整的使用示例，包括：

1. 基本的故障转移测试
2. 流式生成的故障转移
3. 禁用故障转移的行为
4. 统计信息的查看和分析

### 运行示例

```bash
# 设置环境变量
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
export GCP_PROJECT_ID=your-project-id

# 运行示例
python examples/gemini_region_fallback_example.py
```

## 性能考虑

- **延迟增加**: 故障转移会增加请求延迟（通常几百毫秒到几秒）
- **成本**: 故障转移到不同区域可能产生不同的费用
- **缓存清理**: 故障转移会清除模型缓存，首次请求可能稍慢

## 最佳实践

1. **选择合适的主要区域**: 选择离用户最近的区域作为主要区域
2. **监控故障转移率**: 如果故障转移频繁发生，考虑更换主要区域
3. **配置合适的备用区域**: `us-central1` 通常是最稳定的选择
4. **记录和分析**: 定期查看故障转移统计，优化配置

## 配置文件示例

在 `config.yaml` 中配置：

```yaml
providers:
  gemini:
    api_key: ${GEMINI_API_KEY}
    mode: vertex
    project_id: your-project-id
    location: asia-southeast1
    enable_region_fallback: true
    fallback_location: us-central1
    models:
      cheap: gemini-2.0-flash-lite-001
      normal: gemini-2.0-flash-lite-001
```

## 故障排查

### 故障转移失败

如果故障转移也失败，检查：

1. 备用区域是否正确配置
2. 服务账号是否有备用区域的权限
3. 模型在备用区域是否可用

### 性能问题

如果故障转移耗时过长：

1. 检查网络连接
2. 考虑使用更近的备用区域
3. 检查 GCP 服务状态

## API 参考

### FallbackEvent

记录单次故障转移事件的数据类。

**属性**:
- `timestamp`: 事件时间
- `provider`: 提供商名称
- `original_location`: 原始区域
- `fallback_location`: 备用区域
- `original_model`: 模型名称
- `fallback_model`: 备用模型名称
- `error_message`: 原始错误信息
- `fallback_duration_ms`: 故障转移耗时（毫秒）
- `success`: 是否成功

### FallbackStats

故障转移统计信息。

**方法**:
- `add_event(event)`: 添加事件并更新统计
- `get_summary()`: 获取统计摘要

### FallbackTracker

全局故障转移追踪器。

**方法**:
- `record_fallback(...)`: 记录故障转移事件
- `get_stats()`: 获取统计信息
- `get_recent_events(limit)`: 获取最近的事件
- `clear()`: 清除所有数据
