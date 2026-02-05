# HTTP Client 并发处理说明

## 当前实现

所有适配器复用同一个 `httpx.AsyncClient` 实例，这是**推荐的做法**，不会影响并发性能。

## 并发处理机制

### 1. 连接池配置

```python
httpx.Limits(
    max_connections=100,           # 最大并发连接数
    max_keepalive_connections=20,  # 保持活跃的连接数
)
```

### 2. 请求处理流程

当多个请求并发进来时：

```
请求1 进来 → 检查连接池
├─ 有空闲 keep-alive 连接 → 立即复用（最快）
├─ 无空闲但未达上限 → 创建新连接
└─ 达到上限(100) → 排队等待

请求2 进来 → 同时处理（异步非阻塞）
请求3 进来 → 同时处理（异步非阻塞）
...
请求100 进来 → 同时处理
请求101 进来 → 等待前面的请求完成
```

### 3. 异步非阻塞

`AsyncClient` 是异步的，多个请求可以：
- 同时发送（不会互相阻塞）
- 同时等待响应
- 并发处理多达 100 个请求

## 性能优势

### 复用 Client 的好处：

1. **连接复用**：避免重复建立 TCP 连接
2. **SSL/TLS 复用**：避免重复握手（节省 100-200ms）
3. **内存效率**：共享连接池，避免资源浪费
4. **更高吞吐**：keep-alive 连接可以立即使用

### 如果每次创建新 Client：

```python
# ❌ 不推荐
async def generate(self, prompt: str, model: str):
    client = httpx.AsyncClient()  # 每次都创建
    response = await client.post(...)
    await client.aclose()
```

问题：
- 每次都要建立新的 TCP 连接（慢）
- 每次都要 SSL/TLS 握手（慢）
- 无法复用连接
- 资源浪费

## 配置优化

### 在 config.yaml 中配置：

```yaml
# HTTP 客户端配置（可选）
http_client:
  max_connections: 200              # 增加最大并发数
  max_keepalive_connections: 50    # 增加保持活跃的连接数
  timeout: 120.0                    # 超时时间（秒）
```

### 默认值：

如果不配置 `http_client` 部分，将使用以下默认值：

- `max_connections`: 100（足够大多数场景）
- `max_keepalive_connections`: 20（平衡性能和资源）
- `timeout`: 60.0 秒（OpenAI/Gemini/Cloudflare/DashScope）或 120.0 秒（OpenRouter/HuggingFace）

### 配置说明：

- **max_connections**: 最大并发连接数，决定了可以同时处理多少个请求
- **max_keepalive_connections**: 保持活跃的连接数，这些连接可以被立即复用，提升性能
- **timeout**: 单个请求的超时时间（秒），超时后会抛出 `TimeoutException`

## 实际场景示例

### 场景 1：低并发（< 20 请求/秒）

```
配置：默认值即可
结果：所有请求都能复用 keep-alive 连接，性能最优
```

### 场景 2：中等并发（20-100 请求/秒）

```
配置：默认值即可
结果：部分请求复用连接，部分创建新连接，性能良好
```

### 场景 3：高并发（> 100 请求/秒）

```
配置：增加 max_connections 到 200-500
结果：更多请求可以并发处理，减少排队等待
```

### 场景 4：超高并发（> 500 请求/秒）

```
建议：
1. 增加 max_connections 到 1000+
2. 考虑使用多个 adapter 实例（负载均衡）
3. 考虑使用消息队列（如 Redis/RabbitMQ）
4. 监控 API 提供商的速率限制
```

## 监控建议

可以添加日志来监控连接池使用情况：

```python
# 在 adapter 中添加
async def generate(self, prompt: str, model: str):
    # 记录当前连接数
    pool_info = self._client._pool  # 内部 API，仅用于调试
    logger.debug(f"Active connections: {len(pool_info._connections)}")
    
    response = await self._client.post(...)
    return response
```

## 总结

✅ **复用 client 是正确的做法**
- 不会影响并发性能
- 反而提升性能（连接复用）
- 节省资源

✅ **当前配置足够大多数场景**
- max_connections=100 支持 100 个并发请求
- 如需更高并发，可在配置文件中调整

✅ **异步非阻塞**
- 多个请求不会互相阻塞
- 可以同时处理多个请求
