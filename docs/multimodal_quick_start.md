# Multimodal Quick Start Guide

快速开始使用多模态（文本+图片）功能。

## 安装

确保已安装必要的依赖：

```bash
# 基础依赖
pip install httpx

# DashScope SDK（可选，用于 DashScope SDK 模式）
pip install dashscope
```

## 基本用法

### 1. OpenRouter - URL 模式

最简单的方式，使用图片 URL：

```python
import asyncio
import os
from llm_adapter.adapters.openrouter_adapter import OpenRouterAdapter
from llm_adapter.adapters.base import ImageInput, MultimodalContent

async def main():
    # 初始化 adapter
    adapter = OpenRouterAdapter(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
    
    # 创建多模态内容
    content = MultimodalContent(
        text="What's in this image?",
        images=[ImageInput.from_url("https://example.com/image.jpg")]
    )
    
    # 生成响应
    result = await adapter.generate_multimodal(
        content=content,
        model="google/gemini-2.0-flash-exp:free"
    )
    
    print(result.text)
    await adapter.aclose()

asyncio.run(main())
```

### 2. DashScope - URL 模式

使用阿里云百炼的视觉模型：

```python
from llm_adapter.adapters.dashscope_adapter import DashScopeAdapter

async def main():
    adapter = DashScopeAdapter(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        mode="dashscope"  # 使用 SDK 模式
    )
    
    content = MultimodalContent(
        text="请描述这张图片的内容。",
        images=[ImageInput.from_url("https://example.com/image.jpg")]
    )
    
    result = await adapter.generate_multimodal(content, "qwen-vl-plus")
    print(result.text)
    await adapter.aclose()

asyncio.run(main())
```

### 3. 流式输出

实时获取响应：

```python
async def stream_example():
    adapter = OpenRouterAdapter(api_key=os.getenv("OPENROUTER_API_KEY"))
    
    content = MultimodalContent(
        text="Describe this image in detail",
        images=[ImageInput.from_url("https://example.com/image.jpg")]
    )
    
    # 流式输出
    async for chunk in adapter.stream_multimodal(content, "google/gemini-2.0-flash-exp:free"):
        print(chunk, end="", flush=True)
    
    await adapter.aclose()

asyncio.run(stream_example())
```

### 4. Base64 模式

使用本地图片：

```python
import base64
from pathlib import Path

async def base64_example():
    # 读取本地图片
    image_path = Path("local_image.jpg")
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    
    adapter = OpenRouterAdapter(api_key=os.getenv("OPENROUTER_API_KEY"))
    
    content = MultimodalContent(
        text="What do you see?",
        images=[ImageInput.from_base64(image_data, "image/jpeg")]
    )
    
    result = await adapter.generate_multimodal(content, "google/gemini-2.0-flash-exp:free")
    print(result.text)
    await adapter.aclose()

asyncio.run(base64_example())
```

### 5. 多张图片

一次分析多张图片：

```python
async def multiple_images():
    adapter = OpenRouterAdapter(api_key=os.getenv("OPENROUTER_API_KEY"))
    
    content = MultimodalContent(
        text="Compare these images",
        images=[
            ImageInput.from_url("https://example.com/image1.jpg"),
            ImageInput.from_url("https://example.com/image2.jpg"),
            ImageInput.from_url("https://example.com/image3.jpg"),
        ]
    )
    
    result = await adapter.generate_multimodal(content, "anthropic/claude-3-opus")
    print(result.text)
    await adapter.aclose()

asyncio.run(multiple_images())
```

## 支持的模型

### OpenRouter

- `google/gemini-2.0-flash-exp:free` - 免费，速度快
- `google/gemini-2.0-flash-lite-001` - 轻量级
- `anthropic/claude-3-opus` - 高质量分析
- `anthropic/claude-3-sonnet` - 平衡性能
- `anthropic/claude-3-haiku:beta` - 快速响应
- `openai/gpt-4-vision-preview` - GPT-4 视觉

### DashScope

- `qwen-vl-plus` - 通义千问视觉增强版
- `qwen-vl-max` - 通义千问视觉旗舰版
- `qwen3-vl-30b-a3b-instruct` - Qwen3 视觉模型

## 完整示例

```python
import asyncio
import os
from llm_adapter.adapters.openrouter_adapter import OpenRouterAdapter
from llm_adapter.adapters.base import ImageInput, MultimodalContent

async def analyze_image(image_url: str, question: str):
    """分析图片并回答问题"""
    
    adapter = OpenRouterAdapter(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )
    
    try:
        content = MultimodalContent(
            text=question,
            images=[ImageInput.from_url(image_url)]
        )
        
        result = await adapter.generate_multimodal(
            content=content,
            model="google/gemini-2.0-flash-exp:free"
        )
        
        return {
            "answer": result.text,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": result.cost_usd
        }
        
    finally:
        await adapter.aclose()

# 使用示例
result = asyncio.run(analyze_image(
    image_url="https://example.com/chart.png",
    question="What are the key trends in this chart?"
))

print(f"Answer: {result['answer']}")
print(f"Tokens: {result['input_tokens']} in, {result['output_tokens']} out")
if result['cost_usd']:
    print(f"Cost: ${result['cost_usd']:.6f}")
```

## 错误处理

```python
from llm_adapter.adapters.base import ProviderError

async def safe_generate():
    adapter = OpenRouterAdapter(api_key=os.getenv("OPENROUTER_API_KEY"))
    
    try:
        content = MultimodalContent(
            text="Describe this",
            images=[ImageInput.from_url("https://example.com/image.jpg")]
        )
        
        result = await adapter.generate_multimodal(content, "google/gemini-2.0-flash-exp:free")
        return result.text
        
    except NotImplementedError:
        print("This adapter doesn't support multimodal")
    except ProviderError as e:
        print(f"API error: {e.message}")
        if e.status_code:
            print(f"Status code: {e.status_code}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        await adapter.aclose()

asyncio.run(safe_generate())
```

## 最佳实践

1. **优先使用 URL 模式**：更高效，不增加请求大小
2. **优化图片大小**：发送前调整图片大小以减少延迟和成本
3. **检查模型能力**：不是所有模型都支持多张图片
4. **处理错误**：不同提供商可能有不同的限制
5. **注意成本**：视觉模型通常比纯文本模型更贵

## 环境变量

```bash
# OpenRouter
export OPENROUTER_API_KEY=sk-or-v1-xxx
export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# DashScope
export DASHSCOPE_API_KEY=sk-xxx
```

## 下一步

- 查看 [完整文档](./multimodal_interface.md)
- 运行 [测试示例](../examples/test_multimodal_implementation.py)
- 查看 [更多示例](../examples/multimodal_example.py)
