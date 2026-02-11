"""
Gemini API 调用示例

演示两种调用方式:
1. HTTP 模式 - 直接调用 API，无需额外依赖
2. SDK 模式 - 使用官方 google-generativeai 库

使用前请确保设置环境变量:
    set GEMINI_API_KEY=your_api_key_here  (Windows CMD)
    $env:GEMINI_API_KEY="your_api_key_here"  (PowerShell)

SDK 模式需要安装:
    pip install google-generativeai
"""

import asyncio
import os
from llm_adapter.adapters.gemini_adapter import GeminiAdapter, ProviderError


async def call_gemini_http():
    """HTTP 模式调用 (默认，无需额外依赖)"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("请设置 GEMINI_API_KEY 环境变量")
        return
    
    # mode="http" 是默认值，可以省略
    adapter = GeminiAdapter(api_key=api_key, mode="http")
    
    result = await adapter.generate(
        prompt="用一句话解释什么是人工智能",
        model="gemini-2.5-flash"
    )
    
    print("=== HTTP 模式 ===")
    print(f"回答: {result.text}")
    print(f"输入 tokens: {result.input_tokens}")
    print(f"输出 tokens: {result.output_tokens}")


async def call_gemini_sdk():
    """SDK 模式调用 (需要 pip install google-generativeai)"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("请设置 GEMINI_API_KEY 环境变量")
        return
    
    try:
        adapter = GeminiAdapter(api_key=api_key, mode="sdk")
    except ProviderError as e:
        print(f"\n=== SDK 模式 ===")
        print(f"跳过: {e}")
        print("安装命令: pip install google-generativeai")
        return
    
    result = await adapter.generate(
        prompt="Python 有哪些优点？请简要列出3点。",
        model="gemini-2.0-flash"
    )
    
    print("\n=== SDK 模式 ===")
    print(f"回答: {result.text}")
    print(f"输入 tokens: {result.input_tokens}")
    print(f"输出 tokens: {result.output_tokens}")


async def main():
    """运行示例"""
    try:
        await call_gemini_http()
        # await call_gemini_sdk()
    except ProviderError as e:
        print(f"调用失败: {e}")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())
