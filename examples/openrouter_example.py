"""
OpenRouter 使用示例

OpenRouter 是一个聚合多个 LLM 的平台，支持 OpenAI、Anthropic、Google、Meta 等多家模型。

使用前请设置环境变量:
    export OPENROUTER_API_KEY=your_api_key

获取 API Key: https://openrouter.ai/keys
"""

import asyncio
import os

from llm_adapter.adapters.openrouter_adapter import OpenRouterAdapter


async def main():
    # 从环境变量获取 API Key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("请设置 OPENROUTER_API_KEY 环境变量")
        print("获取 API Key: https://openrouter.ai/keys")
        return
    
    # 创建适配器
    adapter = OpenRouterAdapter(
        api_key=api_key,
        site_name="LLM Adapter Demo"  # 可选：用于 OpenRouter 统计
    )
    
    # 测试不同模型
    # OpenRouter 支持多家厂商的模型，格式为 provider/model
    models = [
        ("meta-llama/llama-3.1-8b-instruct", "Meta Llama 3.1 8B - 免费/低成本"),
        ("anthropic/claude-3.5-sonnet", "Claude 3.5 Sonnet - 性价比高"),
        # ("anthropic/claude-3-opus", "Claude 3 Opus - 最强性能"),
        # ("openai/gpt-4o", "GPT-4o - OpenAI 最新"),
    ]
    
    prompt = "用一句话介绍你自己"
    
    for model, description in models:
        print(f"\n{'='*50}")
        print(f"模型: {model}")
        print(f"描述: {description}")
        print(f"{'='*50}")
        
        try:
            result = await adapter.generate(prompt, model)
            print(f"回复: {result.text}")
            print(f"输入Token: {result.input_tokens}")
            print(f"输出Token: {result.output_tokens}")
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())
