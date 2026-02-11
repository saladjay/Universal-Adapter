"""
阿里百炼(DashScope) 使用示例

使用前请设置环境变量:
    export DASHSCOPE_API_KEY=your_api_key

或者在代码中直接设置 api_key
"""

import asyncio
import os

from llm_adapter.adapters.dashscope_adapter import DashScopeAdapter


async def main():
    # 从环境变量获取 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("请设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    # 创建适配器
    adapter = DashScopeAdapter(api_key=api_key)
    
    # 测试不同模型
    models = [
        ("qwen-flash", "经济实惠，适合简单任务"),
        # ("qwen-plus", "性价比高，适合大多数场景"),
        # ("qwen-max", "最强性能，适合复杂任务"),
    ]
    
    prompt = "用一句话介绍你自己"
    
    for model, description in models:
        print(f"\n{'='*50}")
        print(f"模型: {model} ({description})")
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
