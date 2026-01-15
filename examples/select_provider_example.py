"""
LLMAdapter 使用示例 - 指定平台和模型

本示例展示如何使用 LLMAdapter 选择特定的平台和模型进行调用。
系统会自动跳过没有配置 API Key 的平台。

使用前请设置相应平台的环境变量，例如:
    export OPENAI_API_KEY=your_key
    export DASHSCOPE_API_KEY=your_key
    export OPENROUTER_API_KEY=your_key
"""

import asyncio
import os
import sys
from pathlib import Path

# 确保能找到 llm_adapter 模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_adapter.adapter import LLMAdapter
from llm_adapter.config import ConfigManager


def get_config_path() -> str:
    """获取配置文件路径"""
    # 尝试多个可能的位置
    possible_paths = [
        Path(__file__).parent.parent / "config.yaml",  # 项目根目录
        Path.cwd() / "config.yaml",  # 当前工作目录
    ]
    print(possible_paths)
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    raise FileNotFoundError("找不到 config.yaml 配置文件")


def show_available_providers(adapter: LLMAdapter) -> None:
    """显示所有可用的平台和模型"""
    providers = adapter.get_available_providers()
    
    if not providers:
        print("没有可用的平台，请检查环境变量配置")
        return
    
    print("\n可用的平台和模型:")
    print("=" * 60)
    
    for provider in providers:
        models = adapter.get_provider_models(provider)
        print(f"\n【{provider}】")
        for tier, model in models.items():
            print(f"  - {tier}: {model}")


async def call_with_provider(
    adapter: LLMAdapter,
    provider: str,
    model: str,
    prompt: str
) -> None:
    """使用指定平台和模型调用"""
    print(f"\n{'='*60}")
    print(f"平台: {provider}")
    print(f"模型: {model}")
    print(f"提示: {prompt}")
    print("-" * 60)
    
    try:
        response = await adapter.generate_with_provider(
            user_id="demo_user",
            prompt=prompt,
            provider=provider,
            model=model,
        )
        
        print(f"回复: {response.text}")
        print(f"输入Token: {response.input_tokens}")
        print(f"输出Token: {response.output_tokens}")
        print(f"成本: ${response.cost_usd:.6f}")
        
    except Exception as e:
        print(f"错误: {e}")


async def interactive_mode(adapter: LLMAdapter) -> None:
    """交互模式 - 让用户选择平台和模型"""
    providers = adapter.get_available_providers()
    
    if not providers:
        print("没有可用的平台")
        return
    
    # 显示可用平台
    print("\n请选择平台:")
    for i, provider in enumerate(providers, 1):
        print(f"  {i}. {provider}")
    
    try:
        choice = int(input("\n输入数字选择平台: ")) - 1
        if choice < 0 or choice >= len(providers):
            print("无效选择")
            return
        
        selected_provider = providers[choice]
        
        # 显示该平台的模型
        models = adapter.get_provider_models(selected_provider)
        model_list = list(models.items())
        
        print(f"\n【{selected_provider}】可用模型:")
        for i, (tier, model) in enumerate(model_list, 1):
            print(f"  {i}. {model} ({tier})")
        
        model_choice = int(input("\n输入数字选择模型: ")) - 1
        if model_choice < 0 or model_choice >= len(model_list):
            print("无效选择")
            return
        
        selected_model = model_list[model_choice][1]
        
        # 获取用户输入
        prompt = input("\n请输入提示词: ")
        if not prompt.strip():
            prompt = "你好，请用一句话介绍你自己"
        
        # 调用
        await call_with_provider(adapter, selected_provider, selected_model, prompt)
        
    except ValueError:
        print("请输入有效数字")
    except KeyboardInterrupt:
        print("\n已取消")


async def demo_all_providers(adapter: LLMAdapter) -> None:
    """演示调用所有可用平台"""
    providers = adapter.get_available_providers()
    print(providers)
    prompt = "用一句话介绍你自己"
    
    for provider in providers:
        models = adapter.get_provider_models(provider)
        # 选择第一个可用模型
        if models:
            model = list(models.values())[0]
            await call_with_provider(adapter, provider, model, prompt)


async def main():
    # 获取配置文件路径
    try:
        config_path = get_config_path()
        print(f"使用配置文件: {config_path}")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return
    
    # 创建 LLMAdapter 实例
    try:
        adapter = LLMAdapter(config_path=config_path)
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    # 显示可用平台
    show_available_providers(adapter)
    
    print("\n" + "=" * 60)
    print("选择模式:")
    print("  1. 交互模式 - 手动选择平台和模型")
    print("  2. 演示模式 - 自动调用所有可用平台")
    print("  3. 退出")
    
    try:
        mode = input("\n输入数字选择模式: ").strip()
        
        if mode == "1":
            await interactive_mode(adapter)
        elif mode == "2":
            await demo_all_providers(adapter)
        else:
            print("退出")
            
    except KeyboardInterrupt:
        print("\n已取消")


if __name__ == "__main__":
    asyncio.run(main())
