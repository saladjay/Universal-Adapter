"""
阿里百炼 (DashScope) 快速开始示例

这是一个简单的 DashScope 使用示例，展示基本的调用方式。

使用方法:
  python examples/dashscope_quick_start.py                    # 使用环境变量中的 API Key
  python examples/dashscope_quick_start.py your-api-key       # 使用命令行参数指定 API Key
"""
import time
import asyncio
import os
import sys
from llm_adapter.adapters.dashscope_adapter import DashScopeAdapter


def get_api_key():
    """获取 API Key，优先使用命令行参数，其次使用环境变量"""
    
    # 1. 检查命令行参数
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
        print(f"✓ 使用命令行参数提供的 API Key")
        return api_key
    
    # 2. 检查环境变量
    api_key = os.getenv("DASHSCOPY_API_KEY")
    if api_key:
        print(f"✓ 使用环境变量 DASHSCOPY_API_KEY")
        return api_key
    
    # 3. 都没有，返回 None
    return None


async def quick_start():
    """快速开始示例"""
    
    # 获取 API Key
    api_key = get_api_key()
    
    if not api_key:
        print("\n" + "=" * 60)
        print("错误: 未找到 API Key")
        print("=" * 60)
        print("\n请使用以下任一方式提供 API Key:\n")
        print("方式 1: 命令行参数")
        print("  python examples/dashscope_quick_start.py your-api-key\n")
        print("方式 2: 环境变量")
        print("  export DASHSCOPY_API_KEY=your-api-key")
        print("  python examples/dashscope_quick_start.py\n")
        print("方式 3: .env 文件")
        print("  在项目根目录的 .env 文件中添加:")
        print("  DASHSCOPY_API_KEY=your-api-key\n")
        print("获取 API Key:")
        print("  https://dashscope.console.aliyun.com/apiKey")
        print("=" * 60)
        return
    
    # 创建 adapter
    adapter = DashScopeAdapter(api_key=api_key)
    
    print(f"Adapter base_url: {adapter.base_url}")
    print(f"Adapter config: {adapter.config}")
    
    # 调用模型生成文本
    print("\n正在调用通义千问...")
    try:
        start = time.time()
        result = await adapter.generate(
            prompt="你好，请介绍一下你自己。",
            model="qwen-turbo"  # 可选: qwen-turbo, qwen-plus, qwen-max
        )
        print(f'所花时间：{time.time() - start}ms')
        print(f"\n回答: {result.text}")
        print(f"\nToken 使用: 输入 {result.input_tokens}, 输出 {result.output_tokens}")
        
    except Exception as e:
        print(f"错误: {e}")
    
    # 关闭连接
    await adapter.aclose()


async def streaming_example():
    """流式输出示例"""
    
    api_key = get_api_key()
    if not api_key:
        return
    
    adapter = DashScopeAdapter(api_key=api_key)
    
    print("\n流式输出示例:")
    print("-" * 50)
    
    try:
        async for chunk in adapter.stream("讲一个笑话", "qwen-turbo"):
            print(chunk, end="", flush=True)
        print("\n" + "-" * 50)
        
    except Exception as e:
        print(f"错误: {e}")
    
    await adapter.aclose()


if __name__ == "__main__":
    print("=" * 50)
    print("DashScope 快速开始")
    print("=" * 50)
    
    asyncio.run(quick_start())
    # asyncio.run(streaming_example())
    
    print("\n提示:")
    print("• 如果遇到 'FreeTierOnly' 错误，说明免费额度已用完")
    print("• 需要在阿里云控制台开通付费模式")
    print("• 控制台地址: https://dashscope.console.aliyun.com/")
