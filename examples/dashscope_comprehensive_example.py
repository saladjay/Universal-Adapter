"""
阿里百炼 (DashScope) 完整使用示例

演示如何使用 DashScope adapter 调用通义千问系列模型。
包括基础调用、流式输出、多模型对比等功能。
"""

import asyncio
import os
from llm_adapter.adapters.dashscope_adapter import DashScopeAdapter


async def basic_generation():
    """基础文本生成示例"""
    
    print("=" * 70)
    print("基础文本生成示例")
    print("=" * 70)
    
    api_key = os.getenv("DASHSCOPY_API_KEY")
    if not api_key:
        print("\n⚠ 错误: 未设置 DASHSCOPY_API_KEY 环境变量")
        print("请设置: export DASHSCOPY_API_KEY=your_api_key")
        return
    
    # 初始化 adapter
    adapter = DashScopeAdapter(api_key=api_key)
    
    # 使用 qwen-turbo 模型（最快最便宜）
    model = "qwen-turbo"
    prompt = "请用一句话介绍一下杭州这座城市。"
    
    print(f"\n模型: {model}")
    print(f"提示词: {prompt}")
    print("\n生成中...")
    
    try:
        result = await adapter.generate(prompt, model)
        
        print("\n" + "=" * 70)
        print("生成结果")
        print("=" * 70)
        
        print(f"\n📝 生成文本:")
        print(f"  {result.text}")
        
        print(f"\n🔢 Token 使用:")
        print(f"  输入 tokens:  {result.input_tokens}")
        print(f"  输出 tokens:  {result.output_tokens}")
        print(f"  总计 tokens:  {(result.input_tokens or 0) + (result.output_tokens or 0)}")
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
    
    await adapter.aclose()


async def streaming_generation():
    """流式输出示例"""
    
    print("\n" + "=" * 70)
    print("流式输出示例")
    print("=" * 70)
    
    api_key = os.getenv("DASHSCOPY_API_KEY")
    if not api_key:
        return
    
    adapter = DashScopeAdapter(api_key=api_key)
    
    model = "qwen-turbo"
    prompt = "请写一首关于春天的五言绝句。"
    
    print(f"\n模型: {model}")
    print(f"提示词: {prompt}")
    print("\n流式输出:")
    print("-" * 70)
    
    try:
        print("  ", end="", flush=True)
        async for chunk in adapter.stream(prompt, model):
            print(chunk, end="", flush=True)
        print("\n" + "-" * 70)
        print("✓ 流式输出完成")
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
    
    await adapter.aclose()


async def multi_model_comparison():
    """多模型对比示例"""
    
    print("\n" + "=" * 70)
    print("多模型对比示例")
    print("=" * 70)
    
    api_key = os.getenv("DASHSCOPY_API_KEY")
    if not api_key:
        return
    
    adapter = DashScopeAdapter(api_key=api_key)
    
    # 通义千问系列模型
    models = [
        ("qwen-turbo", "通义千问-Turbo (最快最便宜)"),
        ("qwen-plus", "通义千问-Plus (平衡性能)"),
        ("qwen-max", "通义千问-Max (最强性能)"),
    ]
    
    prompt = "什么是人工智能？用一句话回答。"
    
    print(f"\n测试提示词: {prompt}")
    print(f"\n测试 {len(models)} 个模型...\n")
    
    results = []
    
    for model_id, model_name in models:
        print(f"测试 {model_name}...", end=" ", flush=True)
        try:
            result = await adapter.generate(prompt, model_id)
            results.append({
                "model_id": model_id,
                "model_name": model_name,
                "text": result.text,
                "input_tokens": result.input_tokens or 0,
                "output_tokens": result.output_tokens or 0,
                "success": True
            })
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")
            results.append({
                "model_id": model_id,
                "model_name": model_name,
                "text": None,
                "input_tokens": 0,
                "output_tokens": 0,
                "success": False
            })
    
    # 打印对比表格
    print("\n" + "=" * 70)
    print("模型对比结果")
    print("=" * 70)
    
    for r in results:
        if r["success"]:
            print(f"\n【{r['model_name']}】")
            print(f"  回答: {r['text']}")
            print(f"  Tokens: 输入 {r['input_tokens']} + 输出 {r['output_tokens']} = {r['input_tokens'] + r['output_tokens']}")
    
    await adapter.aclose()


async def chinese_english_mixed():
    """中英文混合测试"""
    
    print("\n" + "=" * 70)
    print("中英文混合测试")
    print("=" * 70)
    
    api_key = os.getenv("DASHSCOPY_API_KEY")
    if not api_key:
        return
    
    adapter = DashScopeAdapter(api_key=api_key)
    
    model = "qwen-turbo"
    
    test_cases = [
        ("纯中文", "介绍一下北京"),
        ("纯英文", "Introduce Beijing in one sentence"),
        ("中英混合", "请用中英文混合介绍 AI (Artificial Intelligence)"),
    ]
    
    print(f"\n使用模型: {model}\n")
    
    for test_name, prompt in test_cases:
        print(f"【{test_name}测试】")
        print(f"提示词: {prompt}")
        
        try:
            result = await adapter.generate(prompt, model)
            print(f"回答: {result.text}")
            print(f"Tokens: 输入 {result.input_tokens}, 输出 {result.output_tokens}")
            print()
        except Exception as e:
            print(f"✗ 错误: {e}\n")
    
    await adapter.aclose()


async def long_context_test():
    """长文本上下文测试"""
    
    print("\n" + "=" * 70)
    print("长文本上下文测试")
    print("=" * 70)
    
    api_key = os.getenv("DASHSCOPY_API_KEY")
    if not api_key:
        return
    
    adapter = DashScopeAdapter(api_key=api_key)
    
    model = "qwen-turbo"
    
    # 构造一个较长的上下文
    long_context = """
    人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，
    它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    
    人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。
    可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。
    人工智能可以对人的意识、思维的信息过程进行模拟。
    """
    
    prompt = f"{long_context}\n\n请根据上述内容，用一句话总结什么是人工智能。"
    
    print(f"\n模型: {model}")
    print(f"上下文长度: {len(long_context)} 字符")
    print("\n生成中...")
    
    try:
        result = await adapter.generate(prompt, model)
        
        print(f"\n📝 总结:")
        print(f"  {result.text}")
        
        print(f"\n🔢 Token 统计:")
        print(f"  输入 tokens: {result.input_tokens} (包含长上下文)")
        print(f"  输出 tokens: {result.output_tokens}")
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
    
    await adapter.aclose()


async def error_handling_demo():
    """错误处理示例"""
    
    print("\n" + "=" * 70)
    print("错误处理示例")
    print("=" * 70)
    
    api_key = os.getenv("DASHSCOPY_API_KEY")
    if not api_key:
        return
    
    adapter = DashScopeAdapter(api_key=api_key)
    
    # 测试 1: 无效的模型名称
    print("\n测试 1: 使用无效的模型名称")
    try:
        result = await adapter.generate("你好", "invalid-model-name")
        print(f"✓ 成功: {result.text}")
    except Exception as e:
        print(f"✗ 预期的错误: {e}")
    
    # 测试 2: 空提示词
    print("\n测试 2: 使用空提示词")
    try:
        result = await adapter.generate("", "qwen-turbo")
        print(f"✓ 成功: {result.text}")
    except Exception as e:
        print(f"✗ 错误: {e}")
    
    # 测试 3: 超长提示词（可能触发限制）
    print("\n测试 3: 使用超长提示词")
    very_long_prompt = "测试" * 10000  # 20000 字符
    try:
        result = await adapter.generate(very_long_prompt, "qwen-turbo")
        print(f"✓ 成功处理超长文本")
    except Exception as e:
        print(f"✗ 预期的错误: {e}")
    
    await adapter.aclose()


async def main():
    """运行所有示例"""
    
    print("\n" + "=" * 70)
    print("阿里百炼 (DashScope) 完整使用示例")
    print("=" * 70)
    
    # 检查 API Key
    if not os.getenv("DASHSCOPY_API_KEY"):
        print("\n⚠ 错误: 未设置 DASHSCOPY_API_KEY 环境变量")
        print("\n请先设置环境变量:")
        print("  export DASHSCOPY_API_KEY=your_api_key")
        print("\n或在 .env 文件中添加:")
        print("  DASHSCOPY_API_KEY=your_api_key")
        return
    
    # 运行所有示例
    await basic_generation()
    await streaming_generation()
    await multi_model_comparison()
    await chinese_english_mixed()
    await long_context_test()
    await error_handling_demo()
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("\n✅ DashScope (通义千问) 主要特点:")
    print("  • 支持中文优化的大语言模型")
    print("  • 提供多个性能级别: turbo (快), plus (平衡), max (强)")
    print("  • 支持流式输出，适合实时交互")
    print("  • 返回准确的 token 使用统计")
    print("  • 适合中文场景和中英文混合场景")
    print("\n💡 推荐使用场景:")
    print("  • qwen-turbo: 快速响应、成本敏感的场景")
    print("  • qwen-plus: 需要平衡性能和成本的场景")
    print("  • qwen-max: 需要最佳质量的场景")
    print("\n📚 更多信息:")
    print("  • 官方文档: https://help.aliyun.com/zh/dashscope/")
    print("  • API 参考: https://help.aliyun.com/zh/dashscope/developer-reference/api-details")


if __name__ == "__main__":
    asyncio.run(main())
