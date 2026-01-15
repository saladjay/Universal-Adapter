# Implementation Plan: LLM Adapter & Cost Controller

## Overview

本实现计划将多海外LLM统一接入与计费监控系统分解为可执行的编码任务。采用Python语言实现，使用异步编程模式处理LLM API调用。

## Tasks

- [x] 1. 项目结构和核心数据模型
  - [x] 1.1 创建项目目录结构和依赖配置
    - 创建 `llm_adapter/` 包目录
    - 更新 `pyproject.toml` 添加依赖（httpx, pyyaml, hypothesis）
    - _Requirements: 6.1_

  - [x] 1.2 实现核心数据类
    - 创建 `llm_adapter/models.py`
    - 实现 LLMRequest, LLMResponse, TokenUsage, PricingRule, UsageLog 数据类
    - _Requirements: 1.1, 1.2, 3.5, 4.1, 7.1_

  - [x] 1.3 编写数据模型属性测试

    - **Property 3: Token统计一致性**
    - **Validates: Requirements 3.5**

- [x] 2. 配置管理模块
  - [x] 2.1 实现ConfigManager
    - 创建 `llm_adapter/config.py`
    - 实现YAML配置加载
    - 实现环境变量替换
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x] 2.2 创建默认配置文件
    - 创建 `config.yaml` 模板
    - 包含所有provider配置和定价规则
    - _Requirements: 6.1, 6.3, 6.4_

  - [x] 2.3 编写配置管理属性测试

    - **Property 6: 配置往返一致性**
    - **Property 7: 无效配置错误处理**
    - **Validates: Requirements 6.1, 6.5**

- [x] 3. Checkpoint - 确保基础模块测试通过
  - 确保所有测试通过，如有问题请询问用户

- [x] 4. Provider适配器实现
  - [x] 4.1 实现ProviderAdapter抽象基类
    - 创建 `llm_adapter/adapters/base.py`
    - 定义 generate() 和 estimate_tokens() 抽象方法
    - _Requirements: 2.5_

  - [x] 4.2 实现OpenAI适配器
    - 创建 `llm_adapter/adapters/openai_adapter.py`
    - 实现API调用和Token提取
    - _Requirements: 2.1, 3.1_

  - [x] 4.3 实现Gemini适配器
    - 创建 `llm_adapter/adapters/gemini_adapter.py`
    - 实现API调用和Token提取
    - _Requirements: 2.2, 3.2_

  - [x] 4.4 实现Cloudflare适配器
    - 创建 `llm_adapter/adapters/cloudflare_adapter.py`
    - 实现API调用和Token估算
    - _Requirements: 2.3, 3.3_

  - [x] 4.5 实现HuggingFace适配器
    - 创建 `llm_adapter/adapters/huggingface_adapter.py`
    - 实现API调用和本地Token估算
    - _Requirements: 2.4, 3.4_

- [x] 5. 计费引擎
  - [x] 5.1 实现BillingEngine
    - 创建 `llm_adapter/billing.py`
    - 实现成本计算公式
    - 从配置加载定价规则
    - _Requirements: 4.1, 4.2, 4.3_

  - [x] 5.2 编写计费引擎属性测试

    - **Property 4: 计费公式正确性**
    - **Validates: Requirements 4.2, 4.3**

- [x] 6. 路由模块
  - [x] 6.1 实现Router
    - 创建 `llm_adapter/router.py`
    - 实现基于quality的路由策略
    - 实现降级逻辑
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [x] 6.2 编写路由策略属性测试

    - **Property 5: 路由策略正确性**
    - **Validates: Requirements 5.1, 5.2, 5.3**

- [x] 7. Checkpoint - 确保核心模块测试通过
  - 确保所有测试通过，如有问题请询问用户

- [x] 8. 使用日志模块
  - [x] 8.1 实现UsageLogger
    - 创建 `llm_adapter/logger.py`
    - 实现日志记录功能
    - 实现按用户和时间范围查询
    - _Requirements: 7.1, 7.2, 7.3_

  - [x] 8.2 编写日志完整性属性测试

    - **Property 8: 使用日志完整性**
    - **Validates: Requirements 7.1**

- [x] 9. LLMAdapter统一入口
  - [x] 9.1 实现LLMAdapter主类
    - 创建 `llm_adapter/adapter.py`
    - 整合Router、Provider、Billing、Logger
    - 实现 generate() 方法
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 9.2 编写请求验证属性测试

    - **Property 1: 请求参数验证**
    - **Property 2: 响应结构完整性**
    - **Validates: Requirements 1.1, 1.2, 1.3**

- [x] 10. 集成和入口点
  - [x] 10.1 创建包入口点
    - 更新 `llm_adapter/__init__.py`
    - 导出主要类和函数
    - _Requirements: 1.1_

  - [x] 10.2 更新main.py示例
    - 添加使用示例代码
    - 演示完整调用流程
    - _Requirements: 1.1, 1.2_

- [x] 11. Final Checkpoint - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户

## Notes

- 标记 `*` 的任务为可选任务，可跳过以加快MVP开发
- 每个任务都引用了具体的需求以便追溯
- 属性测试验证核心正确性属性
- 单元测试验证具体示例和边界情况
