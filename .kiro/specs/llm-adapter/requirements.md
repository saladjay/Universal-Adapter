# Requirements Document

## Introduction

本系统是一个多海外LLM统一接入与计费监控系统（LLM Adapter & Cost Controller），旨在为上层应用提供统一的LLM调用接口，自动完成模型选择、调用、Token统计和成本计算。系统支持OpenAI、Google Gemini、Cloudflare Workers AI、HuggingFace等多个LLM平台。

## Glossary

- **LLM_Adapter**: 统一LLM接入层，对外提供单一API，对内调度多个海外大模型平台
- **Provider_Adapter**: 各LLM平台的适配器，实现统一接口规范
- **Token_Counter**: Token统计模块，负责统计各平台的Token使用量
- **Billing_Engine**: 计费引擎，根据Token使用量实时计算成本
- **Router**: 路由模块，根据质量等级和场景选择最优LLM平台
- **Usage_Logger**: 使用日志记录器，记录每次调用的详细信息
- **Config_Manager**: 配置管理器，管理各平台的API密钥和模型配置

## Requirements

### Requirement 1: 统一LLM调用接口

**User Story:** As a 上层应用开发者, I want 通过统一的API调用多个LLM平台, so that 无需关心底层平台差异即可获得LLM响应。

#### Acceptance Criteria

1. WHEN 应用发起LLM请求时 THEN THE LLM_Adapter SHALL 接收包含userId、prompt、scene和quality的请求参数
2. WHEN LLM调用成功时 THEN THE LLM_Adapter SHALL 返回包含text、model、provider、inputTokens、outputTokens和costUSD的响应
3. IF 请求参数缺失或无效 THEN THE LLM_Adapter SHALL 返回明确的错误信息

### Requirement 2: Provider适配器实现

**User Story:** As a 系统管理员, I want 系统支持多个LLM平台, so that 可以根据需求灵活选择不同的模型提供商。

#### Acceptance Criteria

1. THE Provider_Adapter SHALL 为OpenAI平台实现统一的generate接口
2. THE Provider_Adapter SHALL 为Gemini平台实现统一的generate接口
3. THE Provider_Adapter SHALL 为Cloudflare平台实现统一的generate接口
4. THE Provider_Adapter SHALL 为HuggingFace平台实现统一的generate接口
5. WHEN 新增LLM平台时 THEN THE Provider_Adapter SHALL 遵循统一的接口规范实现适配器

### Requirement 3: Token统计功能

**User Story:** As a 运营人员, I want 准确统计每次调用的Token使用量, so that 可以进行成本分析和用户额度管理。

#### Acceptance Criteria

1. WHEN 调用OpenAI时 THEN THE Token_Counter SHALL 从API响应中提取prompt_tokens和completion_tokens
2. WHEN 调用Gemini时 THEN THE Token_Counter SHALL 从API响应中提取tokenMetadata
3. WHEN 调用Cloudflare时 THEN THE Token_Counter SHALL 通过neurons到token的估算表计算Token数量
4. WHEN 调用HuggingFace时 THEN THE Token_Counter SHALL 使用本地tokenizer估算Token数量
5. THE Token_Counter SHALL 返回统一格式的Token统计结果，包含inputTokens、outputTokens和totalTokens

### Requirement 4: 实时计费功能

**User Story:** As a 财务人员, I want 实时计算每次LLM调用的成本, so that 可以进行成本控制和账单生成。

#### Acceptance Criteria

1. THE Billing_Engine SHALL 维护各平台各模型的定价规则，包含inputCostPer1M和outputCostPer1M
2. WHEN Token统计完成时 THEN THE Billing_Engine SHALL 根据公式计算成本：cost = (inputTokens / 1_000_000) * inputPrice + (outputTokens / 1_000_000) * outputPrice
3. THE Billing_Engine SHALL 返回以USD为单位的成本值

### Requirement 5: 智能路由策略

**User Story:** As a 产品经理, I want 系统根据质量等级自动选择最优LLM, so that 可以在成本和质量之间取得平衡。

#### Acceptance Criteria

1. WHEN quality为"low"时 THEN THE Router SHALL 选择Cloudflare或HuggingFace平台
2. WHEN quality为"medium"时 THEN THE Router SHALL 选择OpenAI-mini或Gemini-flash模型
3. WHEN quality为"high"时 THEN THE Router SHALL 选择OpenAI或Gemini-pro模型
4. IF 首选平台不可用 THEN THE Router SHALL 自动降级到备选平台

### Requirement 6: 配置管理

**User Story:** As a 系统管理员, I want 通过配置文件管理各平台的API密钥和模型设置, so that 可以灵活调整系统配置而无需修改代码。

#### Acceptance Criteria

1. THE Config_Manager SHALL 支持从YAML配置文件加载配置
2. THE Config_Manager SHALL 支持通过环境变量注入敏感信息（如API密钥）
3. THE Config_Manager SHALL 支持配置默认provider
4. THE Config_Manager SHALL 支持为每个provider配置多个模型（cheap、normal、premium）
5. WHEN 配置文件格式错误时 THEN THE Config_Manager SHALL 返回明确的错误信息

### Requirement 7: 使用日志记录

**User Story:** As a 数据分析师, I want 记录每次LLM调用的详细信息, so that 可以进行用户行为分析和成本优化。

#### Acceptance Criteria

1. WHEN LLM调用完成时 THEN THE Usage_Logger SHALL 记录userId、provider、model、inputTokens、outputTokens、cost和timestamp
2. THE Usage_Logger SHALL 支持按用户维度查询使用记录
3. THE Usage_Logger SHALL 支持按时间范围查询使用记录
