"""
LLM Request Logger

记录每个 adapter 的请求日志，包括：
- 触发时间
- 耗时
- Token 数量（输入/输出）
- 模型信息
- 成功/失败状态

日志按日期分文件存储在 logs/{adapter_name}/{YYYY-MM-DD}.jsonl
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class RequestLogger:
    """请求日志记录器"""
    
    def __init__(self, adapter_name: str, enabled: Optional[bool] = None):
        """
        初始化日志记录器
        
        Args:
            adapter_name: Adapter 名称（用于日志文件路径）
            enabled: 是否启用日志，None 表示从环境变量读取
        """
        self.adapter_name = adapter_name
        
        # 从环境变量读取配置，默认启用
        if enabled is None:
            env_value = os.getenv("LLM_ADAPTER_LOGGING", "true").lower()
            self.enabled = env_value in ("true", "1", "yes", "on")
        else:
            self.enabled = enabled
        
        # 日志根目录
        self.log_root = Path(os.getenv("LLM_ADAPTER_LOG_DIR", "logs"))
        
        # 当前 adapter 的日志目录
        self.adapter_log_dir = self.log_root / adapter_name
        
        # 确保目录存在
        if self.enabled:
            self.adapter_log_dir.mkdir(parents=True, exist_ok=True)
    
    def log_request(
        self,
        model: str,
        prompt: str,
        response_text: Optional[str],
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        duration_ms: float,
        success: bool,
        error_message: Optional[str] = None,
        cost_usd: Optional[float] = None,
        provider: Optional[str] = None,
        **extra_fields
    ):
        """
        记录一次请求
        
        Args:
            model: 模型名称
            prompt: 输入提示词
            response_text: 响应文本
            input_tokens: 输入 token 数
            output_tokens: 输出 token 数
            duration_ms: 请求耗时（毫秒）
            success: 是否成功
            error_message: 错误信息（如果失败）
            cost_usd: 成本（USD）
            provider: 实际提供商
            **extra_fields: 其他额外字段
        """
        if not self.enabled:
            return
        
        try:
            # 获取当前日期作为文件名
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = self.adapter_log_dir / f"{today}.jsonl"
            
            # 构建日志记录
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "adapter": self.adapter_name,
                "model": model,
                "prompt_length": len(prompt) if prompt else 0,
                "prompt_preview": prompt[:100] if prompt else None,  # 只记录前100字符
                "response_length": len(response_text) if response_text else 0,
                "response_preview": response_text[:100] if response_text else None,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": (input_tokens or 0) + (output_tokens or 0),
                "duration_ms": round(duration_ms, 2),
                "success": success,
                "error_message": error_message,
                "cost_usd": cost_usd,
                "provider": provider,
            }
            
            # 添加额外字段
            log_entry.update(extra_fields)
            
            # 写入日志文件（追加模式）
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                
        except Exception as e:
            # 日志记录失败不应该影响主流程
            # 可以选择打印警告或静默失败
            print(f"Warning: Failed to write log: {e}")
    
    def log_stream_request(
        self,
        model: str,
        prompt: str,
        total_chunks: int,
        total_text_length: int,
        duration_ms: float,
        success: bool,
        error_message: Optional[str] = None,
        **extra_fields
    ):
        """
        记录流式请求
        
        Args:
            model: 模型名称
            prompt: 输入提示词
            total_chunks: 总块数
            total_text_length: 总文本长度
            duration_ms: 请求耗时（毫秒）
            success: 是否成功
            error_message: 错误信息
            **extra_fields: 其他额外字段
        """
        if not self.enabled:
            return
        
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = self.adapter_log_dir / f"{today}.jsonl"
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "adapter": self.adapter_name,
                "model": model,
                "prompt_length": len(prompt) if prompt else 0,
                "prompt_preview": prompt[:100] if prompt else None,
                "stream": True,
                "total_chunks": total_chunks,
                "total_text_length": total_text_length,
                "duration_ms": round(duration_ms, 2),
                "success": success,
                "error_message": error_message,
            }
            
            log_entry.update(extra_fields)
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                
        except Exception as e:
            print(f"Warning: Failed to write stream log: {e}")


# 全局日志记录器缓存
_loggers = {}


def get_logger(adapter_name: str) -> RequestLogger:
    """
    获取或创建指定 adapter 的日志记录器
    
    Args:
        adapter_name: Adapter 名称
        
    Returns:
        RequestLogger 实例
    """
    if adapter_name not in _loggers:
        _loggers[adapter_name] = RequestLogger(adapter_name)
    return _loggers[adapter_name]
