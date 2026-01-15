"""
Usage Logger module for recording and querying LLM usage logs.
"""

from datetime import datetime
from typing import Optional

from llm_adapter.models import UsageLog


class UsageLogger:
    """
    使用日志记录器，记录每次LLM调用的详细信息。
    
    支持功能：
    - 记录LLM调用日志
    - 按用户维度查询使用记录
    - 按时间范围查询使用记录
    """

    def __init__(self) -> None:
        """初始化日志记录器，使用内存存储"""
        self._logs: list[UsageLog] = []

    def log(
        self,
        user_id: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        timestamp: Optional[datetime] = None,
    ) -> UsageLog:
        """
        记录一次LLM调用的使用日志。
        
        Args:
            user_id: 用户ID
            provider: LLM提供商名称
            model: 使用的模型名称
            input_tokens: 输入Token数量
            output_tokens: 输出Token数量
            cost: 调用成本(USD)
            timestamp: 时间戳，默认为当前时间
            
        Returns:
            创建的UsageLog记录
        """
        if timestamp is None:
            timestamp = datetime.now()

        log_entry = UsageLog(
            user_id=user_id,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            timestamp=timestamp,
        )
        self._logs.append(log_entry)
        return log_entry

    def get_logs_by_user(self, user_id: str) -> list[UsageLog]:
        """
        按用户ID查询使用记录。
        
        Args:
            user_id: 用户ID
            
        Returns:
            该用户的所有使用日志列表
        """
        return [log for log in self._logs if log.user_id == user_id]

    def get_logs_by_time_range(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[UsageLog]:
        """
        按时间范围查询使用记录。
        
        Args:
            start_time: 开始时间（包含），None表示不限制开始时间
            end_time: 结束时间（包含），None表示不限制结束时间
            
        Returns:
            在指定时间范围内的使用日志列表
        """
        result = []
        for log in self._logs:
            if start_time is not None and log.timestamp < start_time:
                continue
            if end_time is not None and log.timestamp > end_time:
                continue
            result.append(log)
        return result

    def get_logs_by_user_and_time_range(
        self,
        user_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[UsageLog]:
        """
        按用户ID和时间范围查询使用记录。
        
        Args:
            user_id: 用户ID
            start_time: 开始时间（包含），None表示不限制开始时间
            end_time: 结束时间（包含），None表示不限制结束时间
            
        Returns:
            符合条件的使用日志列表
        """
        user_logs = self.get_logs_by_user(user_id)
        result = []
        for log in user_logs:
            if start_time is not None and log.timestamp < start_time:
                continue
            if end_time is not None and log.timestamp > end_time:
                continue
            result.append(log)
        return result

    def get_all_logs(self) -> list[UsageLog]:
        """
        获取所有使用日志。
        
        Returns:
            所有使用日志列表的副本
        """
        return list(self._logs)

    def get_user_total_cost(self, user_id: str) -> float:
        """
        获取用户的总成本。
        
        Args:
            user_id: 用户ID
            
        Returns:
            该用户的总成本(USD)
        """
        return sum(log.cost for log in self.get_logs_by_user(user_id))

    def get_user_total_tokens(self, user_id: str) -> tuple[int, int]:
        """
        获取用户的总Token使用量。
        
        Args:
            user_id: 用户ID
            
        Returns:
            (总输入Token数, 总输出Token数)
        """
        user_logs = self.get_logs_by_user(user_id)
        total_input = sum(log.input_tokens for log in user_logs)
        total_output = sum(log.output_tokens for log in user_logs)
        return total_input, total_output

    def clear(self) -> None:
        """清空所有日志记录"""
        self._logs.clear()
