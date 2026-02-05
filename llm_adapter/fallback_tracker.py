"""
Fallback tracker for recording region fallback events and timing.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class FallbackEvent:
    """Record of a single fallback event."""
    timestamp: datetime
    provider: str
    original_location: str
    fallback_location: str
    original_model: str
    fallback_model: str
    error_message: str
    fallback_duration_ms: float
    success: bool


@dataclass
class FallbackStats:
    """Statistics for fallback events."""
    total_fallbacks: int = 0
    successful_fallbacks: int = 0
    failed_fallbacks: int = 0
    total_duration_ms: float = 0.0
    average_duration_ms: float = 0.0
    events: List[FallbackEvent] = field(default_factory=list)
    
    def add_event(self, event: FallbackEvent):
        """Add a fallback event and update statistics."""
        self.events.append(event)
        self.total_fallbacks += 1
        self.total_duration_ms += event.fallback_duration_ms
        
        if event.success:
            self.successful_fallbacks += 1
        else:
            self.failed_fallbacks += 1
        
        if self.total_fallbacks > 0:
            self.average_duration_ms = self.total_duration_ms / self.total_fallbacks
    
    def get_summary(self) -> dict:
        """Get a summary of fallback statistics."""
        return {
            "total_fallbacks": self.total_fallbacks,
            "successful_fallbacks": self.successful_fallbacks,
            "failed_fallbacks": self.failed_fallbacks,
            "success_rate": (
                self.successful_fallbacks / self.total_fallbacks 
                if self.total_fallbacks > 0 else 0.0
            ),
            "total_duration_ms": round(self.total_duration_ms, 2),
            "average_duration_ms": round(self.average_duration_ms, 2),
        }


class FallbackTracker:
    """Global tracker for region fallback events."""
    
    def __init__(self):
        self.stats = FallbackStats()
    
    def record_fallback(
        self,
        provider: str,
        original_location: str,
        fallback_location: str,
        original_model: str,
        fallback_model: str,
        error_message: str,
        fallback_duration_ms: float,
        success: bool
    ) -> FallbackEvent:
        """
        Record a fallback event.
        
        Args:
            provider: Provider name (e.g., "gemini")
            original_location: Original region that failed
            fallback_location: Fallback region used
            original_model: Original model name
            fallback_model: Fallback model name
            error_message: Error message from original attempt
            fallback_duration_ms: Time taken for fallback in milliseconds
            success: Whether the fallback succeeded
            
        Returns:
            FallbackEvent object
        """
        event = FallbackEvent(
            timestamp=datetime.now(),
            provider=provider,
            original_location=original_location,
            fallback_location=fallback_location,
            original_model=original_model,
            fallback_model=fallback_model,
            error_message=error_message,
            fallback_duration_ms=fallback_duration_ms,
            success=success
        )
        
        self.stats.add_event(event)
        return event
    
    def get_stats(self) -> FallbackStats:
        """Get current fallback statistics."""
        return self.stats
    
    def get_recent_events(self, limit: int = 10) -> List[FallbackEvent]:
        """Get the most recent fallback events."""
        return self.stats.events[-limit:]
    
    def clear(self):
        """Clear all recorded events and statistics."""
        self.stats = FallbackStats()


# Global fallback tracker instance
_global_tracker: Optional[FallbackTracker] = None


def get_fallback_tracker() -> FallbackTracker:
    """Get the global fallback tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = FallbackTracker()
    return _global_tracker
