"""
Token usage tracking and cost optimization utilities.
"""
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from .interfaces import TokenUsage, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class UsageSession:
    """Represents a usage session with multiple LLM calls."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_tokens: int = 0
    total_cost: float = 0.0
    call_count: int = 0
    model: Optional[str] = None
    provider: Optional[str] = None


@dataclass
class UsageStats:
    """Aggregated usage statistics."""
    total_tokens: int
    total_cost: float
    total_calls: int
    average_tokens_per_call: float
    average_cost_per_call: float
    most_expensive_call: float
    peak_usage_hour: Optional[int] = None


class TokenUsageTracker:
    """Tracks token usage and costs across LLM calls."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or "usage_data.json"
        self.current_session: Optional[UsageSession] = None
        self.usage_history: List[Dict[str, Any]] = []
        self._load_history()
    
    def start_session(self, session_id: str, model: str, provider: str) -> None:
        """Start a new usage tracking session."""
        if self.current_session and not self.current_session.end_time:
            self.end_session()
        
        self.current_session = UsageSession(
            session_id=session_id,
            start_time=datetime.now(),
            model=model,
            provider=provider
        )
        logger.info(f"Started usage tracking session: {session_id}")
    
    def track_response(self, response: LLMResponse) -> None:
        """Track token usage from an LLM response."""
        if not self.current_session:
            logger.warning("No active session. Starting default session.")
            self.start_session("default", response.model, "unknown")
        
        usage = response.token_usage
        self.current_session.total_tokens += usage.total_tokens
        self.current_session.call_count += 1
        
        if usage.cost_usd:
            self.current_session.total_cost += usage.cost_usd
        
        logger.debug(f"Tracked {usage.total_tokens} tokens, cost: ${usage.cost_usd or 0:.4f}")
    
    def end_session(self) -> UsageSession:
        """End the current session and save to history."""
        if not self.current_session:
            raise ValueError("No active session to end")
        
        self.current_session.end_time = datetime.now()
        
        # Save to history
        self.usage_history.append(asdict(self.current_session))
        self._save_history()
        
        session = self.current_session
        self.current_session = None
        
        logger.info(f"Ended session {session.session_id}: "
                   f"{session.total_tokens} tokens, ${session.total_cost:.4f}")
        
        return session
    
    def get_current_session_stats(self) -> Optional[UsageSession]:
        """Get current session statistics."""
        return self.current_session
    
    def get_usage_stats(self, 
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       model: Optional[str] = None) -> UsageStats:
        """Get aggregated usage statistics."""
        filtered_history = self._filter_history(start_date, end_date, model)
        
        if not filtered_history:
            return UsageStats(0, 0.0, 0, 0.0, 0.0, 0.0)
        
        total_tokens = sum(session["total_tokens"] for session in filtered_history)
        total_cost = sum(session["total_cost"] for session in filtered_history)
        total_calls = sum(session["call_count"] for session in filtered_history)
        
        avg_tokens = total_tokens / len(filtered_history) if filtered_history else 0
        avg_cost = total_cost / len(filtered_history) if filtered_history else 0
        max_cost = max(session["total_cost"] for session in filtered_history)
        
        # Find peak usage hour
        peak_hour = self._find_peak_usage_hour(filtered_history)
        
        return UsageStats(
            total_tokens=total_tokens,
            total_cost=total_cost,
            total_calls=total_calls,
            average_tokens_per_call=avg_tokens,
            average_cost_per_call=avg_cost,
            most_expensive_call=max_cost,
            peak_usage_hour=peak_hour
        )
    
    def get_cost_breakdown_by_model(self) -> Dict[str, Dict[str, float]]:
        """Get cost breakdown by model."""
        breakdown = {}
        
        for session in self.usage_history:
            model = session.get("model", "unknown")
            if model not in breakdown:
                breakdown[model] = {"tokens": 0, "cost": 0.0, "calls": 0}
            
            breakdown[model]["tokens"] += session["total_tokens"]
            breakdown[model]["cost"] += session["total_cost"]
            breakdown[model]["calls"] += session["call_count"]
        
        return breakdown
    
    def get_daily_usage(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get daily usage for the last N days."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        daily_usage = {}
        
        for session in self.usage_history:
            start_time = session["start_time"]
            if isinstance(start_time, str):
                session_date = datetime.fromisoformat(start_time).date()
            else:
                session_date = start_time.date()
            
            if start_date.date() <= session_date <= end_date.date():
                date_str = session_date.isoformat()
                
                if date_str not in daily_usage:
                    daily_usage[date_str] = {"tokens": 0, "cost": 0.0, "calls": 0}
                
                daily_usage[date_str]["tokens"] += session["total_tokens"]
                daily_usage[date_str]["cost"] += session["total_cost"]
                daily_usage[date_str]["calls"] += session["call_count"]
        
        return [{"date": date, **stats} for date, stats in sorted(daily_usage.items())]
    
    def optimize_costs(self) -> Dict[str, Any]:
        """Provide cost optimization recommendations."""
        stats = self.get_usage_stats()
        model_breakdown = self.get_cost_breakdown_by_model()
        
        recommendations = []
        
        # Check for high-cost models
        if model_breakdown:
            most_expensive_model = max(model_breakdown.items(), 
                                     key=lambda x: x[1]["cost"])
            
            if most_expensive_model[1]["cost"] > 10.0:  # $10 threshold
                recommendations.append({
                    "type": "model_optimization",
                    "message": f"Consider switching from {most_expensive_model[0]} to a cheaper model",
                    "potential_savings": most_expensive_model[1]["cost"] * 0.3  # Estimate 30% savings
                })
        
        # Check for high token usage
        if stats.average_tokens_per_call > 2000:
            recommendations.append({
                "type": "prompt_optimization",
                "message": "Average token usage is high. Consider optimizing prompts to be more concise",
                "current_avg": stats.average_tokens_per_call
            })
        
        # Check for peak usage patterns
        if stats.peak_usage_hour is not None:
            recommendations.append({
                "type": "usage_pattern",
                "message": f"Peak usage at hour {stats.peak_usage_hour}. Consider load balancing",
                "peak_hour": stats.peak_usage_hour
            })
        
        return {
            "total_cost": stats.total_cost,
            "recommendations": recommendations,
            "potential_monthly_savings": sum(r.get("potential_savings", 0) for r in recommendations)
        }
    
    def export_usage_report(self, filepath: str, format: str = "json") -> None:
        """Export usage report to file."""
        stats = self.get_usage_stats()
        model_breakdown = self.get_cost_breakdown_by_model()
        daily_usage = self.get_daily_usage()
        optimization = self.optimize_costs()
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": asdict(stats),
            "model_breakdown": model_breakdown,
            "daily_usage": daily_usage,
            "optimization": optimization,
            "raw_sessions": self.usage_history
        }
        
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Usage report exported to {filepath}")
    
    def _filter_history(self, 
                       start_date: Optional[datetime],
                       end_date: Optional[datetime],
                       model: Optional[str]) -> List[Dict[str, Any]]:
        """Filter usage history by date range and model."""
        filtered = self.usage_history
        
        if start_date:
            filtered = [s for s in filtered 
                       if (datetime.fromisoformat(s["start_time"]) if isinstance(s["start_time"], str) 
                           else s["start_time"]) >= start_date]
        
        if end_date:
            filtered = [s for s in filtered 
                       if (datetime.fromisoformat(s["start_time"]) if isinstance(s["start_time"], str) 
                           else s["start_time"]) <= end_date]
        
        if model:
            filtered = [s for s in filtered if s.get("model") == model]
        
        return filtered
    
    def _find_peak_usage_hour(self, sessions: List[Dict[str, Any]]) -> Optional[int]:
        """Find the hour with peak usage."""
        if not sessions:
            return None
        
        hourly_usage = {}
        
        for session in sessions:
            start_time = session["start_time"]
            if isinstance(start_time, str):
                hour = datetime.fromisoformat(start_time).hour
            else:
                hour = start_time.hour
            hourly_usage[hour] = hourly_usage.get(hour, 0) + session["total_tokens"]
        
        if not hourly_usage:
            return None
        
        return max(hourly_usage.items(), key=lambda x: x[1])[0]
    
    def _load_history(self) -> None:
        """Load usage history from storage."""
        try:
            if Path(self.storage_path).exists():
                with open(self.storage_path, 'r') as f:
                    self.usage_history = json.load(f)
                logger.debug(f"Loaded {len(self.usage_history)} usage sessions")
        except Exception as e:
            logger.warning(f"Failed to load usage history: {e}")
            self.usage_history = []
    
    def _save_history(self) -> None:
        """Save usage history to storage."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.usage_history, f, indent=2, default=str)
            logger.debug("Usage history saved")
        except Exception as e:
            logger.error(f"Failed to save usage history: {e}")


class PromptOptimizer:
    """Utilities for optimizing prompts to reduce token usage."""
    
    @staticmethod
    def estimate_token_reduction(original_prompt: str, optimized_prompt: str) -> Dict[str, Any]:
        """Estimate token reduction from prompt optimization."""
        # Simple character-based estimation (rough approximation)
        original_tokens = len(original_prompt) // 4
        optimized_tokens = len(optimized_prompt) // 4
        
        reduction = original_tokens - optimized_tokens
        reduction_percent = (reduction / original_tokens * 100) if original_tokens > 0 else 0
        
        return {
            "original_tokens": original_tokens,
            "optimized_tokens": optimized_tokens,
            "token_reduction": reduction,
            "reduction_percent": reduction_percent
        }
    
    @staticmethod
    def suggest_prompt_optimizations(prompt: str) -> List[Dict[str, str]]:
        """Suggest optimizations for a given prompt."""
        suggestions = []
        
        # Check for redundant phrases
        redundant_phrases = [
            "please", "kindly", "if you would", "I would like you to",
            "could you please", "would you mind"
        ]
        
        for phrase in redundant_phrases:
            if phrase.lower() in prompt.lower():
                suggestions.append({
                    "type": "remove_redundancy",
                    "suggestion": f"Remove redundant phrase: '{phrase}'",
                    "impact": "low"
                })
        
        # Check for verbose instructions
        if len(prompt) > 300:  # Lower threshold for testing
            suggestions.append({
                "type": "length_reduction",
                "suggestion": "Consider breaking down long prompts into shorter, focused ones",
                "impact": "high"
            })
        
        # Check for repetitive examples
        if prompt.count("Example") > 3:
            suggestions.append({
                "type": "example_optimization",
                "suggestion": "Consider reducing the number of examples or making them more concise",
                "impact": "medium"
            })
        
        return suggestions
    
    @staticmethod
    def compress_prompt(prompt: str, target_reduction: float = 0.2) -> str:
        """Compress a prompt by removing unnecessary words."""
        # Simple compression by removing common filler words
        filler_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "very", "really", "quite", "rather", "pretty"
        ]
        
        words = prompt.split()
        original_length = len(words)
        target_length = int(original_length * (1 - target_reduction))
        
        # Remove filler words until we reach target length
        compressed_words = []
        important_words = []
        filler_removed = []
        
        # First pass: separate important words from filler words
        for word in words:
            if word.lower() in filler_words:
                filler_removed.append(word)
            else:
                important_words.append(word)
        
        # Add important words first
        compressed_words.extend(important_words)
        
        # Add back some filler words if we haven't reached minimum length
        remaining_space = target_length - len(compressed_words)
        if remaining_space > 0:
            compressed_words.extend(filler_removed[:remaining_space])
        
        return " ".join(compressed_words)