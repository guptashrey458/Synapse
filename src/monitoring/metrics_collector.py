"""
Comprehensive metrics collection system for resolution success rates and system performance.
"""
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
import json


class MetricCategory(Enum):
    """Categories of metrics collected."""
    SYSTEM = "system"
    RESOLUTION = "resolution"
    TOOL = "tool"
    LLM = "llm"
    REASONING = "reasoning"
    ERROR = "error"


@dataclass
class SystemMetrics:
    """System-level performance metrics."""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_usage_mb: float
    active_threads: int
    disk_usage_percent: float
    network_io_bytes: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_percent": self.memory_usage_percent,
            "memory_usage_mb": self.memory_usage_mb,
            "active_threads": self.active_threads,
            "disk_usage_percent": self.disk_usage_percent,
            "network_io_bytes": self.network_io_bytes
        }


@dataclass
class ResolutionMetrics:
    """Metrics for delivery disruption resolution."""
    timestamp: datetime
    scenario_id: str
    scenario_type: str
    resolution_time_seconds: float
    success: bool
    confidence_score: float
    steps_count: int
    tools_used: List[str]
    llm_calls: int
    total_tokens: int
    error_count: int
    customer_satisfaction: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "scenario_id": self.scenario_id,
            "scenario_type": self.scenario_type,
            "resolution_time_seconds": self.resolution_time_seconds,
            "success": self.success,
            "confidence_score": self.confidence_score,
            "steps_count": self.steps_count,
            "tools_used": self.tools_used,
            "llm_calls": self.llm_calls,
            "total_tokens": self.total_tokens,
            "error_count": self.error_count,
            "customer_satisfaction": self.customer_satisfaction
        }


@dataclass
class ToolMetrics:
    """Metrics for individual tool performance."""
    timestamp: datetime
    tool_name: str
    operation: str
    execution_time_seconds: float
    success: bool
    input_size_bytes: int
    output_size_bytes: int
    retry_count: int
    error_message: Optional[str] = None
    cache_hit: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "tool_name": self.tool_name,
            "operation": self.operation,
            "execution_time_seconds": self.execution_time_seconds,
            "success": self.success,
            "input_size_bytes": self.input_size_bytes,
            "output_size_bytes": self.output_size_bytes,
            "retry_count": self.retry_count,
            "error_message": self.error_message,
            "cache_hit": self.cache_hit
        }


@dataclass
class LLMMetrics:
    """Metrics for LLM provider interactions."""
    timestamp: datetime
    provider: str
    model: str
    operation: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time_seconds: float
    success: bool
    temperature: float
    max_tokens: int
    cost_estimate: Optional[float] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "provider": self.provider,
            "model": self.model,
            "operation": self.operation,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "response_time_seconds": self.response_time_seconds,
            "success": self.success,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "cost_estimate": self.cost_estimate,
            "error_message": self.error_message
        }


class MetricsAggregator:
    """Aggregates and analyzes collected metrics."""
    
    def __init__(self, max_metrics_per_category: int = 10000):
        self.max_metrics_per_category = max_metrics_per_category
        
        # Storage for different metric types
        self.system_metrics: deque = deque(maxlen=max_metrics_per_category)
        self.resolution_metrics: deque = deque(maxlen=max_metrics_per_category)
        self.tool_metrics: deque = deque(maxlen=max_metrics_per_category)
        self.llm_metrics: deque = deque(maxlen=max_metrics_per_category)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Aggregated statistics cache
        self._stats_cache: Dict[str, Any] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=5)
    
    def record_system_metrics(self, metrics: SystemMetrics):
        """Record system performance metrics."""
        with self.lock:
            self.system_metrics.append(metrics)
            self._invalidate_cache("system")
    
    def record_resolution_metrics(self, metrics: ResolutionMetrics):
        """Record resolution performance metrics."""
        with self.lock:
            self.resolution_metrics.append(metrics)
            self._invalidate_cache("resolution")
    
    def record_tool_metrics(self, metrics: ToolMetrics):
        """Record tool execution metrics."""
        with self.lock:
            self.tool_metrics.append(metrics)
            self._invalidate_cache("tool")
    
    def record_llm_metrics(self, metrics: LLMMetrics):
        """Record LLM interaction metrics."""
        with self.lock:
            self.llm_metrics.append(metrics)
            self._invalidate_cache("llm")
    
    def get_resolution_success_rate(self, hours: int = 24) -> Dict[str, Any]:
        """Get resolution success rate statistics."""
        cache_key = f"resolution_success_rate_{hours}"
        cached = self._get_cached_stats(cache_key)
        if cached:
            return cached
        
        since = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            recent_resolutions = [
                m for m in self.resolution_metrics 
                if m.timestamp >= since
            ]
        
        if not recent_resolutions:
            return {"no_data": True, "period_hours": hours}
        
        total_resolutions = len(recent_resolutions)
        successful_resolutions = sum(1 for m in recent_resolutions if m.success)
        
        # Group by scenario type
        by_scenario_type = defaultdict(list)
        for m in recent_resolutions:
            by_scenario_type[m.scenario_type].append(m)
        
        scenario_stats = {}
        for scenario_type, metrics in by_scenario_type.items():
            successful = sum(1 for m in metrics if m.success)
            scenario_stats[scenario_type] = {
                "total": len(metrics),
                "successful": successful,
                "success_rate": successful / len(metrics) * 100,
                "avg_resolution_time": statistics.mean([m.resolution_time_seconds for m in metrics]),
                "avg_confidence": statistics.mean([m.confidence_score for m in metrics])
            }
        
        stats = {
            "period_hours": hours,
            "total_resolutions": total_resolutions,
            "successful_resolutions": successful_resolutions,
            "overall_success_rate": successful_resolutions / total_resolutions * 100,
            "average_resolution_time": statistics.mean([m.resolution_time_seconds for m in recent_resolutions]),
            "median_resolution_time": statistics.median([m.resolution_time_seconds for m in recent_resolutions]),
            "average_confidence": statistics.mean([m.confidence_score for m in recent_resolutions]),
            "by_scenario_type": scenario_stats
        }
        
        self._cache_stats(cache_key, stats)
        return stats
    
    def get_tool_performance_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get tool performance statistics."""
        cache_key = f"tool_performance_{hours}"
        cached = self._get_cached_stats(cache_key)
        if cached:
            return cached
        
        since = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            recent_tools = [
                m for m in self.tool_metrics 
                if m.timestamp >= since
            ]
        
        if not recent_tools:
            return {"no_data": True, "period_hours": hours}
        
        # Group by tool name
        by_tool = defaultdict(list)
        for m in recent_tools:
            by_tool[m.tool_name].append(m)
        
        tool_stats = {}
        for tool_name, metrics in by_tool.items():
            successful = [m for m in metrics if m.success]
            failed = [m for m in metrics if not m.success]
            
            tool_stats[tool_name] = {
                "total_calls": len(metrics),
                "successful_calls": len(successful),
                "failed_calls": len(failed),
                "success_rate": len(successful) / len(metrics) * 100,
                "avg_execution_time": statistics.mean([m.execution_time_seconds for m in metrics]),
                "p95_execution_time": self._percentile([m.execution_time_seconds for m in metrics], 95),
                "avg_retry_count": statistics.mean([m.retry_count for m in metrics]),
                "cache_hit_rate": sum(1 for m in metrics if m.cache_hit) / len(metrics) * 100,
                "common_errors": self._get_common_errors([m for m in metrics if not m.success])
            }
        
        stats = {
            "period_hours": hours,
            "total_tool_calls": len(recent_tools),
            "unique_tools": len(by_tool),
            "overall_success_rate": sum(1 for m in recent_tools if m.success) / len(recent_tools) * 100,
            "by_tool": tool_stats
        }
        
        self._cache_stats(cache_key, stats)
        return stats
    
    def get_llm_usage_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get LLM usage statistics."""
        cache_key = f"llm_usage_{hours}"
        cached = self._get_cached_stats(cache_key)
        if cached:
            return cached
        
        since = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            recent_llm = [
                m for m in self.llm_metrics 
                if m.timestamp >= since
            ]
        
        if not recent_llm:
            return {"no_data": True, "period_hours": hours}
        
        total_tokens = sum(m.total_tokens for m in recent_llm)
        total_cost = sum(m.cost_estimate for m in recent_llm if m.cost_estimate)
        
        # Group by provider and model
        by_provider = defaultdict(list)
        by_model = defaultdict(list)
        
        for m in recent_llm:
            by_provider[m.provider].append(m)
            by_model[f"{m.provider}:{m.model}"].append(m)
        
        provider_stats = {}
        for provider, metrics in by_provider.items():
            provider_stats[provider] = {
                "calls": len(metrics),
                "total_tokens": sum(m.total_tokens for m in metrics),
                "avg_response_time": statistics.mean([m.response_time_seconds for m in metrics]),
                "success_rate": sum(1 for m in metrics if m.success) / len(metrics) * 100,
                "cost_estimate": sum(m.cost_estimate for m in metrics if m.cost_estimate)
            }
        
        model_stats = {}
        for model_key, metrics in by_model.items():
            model_stats[model_key] = {
                "calls": len(metrics),
                "total_tokens": sum(m.total_tokens for m in metrics),
                "avg_tokens_per_call": statistics.mean([m.total_tokens for m in metrics]),
                "avg_response_time": statistics.mean([m.response_time_seconds for m in metrics]),
                "success_rate": sum(1 for m in metrics if m.success) / len(metrics) * 100
            }
        
        stats = {
            "period_hours": hours,
            "total_calls": len(recent_llm),
            "total_tokens": total_tokens,
            "total_cost_estimate": total_cost,
            "avg_tokens_per_call": total_tokens / len(recent_llm),
            "avg_response_time": statistics.mean([m.response_time_seconds for m in recent_llm]),
            "overall_success_rate": sum(1 for m in recent_llm if m.success) / len(recent_llm) * 100,
            "by_provider": provider_stats,
            "by_model": model_stats
        }
        
        self._cache_stats(cache_key, stats)
        return stats
    
    def get_system_health_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get system health statistics."""
        cache_key = f"system_health_{hours}"
        cached = self._get_cached_stats(cache_key)
        if cached:
            return cached
        
        since = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            recent_system = [
                m for m in self.system_metrics 
                if m.timestamp >= since
            ]
        
        if not recent_system:
            return {"no_data": True, "period_hours": hours}
        
        cpu_values = [m.cpu_usage_percent for m in recent_system]
        memory_values = [m.memory_usage_percent for m in recent_system]
        thread_values = [m.active_threads for m in recent_system]
        
        stats = {
            "period_hours": hours,
            "data_points": len(recent_system),
            "cpu_usage": {
                "current": cpu_values[-1] if cpu_values else 0,
                "average": statistics.mean(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values),
                "p95": self._percentile(cpu_values, 95)
            },
            "memory_usage": {
                "current": memory_values[-1] if memory_values else 0,
                "average": statistics.mean(memory_values),
                "max": max(memory_values),
                "min": min(memory_values),
                "p95": self._percentile(memory_values, 95)
            },
            "active_threads": {
                "current": thread_values[-1] if thread_values else 0,
                "average": statistics.mean(thread_values),
                "max": max(thread_values),
                "min": min(thread_values)
            }
        }
        
        self._cache_stats(cache_key, stats)
        return stats
    
    def get_comprehensive_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive metrics report."""
        return {
            "report_timestamp": datetime.now().isoformat(),
            "period_hours": hours,
            "resolution_performance": self.get_resolution_success_rate(hours),
            "tool_performance": self.get_tool_performance_stats(hours),
            "llm_usage": self.get_llm_usage_stats(hours),
            "system_health": self.get_system_health_stats(hours)
        }
    
    def get_trend_analysis(self, metric_type: str, hours: int = 24, 
                          window_size: int = 60) -> Dict[str, Any]:
        """Get trend analysis for a specific metric type."""
        since = datetime.now() - timedelta(hours=hours)
        
        if metric_type == "resolution_success_rate":
            return self._analyze_resolution_trends(since, window_size)
        elif metric_type == "tool_performance":
            return self._analyze_tool_trends(since, window_size)
        elif metric_type == "llm_usage":
            return self._analyze_llm_trends(since, window_size)
        elif metric_type == "system_health":
            return self._analyze_system_trends(since, window_size)
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
    
    def export_metrics(self, filepath: str, hours: int = 24, format: str = "json"):
        """Export metrics to file."""
        if format.lower() == "json":
            self._export_json(filepath, hours)
        elif format.lower() == "csv":
            self._export_csv(filepath, hours)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def _get_common_errors(self, failed_metrics: List[ToolMetrics]) -> List[Dict[str, Any]]:
        """Get common error patterns from failed metrics."""
        error_counts = defaultdict(int)
        
        for metric in failed_metrics:
            if metric.error_message:
                # Simplify error message for grouping
                error_key = metric.error_message[:100]  # First 100 chars
                error_counts[error_key] += 1
        
        # Return top 5 errors
        return [
            {"error": error, "count": count}
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
    
    def _get_cached_stats(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached statistics if not expired."""
        if cache_key in self._stats_cache:
            if cache_key in self._cache_expiry:
                if datetime.now() < self._cache_expiry[cache_key]:
                    return self._stats_cache[cache_key]
                else:
                    # Expired, remove from cache
                    del self._stats_cache[cache_key]
                    del self._cache_expiry[cache_key]
        return None
    
    def _cache_stats(self, cache_key: str, stats: Dict[str, Any]):
        """Cache statistics with expiry."""
        self._stats_cache[cache_key] = stats
        self._cache_expiry[cache_key] = datetime.now() + self._cache_ttl
    
    def _invalidate_cache(self, category: str):
        """Invalidate cache entries for a category."""
        keys_to_remove = [key for key in self._stats_cache.keys() if category in key]
        for key in keys_to_remove:
            if key in self._stats_cache:
                del self._stats_cache[key]
            if key in self._cache_expiry:
                del self._cache_expiry[key]
    
    def _analyze_resolution_trends(self, since: datetime, window_minutes: int) -> Dict[str, Any]:
        """Analyze resolution success rate trends."""
        with self.lock:
            metrics = [m for m in self.resolution_metrics if m.timestamp >= since]
        
        if not metrics:
            return {"no_data": True}
        
        # Group by time windows
        windows = self._group_by_time_windows(metrics, window_minutes)
        
        trend_data = []
        for window_start, window_metrics in windows.items():
            successful = sum(1 for m in window_metrics if m.success)
            total = len(window_metrics)
            success_rate = successful / total * 100 if total > 0 else 0
            
            trend_data.append({
                "timestamp": window_start.isoformat(),
                "success_rate": success_rate,
                "total_resolutions": total,
                "avg_resolution_time": statistics.mean([m.resolution_time_seconds for m in window_metrics])
            })
        
        # Calculate trend direction
        if len(trend_data) >= 2:
            recent_rate = statistics.mean([d["success_rate"] for d in trend_data[-3:]])
            earlier_rate = statistics.mean([d["success_rate"] for d in trend_data[:3]])
            trend_direction = "improving" if recent_rate > earlier_rate else "declining"
        else:
            trend_direction = "stable"
        
        return {
            "window_minutes": window_minutes,
            "trend_direction": trend_direction,
            "data_points": trend_data
        }
    
    def _analyze_tool_trends(self, since: datetime, window_minutes: int) -> Dict[str, Any]:
        """Analyze tool performance trends."""
        with self.lock:
            metrics = [m for m in self.tool_metrics if m.timestamp >= since]
        
        if not metrics:
            return {"no_data": True}
        
        windows = self._group_by_time_windows(metrics, window_minutes)
        
        trend_data = []
        for window_start, window_metrics in windows.items():
            successful = sum(1 for m in window_metrics if m.success)
            total = len(window_metrics)
            success_rate = successful / total * 100 if total > 0 else 0
            
            trend_data.append({
                "timestamp": window_start.isoformat(),
                "success_rate": success_rate,
                "total_calls": total,
                "avg_execution_time": statistics.mean([m.execution_time_seconds for m in window_metrics])
            })
        
        return {
            "window_minutes": window_minutes,
            "data_points": trend_data
        }
    
    def _analyze_llm_trends(self, since: datetime, window_minutes: int) -> Dict[str, Any]:
        """Analyze LLM usage trends."""
        with self.lock:
            metrics = [m for m in self.llm_metrics if m.timestamp >= since]
        
        if not metrics:
            return {"no_data": True}
        
        windows = self._group_by_time_windows(metrics, window_minutes)
        
        trend_data = []
        for window_start, window_metrics in windows.items():
            total_tokens = sum(m.total_tokens for m in window_metrics)
            total_calls = len(window_metrics)
            
            trend_data.append({
                "timestamp": window_start.isoformat(),
                "total_calls": total_calls,
                "total_tokens": total_tokens,
                "avg_tokens_per_call": total_tokens / total_calls if total_calls > 0 else 0,
                "avg_response_time": statistics.mean([m.response_time_seconds for m in window_metrics])
            })
        
        return {
            "window_minutes": window_minutes,
            "data_points": trend_data
        }
    
    def _analyze_system_trends(self, since: datetime, window_minutes: int) -> Dict[str, Any]:
        """Analyze system health trends."""
        with self.lock:
            metrics = [m for m in self.system_metrics if m.timestamp >= since]
        
        if not metrics:
            return {"no_data": True}
        
        windows = self._group_by_time_windows(metrics, window_minutes)
        
        trend_data = []
        for window_start, window_metrics in windows.items():
            trend_data.append({
                "timestamp": window_start.isoformat(),
                "avg_cpu_usage": statistics.mean([m.cpu_usage_percent for m in window_metrics]),
                "avg_memory_usage": statistics.mean([m.memory_usage_percent for m in window_metrics]),
                "avg_active_threads": statistics.mean([m.active_threads for m in window_metrics])
            })
        
        return {
            "window_minutes": window_minutes,
            "data_points": trend_data
        }
    
    def _group_by_time_windows(self, metrics: List[Any], window_minutes: int) -> Dict[datetime, List[Any]]:
        """Group metrics by time windows."""
        windows = defaultdict(list)
        
        for metric in metrics:
            # Round timestamp to window boundary
            window_start = metric.timestamp.replace(
                minute=(metric.timestamp.minute // window_minutes) * window_minutes,
                second=0,
                microsecond=0
            )
            windows[window_start].append(metric)
        
        return dict(windows)
    
    def _export_json(self, filepath: str, hours: int):
        """Export metrics as JSON."""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "period_hours": hours,
            "comprehensive_report": self.get_comprehensive_report(hours)
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def _export_csv(self, filepath: str, hours: int):
        """Export metrics as CSV."""
        import csv
        
        since = datetime.now() - timedelta(hours=hours)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Export resolution metrics
            writer.writerow(["=== RESOLUTION METRICS ==="])
            writer.writerow([
                "timestamp", "scenario_id", "scenario_type", "resolution_time_seconds",
                "success", "confidence_score", "steps_count", "tools_used", "llm_calls",
                "total_tokens", "error_count"
            ])
            
            with self.lock:
                for metric in self.resolution_metrics:
                    if metric.timestamp >= since:
                        writer.writerow([
                            metric.timestamp.isoformat(),
                            metric.scenario_id,
                            metric.scenario_type,
                            metric.resolution_time_seconds,
                            metric.success,
                            metric.confidence_score,
                            metric.steps_count,
                            ",".join(metric.tools_used),
                            metric.llm_calls,
                            metric.total_tokens,
                            metric.error_count
                        ])
            
            # Export tool metrics
            writer.writerow([])
            writer.writerow(["=== TOOL METRICS ==="])
            writer.writerow([
                "timestamp", "tool_name", "operation", "execution_time_seconds",
                "success", "retry_count", "cache_hit", "error_message"
            ])
            
            with self.lock:
                for metric in self.tool_metrics:
                    if metric.timestamp >= since:
                        writer.writerow([
                            metric.timestamp.isoformat(),
                            metric.tool_name,
                            metric.operation,
                            metric.execution_time_seconds,
                            metric.success,
                            metric.retry_count,
                            metric.cache_hit,
                            metric.error_message or ""
                        ])