"""
Performance monitoring system for tracking response times and resource usage.
"""
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import json
from contextlib import contextmanager


class MetricType(Enum):
    """Types of metrics that can be collected."""
    RESPONSE_TIME = "response_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    TOOL_EXECUTION_TIME = "tool_execution_time"
    LLM_TOKEN_USAGE = "llm_token_usage"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"
    THROUGHPUT = "throughput"


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    metric_type: MetricType
    value: float
    component: str
    operation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a point in time."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    active_threads: int
    response_times: Dict[str, float]
    error_rates: Dict[str, float]
    throughput: Dict[str, float]


class MetricsCollector:
    """Collects and stores performance metrics."""
    
    def __init__(self, max_points: int = 10000, retention_hours: int = 24):
        self.max_points = max_points
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Start background cleanup task
        self._cleanup_thread = threading.Thread(target=self._cleanup_old_metrics, daemon=True)
        self._cleanup_thread.start()
    
    def record_metric(self, metric_type: MetricType, value: float, component: str,
                     operation: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Record a metric data point."""
        point = MetricPoint(
            timestamp=datetime.now(),
            metric_type=metric_type,
            value=value,
            component=component,
            operation=operation,
            metadata=metadata or {}
        )
        
        key = self._get_metric_key(metric_type, component, operation)
        
        with self.lock:
            self.metrics[key].append(point)
    
    def get_metrics(self, metric_type: MetricType, component: str,
                   operation: Optional[str] = None, 
                   since: Optional[datetime] = None) -> List[MetricPoint]:
        """Get metrics for a specific type and component."""
        key = self._get_metric_key(metric_type, component, operation)
        
        with self.lock:
            points = list(self.metrics.get(key, []))
        
        if since:
            points = [p for p in points if p.timestamp >= since]
        
        return points
    
    def get_average(self, metric_type: MetricType, component: str,
                   operation: Optional[str] = None,
                   since: Optional[datetime] = None) -> Optional[float]:
        """Get average value for a metric."""
        points = self.get_metrics(metric_type, component, operation, since)
        
        if not points:
            return None
        
        return sum(p.value for p in points) / len(points)
    
    def get_percentile(self, metric_type: MetricType, component: str,
                      percentile: float, operation: Optional[str] = None,
                      since: Optional[datetime] = None) -> Optional[float]:
        """Get percentile value for a metric."""
        points = self.get_metrics(metric_type, component, operation, since)
        
        if not points:
            return None
        
        values = sorted([p.value for p in points])
        index = int(len(values) * percentile / 100)
        return values[min(index, len(values) - 1)]
    
    def get_trend(self, metric_type: MetricType, component: str,
                 operation: Optional[str] = None,
                 window_minutes: int = 30) -> Dict[str, float]:
        """Get trend information for a metric."""
        since = datetime.now() - timedelta(minutes=window_minutes)
        points = self.get_metrics(metric_type, component, operation, since)
        
        if len(points) < 2:
            return {"trend": 0.0, "change_rate": 0.0}
        
        # Calculate simple linear trend
        values = [p.value for p in points]
        n = len(values)
        
        # Simple trend calculation
        first_half = sum(values[:n//2]) / (n//2) if n//2 > 0 else 0
        second_half = sum(values[n//2:]) / (n - n//2) if n - n//2 > 0 else 0
        
        trend = second_half - first_half
        change_rate = (trend / first_half * 100) if first_half > 0 else 0
        
        return {
            "trend": trend,
            "change_rate": change_rate,
            "first_half_avg": first_half,
            "second_half_avg": second_half
        }
    
    def get_all_components(self) -> List[str]:
        """Get list of all components that have metrics."""
        components = set()
        with self.lock:
            for key in self.metrics.keys():
                parts = key.split(":")
                if len(parts) >= 2:
                    components.add(parts[1])
        return list(components)
    
    def _get_metric_key(self, metric_type: MetricType, component: str,
                       operation: Optional[str] = None) -> str:
        """Generate key for storing metrics."""
        if operation:
            return f"{metric_type.value}:{component}:{operation}"
        return f"{metric_type.value}:{component}"
    
    def _cleanup_old_metrics(self):
        """Background task to clean up old metrics."""
        while True:
            try:
                cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
                
                with self.lock:
                    for key, points in self.metrics.items():
                        # Remove old points
                        while points and points[0].timestamp < cutoff_time:
                            points.popleft()
                
                # Sleep for 1 hour before next cleanup
                time.sleep(3600)
                
            except Exception as e:
                self.logger.error(f"Error in metrics cleanup: {e}")
                time.sleep(300)  # Sleep 5 minutes on error


class PerformanceMonitor:
    """Main performance monitoring system."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.logger = logging.getLogger(__name__)
        
        # System monitoring
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_interval = 30  # seconds
        
        # Performance thresholds
        self.thresholds = {
            "response_time_warning": 10.0,  # seconds
            "response_time_critical": 30.0,  # seconds
            "memory_warning": 80.0,  # percent
            "memory_critical": 95.0,  # percent
            "cpu_warning": 80.0,  # percent
            "cpu_critical": 95.0,  # percent
            "error_rate_warning": 5.0,  # percent
            "error_rate_critical": 15.0  # percent
        }
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
    
    def start_monitoring(self):
        """Start continuous system monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous system monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Performance monitoring stopped")
    
    @contextmanager
    def measure_operation(self, component: str, operation: str):
        """Context manager to measure operation performance."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
            success = True
        except Exception as e:
            success = False
            raise
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Record metrics
            response_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self.metrics_collector.record_metric(
                MetricType.RESPONSE_TIME, response_time, component, operation
            )
            
            if memory_delta > 0:
                self.metrics_collector.record_metric(
                    MetricType.MEMORY_USAGE, memory_delta, component, operation
                )
            
            # Record success/error rate
            rate_metric = MetricType.SUCCESS_RATE if success else MetricType.ERROR_RATE
            self.metrics_collector.record_metric(
                rate_metric, 1.0, component, operation
            )
            
            # Check for threshold violations
            self._check_thresholds(component, operation, response_time)
    
    def record_llm_usage(self, component: str, operation: str, 
                        prompt_tokens: int, completion_tokens: int,
                        response_time: float):
        """Record LLM usage metrics."""
        total_tokens = prompt_tokens + completion_tokens
        
        self.metrics_collector.record_metric(
            MetricType.LLM_TOKEN_USAGE, total_tokens, component, operation,
            metadata={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "response_time": response_time
            }
        )
        
        self.metrics_collector.record_metric(
            MetricType.RESPONSE_TIME, response_time, component, operation
        )
    
    def record_tool_execution(self, tool_name: str, execution_time: float, 
                            success: bool, error_message: Optional[str] = None):
        """Record tool execution metrics."""
        self.metrics_collector.record_metric(
            MetricType.TOOL_EXECUTION_TIME, execution_time, "tool_manager", tool_name
        )
        
        rate_metric = MetricType.SUCCESS_RATE if success else MetricType.ERROR_RATE
        self.metrics_collector.record_metric(
            rate_metric, 1.0, "tool_manager", tool_name,
            metadata={"error_message": error_message} if error_message else {}
        )
    
    def get_performance_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        # Response times for different components
        response_times = {}
        for component in self.metrics_collector.get_all_components():
            avg_time = self.metrics_collector.get_average(
                MetricType.RESPONSE_TIME, component,
                since=datetime.now() - timedelta(minutes=5)
            )
            if avg_time is not None:
                response_times[component] = avg_time
        
        # Error rates
        error_rates = {}
        for component in self.metrics_collector.get_all_components():
            error_points = self.metrics_collector.get_metrics(
                MetricType.ERROR_RATE, component,
                since=datetime.now() - timedelta(minutes=5)
            )
            success_points = self.metrics_collector.get_metrics(
                MetricType.SUCCESS_RATE, component,
                since=datetime.now() - timedelta(minutes=5)
            )
            
            total_operations = len(error_points) + len(success_points)
            if total_operations > 0:
                error_rates[component] = len(error_points) / total_operations * 100
        
        # Throughput (operations per minute)
        throughput = {}
        for component in self.metrics_collector.get_all_components():
            since = datetime.now() - timedelta(minutes=1)
            operations = len(self.metrics_collector.get_metrics(
                MetricType.SUCCESS_RATE, component, since=since
            )) + len(self.metrics_collector.get_metrics(
                MetricType.ERROR_RATE, component, since=since
            ))
            throughput[component] = operations
        
        return PerformanceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_mb=process.memory_info().rss / 1024 / 1024,
            active_threads=threading.active_count(),
            response_times=response_times,
            error_rates=error_rates,
            throughput=throughput
        )
    
    def get_performance_report(self, hours: int = 1) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        since = datetime.now() - timedelta(hours=hours)
        
        report = {
            "period": f"Last {hours} hour(s)",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        for component in self.metrics_collector.get_all_components():
            component_report = {
                "response_time": {
                    "average": self.metrics_collector.get_average(
                        MetricType.RESPONSE_TIME, component, since=since
                    ),
                    "p95": self.metrics_collector.get_percentile(
                        MetricType.RESPONSE_TIME, component, 95, since=since
                    ),
                    "p99": self.metrics_collector.get_percentile(
                        MetricType.RESPONSE_TIME, component, 99, since=since
                    )
                },
                "error_rate": self._calculate_error_rate(component, since),
                "throughput": self._calculate_throughput(component, since),
                "trends": self.metrics_collector.get_trend(
                    MetricType.RESPONSE_TIME, component, window_minutes=hours*60
                )
            }
            
            # Add LLM-specific metrics if available
            llm_tokens = self.metrics_collector.get_metrics(
                MetricType.LLM_TOKEN_USAGE, component, since=since
            )
            if llm_tokens:
                total_tokens = sum(p.value for p in llm_tokens)
                component_report["llm_usage"] = {
                    "total_tokens": total_tokens,
                    "average_tokens_per_request": total_tokens / len(llm_tokens),
                    "requests": len(llm_tokens)
                }
            
            report["components"][component] = component_report
        
        return report
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback for performance alerts."""
        self.alert_callbacks.append(callback)
    
    def _monitor_system(self):
        """Background system monitoring loop."""
        while self._monitoring_active:
            try:
                # Record system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                self.metrics_collector.record_metric(
                    MetricType.CPU_USAGE, cpu_percent, "system"
                )
                self.metrics_collector.record_metric(
                    MetricType.MEMORY_USAGE, memory.percent, "system"
                )
                
                # Check system thresholds
                self._check_system_thresholds(cpu_percent, memory.percent)
                
                time.sleep(self._monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Error in system monitoring: {e}")
                time.sleep(60)  # Sleep longer on error
    
    def _check_thresholds(self, component: str, operation: str, response_time: float):
        """Check if performance thresholds are violated."""
        if response_time > self.thresholds["response_time_critical"]:
            self._trigger_alert("response_time_critical", {
                "component": component,
                "operation": operation,
                "response_time": response_time,
                "threshold": self.thresholds["response_time_critical"]
            })
        elif response_time > self.thresholds["response_time_warning"]:
            self._trigger_alert("response_time_warning", {
                "component": component,
                "operation": operation,
                "response_time": response_time,
                "threshold": self.thresholds["response_time_warning"]
            })
    
    def _check_system_thresholds(self, cpu_percent: float, memory_percent: float):
        """Check system resource thresholds."""
        if cpu_percent > self.thresholds["cpu_critical"]:
            self._trigger_alert("cpu_critical", {
                "cpu_percent": cpu_percent,
                "threshold": self.thresholds["cpu_critical"]
            })
        elif cpu_percent > self.thresholds["cpu_warning"]:
            self._trigger_alert("cpu_warning", {
                "cpu_percent": cpu_percent,
                "threshold": self.thresholds["cpu_warning"]
            })
        
        if memory_percent > self.thresholds["memory_critical"]:
            self._trigger_alert("memory_critical", {
                "memory_percent": memory_percent,
                "threshold": self.thresholds["memory_critical"]
            })
        elif memory_percent > self.thresholds["memory_warning"]:
            self._trigger_alert("memory_warning", {
                "memory_percent": memory_percent,
                "threshold": self.thresholds["memory_warning"]
            })
    
    def _trigger_alert(self, alert_type: str, data: Dict[str, Any]):
        """Trigger performance alert."""
        self.logger.warning(f"Performance alert: {alert_type} - {data}")
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, data)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def _calculate_error_rate(self, component: str, since: datetime) -> float:
        """Calculate error rate for a component."""
        error_points = self.metrics_collector.get_metrics(
            MetricType.ERROR_RATE, component, since=since
        )
        success_points = self.metrics_collector.get_metrics(
            MetricType.SUCCESS_RATE, component, since=since
        )
        
        total_operations = len(error_points) + len(success_points)
        if total_operations == 0:
            return 0.0
        
        return len(error_points) / total_operations * 100
    
    def _calculate_throughput(self, component: str, since: datetime) -> float:
        """Calculate throughput (operations per minute) for a component."""
        error_points = self.metrics_collector.get_metrics(
            MetricType.ERROR_RATE, component, since=since
        )
        success_points = self.metrics_collector.get_metrics(
            MetricType.SUCCESS_RATE, component, since=since
        )
        
        total_operations = len(error_points) + len(success_points)
        time_diff = (datetime.now() - since).total_seconds() / 60  # minutes
        
        if time_diff == 0:
            return 0.0
        
        return total_operations / time_diff
    
    def export_metrics(self, filepath: str, format: str = "json"):
        """Export metrics to file."""
        if format.lower() == "json":
            self._export_json(filepath)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, filepath: str):
        """Export metrics as JSON."""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "metrics": {}
        }
        
        with self.metrics_collector.lock:
            for key, points in self.metrics_collector.metrics.items():
                export_data["metrics"][key] = [
                    {
                        "timestamp": point.timestamp.isoformat(),
                        "value": point.value,
                        "metadata": point.metadata
                    }
                    for point in points
                ]
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Metrics exported to {filepath}")