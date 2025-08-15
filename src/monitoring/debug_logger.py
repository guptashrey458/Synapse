"""
Advanced debugging and logging capabilities for complex reasoning scenarios.
"""
import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import threading
from contextlib import contextmanager

from typing import Protocol
from ..tools.interfaces import ToolResult


class LogLevel(Enum):
    """Debug log levels."""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DebugCategory(Enum):
    """Categories for debug information."""
    REASONING = "reasoning"
    TOOL_EXECUTION = "tool_execution"
    LLM_INTERACTION = "llm_interaction"
    ERROR_HANDLING = "error_handling"
    PERFORMANCE = "performance"
    SYSTEM = "system"


@dataclass
class DebugEntry:
    """A single debug log entry."""
    timestamp: datetime
    level: LogLevel
    category: DebugCategory
    component: str
    operation: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    correlation_id: Optional[str] = None


@dataclass
class ReasoningDebugInfo:
    """Debug information for reasoning processes."""
    trace_id: str
    scenario_description: str
    start_time: datetime
    end_time: Optional[datetime]
    total_steps: int
    successful_steps: int
    failed_steps: int
    tool_calls: List[Dict[str, Any]]
    decision_points: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    error_summary: Optional[Dict[str, Any]] = None


class DebugLogger:
    """Advanced debug logging system."""
    
    def __init__(self, log_dir: str = "logs/debug", max_log_files: int = 100,
                 max_file_size_mb: int = 50, enable_console: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_log_files = max_log_files
        self.max_file_size_mb = max_file_size_mb
        self.enable_console = enable_console
        
        # In-memory debug entries for quick access
        self.debug_entries: List[DebugEntry] = []
        self.max_memory_entries = 1000
        self.lock = threading.RLock()
        
        # Setup file logging
        self._setup_file_logging()
        
        # Setup console logging if enabled
        if enable_console:
            self._setup_console_logging()
        
        # Correlation ID tracking
        self._correlation_ids: Dict[str, List[DebugEntry]] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def log(self, level: LogLevel, category: DebugCategory, component: str,
            operation: str, message: str, data: Optional[Dict[str, Any]] = None,
            correlation_id: Optional[str] = None, include_stack: bool = False):
        """Log a debug entry."""
        entry = DebugEntry(
            timestamp=datetime.now(),
            level=level,
            category=category,
            component=component,
            operation=operation,
            message=message,
            data=data or {},
            stack_trace=traceback.format_stack() if include_stack else None,
            correlation_id=correlation_id
        )
        
        with self.lock:
            # Add to memory storage
            self.debug_entries.append(entry)
            if len(self.debug_entries) > self.max_memory_entries:
                self.debug_entries.pop(0)
            
            # Track by correlation ID
            if correlation_id:
                if correlation_id not in self._correlation_ids:
                    self._correlation_ids[correlation_id] = []
                self._correlation_ids[correlation_id].append(entry)
        
        # Log to file
        self._log_to_file(entry)
        
        # Log to console if enabled
        if self.enable_console:
            self._log_to_console(entry)
    
    def trace(self, category: DebugCategory, component: str, operation: str,
             message: str, data: Optional[Dict[str, Any]] = None,
             correlation_id: Optional[str] = None):
        """Log trace level message."""
        self.log(LogLevel.TRACE, category, component, operation, message, data, correlation_id)
    
    def debug(self, category: DebugCategory, component: str, operation: str,
             message: str, data: Optional[Dict[str, Any]] = None,
             correlation_id: Optional[str] = None):
        """Log debug level message."""
        self.log(LogLevel.DEBUG, category, component, operation, message, data, correlation_id)
    
    def info(self, category: DebugCategory, component: str, operation: str,
            message: str, data: Optional[Dict[str, Any]] = None,
            correlation_id: Optional[str] = None):
        """Log info level message."""
        self.log(LogLevel.INFO, category, component, operation, message, data, correlation_id)
    
    def warning(self, category: DebugCategory, component: str, operation: str,
               message: str, data: Optional[Dict[str, Any]] = None,
               correlation_id: Optional[str] = None):
        """Log warning level message."""
        self.log(LogLevel.WARNING, category, component, operation, message, data, correlation_id)
    
    def error(self, category: DebugCategory, component: str, operation: str,
             message: str, data: Optional[Dict[str, Any]] = None,
             correlation_id: Optional[str] = None, include_stack: bool = True):
        """Log error level message."""
        self.log(LogLevel.ERROR, category, component, operation, message, data, 
                correlation_id, include_stack)
    
    def critical(self, category: DebugCategory, component: str, operation: str,
                message: str, data: Optional[Dict[str, Any]] = None,
                correlation_id: Optional[str] = None, include_stack: bool = True):
        """Log critical level message."""
        self.log(LogLevel.CRITICAL, category, component, operation, message, data,
                correlation_id, include_stack)
    
    @contextmanager
    def debug_context(self, category: DebugCategory, component: str, operation: str,
                     correlation_id: Optional[str] = None):
        """Context manager for debugging operations."""
        start_time = datetime.now()
        
        self.debug(category, component, operation, f"Starting {operation}",
                  {"start_time": start_time.isoformat()}, correlation_id)
        
        try:
            yield correlation_id
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.debug(category, component, operation, f"Completed {operation}",
                      {"end_time": end_time.isoformat(), "duration_seconds": duration},
                      correlation_id)
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.error(category, component, operation, f"Failed {operation}: {str(e)}",
                      {"end_time": end_time.isoformat(), "duration_seconds": duration,
                       "error": str(e)}, correlation_id)
            raise
    
    def get_entries(self, level: Optional[LogLevel] = None,
                   category: Optional[DebugCategory] = None,
                   component: Optional[str] = None,
                   since: Optional[datetime] = None,
                   correlation_id: Optional[str] = None) -> List[DebugEntry]:
        """Get debug entries matching criteria."""
        with self.lock:
            entries = self.debug_entries.copy()
        
        # Filter by correlation ID first if specified
        if correlation_id:
            entries = self._correlation_ids.get(correlation_id, [])
        
        # Apply other filters
        if level:
            entries = [e for e in entries if e.level == level]
        
        if category:
            entries = [e for e in entries if e.category == category]
        
        if component:
            entries = [e for e in entries if e.component == component]
        
        if since:
            entries = [e for e in entries if e.timestamp >= since]
        
        return entries
    
    def get_correlation_trace(self, correlation_id: str) -> List[DebugEntry]:
        """Get all entries for a specific correlation ID."""
        with self.lock:
            return self._correlation_ids.get(correlation_id, []).copy()
    
    def export_debug_session(self, correlation_id: str, filepath: str):
        """Export all debug information for a session to file."""
        entries = self.get_correlation_trace(correlation_id)
        
        export_data = {
            "correlation_id": correlation_id,
            "export_timestamp": datetime.now().isoformat(),
            "total_entries": len(entries),
            "entries": [
                {
                    "timestamp": entry.timestamp.isoformat(),
                    "level": entry.level.value,
                    "category": entry.category.value,
                    "component": entry.component,
                    "operation": entry.operation,
                    "message": entry.message,
                    "data": entry.data,
                    "stack_trace": entry.stack_trace
                }
                for entry in entries
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Debug session exported to {filepath}")
    
    def analyze_performance_patterns(self, hours: int = 1) -> Dict[str, Any]:
        """Analyze performance patterns from debug logs."""
        since = datetime.now() - timedelta(hours=hours)
        entries = self.get_entries(since=since)
        
        # Group by component and operation
        operations = {}
        for entry in entries:
            key = f"{entry.component}:{entry.operation}"
            if key not in operations:
                operations[key] = []
            operations[key].append(entry)
        
        analysis = {
            "period_hours": hours,
            "total_entries": len(entries),
            "operations": {}
        }
        
        for op_key, op_entries in operations.items():
            # Calculate timing information
            start_entries = [e for e in op_entries if "Starting" in e.message]
            complete_entries = [e for e in op_entries if "Completed" in e.message]
            error_entries = [e for e in op_entries if e.level in [LogLevel.ERROR, LogLevel.CRITICAL]]
            
            durations = []
            for entry in complete_entries:
                if "duration_seconds" in entry.data:
                    durations.append(entry.data["duration_seconds"])
            
            analysis["operations"][op_key] = {
                "total_calls": len(start_entries),
                "completed_calls": len(complete_entries),
                "error_calls": len(error_entries),
                "success_rate": len(complete_entries) / len(start_entries) * 100 if start_entries else 0,
                "average_duration": sum(durations) / len(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0,
                "min_duration": min(durations) if durations else 0
            }
        
        return analysis
    
    def _setup_file_logging(self):
        """Setup file-based logging."""
        log_file = self.log_dir / f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Create file handler
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.file_handler.setFormatter(formatter)
        
        # Add to logger
        file_logger = logging.getLogger('debug_file')
        file_logger.addHandler(self.file_handler)
        file_logger.setLevel(logging.DEBUG)
        
        self.file_logger = file_logger
    
    def _setup_console_logging(self):
        """Setup console logging."""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        console_logger = logging.getLogger('debug_console')
        console_logger.addHandler(console_handler)
        console_logger.setLevel(logging.INFO)
        
        self.console_logger = console_logger
    
    def _log_to_file(self, entry: DebugEntry):
        """Log entry to file."""
        log_data = {
            "timestamp": entry.timestamp.isoformat(),
            "level": entry.level.value,
            "category": entry.category.value,
            "component": entry.component,
            "operation": entry.operation,
            "message": entry.message,
            "data": entry.data
        }
        
        if entry.correlation_id:
            log_data["correlation_id"] = entry.correlation_id
        
        if entry.stack_trace:
            log_data["stack_trace"] = entry.stack_trace
        
        self.file_logger.info(json.dumps(log_data))
    
    def _log_to_console(self, entry: DebugEntry):
        """Log entry to console."""
        if entry.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            level = logging.ERROR
        elif entry.level == LogLevel.WARNING:
            level = logging.WARNING
        else:
            level = logging.INFO
        
        message = f"[{entry.category.value}] {entry.component}:{entry.operation} - {entry.message}"
        if entry.data:
            message += f" | Data: {json.dumps(entry.data, default=str)}"
        
        self.console_logger.log(level, message)


class ReasoningDebugger:
    """Specialized debugger for reasoning processes."""
    
    def __init__(self, debug_logger: DebugLogger):
        self.debug_logger = debug_logger
        self.active_traces: Dict[str, ReasoningDebugInfo] = {}
        self.lock = threading.RLock()
    
    def start_reasoning_trace(self, trace_id: str, scenario_description: str) -> str:
        """Start debugging a reasoning trace."""
        debug_info = ReasoningDebugInfo(
            trace_id=trace_id,
            scenario_description=scenario_description,
            start_time=datetime.now(),
            end_time=None,
            total_steps=0,
            successful_steps=0,
            failed_steps=0,
            tool_calls=[],
            decision_points=[],
            performance_metrics={}
        )
        
        with self.lock:
            self.active_traces[trace_id] = debug_info
        
        self.debug_logger.info(
            DebugCategory.REASONING, "reasoning_engine", "start_trace",
            f"Started reasoning trace for scenario: {scenario_description[:100]}...",
            {"trace_id": trace_id, "scenario": scenario_description},
            correlation_id=trace_id
        )
        
        return trace_id
    
    def log_reasoning_step(self, trace_id: str, step: Any):
        """Log a reasoning step."""
        with self.lock:
            if trace_id not in self.active_traces:
                return
            
            debug_info = self.active_traces[trace_id]
            debug_info.total_steps += 1
        
        step_data = {
            "step_number": getattr(step, 'step_number', debug_info.total_steps),
            "thought": step.thought,
            "action": step.action.tool_name if step.action else None,
            "observation": step.observation,
            "timestamp": getattr(step, 'timestamp', datetime.now()).isoformat()
        }
        
        # Log tool results if available
        if hasattr(step, 'tool_results') and step.tool_results:
            tool_results_data = []
            for result in step.tool_results:
                tool_data = {
                    "tool_name": result.tool_name,
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "data_keys": list(result.data.keys()) if result.data else [],
                    "error_message": result.error_message
                }
                tool_results_data.append(tool_data)
                
                # Track tool calls
                with self.lock:
                    debug_info.tool_calls.append(tool_data)
                    if result.success:
                        debug_info.successful_steps += 1
                    else:
                        debug_info.failed_steps += 1
            
            step_data["tool_results"] = tool_results_data
        
        self.debug_logger.debug(
            DebugCategory.REASONING, "reasoning_engine", "reasoning_step",
            f"Step {step_data['step_number']}: {step.thought[:100]}...",
            step_data,
            correlation_id=trace_id
        )
    
    def log_decision_point(self, trace_id: str, decision: str, 
                          reasoning: str, confidence: float,
                          alternatives: Optional[List[str]] = None):
        """Log a decision point in reasoning."""
        decision_data = {
            "decision": decision,
            "reasoning": reasoning,
            "confidence": confidence,
            "alternatives": alternatives or [],
            "timestamp": datetime.now().isoformat()
        }
        
        with self.lock:
            if trace_id in self.active_traces:
                self.active_traces[trace_id].decision_points.append(decision_data)
        
        self.debug_logger.info(
            DebugCategory.REASONING, "reasoning_engine", "decision_point",
            f"Decision: {decision} (confidence: {confidence:.2f})",
            decision_data,
            correlation_id=trace_id
        )
    
    def log_reasoning_error(self, trace_id: str, error: Exception, 
                           context: Optional[Dict[str, Any]] = None):
        """Log a reasoning error."""
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }
        
        with self.lock:
            if trace_id in self.active_traces:
                self.active_traces[trace_id].error_summary = error_data
        
        self.debug_logger.error(
            DebugCategory.REASONING, "reasoning_engine", "reasoning_error",
            f"Reasoning error: {str(error)}",
            error_data,
            correlation_id=trace_id
        )
    
    def end_reasoning_trace(self, trace_id: str, success: bool = True,
                           final_plan: Optional[Dict[str, Any]] = None):
        """End a reasoning trace."""
        with self.lock:
            if trace_id not in self.active_traces:
                return
            
            debug_info = self.active_traces[trace_id]
            debug_info.end_time = datetime.now()
            
            # Calculate performance metrics
            duration = (debug_info.end_time - debug_info.start_time).total_seconds()
            debug_info.performance_metrics = {
                "total_duration_seconds": duration,
                "steps_per_second": debug_info.total_steps / duration if duration > 0 else 0,
                "tool_success_rate": (debug_info.successful_steps / debug_info.total_steps * 100 
                                    if debug_info.total_steps > 0 else 0),
                "average_step_time": duration / debug_info.total_steps if debug_info.total_steps > 0 else 0
            }
        
        summary_data = {
            "success": success,
            "duration_seconds": duration,
            "total_steps": debug_info.total_steps,
            "successful_steps": debug_info.successful_steps,
            "failed_steps": debug_info.failed_steps,
            "tool_calls": len(debug_info.tool_calls),
            "decision_points": len(debug_info.decision_points),
            "performance_metrics": debug_info.performance_metrics
        }
        
        if final_plan:
            summary_data["final_plan"] = final_plan
        
        self.debug_logger.info(
            DebugCategory.REASONING, "reasoning_engine", "end_trace",
            f"Completed reasoning trace - Success: {success}, Duration: {duration:.2f}s",
            summary_data,
            correlation_id=trace_id
        )
    
    def get_trace_summary(self, trace_id: str) -> Optional[ReasoningDebugInfo]:
        """Get summary of a reasoning trace."""
        with self.lock:
            return self.active_traces.get(trace_id)
    
    def export_trace_analysis(self, trace_id: str, filepath: str):
        """Export detailed trace analysis."""
        with self.lock:
            debug_info = self.active_traces.get(trace_id)
        
        if not debug_info:
            raise ValueError(f"No trace found with ID: {trace_id}")
        
        # Get all debug entries for this trace
        entries = self.debug_logger.get_correlation_trace(trace_id)
        
        analysis = {
            "trace_id": trace_id,
            "scenario": debug_info.scenario_description,
            "timeline": {
                "start_time": debug_info.start_time.isoformat(),
                "end_time": debug_info.end_time.isoformat() if debug_info.end_time else None,
                "duration_seconds": debug_info.performance_metrics.get("total_duration_seconds", 0)
            },
            "summary": {
                "total_steps": debug_info.total_steps,
                "successful_steps": debug_info.successful_steps,
                "failed_steps": debug_info.failed_steps,
                "tool_calls": len(debug_info.tool_calls),
                "decision_points": len(debug_info.decision_points)
            },
            "performance_metrics": debug_info.performance_metrics,
            "tool_calls": debug_info.tool_calls,
            "decision_points": debug_info.decision_points,
            "error_summary": debug_info.error_summary,
            "detailed_log": [
                {
                    "timestamp": entry.timestamp.isoformat(),
                    "level": entry.level.value,
                    "operation": entry.operation,
                    "message": entry.message,
                    "data": entry.data
                }
                for entry in entries
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        self.debug_logger.info(
            DebugCategory.SYSTEM, "reasoning_debugger", "export_analysis",
            f"Exported trace analysis to {filepath}",
            {"trace_id": trace_id, "filepath": filepath}
        )
    
    def analyze_reasoning_patterns(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze reasoning patterns across multiple traces."""
        since = datetime.now() - timedelta(hours=hours)
        
        # Get all reasoning entries
        entries = self.debug_logger.get_entries(
            category=DebugCategory.REASONING,
            since=since
        )
        
        # Group by trace ID
        traces = {}
        for entry in entries:
            if entry.correlation_id:
                if entry.correlation_id not in traces:
                    traces[entry.correlation_id] = []
                traces[entry.correlation_id].append(entry)
        
        analysis = {
            "period_hours": hours,
            "total_traces": len(traces),
            "patterns": {
                "average_steps_per_trace": 0,
                "average_duration": 0,
                "common_decision_patterns": [],
                "frequent_errors": [],
                "tool_usage_patterns": {}
            }
        }
        
        if not traces:
            return analysis
        
        # Analyze patterns
        total_steps = 0
        total_duration = 0
        decision_patterns = {}
        error_patterns = {}
        tool_usage = {}
        
        for trace_id, trace_entries in traces.items():
            # Count steps
            step_entries = [e for e in trace_entries if e.operation == "reasoning_step"]
            total_steps += len(step_entries)
            
            # Calculate duration
            start_entry = next((e for e in trace_entries if e.operation == "start_trace"), None)
            end_entry = next((e for e in trace_entries if e.operation == "end_trace"), None)
            
            if start_entry and end_entry:
                duration = (end_entry.timestamp - start_entry.timestamp).total_seconds()
                total_duration += duration
            
            # Analyze decisions
            decision_entries = [e for e in trace_entries if e.operation == "decision_point"]
            for entry in decision_entries:
                decision = entry.data.get("decision", "unknown")
                decision_patterns[decision] = decision_patterns.get(decision, 0) + 1
            
            # Analyze errors
            error_entries = [e for e in trace_entries if e.level == LogLevel.ERROR]
            for entry in error_entries:
                error_type = entry.data.get("error_type", "unknown")
                error_patterns[error_type] = error_patterns.get(error_type, 0) + 1
            
            # Analyze tool usage
            for entry in step_entries:
                if "tool_results" in entry.data:
                    for tool_result in entry.data["tool_results"]:
                        tool_name = tool_result.get("tool_name", "unknown")
                        if tool_name not in tool_usage:
                            tool_usage[tool_name] = {"calls": 0, "successes": 0}
                        tool_usage[tool_name]["calls"] += 1
                        if tool_result.get("success", False):
                            tool_usage[tool_name]["successes"] += 1
        
        # Calculate averages and patterns
        analysis["patterns"]["average_steps_per_trace"] = total_steps / len(traces)
        analysis["patterns"]["average_duration"] = total_duration / len(traces)
        
        # Top decision patterns
        analysis["patterns"]["common_decision_patterns"] = sorted(
            decision_patterns.items(), key=lambda x: x[1], reverse=True
        )[:5]
        
        # Top error patterns
        analysis["patterns"]["frequent_errors"] = sorted(
            error_patterns.items(), key=lambda x: x[1], reverse=True
        )[:5]
        
        # Tool usage with success rates
        for tool_name, stats in tool_usage.items():
            stats["success_rate"] = stats["successes"] / stats["calls"] * 100 if stats["calls"] > 0 else 0
        
        analysis["patterns"]["tool_usage_patterns"] = tool_usage
        
        return analysis