"""
Comprehensive tests for monitoring and debugging capabilities.
"""
import pytest
import time
import tempfile
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.monitoring.performance_monitor import (
    PerformanceMonitor, MetricsCollector, MetricType, PerformanceSnapshot
)
from src.monitoring.debug_logger import (
    DebugLogger, ReasoningDebugger, LogLevel, DebugCategory
)
from src.monitoring.metrics_collector import (
    MetricsAggregator, SystemMetrics, ResolutionMetrics, 
    ToolMetrics, LLMMetrics
)
from src.agent.models import ValidatedReasoningStep
from src.tools.interfaces import ToolResult


class TestMetricsCollector:
    """Test metrics collection functionality."""
    
    def test_initialization(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector(max_points=1000, retention_hours=12)
        
        assert collector.max_points == 1000
        assert collector.retention_hours == 12
        assert len(collector.metrics) == 0
    
    def test_record_metric(self):
        """Test recording a metric."""
        collector = MetricsCollector()
        
        collector.record_metric(
            MetricType.RESPONSE_TIME, 1.5, "test_component", "test_operation"
        )
        
        metrics = collector.get_metrics(
            MetricType.RESPONSE_TIME, "test_component", "test_operation"
        )
        
        assert len(metrics) == 1
        assert metrics[0].value == 1.5
        assert metrics[0].component == "test_component"
        assert metrics[0].operation == "test_operation"
    
    def test_get_average(self):
        """Test calculating average metric values."""
        collector = MetricsCollector()
        
        # Record multiple metrics
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            collector.record_metric(
                MetricType.RESPONSE_TIME, value, "test_component"
            )
        
        average = collector.get_average(MetricType.RESPONSE_TIME, "test_component")
        assert average == 3.0
    
    def test_get_percentile(self):
        """Test calculating percentile values."""
        collector = MetricsCollector()
        
        # Record metrics with known distribution
        values = list(range(1, 101))  # 1 to 100
        for value in values:
            collector.record_metric(
                MetricType.RESPONSE_TIME, float(value), "test_component"
            )
        
        p95 = collector.get_percentile(MetricType.RESPONSE_TIME, "test_component", 95)
        assert p95 == 95.0
        
        p50 = collector.get_percentile(MetricType.RESPONSE_TIME, "test_component", 50)
        assert p50 == 50.0
    
    def test_get_trend(self):
        """Test trend calculation."""
        collector = MetricsCollector()
        
        # Record increasing values
        for i in range(10):
            collector.record_metric(
                MetricType.RESPONSE_TIME, float(i), "test_component"
            )
        
        trend = collector.get_trend(MetricType.RESPONSE_TIME, "test_component")
        
        assert trend["trend"] > 0  # Should be positive trend
        assert trend["change_rate"] > 0
    
    def test_metric_filtering_by_time(self):
        """Test filtering metrics by time."""
        collector = MetricsCollector()
        
        # Record old metric
        old_time = datetime.now() - timedelta(hours=2)
        collector.record_metric(
            MetricType.RESPONSE_TIME, 1.0, "test_component"
        )
        
        # Manually set timestamp to old time
        collector.metrics[collector._get_metric_key(MetricType.RESPONSE_TIME, "test_component")][0].timestamp = old_time
        
        # Record new metric
        collector.record_metric(
            MetricType.RESPONSE_TIME, 2.0, "test_component"
        )
        
        # Get metrics from last hour
        since = datetime.now() - timedelta(hours=1)
        recent_metrics = collector.get_metrics(
            MetricType.RESPONSE_TIME, "test_component", since=since
        )
        
        assert len(recent_metrics) == 1
        assert recent_metrics[0].value == 2.0


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""
    
    def test_initialization(self):
        """Test performance monitor initialization."""
        monitor = PerformanceMonitor()
        
        assert monitor.metrics_collector is not None
        assert not monitor._monitoring_active
        assert monitor.thresholds["response_time_warning"] == 10.0
    
    def test_measure_operation_context_manager(self):
        """Test operation measurement context manager."""
        monitor = PerformanceMonitor()
        
        with monitor.measure_operation("test_component", "test_operation"):
            time.sleep(0.1)  # Simulate work
        
        # Check that metrics were recorded
        response_time = monitor.metrics_collector.get_average(
            MetricType.RESPONSE_TIME, "test_component", "test_operation"
        )
        
        assert response_time is not None
        assert response_time >= 0.1
    
    def test_measure_operation_with_exception(self):
        """Test operation measurement when exception occurs."""
        monitor = PerformanceMonitor()
        
        with pytest.raises(ValueError):
            with monitor.measure_operation("test_component", "test_operation"):
                raise ValueError("Test error")
        
        # Should still record error rate metric
        error_metrics = monitor.metrics_collector.get_metrics(
            MetricType.ERROR_RATE, "test_component", "test_operation"
        )
        
        assert len(error_metrics) == 1
    
    def test_record_llm_usage(self):
        """Test recording LLM usage metrics."""
        monitor = PerformanceMonitor()
        
        monitor.record_llm_usage(
            "llm_provider", "generate_response", 
            prompt_tokens=100, completion_tokens=50, response_time=2.5
        )
        
        # Check token usage metric
        token_usage = monitor.metrics_collector.get_average(
            MetricType.LLM_TOKEN_USAGE, "llm_provider", "generate_response"
        )
        assert token_usage == 150  # 100 + 50
        
        # Check response time metric
        response_time = monitor.metrics_collector.get_average(
            MetricType.RESPONSE_TIME, "llm_provider", "generate_response"
        )
        assert response_time == 2.5
    
    def test_record_tool_execution(self):
        """Test recording tool execution metrics."""
        monitor = PerformanceMonitor()
        
        monitor.record_tool_execution(
            "test_tool", execution_time=1.5, success=True
        )
        
        # Check execution time metric
        exec_time = monitor.metrics_collector.get_average(
            MetricType.TOOL_EXECUTION_TIME, "tool_manager", "test_tool"
        )
        assert exec_time == 1.5
        
        # Check success rate metric
        success_metrics = monitor.metrics_collector.get_metrics(
            MetricType.SUCCESS_RATE, "tool_manager", "test_tool"
        )
        assert len(success_metrics) == 1
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.Process')
    def test_get_performance_snapshot(self, mock_process, mock_memory, mock_cpu):
        """Test getting performance snapshot."""
        # Mock system metrics
        mock_cpu.return_value = 45.0
        mock_memory.return_value.percent = 60.0
        mock_process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
        
        monitor = PerformanceMonitor()
        
        # Add some test metrics
        monitor.record_tool_execution("test_tool", 1.0, True)
        
        snapshot = monitor.get_performance_snapshot()
        
        assert snapshot.cpu_percent == 45.0
        assert snapshot.memory_percent == 60.0
        assert snapshot.memory_mb == 100.0
        assert "tool_manager" in snapshot.response_times
    
    def test_alert_callbacks(self):
        """Test performance alert callbacks."""
        monitor = PerformanceMonitor()
        
        # Add alert callback
        alerts_received = []
        def alert_callback(alert_type, data):
            alerts_received.append((alert_type, data))
        
        monitor.add_alert_callback(alert_callback)
        
        # Trigger alert by exceeding threshold
        monitor._check_thresholds("test_component", "test_operation", 35.0)  # Above critical threshold
        
        assert len(alerts_received) == 1
        assert alerts_received[0][0] == "response_time_critical"
    
    def test_performance_report_generation(self):
        """Test comprehensive performance report generation."""
        monitor = PerformanceMonitor()
        
        # Add some test data
        monitor.record_llm_usage("openai", "generate", 100, 50, 2.0)
        monitor.record_tool_execution("test_tool", 1.5, True)
        
        report = monitor.get_performance_report(hours=1)
        
        assert "period" in report
        assert "components" in report
        assert "tool_manager" in report["components"]


class TestDebugLogger:
    """Test debug logging functionality."""
    
    def test_initialization(self):
        """Test debug logger initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = DebugLogger(log_dir=temp_dir, enable_console=False)
            
            assert logger.log_dir == Path(temp_dir)
            assert len(logger.debug_entries) == 0
    
    def test_log_entry_creation(self):
        """Test creating log entries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = DebugLogger(log_dir=temp_dir, enable_console=False)
            
            logger.debug(
                DebugCategory.REASONING, "test_component", "test_operation",
                "Test message", {"key": "value"}, "correlation_123"
            )
            
            assert len(logger.debug_entries) == 1
            entry = logger.debug_entries[0]
            
            assert entry.level == LogLevel.DEBUG
            assert entry.category == DebugCategory.REASONING
            assert entry.component == "test_component"
            assert entry.operation == "test_operation"
            assert entry.message == "Test message"
            assert entry.data["key"] == "value"
            assert entry.correlation_id == "correlation_123"
    
    def test_correlation_id_tracking(self):
        """Test correlation ID tracking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = DebugLogger(log_dir=temp_dir, enable_console=False)
            
            correlation_id = "test_correlation_123"
            
            # Log multiple entries with same correlation ID
            logger.info(DebugCategory.REASONING, "comp1", "op1", "Message 1", correlation_id=correlation_id)
            logger.debug(DebugCategory.TOOL_EXECUTION, "comp2", "op2", "Message 2", correlation_id=correlation_id)
            
            # Get entries by correlation ID
            correlated_entries = logger.get_correlation_trace(correlation_id)
            
            assert len(correlated_entries) == 2
            assert all(entry.correlation_id == correlation_id for entry in correlated_entries)
    
    def test_debug_context_manager(self):
        """Test debug context manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = DebugLogger(log_dir=temp_dir, enable_console=False)
            
            with logger.debug_context(
                DebugCategory.REASONING, "test_component", "test_operation"
            ) as correlation_id:
                time.sleep(0.1)  # Simulate work
            
            # Should have start and end entries
            entries = logger.get_correlation_trace(correlation_id)
            assert len(entries) == 2
            
            start_entry = entries[0]
            end_entry = entries[1]
            
            assert "Starting" in start_entry.message
            assert "Completed" in end_entry.message
            assert "duration_seconds" in end_entry.data
    
    def test_debug_context_with_exception(self):
        """Test debug context manager with exception."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = DebugLogger(log_dir=temp_dir, enable_console=False)
            
            with pytest.raises(ValueError):
                with logger.debug_context(
                    DebugCategory.REASONING, "test_component", "test_operation"
                ) as correlation_id:
                    raise ValueError("Test error")
            
            # Should have start and error entries
            entries = logger.get_correlation_trace(correlation_id)
            assert len(entries) == 2
            
            error_entry = entries[1]
            assert "Failed" in error_entry.message
            assert error_entry.level == LogLevel.ERROR
    
    def test_entry_filtering(self):
        """Test filtering debug entries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = DebugLogger(log_dir=temp_dir, enable_console=False)
            
            # Log entries with different levels and categories
            logger.debug(DebugCategory.REASONING, "comp1", "op1", "Debug message")
            logger.info(DebugCategory.TOOL_EXECUTION, "comp2", "op2", "Info message")
            logger.error(DebugCategory.ERROR_HANDLING, "comp3", "op3", "Error message")
            
            # Filter by level
            error_entries = logger.get_entries(level=LogLevel.ERROR)
            assert len(error_entries) == 1
            assert error_entries[0].level == LogLevel.ERROR
            
            # Filter by category
            reasoning_entries = logger.get_entries(category=DebugCategory.REASONING)
            assert len(reasoning_entries) == 1
            assert reasoning_entries[0].category == DebugCategory.REASONING
            
            # Filter by component
            comp1_entries = logger.get_entries(component="comp1")
            assert len(comp1_entries) == 1
            assert comp1_entries[0].component == "comp1"
    
    def test_export_debug_session(self):
        """Test exporting debug session."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = DebugLogger(log_dir=temp_dir, enable_console=False)
            
            correlation_id = "test_session_123"
            logger.info(DebugCategory.REASONING, "comp1", "op1", "Message 1", correlation_id=correlation_id)
            logger.debug(DebugCategory.TOOL_EXECUTION, "comp2", "op2", "Message 2", correlation_id=correlation_id)
            
            export_file = Path(temp_dir) / "session_export.json"
            logger.export_debug_session(correlation_id, str(export_file))
            
            # Verify export file
            assert export_file.exists()
            
            with open(export_file) as f:
                export_data = json.load(f)
            
            assert export_data["correlation_id"] == correlation_id
            assert export_data["total_entries"] == 2
            assert len(export_data["entries"]) == 2


class TestReasoningDebugger:
    """Test reasoning-specific debugging functionality."""
    
    def test_initialization(self):
        """Test reasoning debugger initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            debug_logger = DebugLogger(log_dir=temp_dir, enable_console=False)
            debugger = ReasoningDebugger(debug_logger)
            
            assert debugger.debug_logger == debug_logger
            assert len(debugger.active_traces) == 0
    
    def test_start_reasoning_trace(self):
        """Test starting a reasoning trace."""
        with tempfile.TemporaryDirectory() as temp_dir:
            debug_logger = DebugLogger(log_dir=temp_dir, enable_console=False)
            debugger = ReasoningDebugger(debug_logger)
            
            trace_id = debugger.start_reasoning_trace(
                "trace_123", "Test delivery scenario"
            )
            
            assert trace_id == "trace_123"
            assert trace_id in debugger.active_traces
            
            debug_info = debugger.active_traces[trace_id]
            assert debug_info.scenario_description == "Test delivery scenario"
            assert debug_info.total_steps == 0
    
    def test_log_reasoning_step(self):
        """Test logging reasoning steps."""
        with tempfile.TemporaryDirectory() as temp_dir:
            debug_logger = DebugLogger(log_dir=temp_dir, enable_console=False)
            debugger = ReasoningDebugger(debug_logger)
            
            trace_id = debugger.start_reasoning_trace("trace_123", "Test scenario")
            
            # Create a reasoning step
            step = ValidatedReasoningStep(
                step_number=1,
                thought="I need to check traffic conditions",
                timestamp=datetime.now()
            )
            
            debugger.log_reasoning_step(trace_id, step)
            
            debug_info = debugger.active_traces[trace_id]
            assert debug_info.total_steps == 1
    
    def test_log_decision_point(self):
        """Test logging decision points."""
        with tempfile.TemporaryDirectory() as temp_dir:
            debug_logger = DebugLogger(log_dir=temp_dir, enable_console=False)
            debugger = ReasoningDebugger(debug_logger)
            
            trace_id = debugger.start_reasoning_trace("trace_123", "Test scenario")
            
            debugger.log_decision_point(
                trace_id, "Use alternative route", 
                "Traffic is heavy on main route", 0.85,
                ["Wait for traffic to clear", "Contact customer"]
            )
            
            debug_info = debugger.active_traces[trace_id]
            assert len(debug_info.decision_points) == 1
            
            decision = debug_info.decision_points[0]
            assert decision["decision"] == "Use alternative route"
            assert decision["confidence"] == 0.85
    
    def test_log_reasoning_error(self):
        """Test logging reasoning errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            debug_logger = DebugLogger(log_dir=temp_dir, enable_console=False)
            debugger = ReasoningDebugger(debug_logger)
            
            trace_id = debugger.start_reasoning_trace("trace_123", "Test scenario")
            
            error = ValueError("Test reasoning error")
            debugger.log_reasoning_error(trace_id, error, {"context": "test"})
            
            debug_info = debugger.active_traces[trace_id]
            assert debug_info.error_summary is not None
            assert debug_info.error_summary["error_type"] == "ValueError"
    
    def test_end_reasoning_trace(self):
        """Test ending a reasoning trace."""
        with tempfile.TemporaryDirectory() as temp_dir:
            debug_logger = DebugLogger(log_dir=temp_dir, enable_console=False)
            debugger = ReasoningDebugger(debug_logger)
            
            trace_id = debugger.start_reasoning_trace("trace_123", "Test scenario")
            
            # Add some steps
            step = ValidatedReasoningStep(
                step_number=1,
                thought="Test thought",
                timestamp=datetime.now()
            )
            debugger.log_reasoning_step(trace_id, step)
            
            time.sleep(0.1)  # Ensure some duration
            
            debugger.end_reasoning_trace(trace_id, success=True)
            
            debug_info = debugger.active_traces[trace_id]
            assert debug_info.end_time is not None
            assert "total_duration_seconds" in debug_info.performance_metrics
            assert debug_info.performance_metrics["total_duration_seconds"] > 0
    
    def test_export_trace_analysis(self):
        """Test exporting trace analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            debug_logger = DebugLogger(log_dir=temp_dir, enable_console=False)
            debugger = ReasoningDebugger(debug_logger)
            
            trace_id = debugger.start_reasoning_trace("trace_123", "Test scenario")
            
            # Add some test data
            step = ValidatedReasoningStep(
                step_number=1,
                thought="Test thought",
                timestamp=datetime.now()
            )
            debugger.log_reasoning_step(trace_id, step)
            debugger.log_decision_point(trace_id, "Test decision", "Test reasoning", 0.8)
            debugger.end_reasoning_trace(trace_id, success=True)
            
            export_file = Path(temp_dir) / "trace_analysis.json"
            debugger.export_trace_analysis(trace_id, str(export_file))
            
            # Verify export
            assert export_file.exists()
            
            with open(export_file) as f:
                analysis = json.load(f)
            
            assert analysis["trace_id"] == trace_id
            assert analysis["scenario"] == "Test scenario"
            assert "timeline" in analysis
            assert "summary" in analysis
            assert "performance_metrics" in analysis


class TestMetricsAggregator:
    """Test metrics aggregation functionality."""
    
    def test_initialization(self):
        """Test metrics aggregator initialization."""
        aggregator = MetricsAggregator(max_metrics_per_category=5000)
        
        assert aggregator.max_metrics_per_category == 5000
        assert len(aggregator.system_metrics) == 0
        assert len(aggregator.resolution_metrics) == 0
    
    def test_record_resolution_metrics(self):
        """Test recording resolution metrics."""
        aggregator = MetricsAggregator()
        
        metrics = ResolutionMetrics(
            timestamp=datetime.now(),
            scenario_id="scenario_123",
            scenario_type="traffic",
            resolution_time_seconds=45.5,
            success=True,
            confidence_score=0.85,
            steps_count=5,
            tools_used=["check_traffic", "notify_customer"],
            llm_calls=3,
            total_tokens=150,
            error_count=0
        )
        
        aggregator.record_resolution_metrics(metrics)
        
        assert len(aggregator.resolution_metrics) == 1
        assert aggregator.resolution_metrics[0] == metrics
    
    def test_get_resolution_success_rate(self):
        """Test calculating resolution success rates."""
        aggregator = MetricsAggregator()
        
        # Add successful resolution
        success_metrics = ResolutionMetrics(
            timestamp=datetime.now(),
            scenario_id="success_1",
            scenario_type="traffic",
            resolution_time_seconds=30.0,
            success=True,
            confidence_score=0.9,
            steps_count=3,
            tools_used=["check_traffic"],
            llm_calls=2,
            total_tokens=100,
            error_count=0
        )
        
        # Add failed resolution
        failure_metrics = ResolutionMetrics(
            timestamp=datetime.now(),
            scenario_id="failure_1",
            scenario_type="merchant",
            resolution_time_seconds=60.0,
            success=False,
            confidence_score=0.3,
            steps_count=5,
            tools_used=["get_merchant_status"],
            llm_calls=4,
            total_tokens=200,
            error_count=2
        )
        
        aggregator.record_resolution_metrics(success_metrics)
        aggregator.record_resolution_metrics(failure_metrics)
        
        stats = aggregator.get_resolution_success_rate(hours=1)
        
        assert stats["total_resolutions"] == 2
        assert stats["successful_resolutions"] == 1
        assert stats["overall_success_rate"] == 50.0
        assert "by_scenario_type" in stats
        assert "traffic" in stats["by_scenario_type"]
        assert "merchant" in stats["by_scenario_type"]
    
    def test_get_tool_performance_stats(self):
        """Test getting tool performance statistics."""
        aggregator = MetricsAggregator()
        
        # Add successful tool execution
        success_tool = ToolMetrics(
            timestamp=datetime.now(),
            tool_name="check_traffic",
            operation="execute",
            execution_time_seconds=2.5,
            success=True,
            input_size_bytes=100,
            output_size_bytes=500,
            retry_count=0
        )
        
        # Add failed tool execution
        failure_tool = ToolMetrics(
            timestamp=datetime.now(),
            tool_name="check_traffic",
            operation="execute",
            execution_time_seconds=5.0,
            success=False,
            input_size_bytes=100,
            output_size_bytes=0,
            retry_count=2,
            error_message="Network timeout"
        )
        
        aggregator.record_tool_metrics(success_tool)
        aggregator.record_tool_metrics(failure_tool)
        
        stats = aggregator.get_tool_performance_stats(hours=1)
        
        assert stats["total_tool_calls"] == 2
        assert stats["unique_tools"] == 1
        assert "by_tool" in stats
        assert "check_traffic" in stats["by_tool"]
        
        tool_stats = stats["by_tool"]["check_traffic"]
        assert tool_stats["total_calls"] == 2
        assert tool_stats["successful_calls"] == 1
        assert tool_stats["success_rate"] == 50.0
    
    def test_get_llm_usage_stats(self):
        """Test getting LLM usage statistics."""
        aggregator = MetricsAggregator()
        
        llm_metrics = LLMMetrics(
            timestamp=datetime.now(),
            provider="openai",
            model="gpt-4",
            operation="generate_response",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            response_time_seconds=2.5,
            success=True,
            temperature=0.1,
            max_tokens=200,
            cost_estimate=0.003
        )
        
        aggregator.record_llm_metrics(llm_metrics)
        
        stats = aggregator.get_llm_usage_stats(hours=1)
        
        assert stats["total_calls"] == 1
        assert stats["total_tokens"] == 150
        assert stats["total_cost_estimate"] == 0.003
        assert "by_provider" in stats
        assert "openai" in stats["by_provider"]
    
    def test_get_comprehensive_report(self):
        """Test generating comprehensive metrics report."""
        aggregator = MetricsAggregator()
        
        # Add some test data
        resolution_metrics = ResolutionMetrics(
            timestamp=datetime.now(),
            scenario_id="test_1",
            scenario_type="traffic",
            resolution_time_seconds=30.0,
            success=True,
            confidence_score=0.8,
            steps_count=3,
            tools_used=["check_traffic"],
            llm_calls=2,
            total_tokens=100,
            error_count=0
        )
        aggregator.record_resolution_metrics(resolution_metrics)
        
        report = aggregator.get_comprehensive_report(hours=1)
        
        assert "report_timestamp" in report
        assert "period_hours" in report
        assert "resolution_performance" in report
        assert "tool_performance" in report
        assert "llm_usage" in report
        assert "system_health" in report
    
    def test_export_metrics_json(self):
        """Test exporting metrics to JSON."""
        aggregator = MetricsAggregator()
        
        # Add test data
        resolution_metrics = ResolutionMetrics(
            timestamp=datetime.now(),
            scenario_id="test_1",
            scenario_type="traffic",
            resolution_time_seconds=30.0,
            success=True,
            confidence_score=0.8,
            steps_count=3,
            tools_used=["check_traffic"],
            llm_calls=2,
            total_tokens=100,
            error_count=0
        )
        aggregator.record_resolution_metrics(resolution_metrics)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_file = f.name
        
        try:
            aggregator.export_metrics(export_file, hours=1, format="json")
            
            # Verify export
            with open(export_file) as f:
                export_data = json.load(f)
            
            assert "export_timestamp" in export_data
            assert "comprehensive_report" in export_data
            
        finally:
            Path(export_file).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])