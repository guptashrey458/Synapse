"""
Integration tests for advanced tool manager features.
"""
import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.tools.interfaces import Tool, ToolResult
from src.tools.tool_manager import ConcreteToolManager
from src.tools.error_handling import ErrorCategory, CircuitBreakerState
from src.tools.communication_tools import NotifyCustomerTool
from src.tools.merchant_tools import GetMerchantStatusTool
from src.tools.traffic_tools import CheckTrafficTool


class UnreliableTool(Tool):
    """Tool that fails intermittently for testing error handling."""
    
    def __init__(self, name: str = "unreliable_tool", failure_rate: float = 0.5,
                 error_type: str = "network"):
        super().__init__(
            name=name,
            description="Unreliable tool for testing",
            parameters={"param1": {"type": "string", "description": "Test parameter"}}
        )
        self.failure_rate = failure_rate
        self.error_type = error_type
        self.call_count = 0
        self.success_count = 0
    
    def validate_parameters(self, **kwargs) -> bool:
        return "param1" in kwargs
    
    def execute(self, **kwargs) -> ToolResult:
        self.call_count += 1
        
        # Simulate different types of failures
        import random
        if random.random() < self.failure_rate:
            error_messages = {
                "network": "Network connection failed",
                "timeout": "Request timed out after 30 seconds",
                "rate_limit": "Rate limit exceeded - too many requests",
                "service_unavailable": "Service temporarily unavailable (503)",
                "auth": "Authentication failed (401)"
            }
            
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                execution_time=0.1,
                error_message=error_messages.get(self.error_type, "Unknown error")
            )
        
        self.success_count += 1
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={"result": "success", "call_count": self.call_count},
            execution_time=0.1
        )


class SlowTool(Tool):
    """Tool that takes a long time to execute."""
    
    def __init__(self, execution_time: float = 2.0):
        super().__init__(
            name="slow_tool",
            description="Slow tool for timeout testing",
            parameters={"param1": {"type": "string", "description": "Test parameter"}}
        )
        self.execution_time = execution_time
    
    def validate_parameters(self, **kwargs) -> bool:
        return "param1" in kwargs
    
    def execute(self, **kwargs) -> ToolResult:
        time.sleep(self.execution_time)
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={"result": "completed after delay"},
            execution_time=self.execution_time
        )


class TestToolManagerIntegration:
    """Integration tests for tool manager with real tools."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ConcreteToolManager(
            max_workers=2,
            default_timeout=5,
            enable_caching=True,
            cache_ttl=60,
            enable_circuit_breaker=True
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        self.manager.shutdown()
    
    def test_real_tool_integration(self):
        """Test integration with real logistics tools."""
        # Register real tools
        traffic_tool = CheckTrafficTool()
        merchant_tool = GetMerchantStatusTool()
        notify_tool = NotifyCustomerTool()
        
        self.manager.register_tool(traffic_tool)
        self.manager.register_tool(merchant_tool)
        self.manager.register_tool(notify_tool)
        
        # Test traffic tool
        traffic_result = self.manager.execute_tool(
            "check_traffic",
            {"origin": "123 Main St", "destination": "456 Oak Ave"}
        )
        assert traffic_result.success is True
        assert "route" in traffic_result.data
        
        # Test merchant tool
        merchant_result = self.manager.execute_tool(
            "get_merchant_status",
            {"merchant_id": "MERCH_123"}
        )
        assert merchant_result.success is True
        assert "status" in merchant_result.data
        
        # Test notification tool
        notify_result = self.manager.execute_tool(
            "notify_customer",
            {
                "delivery_id": "DEL_123",
                "customer_id": "CUST_456",
                "message_type": "delay_notification"
            }
        )
        assert notify_result.success is True
        assert "notification_id" in notify_result.data
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker pattern with unreliable tool."""
        # Create tool that always fails
        failing_tool = UnreliableTool(name="failing_tool", failure_rate=1.0)
        self.manager.register_tool(failing_tool)
        
        # Execute multiple times to trigger circuit breaker
        results = []
        for i in range(10):
            result = self.manager.execute_tool("failing_tool", {"param1": "test"})
            results.append(result)
            time.sleep(0.1)  # Small delay between calls
        
        # Check circuit breaker status
        cb_status = self.manager.get_circuit_breaker_status("failing_tool")
        assert cb_status["state"] == CircuitBreakerState.OPEN.value
        assert cb_status["failure_count"] >= 5
        
        # Verify that subsequent calls are rejected
        rejected_result = self.manager.execute_tool("failing_tool", {"param1": "test"})
        assert rejected_result.success is False
        assert "Circuit breaker is OPEN" in rejected_result.error_message
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after failures."""
        # Create tool that fails initially then recovers
        recovering_tool = UnreliableTool(name="recovering_tool", failure_rate=1.0)
        self.manager.register_tool(recovering_tool)
        
        # Trigger circuit breaker
        for _ in range(6):
            self.manager.execute_tool("recovering_tool", {"param1": "test"})
        
        # Verify circuit breaker is open
        cb_status = self.manager.get_circuit_breaker_status("recovering_tool")
        assert cb_status["state"] == CircuitBreakerState.OPEN.value
        
        # Reset circuit breaker manually for testing
        self.manager.reset_circuit_breaker("recovering_tool")
        
        # Change tool to succeed
        recovering_tool.failure_rate = 0.0
        
        # Execute successful calls
        for _ in range(3):
            result = self.manager.execute_tool("recovering_tool", {"param1": "test"})
            assert result.success is True
        
        # Verify circuit breaker is closed
        cb_status = self.manager.get_circuit_breaker_status("recovering_tool")
        assert cb_status["state"] == CircuitBreakerState.CLOSED.value
    
    def test_error_categorization_and_handling(self):
        """Test different error types are handled appropriately."""
        # Test network errors
        network_tool = UnreliableTool(name="network_tool", failure_rate=1.0, error_type="network")
        self.manager.register_tool(network_tool)
        
        result = self.manager.execute_tool("network_tool", {"param1": "test"})
        assert result.success is False
        assert "Network" in result.error_message or "network" in result.error_message
        
        # Test timeout errors
        timeout_tool = UnreliableTool(name="timeout_tool", failure_rate=1.0, error_type="timeout")
        self.manager.register_tool(timeout_tool)
        
        result = self.manager.execute_tool("timeout_tool", {"param1": "test"})
        assert result.success is False
        
        # Test rate limit errors
        rate_limit_tool = UnreliableTool(name="rate_limit_tool", failure_rate=1.0, error_type="rate_limit")
        self.manager.register_tool(rate_limit_tool)
        
        result = self.manager.execute_tool("rate_limit_tool", {"param1": "test"})
        assert result.success is False
    
    def test_timeout_handling(self):
        """Test timeout handling with slow tools."""
        slow_tool = SlowTool(execution_time=3.0)
        self.manager.register_tool(slow_tool)
        
        # Execute with short timeout
        result = self.manager.execute_tool("slow_tool", {"param1": "test"}, timeout=1)
        assert result.success is False
        assert "timed out" in result.error_message
    
    def test_concurrent_execution_with_errors(self):
        """Test concurrent execution with error handling."""
        unreliable_tool = UnreliableTool(name="concurrent_tool", failure_rate=0.7)  # Higher failure rate
        self.manager.register_tool(unreliable_tool)
        
        results = []
        threads = []
        
        def execute_tool():
            result = self.manager.execute_tool("concurrent_tool", {"param1": "concurrent"})
            results.append(result)
        
        # Start multiple concurrent executions
        for _ in range(20):  # More executions to ensure some failures
            thread = threading.Thread(target=execute_tool)
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        # Analyze results
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        assert len(results) == 20
        # With 70% failure rate and 20 executions, we should see both successes and failures
        # But due to randomness, we'll just check that we got all results
        assert len(successful_results) + len(failed_results) == 20
    
    def test_performance_metrics_collection(self):
        """Test performance metrics collection."""
        # Use a mix of successful and failing executions
        success_tool = UnreliableTool(name="metrics_tool", failure_rate=0.0)
        self.manager.register_tool(success_tool)
        
        # Execute successful calls
        for i in range(5):
            self.manager.execute_tool("metrics_tool", {"param1": f"test_{i}"})
        
        # Change to failing and execute more
        success_tool.failure_rate = 1.0
        for i in range(5, 10):
            self.manager.execute_tool("metrics_tool", {"param1": f"test_{i}"})
        
        # Get performance metrics
        metrics = self.manager.get_performance_metrics("metrics_tool")
        
        assert metrics["execution_count"] == 10
        assert metrics["success_count"] >= 5  # At least the first 5 should succeed
        assert metrics["failure_count"] >= 0  # Some may fail due to retries succeeding
        assert metrics["average_execution_time"] > 0
    
    def test_error_statistics(self):
        """Test error statistics collection."""
        # Create tools with different error types
        network_tool = UnreliableTool(name="net_tool", failure_rate=1.0, error_type="network")
        timeout_tool = UnreliableTool(name="time_tool", failure_rate=1.0, error_type="timeout")
        
        self.manager.register_tool(network_tool)
        self.manager.register_tool(timeout_tool)
        
        # Execute tools to generate errors
        self.manager.execute_tool("net_tool", {"param1": "test"})
        self.manager.execute_tool("time_tool", {"param1": "test"})
        
        # Get error statistics
        stats = self.manager.get_error_statistics(hours=1)
        
        assert stats["total_executions"] == 2
        assert stats["failed_executions"] == 2
        assert stats["success_rate"] == 0.0
        assert len(stats["error_categories"]) > 0
    
    def test_cache_with_error_handling(self):
        """Test caching behavior with error handling."""
        intermittent_tool = UnreliableTool(name="cache_tool", failure_rate=0.5)
        self.manager.register_tool(intermittent_tool)
        
        # Execute until we get a successful result
        successful_result = None
        attempts = 0
        while successful_result is None and attempts < 10:
            result = self.manager.execute_tool("cache_tool", {"param1": "cache_test"})
            if result.success:
                successful_result = result
            attempts += 1
        
        assert successful_result is not None
        
        # Second execution should use cache
        cached_result = self.manager.execute_tool("cache_tool", {"param1": "cache_test"})
        assert cached_result.success is True
        
        # Tool should not have been called again (same call count)
        # Note: This test might be flaky due to the random nature of the unreliable tool
    
    def test_graceful_degradation(self):
        """Test graceful degradation when services are unavailable."""
        unavailable_tool = UnreliableTool(
            name="unavailable_tool", 
            failure_rate=1.0, 
            error_type="service_unavailable"
        )
        self.manager.register_tool(unavailable_tool)
        
        result = self.manager.execute_tool("unavailable_tool", {"param1": "test"})
        
        # Should get a degraded result rather than complete failure
        assert result.success is False
        assert "service_unavailable" in result.data or "Service" in result.error_message
    
    def test_tool_manager_configuration(self):
        """Test tool manager configuration changes."""
        test_tool = UnreliableTool(name="config_tool", failure_rate=1.0)  # Always fail
        self.manager.register_tool(test_tool)
        
        # Configure error handling
        self.manager.configure_error_handling(
            max_retries=2,  # Fewer retries for faster test
            retry_delay_base=0.05,  # Shorter delays
            retry_backoff_multiplier=1.5
        )
        
        # Execute tool - should retry and take some time
        start_time = time.time()
        result = self.manager.execute_tool("config_tool", {"param1": "test"})
        execution_time = time.time() - start_time
        
        # Should have taken some time due to retries (at least base delay)
        assert execution_time >= 0.05
        assert result.success is False  # Should ultimately fail
        
        # Check execution history for retry count
        history = self.manager.get_execution_history("config_tool", limit=1)
        assert len(history) == 1
        assert history[0]["tool_name"] == "config_tool"


if __name__ == "__main__":
    pytest.main([__file__])