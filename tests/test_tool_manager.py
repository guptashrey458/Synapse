"""
Unit tests for the tool management system.
"""
import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.tools.interfaces import Tool, ToolResult
from src.tools.tool_manager import ConcreteToolManager, ToolExecutionStatus


class MockTool(Tool):
    """Mock tool for testing."""
    
    def __init__(self, name: str = "mock_tool", should_fail: bool = False, 
                 execution_time: float = 0.1, validate_result: bool = True):
        super().__init__(
            name=name,
            description="Mock tool for testing",
            parameters={
                "param1": {"type": "string", "description": "Test parameter"},
                "param2": {"type": "number", "description": "Optional parameter", "required": False}
            }
        )
        self.should_fail = should_fail
        self.execution_time = execution_time
        self.validate_result = validate_result
        self.call_count = 0
        self.last_parameters = None
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate parameters."""
        if not self.validate_result:
            return False
        return "param1" in kwargs and bool(kwargs["param1"])
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute the mock tool."""
        self.call_count += 1
        self.last_parameters = kwargs.copy()
        
        # Simulate execution time
        time.sleep(self.execution_time)
        
        if self.should_fail:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                execution_time=self.execution_time,
                error_message="Mock tool failure"
            )
        
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={"result": "success", "input": kwargs},
            execution_time=self.execution_time
        )


class TestConcreteToolManager:
    """Test cases for ConcreteToolManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ConcreteToolManager(
            max_workers=2,
            default_timeout=5,
            enable_caching=True,
            cache_ttl=60
        )
        self.mock_tool = MockTool()
    
    def teardown_method(self):
        """Clean up after tests."""
        self.manager.shutdown()
    
    def test_tool_registration(self):
        """Test tool registration functionality."""
        # Test successful registration
        self.manager.register_tool(self.mock_tool)
        
        # Verify tool is registered
        tools = self.manager.get_available_tools()
        assert len(tools) == 1
        assert tools[0].name == "mock_tool"
        
        # Test duplicate registration fails
        with pytest.raises(ValueError, match="already registered"):
            self.manager.register_tool(self.mock_tool)
    
    def test_get_tool(self):
        """Test tool retrieval."""
        self.manager.register_tool(self.mock_tool)
        
        # Test successful retrieval
        tool = self.manager.get_tool("mock_tool")
        assert tool is not None
        assert tool.name == "mock_tool"
        
        # Test non-existent tool
        tool = self.manager.get_tool("non_existent")
        assert tool is None
    
    def test_tool_execution_success(self):
        """Test successful tool execution."""
        self.manager.register_tool(self.mock_tool)
        
        parameters = {"param1": "test_value", "param2": 42}
        result = self.manager.execute_tool("mock_tool", parameters)
        
        assert result.success is True
        assert result.tool_name == "mock_tool"
        assert result.data["result"] == "success"
        assert result.data["input"] == parameters
        assert result.execution_time > 0
        
        # Verify tool was called
        assert self.mock_tool.call_count == 1
        assert self.mock_tool.last_parameters == parameters
    
    def test_tool_execution_failure(self):
        """Test tool execution failure handling."""
        failing_tool = MockTool(name="failing_tool", should_fail=True)
        self.manager.register_tool(failing_tool)
        
        result = self.manager.execute_tool("failing_tool", {"param1": "test"})
        
        assert result.success is False
        assert "Mock tool failure" in result.error_message  # Error message may be wrapped by error handler
        assert result.tool_name == "failing_tool"
    
    def test_tool_not_found(self):
        """Test execution of non-existent tool."""
        result = self.manager.execute_tool("non_existent", {"param1": "test"})
        
        assert result.success is False
        assert "not found" in result.error_message
    
    def test_parameter_validation_failure(self):
        """Test parameter validation failure."""
        invalid_tool = MockTool(name="invalid_tool", validate_result=False)
        self.manager.register_tool(invalid_tool)
        
        result = self.manager.execute_tool("invalid_tool", {"param1": "test"})
        
        assert result.success is False
        assert "validation failed" in result.error_message.lower()
    
    def test_tool_timeout(self):
        """Test tool execution timeout."""
        slow_tool = MockTool(name="slow_tool", execution_time=2.0)
        self.manager.register_tool(slow_tool)
        
        result = self.manager.execute_tool("slow_tool", {"param1": "test"}, timeout=1)
        
        assert result.success is False
        assert "timed out" in result.error_message
    
    def test_retry_logic(self):
        """Test retry logic on tool failure."""
        # Create a tool that fails first two times, then succeeds
        class RetryTool(MockTool):
            def __init__(self):
                super().__init__(name="retry_tool")
                self.attempt_count = 0
            
            def execute(self, **kwargs):
                self.attempt_count += 1
                if self.attempt_count < 3:
                    return ToolResult(
                        tool_name=self.name,
                        success=False,
                        data={},
                        execution_time=0.1,
                        error_message=f"Attempt {self.attempt_count} failed"
                    )
                return super().execute(**kwargs)
        
        retry_tool = RetryTool()
        self.manager.register_tool(retry_tool)
        
        result = self.manager.execute_tool("retry_tool", {"param1": "test"})
        
        assert result.success is True
        assert retry_tool.attempt_count == 3
    
    def test_caching(self):
        """Test result caching functionality."""
        self.manager.register_tool(self.mock_tool)
        
        parameters = {"param1": "test_cache"}
        
        # First execution
        result1 = self.manager.execute_tool("mock_tool", parameters)
        assert result1.success is True
        assert self.mock_tool.call_count == 1
        
        # Second execution should use cache
        result2 = self.manager.execute_tool("mock_tool", parameters)
        assert result2.success is True
        assert self.mock_tool.call_count == 1  # Tool not called again
        
        # Different parameters should not use cache
        result3 = self.manager.execute_tool("mock_tool", {"param1": "different"})
        assert result3.success is True
        assert self.mock_tool.call_count == 2  # Tool called again
    
    def test_cache_expiration(self):
        """Test cache expiration."""
        # Create manager with very short cache TTL
        short_cache_manager = ConcreteToolManager(cache_ttl=1)
        short_cache_manager.register_tool(self.mock_tool)
        
        parameters = {"param1": "test_expiry"}
        
        # First execution
        result1 = short_cache_manager.execute_tool("mock_tool", parameters)
        assert result1.success is True
        assert self.mock_tool.call_count == 1
        
        # Wait for cache to expire
        time.sleep(1.1)
        
        # Second execution should not use expired cache
        result2 = short_cache_manager.execute_tool("mock_tool", parameters)
        assert result2.success is True
        assert self.mock_tool.call_count == 2
        
        short_cache_manager.shutdown()
    
    def test_tool_enable_disable(self):
        """Test tool enable/disable functionality."""
        self.manager.register_tool(self.mock_tool)
        
        # Tool should be enabled by default
        tools = self.manager.get_available_tools()
        assert len(tools) == 1
        
        # Disable tool
        assert self.manager.disable_tool("mock_tool") is True
        tools = self.manager.get_available_tools()
        assert len(tools) == 0
        
        # Execution should fail when disabled
        result = self.manager.execute_tool("mock_tool", {"param1": "test"})
        assert result.success is False
        assert "disabled" in result.error_message
        
        # Re-enable tool
        assert self.manager.enable_tool("mock_tool") is True
        tools = self.manager.get_available_tools()
        assert len(tools) == 1
        
        # Execution should work again
        result = self.manager.execute_tool("mock_tool", {"param1": "test"})
        assert result.success is True
    
    def test_tool_metadata(self):
        """Test tool metadata tracking."""
        self.manager.register_tool(self.mock_tool)
        
        # Execute tool a few times
        self.manager.execute_tool("mock_tool", {"param1": "test1"})
        self.manager.execute_tool("mock_tool", {"param1": "test2"})
        
        metadata = self.manager.get_tool_metadata("mock_tool")
        assert metadata is not None
        assert metadata["name"] == "mock_tool"
        assert metadata["execution_count"] == 2
        assert metadata["success_count"] == 2
        assert metadata["failure_count"] == 0
        assert metadata["success_rate"] == 1.0
        assert metadata["average_execution_time"] > 0
        assert metadata["enabled"] is True
    
    def test_execution_history(self):
        """Test execution history tracking."""
        self.manager.register_tool(self.mock_tool)
        
        # Execute tool
        self.manager.execute_tool("mock_tool", {"param1": "test"})
        
        history = self.manager.get_execution_history()
        assert len(history) == 1
        
        entry = history[0]
        assert entry["tool_name"] == "mock_tool"
        assert entry["status"] == ToolExecutionStatus.COMPLETED.value
        assert entry["success"] is True
        assert "start_time" in entry
        assert "end_time" in entry
    
    def test_cache_management(self):
        """Test cache management operations."""
        self.manager.register_tool(self.mock_tool)
        
        # Execute to populate cache
        self.manager.execute_tool("mock_tool", {"param1": "test1"})
        self.manager.execute_tool("mock_tool", {"param1": "test2"})
        
        # Clear cache for specific tool
        cleared = self.manager.clear_cache("mock_tool")
        assert cleared == 2
        
        # Execute again to populate cache
        self.manager.execute_tool("mock_tool", {"param1": "test3"})
        
        # Clear all cache
        cleared = self.manager.clear_cache()
        assert cleared == 1
    
    def test_concurrent_execution(self):
        """Test concurrent tool execution."""
        slow_tool = MockTool(name="slow_tool", execution_time=0.5)
        self.manager.register_tool(slow_tool)
        
        results = []
        threads = []
        
        def execute_tool():
            result = self.manager.execute_tool("slow_tool", {"param1": "concurrent"})
            results.append(result)
        
        # Start multiple concurrent executions
        for _ in range(3):
            thread = threading.Thread(target=execute_tool)
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        # All should succeed
        assert len(results) == 3
        assert all(result.success for result in results)
        
        # Tool should have been called 3 times (no caching for concurrent calls)
        assert slow_tool.call_count == 3
    
    def test_all_tool_metadata(self):
        """Test getting metadata for all tools."""
        tool1 = MockTool(name="tool1")
        tool2 = MockTool(name="tool2")
        
        self.manager.register_tool(tool1)
        self.manager.register_tool(tool2)
        
        all_metadata = self.manager.get_all_tool_metadata()
        assert len(all_metadata) == 2
        assert "tool1" in all_metadata
        assert "tool2" in all_metadata
    
    def test_cleanup_expired_cache(self):
        """Test cleanup of expired cache entries."""
        # Create manager with very short cache TTL
        short_cache_manager = ConcreteToolManager(cache_ttl=1)
        short_cache_manager.register_tool(self.mock_tool)
        
        # Execute to populate cache
        short_cache_manager.execute_tool("mock_tool", {"param1": "test1"})
        short_cache_manager.execute_tool("mock_tool", {"param1": "test2"})
        
        # Wait for cache to expire
        time.sleep(1.1)
        
        # Cleanup expired entries
        cleaned = short_cache_manager.cleanup_expired_cache()
        assert cleaned == 2
        
        short_cache_manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])