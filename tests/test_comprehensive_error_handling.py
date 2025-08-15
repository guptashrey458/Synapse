"""
Comprehensive tests for error handling and resilience features.
"""
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.agent.error_handler import (
    ComprehensiveErrorHandler, AgentErrorContext, AgentErrorType,
    LLMErrorHandler, ToolErrorHandler, ReasoningErrorHandler,
    ReasoningLoopDetector, ReasoningCircuitBreaker,
    create_llm_error_context, create_tool_error_context, create_reasoning_error_context
)
from src.tools.error_handling import ErrorSeverity, CircuitBreakerState
from src.tools.interfaces import ToolResult
from src.reasoning.interfaces import ReasoningStep
from src.agent.models import ValidatedReasoningStep


class TestReasoningLoopDetector:
    """Test reasoning loop detection."""
    
    def test_no_loop_with_different_thoughts(self):
        """Test that different thoughts don't trigger loop detection."""
        detector = ReasoningLoopDetector(max_similar_thoughts=3)
        
        assert not detector.add_thought("I need to check traffic")
        assert not detector.add_thought("Let me get merchant status")
        assert not detector.add_thought("I should notify the customer")
        assert not detector.loop_detected
    
    def test_loop_detection_with_identical_thoughts(self):
        """Test loop detection with identical thoughts."""
        detector = ReasoningLoopDetector(max_similar_thoughts=3)
        
        assert not detector.add_thought("I need to check traffic")
        assert not detector.add_thought("I need to check traffic")
        assert detector.add_thought("I need to check traffic")  # Third identical thought
        assert detector.loop_detected
    
    def test_loop_detection_with_similar_thoughts(self):
        """Test loop detection with semantically similar thoughts."""
        detector = ReasoningLoopDetector(max_similar_thoughts=3, similarity_threshold=0.7)
        
        assert not detector.add_thought("I need to check the traffic situation")
        assert not detector.add_thought("I should check traffic conditions")
        assert detector.add_thought("Let me check traffic status")
        assert detector.loop_detected
    
    def test_reset_functionality(self):
        """Test that reset clears loop detection state."""
        detector = ReasoningLoopDetector(max_similar_thoughts=2)
        
        detector.add_thought("same thought")
        detector.add_thought("same thought")
        assert detector.loop_detected
        
        detector.reset()
        assert not detector.loop_detected
        assert len(detector.recent_thoughts) == 0


class TestReasoningCircuitBreaker:
    """Test reasoning circuit breaker functionality."""
    
    def test_initial_state_closed(self):
        """Test that circuit breaker starts in closed state."""
        breaker = ReasoningCircuitBreaker()
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.can_execute()
    
    def test_loop_detection_triggers_failure(self):
        """Test that loop detection triggers circuit breaker failure."""
        breaker = ReasoningCircuitBreaker()
        
        # Create steps that will trigger loop detection
        step1 = ValidatedReasoningStep(
            step_number=1,
            thought="I need to check traffic",
            timestamp=datetime.now()
        )
        step2 = ValidatedReasoningStep(
            step_number=2,
            thought="I need to check traffic",
            timestamp=datetime.now()
        )
        step3 = ValidatedReasoningStep(
            step_number=3,
            thought="I need to check traffic",
            timestamp=datetime.now()
        )
        
        assert breaker.record_reasoning_step(step1)
        assert breaker.record_reasoning_step(step2)
        assert not breaker.record_reasoning_step(step3)  # Should detect loop and fail
        
        # After enough failures, circuit should open
        assert breaker.state == CircuitBreakerState.OPEN
    
    def test_failed_tools_trigger_failure(self):
        """Test that failed tools trigger circuit breaker failure."""
        breaker = ReasoningCircuitBreaker()
        
        # Create step with mostly failed tool results
        failed_result = ToolResult(
            tool_name="test_tool",
            success=False,
            data={},
            execution_time=1.0,
            error_message="Tool failed"
        )
        
        step = ValidatedReasoningStep(
            step_number=1,
            thought="Testing tool failures",
            timestamp=datetime.now(),
            tool_results=[failed_result, failed_result, failed_result]
        )
        
        assert not breaker.record_reasoning_step(step)  # Should fail due to tool failures
    
    def test_get_loop_info(self):
        """Test getting loop information from circuit breaker."""
        breaker = ReasoningCircuitBreaker()
        
        step = ValidatedReasoningStep(
            step_number=1,
            thought="test thought",
            timestamp=datetime.now()
        )
        
        breaker.record_reasoning_step(step)
        loop_info = breaker.get_loop_info()
        
        assert "loop_detected" in loop_info
        assert "loop_count" in loop_info
        assert "recent_thoughts_count" in loop_info


class TestLLMErrorHandler:
    """Test LLM error handling."""
    
    def test_can_handle_llm_errors(self):
        """Test that handler correctly identifies LLM errors."""
        handler = LLMErrorHandler()
        
        llm_context = AgentErrorContext(
            error_type=AgentErrorType.LLM_ERROR,
            component="llm_provider",
            operation="generate_response",
            error_message="API rate limit exceeded"
        )
        
        tool_context = AgentErrorContext(
            error_type=AgentErrorType.TOOL_ERROR,
            component="tool_manager",
            operation="execute_tool",
            error_message="Tool failed"
        )
        
        assert handler.can_handle(llm_context)
        assert not handler.can_handle(tool_context)
    
    def test_retry_logic_with_backoff(self):
        """Test retry logic with exponential backoff."""
        handler = LLMErrorHandler()
        
        context = AgentErrorContext(
            error_type=AgentErrorType.LLM_ERROR,
            component="llm_provider",
            operation="generate_response",
            error_message="Temporary failure",
            retry_count=1,
            max_retries=3
        )
        
        start_time = time.time()
        result = handler.handle(context)
        end_time = time.time()
        
        # Should return None to signal retry
        assert result is None
        # Should have waited for backoff (2^1 = 2 seconds, but we use min with 30)
        assert end_time - start_time >= 2
    
    def test_fallback_after_max_retries(self):
        """Test fallback response after max retries."""
        handler = LLMErrorHandler()
        
        context = AgentErrorContext(
            error_type=AgentErrorType.LLM_ERROR,
            component="llm_provider",
            operation="generate_response",
            error_message="Persistent failure",
            retry_count=3,
            max_retries=3
        )
        
        result = handler.handle(context)
        
        assert result is not None
        assert result["fallback"] is True
        assert "suggested_actions" in result
    
    def test_recovery_actions_for_rate_limit(self):
        """Test recovery actions for rate limit errors."""
        handler = LLMErrorHandler()
        
        context = AgentErrorContext(
            error_type=AgentErrorType.LLM_ERROR,
            component="llm_provider",
            operation="generate_response",
            error_message="Rate limit exceeded"
        )
        
        actions = handler.get_recovery_actions(context)
        
        assert len(actions) > 0
        assert any("wait_and_retry" in action.action_type for action in actions)
    
    def test_recovery_actions_for_auth_error(self):
        """Test recovery actions for authentication errors."""
        handler = LLMErrorHandler()
        
        context = AgentErrorContext(
            error_type=AgentErrorType.LLM_ERROR,
            component="llm_provider",
            operation="generate_response",
            error_message="Authentication failed"
        )
        
        actions = handler.get_recovery_actions(context)
        
        assert len(actions) > 0
        assert any("check_credentials" in action.action_type for action in actions)


class TestToolErrorHandler:
    """Test tool error handling."""
    
    def test_can_handle_tool_errors(self):
        """Test that handler correctly identifies tool errors."""
        handler = ToolErrorHandler()
        
        tool_context = AgentErrorContext(
            error_type=AgentErrorType.TOOL_ERROR,
            component="tool_manager",
            operation="execute_tool",
            error_message="Tool execution failed"
        )
        
        llm_context = AgentErrorContext(
            error_type=AgentErrorType.LLM_ERROR,
            component="llm_provider",
            operation="generate_response",
            error_message="LLM failed"
        )
        
        assert handler.can_handle(tool_context)
        assert not handler.can_handle(llm_context)
    
    def test_fallback_data_generation(self):
        """Test generation of fallback data for failed tools."""
        handler = ToolErrorHandler()
        
        context = AgentErrorContext(
            error_type=AgentErrorType.TOOL_ERROR,
            component="tool_manager",
            operation="execute_tool",
            error_message="Tool failed",
            context_data={"tool_name": "check_traffic"}
        )
        
        result = handler.handle(context)
        
        assert isinstance(result, ToolResult)
        assert not result.success
        assert "fallback" in result.error_message.lower()
        assert result.data is not None
    
    def test_recovery_actions(self):
        """Test recovery actions for tool errors."""
        handler = ToolErrorHandler()
        
        context = AgentErrorContext(
            error_type=AgentErrorType.TOOL_ERROR,
            component="tool_manager",
            operation="execute_tool",
            error_message="Network timeout",
            context_data={"tool_name": "check_traffic"}
        )
        
        actions = handler.get_recovery_actions(context)
        
        assert len(actions) > 0
        action_types = [action.action_type for action in actions]
        assert "retry_tool" in action_types
        assert "use_fallback_data" in action_types


class TestReasoningErrorHandler:
    """Test reasoning error handling."""
    
    def test_can_handle_reasoning_errors(self):
        """Test that handler correctly identifies reasoning errors."""
        handler = ReasoningErrorHandler()
        
        reasoning_context = AgentErrorContext(
            error_type=AgentErrorType.REASONING_ERROR,
            component="reasoning_engine",
            operation="generate_step",
            error_message="Reasoning loop detected"
        )
        
        tool_context = AgentErrorContext(
            error_type=AgentErrorType.TOOL_ERROR,
            component="tool_manager",
            operation="execute_tool",
            error_message="Tool failed"
        )
        
        assert handler.can_handle(reasoning_context)
        assert not handler.can_handle(tool_context)
    
    def test_loop_breaking(self):
        """Test breaking reasoning loops."""
        handler = ReasoningErrorHandler()
        
        context = AgentErrorContext(
            error_type=AgentErrorType.REASONING_ERROR,
            component="reasoning_engine",
            operation="generate_step",
            error_message="Reasoning loop detected"
        )
        
        result = handler.handle(context)
        
        assert result is not None
        assert result["loop_broken"] is True
        assert "confidence" in result
    
    def test_timeout_handling(self):
        """Test handling of reasoning timeouts."""
        handler = ReasoningErrorHandler()
        
        context = AgentErrorContext(
            error_type=AgentErrorType.REASONING_ERROR,
            component="reasoning_engine",
            operation="generate_step",
            error_message="Reasoning timeout exceeded"
        )
        
        result = handler.handle(context)
        
        assert result is not None
        assert result["timeout_handled"] is True
    
    def test_emergency_plan_creation(self):
        """Test creation of emergency plans when circuit breaker is open."""
        handler = ReasoningErrorHandler()
        
        # Force circuit breaker to open
        handler.circuit_breaker.state = CircuitBreakerState.OPEN
        
        context = AgentErrorContext(
            error_type=AgentErrorType.REASONING_ERROR,
            component="reasoning_engine",
            operation="generate_step",
            error_message="System failure"
        )
        
        result = handler.handle(context)
        
        assert result is not None
        assert result["emergency_plan"] is True
        assert "immediate_actions" in result
    
    def test_record_reasoning_step(self):
        """Test recording reasoning steps for circuit breaker monitoring."""
        handler = ReasoningErrorHandler()
        
        step = ValidatedReasoningStep(
            step_number=1,
            thought="I need to analyze this situation",
            timestamp=datetime.now()
        )
        
        # Should return True for normal step
        assert handler.record_reasoning_step(step) is True
        
        # Create looping steps
        loop_step = ValidatedReasoningStep(
            step_number=2,
            thought="same thought",
            timestamp=datetime.now()
        )
        
        handler.record_reasoning_step(loop_step)
        handler.record_reasoning_step(loop_step)
        # Third identical thought should trigger loop detection
        assert handler.record_reasoning_step(loop_step) is False


class TestComprehensiveErrorHandler:
    """Test the main comprehensive error handler."""
    
    def test_initialization(self):
        """Test proper initialization of comprehensive error handler."""
        handler = ComprehensiveErrorHandler()
        
        assert len(handler.handlers) == 3  # LLM, Tool, Reasoning handlers
        assert handler.global_circuit_breaker is not None
        assert len(handler.error_history) == 0
    
    def test_error_routing_to_correct_handler(self):
        """Test that errors are routed to the correct specialized handler."""
        handler = ComprehensiveErrorHandler()
        
        # Test LLM error routing
        llm_context = create_llm_error_context(
            "generate_response", 
            Exception("LLM failed")
        )
        
        with patch.object(handler.handlers[0], 'handle', return_value={"handled": True}) as mock_handle:
            result = handler.handle_error(llm_context)
            mock_handle.assert_called_once()
            assert result["handled"] is True
    
    def test_error_history_tracking(self):
        """Test that error history is properly tracked."""
        handler = ComprehensiveErrorHandler()
        
        context = create_tool_error_context(
            "test_tool",
            "execute",
            Exception("Tool failed")
        )
        
        handler.handle_error(context)
        
        assert len(handler.error_history) == 1
        assert handler.error_history[0].error_type == AgentErrorType.TOOL_ERROR
    
    def test_global_circuit_breaker_activation(self):
        """Test that global circuit breaker activates after multiple failures."""
        handler = ComprehensiveErrorHandler()
        
        # Force global circuit breaker to open
        handler.global_circuit_breaker.state = CircuitBreakerState.OPEN
        
        context = create_llm_error_context(
            "generate_response",
            Exception("System failure")
        )
        
        result = handler.handle_error(context)
        
        assert result is not None
        assert result["system_failure"] is True
    
    def test_unhandled_error_response(self):
        """Test response for unhandled errors."""
        handler = ComprehensiveErrorHandler()
        
        # Create an error type that no handler can handle
        context = AgentErrorContext(
            error_type=AgentErrorType.SYSTEM_ERROR,  # Not handled by any specialized handler
            component="unknown",
            operation="unknown",
            error_message="Unknown error"
        )
        
        result = handler.handle_error(context)
        
        assert result is not None
        assert result["unhandled_error"] is True
    
    def test_recovery_suggestions_aggregation(self):
        """Test aggregation of recovery suggestions from multiple handlers."""
        handler = ComprehensiveErrorHandler()
        
        context = create_llm_error_context(
            "generate_response",
            Exception("Rate limit exceeded")
        )
        
        suggestions = handler.get_recovery_suggestions(context)
        
        assert len(suggestions) > 0
        # Should be sorted by success probability
        for i in range(len(suggestions) - 1):
            assert suggestions[i].success_probability >= suggestions[i + 1].success_probability
    
    def test_error_statistics_generation(self):
        """Test generation of error statistics."""
        handler = ComprehensiveErrorHandler()
        
        # Add some test errors
        contexts = [
            create_llm_error_context("op1", Exception("LLM error 1")),
            create_llm_error_context("op2", Exception("LLM error 2")),
            create_tool_error_context("tool1", "op3", Exception("Tool error 1")),
        ]
        
        for context in contexts:
            handler.handle_error(context)
        
        stats = handler.get_error_statistics(hours=1)
        
        assert stats["total_errors"] == 3
        assert "error_types" in stats
        assert stats["error_types"]["llm_error"] == 2
        assert stats["error_types"]["tool_error"] == 1
        assert "most_common_error" in stats
    
    def test_circuit_breaker_reset(self):
        """Test resetting all circuit breakers."""
        handler = ComprehensiveErrorHandler()
        
        # Force circuit breakers to open
        handler.global_circuit_breaker.state = CircuitBreakerState.OPEN
        handler.global_circuit_breaker.failure_count = 5
        
        handler.reset_circuit_breakers()
        
        assert handler.global_circuit_breaker.state == CircuitBreakerState.CLOSED
        assert handler.global_circuit_breaker.failure_count == 0
    
    def test_error_handling_context_manager(self):
        """Test the error handling context manager."""
        handler = ComprehensiveErrorHandler()
        
        # Test successful operation
        with handler.error_handling_context(
            AgentErrorType.LLM_ERROR, 
            "test_component", 
            "test_operation"
        ):
            pass  # No exception
        
        # Test exception handling
        with patch.object(handler, 'handle_error', return_value={"recovered": True}):
            with handler.error_handling_context(
                AgentErrorType.LLM_ERROR,
                "test_component", 
                "test_operation"
            ):
                raise Exception("Test error")
        
        # Should have recorded the error
        assert len(handler.error_history) == 1


class TestErrorContextCreation:
    """Test error context creation functions."""
    
    def test_create_llm_error_context(self):
        """Test creation of LLM error context."""
        error = Exception("LLM failed")
        context = create_llm_error_context("generate_response", error)
        
        assert context.error_type == AgentErrorType.LLM_ERROR
        assert context.component == "llm_provider"
        assert context.operation == "generate_response"
        assert context.original_exception == error
    
    def test_create_tool_error_context(self):
        """Test creation of tool error context."""
        error = Exception("Tool failed")
        context = create_tool_error_context("test_tool", "execute", error)
        
        assert context.error_type == AgentErrorType.TOOL_ERROR
        assert context.component == "tool_manager"
        assert context.operation == "execute"
        assert context.context_data["tool_name"] == "test_tool"
        assert context.original_exception == error
    
    def test_create_reasoning_error_context(self):
        """Test creation of reasoning error context."""
        error = Exception("Reasoning failed")
        context = create_reasoning_error_context("generate_step", error)
        
        assert context.error_type == AgentErrorType.REASONING_ERROR
        assert context.component == "reasoning_engine"
        assert context.operation == "generate_step"
        assert context.original_exception == error


class TestErrorRecoveryIntegration:
    """Test integration of error recovery with system components."""
    
    @pytest.fixture
    def mock_tool_manager(self):
        """Create a mock tool manager."""
        tool_manager = Mock()
        tool_manager.execute_tool.side_effect = Exception("Tool failed")
        return tool_manager
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        llm_provider = Mock()
        llm_provider.generate_response.side_effect = Exception("LLM failed")
        return llm_provider
    
    def test_tool_error_recovery_integration(self, mock_tool_manager):
        """Test integration of tool error recovery."""
        handler = ComprehensiveErrorHandler()
        
        # Simulate tool failure and recovery
        try:
            mock_tool_manager.execute_tool("test_tool", {})
        except Exception as e:
            context = create_tool_error_context("test_tool", "execute", e)
            result = handler.handle_error(context)
            
            assert result is not None
            assert isinstance(result, ToolResult)
            assert not result.success  # Fallback result
    
    def test_llm_error_recovery_integration(self, mock_llm_provider):
        """Test integration of LLM error recovery."""
        handler = ComprehensiveErrorHandler()
        
        # Simulate LLM failure and recovery
        try:
            mock_llm_provider.generate_response([])
        except Exception as e:
            context = create_llm_error_context("generate_response", e)
            result = handler.handle_error(context)
            
            # Should get fallback response after max retries
            context.retry_count = context.max_retries
            result = handler.handle_error(context)
            
            assert result is not None
            assert result["fallback"] is True
    
    def test_reasoning_loop_prevention(self):
        """Test prevention of reasoning loops through error handling."""
        handler = ComprehensiveErrorHandler()
        reasoning_handler = None
        
        # Find the reasoning error handler
        for h in handler.handlers:
            if isinstance(h, ReasoningErrorHandler):
                reasoning_handler = h
                break
        
        assert reasoning_handler is not None
        
        # Simulate reasoning steps that would create a loop
        steps = [
            ValidatedReasoningStep(
                step_number=i,
                thought="I need to check traffic status",
                timestamp=datetime.now()
            )
            for i in range(1, 4)
        ]
        
        # First two steps should be fine
        assert reasoning_handler.record_reasoning_step(steps[0])
        assert reasoning_handler.record_reasoning_step(steps[1])
        
        # Third identical step should trigger loop detection
        assert not reasoning_handler.record_reasoning_step(steps[2])
        
        # Circuit breaker should now prevent further reasoning
        assert not reasoning_handler.circuit_breaker.can_execute()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])