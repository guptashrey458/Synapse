"""
Integration tests for reasoning engine and chain-of-thought logger.
"""
import pytest
from datetime import datetime
from unittest.mock import Mock

from src.reasoning.engine import ReActReasoningEngine, ReasoningConfig
from src.reasoning.logger import ConsoleChainOfThoughtLogger, LoggingConfig
from src.reasoning.interfaces import ReasoningContext
from src.agent.interfaces import ScenarioType, UrgencyLevel, EntityType
from src.agent.models import (
    ValidatedEntity, ValidatedDisruptionScenario, ValidatedReasoningTrace
)
from src.llm.interfaces import LLMResponse, Message, MessageRole, TokenUsage
from src.tools.interfaces import Tool, ToolResult


class MockTool(Tool):
    """Mock tool for integration testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.description = f"Mock tool {name}"
        self.parameters = {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "Test parameter"}
            },
            "required": ["param1"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute mock tool."""
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={"result": f"Mock result from {self.name}"},
            execution_time=0.1
        )
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate parameters."""
        return True


@pytest.fixture
def mock_llm_provider():
    """Create mock LLM provider for integration tests."""
    provider = Mock()
    
    # Mock reasoning step response
    provider.generate_response.return_value = LLMResponse(
        content="**Thought:** I need to check the traffic situation first.",
        messages=[],
        token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        model="gpt-4",
        finish_reason="stop",
        response_time=1.0,
        timestamp=datetime.now(),
        tool_calls=[{
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "check_traffic",
                "arguments": '{"location": "Highway 101"}'
            }
        }]
    )
    
    return provider


@pytest.fixture
def mock_tool_manager():
    """Create mock tool manager for integration tests."""
    manager = Mock()
    manager.get_available_tools.return_value = [
        MockTool("check_traffic"),
        MockTool("notify_customer")
    ]
    manager.execute_tool.return_value = ToolResult(
        tool_name="check_traffic",
        success=True,
        data={"traffic_status": "heavy", "estimated_delay": "15 minutes"},
        execution_time=0.5
    )
    return manager


@pytest.fixture
def sample_scenario():
    """Create sample disruption scenario for integration tests."""
    entities = [
        ValidatedEntity(
            text="Highway 101",
            entity_type=EntityType.ADDRESS,
            confidence=0.8,
            normalized_value="Highway 101"
        )
    ]
    
    return ValidatedDisruptionScenario(
        description="Driver stuck in traffic on Highway 101 due to accident",
        entities=entities,
        scenario_type=ScenarioType.TRAFFIC,
        urgency_level=UrgencyLevel.HIGH
    )


class TestReasoningIntegration:
    """Integration tests for reasoning engine and logger."""
    
    def test_reasoning_with_console_logging(self, mock_llm_provider, mock_tool_manager, sample_scenario):
        """Test reasoning engine with console logging integration."""
        # Setup
        reasoning_config = ReasoningConfig(max_reasoning_steps=3, enable_examples=False)
        logging_config = LoggingConfig(enable_console_output=False)  # Disable for test
        
        reasoning_engine = ReActReasoningEngine(mock_llm_provider, mock_tool_manager, reasoning_config)
        logger = ConsoleChainOfThoughtLogger(logging_config)
        
        # Create reasoning context
        context = ReasoningContext(
            scenario=sample_scenario,
            current_step=1,
            previous_steps=[],
            tool_results=[],
            available_tools=["check_traffic", "notify_customer"]
        )
        
        # Generate reasoning step
        step = reasoning_engine.generate_reasoning_step(context)
        
        # Log the step (should not raise any exceptions)
        logger.log_step(step)
        
        # Verify step was generated correctly
        assert step is not None
        assert step.step_number == 1
        assert "traffic situation" in step.thought.lower()
        assert step.action is not None
        assert step.action.tool_name == "check_traffic"
        assert len(step.tool_results) > 0
        assert step.tool_results[0].success is True
    
    def test_full_reasoning_trace_with_logging(self, mock_llm_provider, mock_tool_manager, sample_scenario):
        """Test complete reasoning trace with logging."""
        # Setup
        reasoning_config = ReasoningConfig(max_reasoning_steps=2, enable_examples=False)
        logging_config = LoggingConfig(enable_console_output=False, enable_file_logging=False)
        
        reasoning_engine = ReActReasoningEngine(mock_llm_provider, mock_tool_manager, reasoning_config)
        logger = ConsoleChainOfThoughtLogger(logging_config)
        
        # Create initial trace
        trace = ValidatedReasoningTrace(
            steps=[],
            scenario=sample_scenario,
            start_time=datetime.now()
        )
        
        # Simulate reasoning loop
        step_count = 0
        while reasoning_engine.should_continue_reasoning(trace) and step_count < 2:
            step_count += 1
            
            context = ReasoningContext(
                scenario=sample_scenario,
                current_step=step_count,
                previous_steps=trace.steps,
                tool_results=[],
                available_tools=["check_traffic", "notify_customer"]
            )
            
            # Generate step
            step = reasoning_engine.generate_reasoning_step(context)
            
            # Log step
            logger.log_step(step)
            
            # Add to trace
            trace.add_step(step)
        
        # Complete trace
        trace.complete_trace()
        
        # Test trace formatting
        formatted_trace = logger.format_trace(trace)
        assert "🧠 REASONING TRACE" in formatted_trace
        assert "Driver stuck in traffic" in formatted_trace
        assert f"Steps: {len(trace.steps)}" in formatted_trace
        
        # Test trace summary
        summary = logger.get_trace_summary(trace)
        assert f"{len(trace.steps)} steps" in summary
        assert "check_traffic" in summary
    
    def test_reasoning_evaluation_with_logging(self, mock_llm_provider, mock_tool_manager, sample_scenario):
        """Test reasoning evaluation with logging integration."""
        # Setup
        reasoning_config = ReasoningConfig(enable_examples=False)
        logging_config = LoggingConfig(enable_console_output=False)
        
        reasoning_engine = ReActReasoningEngine(mock_llm_provider, mock_tool_manager, reasoning_config)
        logger = ConsoleChainOfThoughtLogger(logging_config)
        
        # Create tool results
        tool_results = [
            ToolResult(
                tool_name="check_traffic",
                success=True,
                data={"status": "heavy", "delay": "15 min"},
                execution_time=0.5
            ),
            ToolResult(
                tool_name="notify_customer",
                success=True,
                data={"message_sent": True},
                execution_time=0.2
            )
        ]
        
        # Create context
        context = ReasoningContext(
            scenario=sample_scenario,
            current_step=1,
            previous_steps=[],
            tool_results=tool_results,
            available_tools=["check_traffic", "notify_customer"]
        )
        
        # Evaluate results
        evaluation = reasoning_engine.evaluate_tool_results(tool_results, context)
        
        # Verify evaluation
        assert evaluation is not None
        assert evaluation.confidence > 0.5
        assert "confidence" in evaluation.reasoning.lower()
        
        # Test that we can log information about the evaluation
        # (In a real scenario, this might be part of a reasoning step)
        assert evaluation.reasoning is not None
        assert len(evaluation.reasoning) > 0
    
    def test_circuit_breaker_with_logging(self, mock_llm_provider, mock_tool_manager, sample_scenario):
        """Test circuit breaker activation with logging."""
        # Setup with circuit breaker enabled
        reasoning_config = ReasoningConfig(
            enable_circuit_breaker=True,
            circuit_breaker_threshold=2,
            enable_examples=False
        )
        logging_config = LoggingConfig(enable_console_output=False)
        
        reasoning_engine = ReActReasoningEngine(mock_llm_provider, mock_tool_manager, reasoning_config)
        logger = ConsoleChainOfThoughtLogger(logging_config)
        
        # Force consecutive failures
        reasoning_engine._consecutive_failures = reasoning_config.circuit_breaker_threshold
        
        # Create context
        context = ReasoningContext(
            scenario=sample_scenario,
            current_step=1,
            previous_steps=[],
            tool_results=[],
            available_tools=["check_traffic"]
        )
        
        # Generate step (should trigger circuit breaker)
        step = reasoning_engine.generate_reasoning_step(context)
        
        # Log the step
        logger.log_step(step)
        
        # Verify circuit breaker was triggered
        assert "circuit breaker" in step.thought.lower()
        assert "terminated" in step.observation.lower()
    
    def test_logger_data_summarization(self):
        """Test logger's data summarization functionality."""
        logger = ConsoleChainOfThoughtLogger()
        
        # Test various data types
        test_cases = [
            ({}, "No data"),
            ({"status": "ok"}, '{"status": "ok"}'),
            ({"a": 1, "b": 2}, '{"a": 1, "b": 2}'),
            ({f"field_{i}": i for i in range(10)}, "10 fields")
        ]
        
        for data, expected in test_cases:
            result = logger._summarize_data(data)
            if "fields" in expected:
                assert "fields" in result
            else:
                assert result == expected