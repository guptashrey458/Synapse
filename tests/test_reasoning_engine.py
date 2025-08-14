"""
Unit tests for the reasoning engine implementation.
"""
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

from src.reasoning.engine import ReActReasoningEngine, ReasoningConfig
from src.reasoning.interfaces import ReasoningContext, Evaluation
from src.agent.interfaces import (
    DisruptionScenario, ScenarioType, UrgencyLevel, EntityType
)
from src.agent.models import (
    ValidatedEntity, ValidatedDisruptionScenario, ValidatedReasoningStep,
    ValidatedReasoningTrace, ToolAction, ToolResult
)
from src.llm.interfaces import LLMResponse, Message, MessageRole, TokenUsage
from src.tools.interfaces import Tool, ToolResult as BaseToolResult


class MockTool(Tool):
    """Mock tool for testing."""
    
    def __init__(self, name: str, should_succeed: bool = True):
        self.name = name
        self.description = f"Mock tool {name}"
        self.parameters = {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "Test parameter"}
            },
            "required": ["param1"]
        }
        self.should_succeed = should_succeed
    
    def execute(self, **kwargs) -> BaseToolResult:
        """Execute mock tool."""
        if self.should_succeed:
            return BaseToolResult(
                tool_name=self.name,
                success=True,
                data={"result": f"Mock result from {self.name}", "param1": kwargs.get("param1")},
                execution_time=0.1
            )
        else:
            return BaseToolResult(
                tool_name=self.name,
                success=False,
                data={},
                execution_time=0.1,
                error_message="Mock tool failure"
            )
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate parameters."""
        return "param1" in kwargs


@pytest.fixture
def mock_llm_provider():
    """Create mock LLM provider."""
    provider = Mock()
    provider.generate_response.return_value = LLMResponse(
        content="**Thought:** I need to check the traffic situation to understand the delay.",
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
    """Create mock tool manager."""
    manager = Mock()
    manager.get_available_tools.return_value = [
        MockTool("check_traffic"),
        MockTool("notify_customer"),
        MockTool("get_merchant_status")
    ]
    manager.execute_tool.return_value = BaseToolResult(
        tool_name="check_traffic",
        success=True,
        data={"traffic_status": "heavy", "estimated_delay": "15 minutes"},
        execution_time=0.5
    )
    return manager


@pytest.fixture
def sample_scenario():
    """Create sample disruption scenario."""
    entities = [
        ValidatedEntity(
            text="Highway 101",
            entity_type=EntityType.ADDRESS,
            confidence=0.8,
            normalized_value="Highway 101"
        ),
        ValidatedEntity(
            text="John",
            entity_type=EntityType.PERSON,
            confidence=0.9,
            normalized_value="John"
        )
    ]
    
    return ValidatedDisruptionScenario(
        description="Driver John is stuck in traffic on Highway 101 due to an accident.",
        entities=entities,
        scenario_type=ScenarioType.TRAFFIC,
        urgency_level=UrgencyLevel.HIGH
    )


@pytest.fixture
def reasoning_context(sample_scenario):
    """Create reasoning context."""
    return ReasoningContext(
        scenario=sample_scenario,
        current_step=1,
        previous_steps=[],
        tool_results=[],
        available_tools=["check_traffic", "notify_customer", "get_merchant_status"]
    )


@pytest.fixture
def reasoning_engine(mock_llm_provider, mock_tool_manager):
    """Create reasoning engine with mocks."""
    config = ReasoningConfig(max_reasoning_steps=5, enable_examples=False)
    return ReActReasoningEngine(mock_llm_provider, mock_tool_manager, config)


class TestReActReasoningEngine:
    """Test cases for ReAct reasoning engine."""
    
    def test_initialization(self, mock_llm_provider, mock_tool_manager):
        """Test reasoning engine initialization."""
        config = ReasoningConfig(max_reasoning_steps=10)
        engine = ReActReasoningEngine(mock_llm_provider, mock_tool_manager, config)
        
        assert engine.llm_provider == mock_llm_provider
        assert engine.tool_manager == mock_tool_manager
        assert engine.config.max_reasoning_steps == 10
        assert engine._consecutive_failures == 0
    
    def test_generate_reasoning_step_success(self, reasoning_engine, reasoning_context):
        """Test successful reasoning step generation."""
        step = reasoning_engine.generate_reasoning_step(reasoning_context)
        
        assert isinstance(step, ValidatedReasoningStep)
        assert step.step_number == 1
        assert "traffic situation" in step.thought.lower()
        assert step.action is not None
        assert step.action.tool_name == "check_traffic"
        assert step.observation is not None
        assert len(step.tool_results) > 0
    
    def test_generate_reasoning_step_with_tool_failure(self, reasoning_engine, reasoning_context, mock_tool_manager):
        """Test reasoning step generation when tool fails."""
        # Make tool manager return failure
        mock_tool_manager.execute_tool.return_value = BaseToolResult(
            tool_name="check_traffic",
            success=False,
            data={},
            execution_time=0.1,
            error_message="Network timeout"
        )
        
        step = reasoning_engine.generate_reasoning_step(reasoning_context)
        
        assert isinstance(step, ValidatedReasoningStep)
        assert step.tool_results[0].success is False
        assert "failed" in step.observation.lower()
    
    def test_generate_reasoning_step_no_tool_call(self, reasoning_engine, reasoning_context, mock_llm_provider):
        """Test reasoning step generation without tool call."""
        # Mock LLM response without tool calls
        mock_llm_provider.generate_response.return_value = LLMResponse(
            content="**Thought:** I have enough information to create a plan.",
            messages=[],
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            model="gpt-4",
            finish_reason="stop",
            response_time=1.0,
            timestamp=datetime.now(),
            tool_calls=None
        )
        
        step = reasoning_engine.generate_reasoning_step(reasoning_context)
        
        assert isinstance(step, ValidatedReasoningStep)
        assert step.action is None
        assert len(step.tool_results) == 0
    
    def test_evaluate_tool_results_success(self, reasoning_engine, reasoning_context):
        """Test evaluation of successful tool results."""
        results = [
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
        
        evaluation = reasoning_engine.evaluate_tool_results(results, reasoning_context)
        
        assert isinstance(evaluation, Evaluation)
        assert evaluation.confidence > 0.5
        assert evaluation.should_continue is False  # Traffic scenario with traffic info should be sufficient
        assert "high confidence" in evaluation.reasoning.lower()
    
    def test_evaluate_tool_results_failures(self, reasoning_engine, reasoning_context):
        """Test evaluation of failed tool results."""
        results = [
            ToolResult(
                tool_name="check_traffic",
                success=False,
                data={},
                execution_time=0.1,
                error_message="Network error"
            )
        ]
        
        evaluation = reasoning_engine.evaluate_tool_results(results, reasoning_context)
        
        assert isinstance(evaluation, Evaluation)
        assert evaluation.confidence < 0.5
        assert evaluation.should_continue is True
        assert evaluation.next_action is not None
    
    def test_should_continue_reasoning_max_steps(self, reasoning_engine, sample_scenario):
        """Test reasoning termination due to max steps."""
        # Create trace with maximum steps
        steps = []
        for i in range(reasoning_engine.config.max_reasoning_steps):
            step = ValidatedReasoningStep(
                step_number=i + 1,
                thought=f"Step {i + 1} thought",
                timestamp=datetime.now()
            )
            steps.append(step)
        
        trace = ValidatedReasoningTrace(
            steps=steps,
            scenario=sample_scenario,
            start_time=datetime.now()
        )
        
        should_continue = reasoning_engine.should_continue_reasoning(trace)
        assert should_continue is False
    
    def test_should_continue_reasoning_sufficient_info(self, reasoning_engine, sample_scenario):
        """Test reasoning termination due to sufficient information."""
        # Create trace with successful tool results
        step1 = ValidatedReasoningStep(
            step_number=1,
            thought="Checking traffic",
            timestamp=datetime.now()
        )
        step1.add_tool_result(ToolResult(
            tool_name="check_traffic",
            success=True,
            data={"status": "heavy"},
            execution_time=0.5
        ))
        
        step2 = ValidatedReasoningStep(
            step_number=2,
            thought="Notifying customer",
            timestamp=datetime.now()
        )
        step2.add_tool_result(ToolResult(
            tool_name="notify_customer",
            success=True,
            data={"sent": True},
            execution_time=0.2
        ))
        
        trace = ValidatedReasoningTrace(
            steps=[step1, step2],
            scenario=sample_scenario,
            start_time=datetime.now()
        )
        
        should_continue = reasoning_engine.should_continue_reasoning(trace)
        assert should_continue is False
    
    def test_should_continue_reasoning_timeout(self, reasoning_engine, sample_scenario):
        """Test reasoning termination due to timeout."""
        # Create trace that started long ago
        old_start_time = datetime.now() - timedelta(seconds=400)  # Longer than timeout
        
        trace = ValidatedReasoningTrace(
            steps=[ValidatedReasoningStep(
                step_number=1,
                thought="Initial thought",
                timestamp=old_start_time
            )],
            scenario=sample_scenario,
            start_time=old_start_time
        )
        
        should_continue = reasoning_engine.should_continue_reasoning(trace)
        assert should_continue is False
    
    def test_detect_reasoning_loop(self, reasoning_engine, sample_scenario):
        """Test detection of reasoning loops."""
        # Create trace with repeated similar thoughts
        steps = []
        for i in range(4):
            step = ValidatedReasoningStep(
                step_number=i + 1,
                thought="I need to check traffic status",  # Same thought repeated
                timestamp=datetime.now()
            )
            steps.append(step)
        
        trace = ValidatedReasoningTrace(
            steps=steps,
            scenario=sample_scenario,
            start_time=datetime.now()
        )
        
        should_continue = reasoning_engine.should_continue_reasoning(trace)
        assert should_continue is False
    
    def test_generate_final_plan_success(self, reasoning_engine, sample_scenario, mock_llm_provider):
        """Test successful final plan generation."""
        # Mock LLM response with valid plan JSON
        plan_json = {
            "summary": "Reroute driver and notify customer",
            "steps": [
                {
                    "sequence": 1,
                    "action": "Reroute driver via alternate route",
                    "responsible_party": "Dispatch",
                    "estimated_time": "5 minutes",
                    "success_criteria": "Driver confirms new route"
                }
            ],
            "estimated_duration": "20 minutes",
            "success_probability": 0.9,
            "alternatives": ["Cancel and refund if route unavailable"],
            "stakeholders": ["Driver", "Customer", "Dispatch"]
        }
        
        mock_llm_provider.generate_response.return_value = LLMResponse(
            content=f"```json\n{json.dumps(plan_json)}\n```",
            messages=[],
            token_usage=TokenUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300),
            model="gpt-4",
            finish_reason="stop",
            response_time=2.0,
            timestamp=datetime.now()
        )
        
        # Create trace with some steps
        trace = ValidatedReasoningTrace(
            steps=[ValidatedReasoningStep(
                step_number=1,
                thought="Analyzing traffic situation",
                timestamp=datetime.now()
            )],
            scenario=sample_scenario,
            start_time=datetime.now()
        )
        
        plan = reasoning_engine.generate_final_plan(trace)
        
        assert plan is not None
        assert len(plan.steps) == 1
        assert plan.steps[0].action == "Reroute driver via alternate route"
        assert plan.success_probability == 0.9
        assert "Driver" in plan.stakeholders
    
    def test_generate_final_plan_fallback(self, reasoning_engine, sample_scenario, mock_llm_provider):
        """Test fallback plan generation when LLM fails."""
        # Mock LLM to raise an exception
        mock_llm_provider.generate_response.side_effect = Exception("LLM API error")
        
        trace = ValidatedReasoningTrace(
            steps=[ValidatedReasoningStep(
                step_number=1,
                thought="Initial analysis",
                timestamp=datetime.now()
            )],
            scenario=sample_scenario,
            start_time=datetime.now()
        )
        
        plan = reasoning_engine.generate_final_plan(trace)
        
        # Should get fallback plan
        assert plan is not None
        assert len(plan.steps) >= 3  # Fallback plan has 3 steps
        assert plan.success_probability == 0.6  # Fallback probability
    
    def test_circuit_breaker_activation(self, reasoning_engine, reasoning_context):
        """Test circuit breaker activation after consecutive failures."""
        # Set consecutive failures to threshold
        reasoning_engine._consecutive_failures = reasoning_engine.config.circuit_breaker_threshold
        
        step = reasoning_engine.generate_reasoning_step(reasoning_context)
        
        assert "circuit breaker" in step.thought.lower()
        assert "terminated" in step.observation.lower()
    
    def test_parse_time_duration(self, reasoning_engine):
        """Test time duration parsing."""
        assert reasoning_engine._parse_time_duration("5 minutes") == timedelta(minutes=5)
        assert reasoning_engine._parse_time_duration("2 hours") == timedelta(hours=2)
        assert reasoning_engine._parse_time_duration("30 seconds") == timedelta(seconds=30)
        assert reasoning_engine._parse_time_duration("invalid") == timedelta(minutes=15)  # Default
    
    def test_has_sufficient_information_scenarios(self, reasoning_engine, sample_scenario):
        """Test sufficient information detection for different scenarios."""
        # Traffic scenario - needs traffic info
        traffic_step = ValidatedReasoningStep(
            step_number=1,
            thought="Checking traffic",
            timestamp=datetime.now()
        )
        traffic_step.add_tool_result(ToolResult(
            tool_name="check_traffic",
            success=True,
            data={"status": "heavy"},
            execution_time=0.5
        ))
        
        customer_step = ValidatedReasoningStep(
            step_number=2,
            thought="Notifying customer",
            timestamp=datetime.now()
        )
        customer_step.add_tool_result(ToolResult(
            tool_name="notify_customer",
            success=True,
            data={"sent": True},
            execution_time=0.2
        ))
        
        trace = ValidatedReasoningTrace(
            steps=[traffic_step, customer_step],
            scenario=sample_scenario,
            start_time=datetime.now()
        )
        
        has_sufficient = reasoning_engine._has_sufficient_information(trace)
        assert has_sufficient is True
    
    def test_tool_schema_generation(self, reasoning_engine):
        """Test tool schema generation for LLM function calling."""
        schemas = reasoning_engine._get_tool_schemas()
        
        assert len(schemas) == 3  # check_traffic, notify_customer, get_merchant_status
        assert all("type" in schema and schema["type"] == "function" for schema in schemas)
        assert all("function" in schema for schema in schemas)
        
        # Check specific tool schema
        traffic_schema = next(s for s in schemas if s["function"]["name"] == "check_traffic")
        assert traffic_schema["function"]["description"] == "Mock tool check_traffic"