"""
Test the core interfaces to ensure they're properly defined.
"""
import pytest
from datetime import datetime, timedelta

from src.agent.interfaces import (
    Agent, DisruptionScenario, Entity, EntityType, 
    ScenarioType, UrgencyLevel, ReasoningStep, ResolutionResult
)
from src.tools.interfaces import Tool, ToolManager, ToolResult
from src.reasoning.interfaces import ReasoningEngine, ChainOfThoughtLogger
from src.config.settings import Config, LLMProvider


def test_disruption_scenario_creation():
    """Test that DisruptionScenario can be created with required fields."""
    entities = [
        Entity(
            text="123 Main St",
            entity_type=EntityType.ADDRESS,
            confidence=0.95
        )
    ]
    
    scenario = DisruptionScenario(
        description="Traffic jam affecting delivery",
        entities=entities,
        scenario_type=ScenarioType.TRAFFIC,
        urgency_level=UrgencyLevel.HIGH
    )
    
    assert scenario.description == "Traffic jam affecting delivery"
    assert len(scenario.entities) == 1
    assert scenario.scenario_type == ScenarioType.TRAFFIC
    assert scenario.urgency_level == UrgencyLevel.HIGH


def test_reasoning_step_creation():
    """Test that ReasoningStep can be created with required fields."""
    step = ReasoningStep(
        step_number=1,
        thought="I need to check traffic conditions",
        action="check_traffic",
        observation="Heavy traffic detected",
        timestamp=datetime.now()
    )
    
    assert step.step_number == 1
    assert step.thought == "I need to check traffic conditions"
    assert step.action == "check_traffic"
    assert step.observation == "Heavy traffic detected"


def test_tool_result_creation():
    """Test that ToolResult can be created and auto-timestamps."""
    result = ToolResult(
        tool_name="check_traffic",
        success=True,
        data={"traffic_level": "heavy"},
        execution_time=1.5
    )
    
    assert result.tool_name == "check_traffic"
    assert result.success is True
    assert result.data["traffic_level"] == "heavy"
    assert result.execution_time == 1.5
    assert result.timestamp is not None


def test_config_creation():
    """Test that Config can be created with default values."""
    config = Config.default()
    
    assert config.llm.provider == LLMProvider.OPENAI
    assert config.llm.model == "gpt-4"
    assert config.reasoning.max_steps == 20
    assert config.cli.output_format == "structured"


def test_config_from_dict():
    """Test that Config can be created from dictionary."""
    data = {
        "llm": {
            "provider": "anthropic",
            "model": "claude-3-sonnet",
            "temperature": 0.2
        },
        "reasoning": {
            "max_steps": 15
        }
    }
    
    config = Config.from_dict(data)
    
    assert config.llm.provider == LLMProvider.ANTHROPIC
    assert config.llm.model == "claude-3-sonnet"
    assert config.llm.temperature == 0.2
    assert config.reasoning.max_steps == 15


class MockTool(Tool):
    """Mock tool implementation for testing."""
    
    def __init__(self):
        super().__init__(
            name="mock_tool",
            description="A mock tool for testing",
            parameters={"param1": str}
        )
    
    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={"result": "mock_data"},
            execution_time=0.1
        )
    
    def validate_parameters(self, **kwargs) -> bool:
        return "param1" in kwargs


def test_mock_tool():
    """Test that the mock tool implementation works."""
    tool = MockTool()
    
    assert tool.name == "mock_tool"
    assert tool.validate_parameters(param1="test")
    assert not tool.validate_parameters()
    
    result = tool.execute(param1="test")
    assert result.success
    assert result.data["result"] == "mock_data"


if __name__ == "__main__":
    pytest.main([__file__])