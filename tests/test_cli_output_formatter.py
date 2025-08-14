"""
Tests for CLI output formatting and display functionality.
"""
import pytest
import json
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from io import StringIO

from src.cli.output_formatter import OutputFormatter
from src.config.settings import CLIConfig
from src.agent.interfaces import ResolutionResult, ResolutionPlan, PlanStep
from src.agent.models import ValidatedDisruptionScenario, ValidatedReasoningTrace, ValidatedReasoningStep, ToolAction, ToolResult


class TestOutputFormatter:
    """Test output formatter functionality."""
    
    @pytest.fixture
    def cli_config_structured(self):
        """Create CLI config for structured output."""
        return CLIConfig(
            verbose=False,
            output_format="structured",
            show_reasoning=True,
            show_timing=False
        )
    
    @pytest.fixture
    def cli_config_json(self):
        """Create CLI config for JSON output."""
        return CLIConfig(
            verbose=False,
            output_format="json",
            show_reasoning=True,
            show_timing=False
        )
    
    @pytest.fixture
    def cli_config_verbose(self):
        """Create CLI config for verbose output."""
        return CLIConfig(
            verbose=True,
            output_format="structured",
            show_reasoning=True,
            show_timing=True
        )
    
    @pytest.fixture
    def sample_resolution_result(self):
        """Create sample resolution result for testing."""
        from src.agent.interfaces import ScenarioType, UrgencyLevel, Entity, EntityType
        
        # Create scenario
        scenario = ValidatedDisruptionScenario(
            description="Traffic jam on I-95 affecting delivery to 123 Main St",
            entities=[
                Entity(text="I-95", entity_type=EntityType.ADDRESS, confidence=0.8),
                Entity(text="123 Main St", entity_type=EntityType.ADDRESS, confidence=0.9)
            ],
            scenario_type=ScenarioType.TRAFFIC,
            urgency_level=UrgencyLevel.HIGH
        )
        
        # Create reasoning steps
        step1 = ValidatedReasoningStep(
            step_number=1,
            thought="Need to check current traffic conditions on I-95",
            action=ToolAction(
                tool_name="check_traffic",
                parameters={"location": "I-95", "radius": "5"}
            ),
            observation="Traffic is heavily congested with 45-minute delays",
            timestamp=datetime.now(),
            tool_results=[
                ToolResult(
                    tool_name="check_traffic",
                    success=True,
                    data={"delay_minutes": 45, "congestion_level": "heavy"},
                    execution_time=1.2,
                    error_message=None
                )
            ]
        )
        
        step2 = ValidatedReasoningStep(
            step_number=2,
            thought="Should find alternative route to avoid delays",
            action=ToolAction(
                tool_name="re_route_driver",
                parameters={"destination": "123 Main St", "avoid": "I-95"}
            ),
            observation="Alternative route found via Route 1, adds 10 minutes but avoids traffic",
            timestamp=datetime.now(),
            tool_results=[
                ToolResult(
                    tool_name="re_route_driver",
                    success=True,
                    data={"new_route": "Route 1", "additional_time": 10},
                    execution_time=0.8,
                    error_message=None
                )
            ]
        )
        
        # Create reasoning trace
        trace = ValidatedReasoningTrace(
            steps=[step1, step2],
            scenario=scenario,
            start_time=datetime.now() - timedelta(seconds=30),
            end_time=datetime.now()
        )
        
        # Create resolution plan
        plan = ResolutionPlan(
            steps=[
                PlanStep(
                    sequence=1,
                    action="Notify driver of alternative route via Route 1",
                    responsible_party="Dispatch System",
                    estimated_time=timedelta(minutes=2),
                    dependencies=[],
                    success_criteria="Driver confirms receipt of new route"
                ),
                PlanStep(
                    sequence=2,
                    action="Update customer with revised delivery time",
                    responsible_party="Customer Service",
                    estimated_time=timedelta(minutes=3),
                    dependencies=[1],
                    success_criteria="Customer acknowledges updated ETA"
                )
            ],
            estimated_duration=timedelta(minutes=15),
            success_probability=0.85,
            alternatives=["Wait for traffic to clear", "Reschedule delivery"],
            stakeholders=["Driver", "Customer", "Dispatch"]
        )
        
        return ResolutionResult(
            scenario=scenario,
            reasoning_trace=trace,
            resolution_plan=plan,
            success=True
        )
    
    def test_formatter_initialization(self, cli_config_structured):
        """Test output formatter initializes correctly."""
        formatter = OutputFormatter(cli_config_structured)
        
        assert formatter.config == cli_config_structured
    
    @patch('click.echo')
    def test_display_welcome(self, mock_echo, cli_config_structured):
        """Test welcome message display."""
        formatter = OutputFormatter(cli_config_structured)
        
        formatter.display_welcome()
        
        # Verify welcome message was displayed
        mock_echo.assert_called()
        calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("Autonomous Delivery Coordinator" in call for call in calls)
    
    @patch('click.echo')
    def test_display_structured_result(self, mock_echo, cli_config_structured, sample_resolution_result):
        """Test structured result display."""
        formatter = OutputFormatter(cli_config_structured)
        
        formatter.display_result(sample_resolution_result)
        
        # Verify structured output was displayed
        mock_echo.assert_called()
        calls = [call[0][0] for call in mock_echo.call_args_list]
        
        # Check for key sections
        assert any("RESOLUTION RESULT" in call for call in calls)
        assert any("Traffic jam on I-95" in call for call in calls)
        assert any("REASONING PROCESS" in call for call in calls)
        assert any("RESOLUTION PLAN" in call for call in calls)
    
    @patch('click.echo')
    def test_display_json_result(self, mock_echo, cli_config_json, sample_resolution_result):
        """Test JSON result display."""
        formatter = OutputFormatter(cli_config_json)
        
        formatter.display_result(sample_resolution_result)
        
        # Verify JSON output was displayed
        mock_echo.assert_called_once()
        json_output = mock_echo.call_args[0][0]
        
        # Verify it's valid JSON
        result_dict = json.loads(json_output)
        
        assert result_dict["success"] is True
        assert result_dict["scenario"]["description"] == "Traffic jam on I-95 affecting delivery to 123 Main St"
        assert result_dict["scenario"]["type"] == "traffic"
        assert result_dict["scenario"]["urgency"] == "high"
        assert len(result_dict["resolution_plan"]["steps"]) == 2
    
    @patch('click.echo')
    def test_display_plain_result(self, mock_echo, sample_resolution_result):
        """Test plain text result display."""
        cli_config = CLIConfig(output_format="plain")
        formatter = OutputFormatter(cli_config)
        
        formatter.display_result(sample_resolution_result)
        
        # Verify plain output was displayed
        mock_echo.assert_called()
        calls = [call[0][0] for call in mock_echo.call_args_list]
        
        # Check for key information in plain format
        assert any("Scenario:" in call for call in calls)
        assert any("Type:" in call for call in calls)
        assert any("Success:" in call for call in calls)
        assert any("Resolution Plan" in call for call in calls)
    
    @patch('click.echo')
    def test_display_with_reasoning_disabled(self, mock_echo, sample_resolution_result):
        """Test display with reasoning disabled."""
        cli_config = CLIConfig(show_reasoning=False)
        formatter = OutputFormatter(cli_config)
        
        formatter.display_result(sample_resolution_result)
        
        # Verify reasoning section was not displayed
        calls = [call[0][0] for call in mock_echo.call_args_list]
        assert not any("REASONING PROCESS" in call for call in calls)
    
    @patch('click.echo')
    def test_display_with_timing_enabled(self, mock_echo, cli_config_verbose, sample_resolution_result):
        """Test display with timing information enabled."""
        formatter = OutputFormatter(cli_config_verbose)
        
        formatter.display_result(sample_resolution_result)
        
        # Verify timing information was displayed
        calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("TIMING INFORMATION" in call for call in calls)
    
    @patch('click.echo')
    def test_display_verbose_reasoning(self, mock_echo, cli_config_verbose, sample_resolution_result):
        """Test verbose reasoning display."""
        formatter = OutputFormatter(cli_config_verbose)
        
        formatter.display_result(sample_resolution_result)
        
        # Verify verbose reasoning details were displayed
        calls = [call[0][0] for call in mock_echo.call_args_list]
        
        # Check for tool execution details
        assert any("check_traffic" in call for call in calls)
        assert any("re_route_driver" in call for call in calls)
        assert any("Parameters:" in call for call in calls)
    
    @patch('click.echo')
    def test_display_failed_result(self, mock_echo, cli_config_structured, sample_resolution_result):
        """Test display of failed result."""
        # Modify result to be failed
        sample_resolution_result.success = False
        sample_resolution_result.error_message = "LLM provider unavailable"
        
        formatter = OutputFormatter(cli_config_structured)
        formatter.display_result(sample_resolution_result)
        
        # Verify error information was displayed
        calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("Processing failed" in call for call in calls)
        assert any("LLM provider unavailable" in call for call in calls)
    
    @patch('click.echo')
    def test_display_agent_status(self, mock_echo, cli_config_structured):
        """Test agent status display."""
        formatter = OutputFormatter(cli_config_structured)
        
        state = {
            "status": "processing",
            "current_scenario": "Test scenario description",
            "reasoning_steps": 3,
            "error_message": None
        }
        
        metrics = {
            "total_scenarios_processed": 10,
            "average_processing_time_seconds": 25.5,
            "average_reasoning_steps": 4.2,
            "average_success_probability": 0.87
        }
        
        formatter.display_agent_status(state, metrics)
        
        # Verify status information was displayed
        calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("AGENT STATUS" in call for call in calls)
        assert any("Processing" in call for call in calls)
        assert any("PERFORMANCE METRICS" in call for call in calls)
    
    @patch('click.echo')
    def test_display_configuration(self, mock_echo, cli_config_structured):
        """Test configuration display."""
        from src.config.settings import Config, LLMConfig, ReasoningConfig, LLMProvider
        
        config = Config(
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4",
                max_tokens=4000,
                temperature=0.1
            ),
            reasoning=ReasoningConfig(
                max_steps=20,
                confidence_threshold=0.8,
                enable_chain_of_thought=True
            ),
            cli=cli_config_structured
        )
        
        formatter = OutputFormatter(cli_config_structured)
        formatter.display_configuration(config)
        
        # Verify configuration information was displayed
        calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("CONFIGURATION" in call for call in calls)
        assert any("openai" in call for call in calls)
        assert any("gpt-4" in call for call in calls)
    
    def test_format_duration_seconds(self, cli_config_structured):
        """Test duration formatting for seconds."""
        formatter = OutputFormatter(cli_config_structured)
        
        duration = timedelta(seconds=45)
        result = formatter._format_duration(duration)
        
        assert result == "45s"
    
    def test_format_duration_minutes(self, cli_config_structured):
        """Test duration formatting for minutes."""
        formatter = OutputFormatter(cli_config_structured)
        
        duration = timedelta(minutes=5, seconds=30)
        result = formatter._format_duration(duration)
        
        assert result == "5m 30s"
    
    def test_format_duration_hours(self, cli_config_structured):
        """Test duration formatting for hours."""
        formatter = OutputFormatter(cli_config_structured)
        
        duration = timedelta(hours=2, minutes=15)
        result = formatter._format_duration(duration)
        
        assert result == "2h 15m"
    
    @patch('click.echo')
    def test_display_scenario_with_entities(self, mock_echo, cli_config_verbose, sample_resolution_result):
        """Test scenario display with entity details in verbose mode."""
        formatter = OutputFormatter(cli_config_verbose)
        
        formatter._display_scenario_summary(sample_resolution_result)
        
        # Verify entity information was displayed
        calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("Entities found: 2" in call for call in calls)
        assert any("I-95" in call for call in calls)
        assert any("123 Main St" in call for call in calls)
    
    @patch('click.echo')
    def test_display_plan_step_with_dependencies(self, mock_echo, cli_config_structured):
        """Test plan step display with dependencies."""
        formatter = OutputFormatter(cli_config_structured)
        
        step = PlanStep(
            sequence=2,
            action="Update customer with revised delivery time",
            responsible_party="Customer Service",
            estimated_time=timedelta(minutes=3),
            dependencies=[1],
            success_criteria="Customer acknowledges updated ETA"
        )
        
        formatter._display_plan_step(step)
        
        # Verify dependency information was displayed
        calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("Depends on: Steps 1" in call for call in calls)
    
    @patch('click.echo')
    def test_display_tool_results_with_errors(self, mock_echo, cli_config_verbose):
        """Test display of tool results with errors."""
        formatter = OutputFormatter(cli_config_verbose)
        
        # Create reasoning step with failed tool result
        step = ValidatedReasoningStep(
            step_number=1,
            thought="Attempting to check traffic",
            action=ToolAction(tool_name="check_traffic", parameters={}),
            observation="Traffic check failed",
            timestamp=datetime.now(),
            tool_results=[
                ToolResult(
                    tool_name="check_traffic",
                    success=False,
                    data={},
                    execution_time=0.5,
                    error_message="API timeout"
                )
            ]
        )
        
        # Create a minimal trace to test the step display
        from src.agent.interfaces import ScenarioType, UrgencyLevel
        scenario = ValidatedDisruptionScenario(
            description="Test scenario with sufficient length for validation",
            entities=[],
            scenario_type=ScenarioType.OTHER,
            urgency_level=UrgencyLevel.MEDIUM
        )
        
        trace = ValidatedReasoningTrace(
            steps=[step],
            scenario=scenario,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        formatter._display_reasoning_trace(trace)
        
        # Verify error information was displayed
        calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("❌" in call for call in calls)  # Error icon
        assert any("API timeout" in call for call in calls)
    
    def test_json_output_structure(self, cli_config_json, sample_resolution_result):
        """Test JSON output has correct structure."""
        formatter = OutputFormatter(cli_config_json)
        
        with patch('click.echo') as mock_echo:
            formatter.display_result(sample_resolution_result)
            
            json_output = mock_echo.call_args[0][0]
            result_dict = json.loads(json_output)
            
            # Verify required fields
            required_fields = ["success", "scenario", "reasoning_steps", "resolution_plan"]
            for field in required_fields:
                assert field in result_dict
            
            # Verify scenario structure
            scenario = result_dict["scenario"]
            assert "description" in scenario
            assert "type" in scenario
            assert "urgency" in scenario
            assert "entities" in scenario
            
            # Verify resolution plan structure
            plan = result_dict["resolution_plan"]
            assert "steps" in plan
            assert "estimated_duration_minutes" in plan
            assert "success_probability" in plan
            assert "stakeholders" in plan
            assert "alternatives" in plan
            
            # Verify step structure
            steps = plan["steps"]
            assert len(steps) > 0
            step = steps[0]
            assert "sequence" in step
            assert "action" in step
            assert "responsible_party" in step
            assert "estimated_time_minutes" in step