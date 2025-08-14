"""
Tests for enhanced CLI features including chain-of-thought visualization and error formatting.
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.cli.output_formatter import OutputFormatter
from src.cli.error_handler import CLIErrorHandler
from src.config.settings import CLIConfig
from src.agent.models import ValidatedReasoningStep, ValidatedReasoningTrace, ToolAction, ToolResult


class TestEnhancedOutputFormatting:
    """Test enhanced output formatting features."""
    
    @pytest.fixture
    def verbose_config(self):
        """Create verbose CLI config."""
        return CLIConfig(
            verbose=True,
            output_format="structured",
            show_reasoning=True,
            show_timing=True
        )
    
    @pytest.fixture
    def complex_reasoning_trace(self):
        """Create complex reasoning trace for testing."""
        from src.agent.interfaces import ScenarioType, UrgencyLevel
        from src.agent.models import ValidatedDisruptionScenario
        
        scenario = ValidatedDisruptionScenario(
            description="Complex multi-factor disruption",
            entities=[],
            scenario_type=ScenarioType.TRAFFIC,
            urgency_level=UrgencyLevel.HIGH
        )
        
        steps = []
        
        # Step 1: Successful tool execution
        step1 = ValidatedReasoningStep(
            step_number=1,
            thought="Checking traffic conditions to understand delay severity",
            action=ToolAction(
                tool_name="check_traffic",
                parameters={"location": "I-95", "radius": "10"}
            ),
            observation="Heavy congestion detected with 45-minute delays; alternative routes available",
            timestamp=datetime.now(),
            tool_results=[
                ToolResult(
                    tool_name="check_traffic",
                    success=True,
                    data={
                        "delay_minutes": 45,
                        "congestion_level": "heavy",
                        "alternative_routes": 2,
                        "eta": "2:30 PM"
                    },
                    execution_time=1.2,
                    error_message=None
                )
            ]
        )
        
        # Step 2: Failed tool execution
        step2 = ValidatedReasoningStep(
            step_number=2,
            thought="Attempting to get merchant status for potential delays",
            action=ToolAction(
                tool_name="get_merchant_status",
                parameters={"merchant_id": "pizza_palace_123"}
            ),
            observation="Merchant status check failed due to API timeout",
            timestamp=datetime.now(),
            tool_results=[
                ToolResult(
                    tool_name="get_merchant_status",
                    success=False,
                    data={},
                    execution_time=5.0,
                    error_message="API timeout after 5 seconds"
                )
            ]
        )
        
        # Step 3: Multiple tool executions
        step3 = ValidatedReasoningStep(
            step_number=3,
            thought="Finding alternative solutions and notifying stakeholders",
            action=ToolAction(
                tool_name="re_route_driver",
                parameters={"destination": "123 Main St", "avoid_highways": True}
            ),
            observation="Alternative route found; customer notification sent; driver updated with new route",
            timestamp=datetime.now(),
            tool_results=[
                ToolResult(
                    tool_name="re_route_driver",
                    success=True,
                    data={"new_route": "Route 1", "additional_time": 10, "distance": "5.2 miles"},
                    execution_time=0.8,
                    error_message=None
                ),
                ToolResult(
                    tool_name="notify_customer",
                    success=True,
                    data={"notification_sent": True, "method": "SMS", "customer_response": "acknowledged"},
                    execution_time=0.3,
                    error_message=None
                )
            ]
        )
        
        steps = [step1, step2, step3]
        
        return ValidatedReasoningTrace(
            steps=steps,
            scenario=scenario,
            start_time=datetime.now() - timedelta(seconds=45),
            end_time=datetime.now()
        )
    
    @patch('click.echo')
    def test_enhanced_reasoning_display(self, mock_echo, verbose_config, complex_reasoning_trace):
        """Test enhanced reasoning trace display with visual elements."""
        formatter = OutputFormatter(verbose_config)
        
        formatter._display_reasoning_trace(complex_reasoning_trace)
        
        # Verify enhanced visual elements were displayed
        calls = [call[0][0] for call in mock_echo.call_args_list]
        
        # Check for flow diagram
        assert any("Reasoning Flow:" in call for call in calls)
        
        # Check for visual indicators
        assert any("🔄" in call or "💭" in call for call in calls)
        
        # Check for tool execution details
        assert any("check_traffic" in call for call in calls)
        assert any("45min" in call or "delay_minutes: 45" in call for call in calls)
        
        # Check for error formatting
        assert any("API timeout" in call for call in calls)
        
        # Check for reasoning summary
        assert any("Reasoning Summary:" in call for call in calls)
    
    @patch('click.echo')
    def test_reasoning_flow_diagram(self, mock_echo, verbose_config):
        """Test reasoning flow diagram generation."""
        formatter = OutputFormatter(verbose_config)
        
        # Create simple steps for flow diagram
        steps = [
            Mock(action=Mock(tool_name="check_traffic")),
            Mock(action=Mock(tool_name="notify_customer")),
            Mock(action=None)  # Analysis step
        ]
        
        formatter._display_reasoning_flow_diagram(steps)
        
        # Verify flow diagram was displayed
        calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("🔧check_traffic" in call for call in calls)
        assert any("🔧notify_customer" in call for call in calls)
        assert any("💭Analysis" in call for call in calls)
        assert any("→" in call for call in calls)
    
    def test_duration_color_coding(self, verbose_config):
        """Test duration color coding for performance visualization."""
        formatter = OutputFormatter(verbose_config)
        
        # Test different duration ranges
        fast_color = formatter._get_duration_color(0.5)  # Fast
        medium_color = formatter._get_duration_color(2.0)  # Medium
        slow_color = formatter._get_duration_color(5.0)  # Slow
        
        # Colors should be different (we can't test exact colors easily, but can test they're different)
        assert fast_color != medium_color
        assert medium_color != slow_color
    
    def test_key_data_extraction(self, verbose_config):
        """Test extraction of key data points from tool results."""
        formatter = OutputFormatter(verbose_config)
        
        # Test with traffic data
        traffic_data = {
            "delay_minutes": 30,
            "congestion_level": "moderate",
            "alternative_routes": 3,
            "irrelevant_field": "ignored"
        }
        
        key_points = formatter._extract_key_data_points(traffic_data)
        
        assert "delay minutes: 30min" in key_points
        assert "congestion level: moderate" in key_points
        assert "irrelevant_field" not in key_points
    
    def test_reasoning_summary_generation(self, verbose_config, complex_reasoning_trace):
        """Test reasoning summary generation."""
        formatter = OutputFormatter(verbose_config)
        
        with patch('click.echo') as mock_echo:
            formatter._display_reasoning_summary(complex_reasoning_trace)
            
            calls = [call[0][0] for call in mock_echo.call_args_list]
            
            # Check for tool usage summary
            assert any("Tools used:" in call for call in calls)
            assert any("Success rate:" in call for call in calls)
            assert any("Total reasoning time:" in call for call in calls)


class TestEnhancedErrorHandling:
    """Test enhanced error handling and formatting."""
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler for testing."""
        return CLIErrorHandler()
    
    @patch('click.echo')
    def test_enhanced_runtime_error_display(self, mock_echo, error_handler):
        """Test enhanced runtime error display with visual formatting."""
        test_error = ConnectionError("Failed to connect to API endpoint")
        
        with patch('click.get_current_context') as mock_context:
            mock_context.return_value.params = {'verbose': False}
            error_handler.handle_runtime_error(test_error)
        
        # Verify enhanced error formatting
        calls = [call[0][0] for call in mock_echo.call_args_list]
        
        # Check for visual hierarchy
        assert any("RUNTIME ERROR" in call for call in calls)
        assert any("Error Type:" in call for call in calls)
        assert any("RECOVERY SUGGESTIONS" in call for call in calls)
        assert any("QUICK ACTIONS" in call for call in calls)
    
    def test_error_context_detection(self, error_handler):
        """Test error context detection."""
        with patch('click.echo') as mock_echo:
            # Test API key error context
            api_error = Exception("Invalid API key provided")
            error_handler._show_error_context(api_error)
            
            calls = [call[0][0] for call in mock_echo.call_args_list]
            assert any("API authentication issue" in call for call in calls)
    
    def test_quick_actions_display(self, error_handler):
        """Test quick actions display for different error types."""
        with patch('click.echo') as mock_echo:
            error_handler._show_quick_actions("ConnectionError")
            
            calls = [call[0][0] for call in mock_echo.call_args_list]
            assert any("QUICK ACTIONS" in call for call in calls)
            assert any("Check internet connection" in call for call in calls)
    
    def test_error_message_formatting(self, error_handler):
        """Test error message formatting for display."""
        test_error = ValueError("Invalid input format")
        context = "Processing user scenario"
        
        formatted = error_handler.format_error_message_for_display(test_error, context)
        
        assert "💥 ValueError" in formatted
        assert "📝 Invalid input format" in formatted
        assert "📍 Context: Processing user scenario" in formatted
    
    def test_error_report_creation(self, error_handler):
        """Test comprehensive error report creation."""
        test_error = TimeoutError("Operation timed out")
        scenario = "Traffic disruption on highway"
        
        with patch.object(error_handler, 'validate_environment', return_value={'api_key': True}):
            report = error_handler.create_error_report(test_error, scenario)
        
        assert report["error_type"] == "TimeoutError"
        assert report["error_message"] == "Operation timed out"
        assert report["scenario"] == scenario
        assert "suggestions" in report
        assert "environment_status" in report
        assert "timestamp" in report
    
    @patch('click.echo')
    def test_error_report_display(self, mock_echo, error_handler):
        """Test error report display formatting."""
        report = {
            "timestamp": "2024-01-01T12:00:00",
            "error_type": "ConnectionError",
            "error_message": "Network unreachable",
            "scenario": "Test scenario",
            "suggestions": ["Check network", "Retry operation"],
            "environment_status": {"api_key": True, "config_file": False}
        }
        
        error_handler.display_error_report(report)
        
        calls = [call[0][0] for call in mock_echo.call_args_list]
        
        # Check for report sections
        assert any("ERROR REPORT" in call for call in calls)
        assert any("ConnectionError" in call for call in calls)
        assert any("Network unreachable" in call for call in calls)
        assert any("Check network" in call for call in calls)
        assert any("Environment Issues:" in call for call in calls)


class TestCLIIntegration:
    """Test integration of enhanced CLI features."""
    
    @patch('click.echo')
    def test_verbose_mode_integration(self, mock_echo):
        """Test that verbose mode enables enhanced features."""
        verbose_config = CLIConfig(verbose=True, show_reasoning=True)
        formatter = OutputFormatter(verbose_config)
        
        # Create minimal test data
        from src.agent.interfaces import ScenarioType, UrgencyLevel
        from src.agent.models import ValidatedDisruptionScenario
        
        scenario = ValidatedDisruptionScenario(
            description="Test scenario",
            entities=[],
            scenario_type=ScenarioType.OTHER,
            urgency_level=UrgencyLevel.MEDIUM
        )
        
        step = ValidatedReasoningStep(
            step_number=1,
            thought="Test thought",
            action=ToolAction(tool_name="test_tool", parameters={"param": "value"}),
            observation="Test observation",
            timestamp=datetime.now(),
            tool_results=[
                ToolResult(
                    tool_name="test_tool",
                    success=True,
                    data={"result": "success"},
                    execution_time=1.0,
                    error_message=None
                )
            ]
        )
        
        trace = ValidatedReasoningTrace(
            steps=[step],
            scenario=scenario,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        formatter._display_reasoning_trace(trace)
        
        # Verify verbose features were used
        calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("Parameters:" in call for call in calls)
        assert any("📊 Key findings:" in call for call in calls)
    
    def test_error_recovery_suggestions_accuracy(self):
        """Test that error recovery suggestions are contextually appropriate."""
        handler = CLIErrorHandler()
        
        # Test different error types get appropriate suggestions
        connection_error = ConnectionError("Connection refused")
        connection_suggestions = handler._get_error_suggestions("ConnectionError", connection_error)
        assert any("connection" in s.lower() for s in connection_suggestions)
        
        auth_error = Exception("Invalid API key")
        auth_suggestions = handler._get_error_suggestions("AuthenticationError", auth_error)
        assert any("api key" in s.lower() for s in auth_suggestions)
    
    def test_output_format_consistency(self):
        """Test that output formatting is consistent across different modes."""
        configs = [
            CLIConfig(verbose=False, output_format="structured"),
            CLIConfig(verbose=True, output_format="structured"),
            CLIConfig(verbose=False, output_format="json"),
            CLIConfig(verbose=False, output_format="plain")
        ]
        
        for config in configs:
            formatter = OutputFormatter(config)
            # Should initialize without error
            assert formatter.config == config
            
            # Duration formatting should be consistent
            duration = timedelta(minutes=5, seconds=30)
            formatted = formatter._format_duration(duration)
            assert "5m 30s" == formatted