"""
Tests for CLI application structure and functionality.
"""
import pytest
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

from src.cli.main import main, CLIApplication
from src.cli.progress_display import ProgressDisplay, ProgressTracker
from src.cli.interactive_input import InteractiveInput
from src.cli.error_handler import CLIErrorHandler
from src.config.settings import Config, LLMConfig, CLIConfig, LLMProvider
from src.agent.interfaces import ResolutionResult, ResolutionPlan, PlanStep
from src.agent.models import ValidatedDisruptionScenario, ValidatedReasoningTrace
from datetime import datetime, timedelta


class TestCLIApplication:
    """Test CLI application structure and core functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        return Config(
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4",
                api_key="test-key"
            ),
            cli=CLIConfig(
                verbose=False,
                output_format="structured",
                show_reasoning=True,
                show_timing=False
            )
        )
    
    @pytest.fixture
    def mock_resolution_result(self):
        """Create mock resolution result for testing."""
        from src.agent.interfaces import ScenarioType, UrgencyLevel, Entity, EntityType
        
        scenario = ValidatedDisruptionScenario(
            description="Test traffic disruption",
            entities=[
                Entity(text="123 Main St", entity_type=EntityType.ADDRESS, confidence=0.9)
            ],
            scenario_type=ScenarioType.TRAFFIC,
            urgency_level=UrgencyLevel.HIGH
        )
        
        plan = ResolutionPlan(
            steps=[
                PlanStep(
                    sequence=1,
                    action="Check traffic conditions",
                    responsible_party="System",
                    estimated_time=timedelta(minutes=5),
                    dependencies=[],
                    success_criteria="Traffic data retrieved"
                )
            ],
            estimated_duration=timedelta(minutes=15),
            success_probability=0.85,
            alternatives=["Use alternative route"],
            stakeholders=["Driver", "Customer"]
        )
        
        trace = ValidatedReasoningTrace(
            steps=[],
            scenario=scenario,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=30)
        )
        
        return ResolutionResult(
            scenario=scenario,
            reasoning_trace=trace,
            resolution_plan=plan,
            success=True
        )
    
    def test_cli_application_initialization(self, mock_config):
        """Test CLI application initializes correctly."""
        with patch('src.cli.main.get_llm_provider') as mock_llm, \
             patch('src.cli.main.ToolManager') as mock_tool_manager, \
             patch('src.cli.main.AutonomousAgent') as mock_agent:
            
            mock_llm.return_value = Mock()
            mock_tool_manager.return_value = Mock()
            mock_agent.return_value = Mock()
            
            app = CLIApplication(mock_config)
            
            assert app.config == mock_config
            assert app.progress_display is not None
            assert app.output_formatter is not None
            assert app.interactive_input is not None
            assert app.error_handler is not None
            
            # Verify components were initialized
            mock_llm.assert_called_once()
            mock_tool_manager.assert_called_once()
            mock_agent.assert_called_once()
    
    def test_single_scenario_processing(self, mock_config, mock_resolution_result):
        """Test processing a single scenario."""
        with patch('src.cli.main.get_llm_provider'), \
             patch('src.cli.main.ToolManager'), \
             patch('src.cli.main.AutonomousAgent') as mock_agent_class:
            
            mock_agent = Mock()
            mock_agent.process_scenario.return_value = mock_resolution_result
            mock_agent_class.return_value = mock_agent
            
            app = CLIApplication(mock_config)
            
            # Mock progress display to avoid threading issues in tests
            with patch.object(app, '_process_scenario_with_progress') as mock_process:
                app.run_single_scenario("Test scenario")
                mock_process.assert_called_once_with("Test scenario")
    
    def test_interactive_mode_exit(self, mock_config):
        """Test interactive mode exits gracefully."""
        with patch('src.cli.main.get_llm_provider'), \
             patch('src.cli.main.ToolManager'), \
             patch('src.cli.main.AutonomousAgent'):
            
            app = CLIApplication(mock_config)
            
            # Mock interactive input to return None (exit)
            with patch.object(app.interactive_input, 'get_scenario_input', return_value=None), \
                 patch.object(app.output_formatter, 'display_welcome'):
                
                app.run_interactive_mode()
                # Should exit without error
    
    def test_interactive_mode_single_scenario(self, mock_config, mock_resolution_result):
        """Test interactive mode processes single scenario."""
        with patch('src.cli.main.get_llm_provider'), \
             patch('src.cli.main.ToolManager'), \
             patch('src.cli.main.AutonomousAgent') as mock_agent_class:
            
            mock_agent = Mock()
            mock_agent.process_scenario.return_value = mock_resolution_result
            mock_agent_class.return_value = mock_agent
            
            app = CLIApplication(mock_config)
            
            # Mock interactive input
            with patch.object(app.interactive_input, 'get_scenario_input', side_effect=["Test scenario", None]), \
                 patch.object(app.interactive_input, 'ask_continue', return_value=False), \
                 patch.object(app.output_formatter, 'display_welcome'), \
                 patch.object(app, '_process_scenario_with_progress') as mock_process:
                
                app.run_interactive_mode()
                mock_process.assert_called_once_with("Test scenario")
    
    def test_error_handling_during_processing(self, mock_config):
        """Test error handling during scenario processing."""
        with patch('src.cli.main.get_llm_provider'), \
             patch('src.cli.main.ToolManager'), \
             patch('src.cli.main.AutonomousAgent') as mock_agent_class:
            
            mock_agent = Mock()
            mock_agent.process_scenario.side_effect = Exception("Test error")
            mock_agent_class.return_value = mock_agent
            
            app = CLIApplication(mock_config)
            
            with patch.object(app.error_handler, 'handle_runtime_error') as mock_error_handler:
                with pytest.raises(SystemExit):
                    app.run_single_scenario("Test scenario")
                
                mock_error_handler.assert_called_once()


class TestProgressDisplay:
    """Test progress display functionality."""
    
    def test_progress_tracker_initialization(self):
        """Test progress tracker initializes correctly."""
        tracker = ProgressTracker()
        
        assert tracker.current_stage == "Initializing"
        assert tracker.current_message == "Starting up..."
        assert not tracker.is_complete
        assert not tracker.is_error
        assert tracker.start_time is not None
    
    def test_progress_tracker_update_stage(self):
        """Test progress tracker stage updates."""
        tracker = ProgressTracker()
        
        tracker.update_stage("Processing", "Analyzing scenario...")
        status = tracker.get_status()
        
        assert status["stage"] == "Processing"
        assert status["message"] == "Analyzing scenario..."
        assert not status["is_complete"]
        assert not status["is_error"]
    
    def test_progress_tracker_completion(self):
        """Test progress tracker completion."""
        tracker = ProgressTracker()
        
        tracker.complete()
        status = tracker.get_status()
        
        assert status["stage"] == "Complete"
        assert status["is_complete"]
        assert not status["is_error"]
    
    def test_progress_tracker_error(self):
        """Test progress tracker error handling."""
        tracker = ProgressTracker()
        
        tracker.error("Test error message")
        status = tracker.get_status()
        
        assert status["stage"] == "Error"
        assert status["is_error"]
        assert status["error_message"] == "Test error message"
        assert not status["is_complete"]
    
    def test_progress_display_simple_mode(self):
        """Test progress display in simple mode."""
        display = ProgressDisplay(verbose=False)
        tracker = ProgressTracker()
        
        # Start display in thread
        display_thread = threading.Thread(
            target=display.show_processing_progress,
            args=(tracker,),
            daemon=True
        )
        display_thread.start()
        
        # Update progress
        time.sleep(0.1)
        tracker.update_stage("Processing", "Working...")
        time.sleep(0.1)
        tracker.complete()
        
        # Wait for thread to complete
        display_thread.join(timeout=1.0)
        
        # Thread should complete without error
        assert not display_thread.is_alive()
    
    def test_progress_display_verbose_mode(self):
        """Test progress display in verbose mode."""
        display = ProgressDisplay(verbose=True)
        tracker = ProgressTracker()
        
        # Start display in thread
        display_thread = threading.Thread(
            target=display.show_processing_progress,
            args=(tracker,),
            daemon=True
        )
        display_thread.start()
        
        # Update progress
        time.sleep(0.1)
        tracker.update_stage("Parsing", "Extracting entities...")
        time.sleep(0.1)
        tracker.update_stage("Reasoning", "Analyzing scenario...")
        time.sleep(0.1)
        tracker.complete()
        
        # Wait for thread to complete
        display_thread.join(timeout=1.0)
        
        # Thread should complete without error
        assert not display_thread.is_alive()


class TestInteractiveInput:
    """Test interactive input functionality."""
    
    def test_interactive_input_initialization(self):
        """Test interactive input initializes correctly."""
        input_handler = InteractiveInput()
        
        assert input_handler.example_scenarios is not None
        assert len(input_handler.example_scenarios) > 0
    
    @patch('click.confirm')
    @patch('click.prompt')
    def test_get_scenario_input_success(self, mock_prompt, mock_confirm):
        """Test successful scenario input."""
        mock_confirm.side_effect = [False, True]  # No examples, confirm scenario
        mock_prompt.return_value = "Test traffic disruption scenario"
        
        input_handler = InteractiveInput()
        result = input_handler.get_scenario_input()
        
        assert result == "Test traffic disruption scenario"
        mock_prompt.assert_called()
        assert mock_confirm.call_count == 2
    
    @patch('click.confirm')
    @patch('click.prompt')
    def test_get_scenario_input_quit(self, mock_prompt, mock_confirm):
        """Test quitting scenario input."""
        mock_confirm.return_value = False  # No examples
        mock_prompt.return_value = "quit"
        
        input_handler = InteractiveInput()
        result = input_handler.get_scenario_input()
        
        assert result is None
    
    @patch('click.confirm')
    @patch('click.prompt')
    def test_get_scenario_input_too_short(self, mock_prompt, mock_confirm):
        """Test handling of too short scenario input."""
        mock_confirm.side_effect = [False, True]  # No examples, confirm scenario
        mock_prompt.side_effect = ["short", "This is a longer scenario description"]
        
        input_handler = InteractiveInput()
        result = input_handler.get_scenario_input()
        
        assert result == "This is a longer scenario description"
        assert mock_prompt.call_count == 2
    
    @patch('click.confirm')
    def test_ask_continue(self, mock_confirm):
        """Test ask continue functionality."""
        mock_confirm.return_value = True
        
        input_handler = InteractiveInput()
        result = input_handler.ask_continue()
        
        assert result is True
        mock_confirm.assert_called_once()
    
    def test_select_from_list_single(self):
        """Test selecting single item from list."""
        input_handler = InteractiveInput()
        items = ["Option 1", "Option 2", "Option 3"]
        
        with patch('click.prompt', return_value=2):
            result = input_handler.select_from_list(items, "Select option")
            assert result == ["Option 2"]
    
    def test_select_from_list_multiple(self):
        """Test selecting multiple items from list."""
        input_handler = InteractiveInput()
        items = ["Option 1", "Option 2", "Option 3"]
        
        with patch('click.prompt', return_value="1,3"):
            result = input_handler.select_from_list(items, "Select options", allow_multiple=True)
            assert result == ["Option 1", "Option 3"]


class TestErrorHandler:
    """Test error handler functionality."""
    
    def test_error_handler_initialization(self):
        """Test error handler initializes correctly."""
        handler = CLIErrorHandler()
        
        assert handler.error_suggestions is not None
        assert "ConnectionError" in handler.error_suggestions
        assert "AuthenticationError" in handler.error_suggestions
    
    def test_get_error_suggestions_by_type(self):
        """Test getting error suggestions by error type."""
        handler = CLIErrorHandler()
        
        suggestions = handler._get_error_suggestions("ConnectionError", Exception("Connection failed"))
        
        assert len(suggestions) > 0
        assert any("connection" in s.lower() for s in suggestions)
    
    def test_get_error_suggestions_by_message(self):
        """Test getting error suggestions by error message content."""
        handler = CLIErrorHandler()
        
        error = Exception("API key is invalid")
        suggestions = handler._get_error_suggestions("CustomError", error)
        
        assert len(suggestions) > 0
        assert any("api key" in s.lower() for s in suggestions)
    
    def test_format_error_for_json(self):
        """Test formatting error for JSON output."""
        handler = CLIErrorHandler()
        
        error = ValueError("Invalid input")
        result = handler.format_error_for_json(error)
        
        assert result["error"] is True
        assert result["error_type"] == "ValueError"
        assert result["error_message"] == "Invalid input"
        assert "suggestions" in result
        assert "timestamp" in result
    
    def test_validate_environment(self):
        """Test environment validation."""
        handler = CLIErrorHandler()
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            results = handler.validate_environment()
            
            assert "openai_api_key" in results
            assert results["openai_api_key"] is True
            assert "python_version" in results


class TestCLICommands:
    """Test CLI command-line interface."""
    
    def test_main_command_help(self):
        """Test main command shows help."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert "Autonomous Delivery Coordinator" in result.output
    
    def test_main_command_with_scenario(self):
        """Test main command with scenario option."""
        with patch('src.cli.main.CLIApplication') as mock_app_class:
            mock_app = Mock()
            mock_app_class.return_value = mock_app
            
            runner = CliRunner()
            result = runner.invoke(main, ['--scenario', 'Test scenario'])
            
            assert result.exit_code == 0
            mock_app.run_single_scenario.assert_called_once_with('Test scenario')
    
    def test_status_command(self):
        """Test status subcommand."""
        with patch('src.cli.main.CLIApplication') as mock_app_class:
            mock_app = Mock()
            mock_app_class.return_value = mock_app
            
            runner = CliRunner()
            result = runner.invoke(main, ['status'])
            
            assert result.exit_code == 0
            mock_app.display_agent_status.assert_called_once()
    
    def test_config_info_command(self):
        """Test config-info subcommand."""
        with patch('src.cli.main.CLIApplication') as mock_app_class:
            mock_app = Mock()
            mock_app_class.return_value = mock_app
            
            runner = CliRunner()
            result = runner.invoke(main, ['config-info'])
            
            assert result.exit_code == 0
            mock_app.display_configuration.assert_called_once()
    
    def test_process_command(self):
        """Test process subcommand."""
        with patch('src.cli.main.CLIApplication') as mock_app_class:
            mock_app = Mock()
            mock_app_class.return_value = mock_app
            
            runner = CliRunner()
            result = runner.invoke(main, ['process', 'Test scenario'])
            
            assert result.exit_code == 0
            mock_app.run_single_scenario.assert_called_once_with('Test scenario')
    
    def test_interactive_command(self):
        """Test interactive subcommand."""
        with patch('src.cli.main.CLIApplication') as mock_app_class:
            mock_app = Mock()
            mock_app_class.return_value = mock_app
            
            runner = CliRunner()
            result = runner.invoke(main, ['interactive'])
            
            assert result.exit_code == 0
            mock_app.run_interactive_mode.assert_called_once()
    
    def test_verbose_option(self):
        """Test verbose option affects configuration."""
        with patch('src.cli.main.CLIApplication') as mock_app_class, \
             patch('src.cli.main.load_config') as mock_load_config:
            
            mock_config = Mock()
            mock_config.cli.verbose = False
            mock_load_config.return_value = mock_config
            
            mock_app = Mock()
            mock_app_class.return_value = mock_app
            
            runner = CliRunner()
            result = runner.invoke(main, ['--verbose', '--scenario', 'Test'])
            
            assert result.exit_code == 0
            assert mock_config.cli.verbose is True
    
    def test_output_format_option(self):
        """Test output format option affects configuration."""
        with patch('src.cli.main.CLIApplication') as mock_app_class, \
             patch('src.cli.main.load_config') as mock_load_config:
            
            mock_config = Mock()
            mock_config.cli.output_format = 'structured'
            mock_load_config.return_value = mock_config
            
            mock_app = Mock()
            mock_app_class.return_value = mock_app
            
            runner = CliRunner()
            result = runner.invoke(main, ['--output-format', 'json', '--scenario', 'Test'])
            
            assert result.exit_code == 0
            assert mock_config.cli.output_format == 'json'