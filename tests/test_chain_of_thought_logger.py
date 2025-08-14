"""
Unit tests for chain-of-thought logging functionality.
"""
import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from io import StringIO

from src.reasoning.logger import (
    ConsoleChainOfThoughtLogger, FileChainOfThoughtLogger, 
    CombinedChainOfThoughtLogger, LoggingConfig, create_chain_of_thought_logger
)
from src.agent.interfaces import ScenarioType, UrgencyLevel, EntityType
from src.agent.models import (
    ValidatedEntity, ValidatedDisruptionScenario, ValidatedReasoningStep,
    ValidatedReasoningTrace, ToolAction, ToolResult
)


@pytest.fixture
def sample_scenario():
    """Create sample disruption scenario."""
    entities = [
        ValidatedEntity(
            text="Highway 101",
            entity_type=EntityType.ADDRESS,
            confidence=0.8,
            normalized_value="Highway 101"
        )
    ]
    
    return ValidatedDisruptionScenario(
        description="Driver stuck in traffic on Highway 101",
        entities=entities,
        scenario_type=ScenarioType.TRAFFIC,
        urgency_level=UrgencyLevel.HIGH
    )


@pytest.fixture
def sample_reasoning_step():
    """Create sample reasoning step."""
    step = ValidatedReasoningStep(
        step_number=1,
        thought="I need to check the traffic situation to understand the delay",
        action=ToolAction(tool_name="check_traffic", parameters={"location": "Highway 101"}),
        observation="Traffic is heavy due to accident, estimated delay 20 minutes",
        timestamp=datetime.now()
    )
    
    # Add tool result
    step.add_tool_result(ToolResult(
        tool_name="check_traffic",
        success=True,
        data={"status": "heavy", "delay_minutes": 20, "cause": "accident"},
        execution_time=0.5
    ))
    
    return step


@pytest.fixture
def sample_reasoning_trace(sample_scenario):
    """Create sample reasoning trace."""
    step1 = ValidatedReasoningStep(
        step_number=1,
        thought="Analyzing traffic situation",
        action=ToolAction(tool_name="check_traffic", parameters={"location": "Highway 101"}),
        observation="Heavy traffic confirmed",
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
        thought="Need to notify customer about delay",
        action=ToolAction(tool_name="notify_customer", parameters={"message": "Delay due to traffic"}),
        observation="Customer notified successfully",
        timestamp=datetime.now()
    )
    step2.add_tool_result(ToolResult(
        tool_name="notify_customer",
        success=True,
        data={"sent": True},
        execution_time=0.2
    ))
    
    start_time = datetime.now() - timedelta(minutes=5)
    end_time = datetime.now()
    
    trace = ValidatedReasoningTrace(
        steps=[step1, step2],
        scenario=sample_scenario,
        start_time=start_time,
        end_time=end_time
    )
    
    return trace


class TestConsoleChainOfThoughtLogger:
    """Test cases for console chain-of-thought logger."""
    
    def test_initialization(self):
        """Test logger initialization."""
        config = LoggingConfig(enable_console_output=True, include_timestamps=False)
        logger = ConsoleChainOfThoughtLogger(config)
        
        assert logger.config.enable_console_output is True
        assert logger.config.include_timestamps is False
        assert logger._step_count == 0
    
    @patch('builtins.print')
    def test_log_step_basic(self, mock_print, sample_reasoning_step):
        """Test basic step logging to console."""
        config = LoggingConfig(include_timestamps=False, include_tool_details=False)
        logger = ConsoleChainOfThoughtLogger(config)
        
        logger.log_step(sample_reasoning_step)
        
        # Check that print was called multiple times
        assert mock_print.call_count >= 3
        
        # Check content of print calls
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        
        # Should contain step header
        assert any("Step 1" in call for call in print_calls)
        
        # Should contain thought
        assert any("💭 Thought:" in call for call in print_calls)
        assert any("traffic situation" in call for call in print_calls)
        
        # Should contain action
        assert any("🔧 Action:" in call for call in print_calls)
        assert any("check_traffic" in call for call in print_calls)
        
        # Should contain observation
        assert any("👁️  Observation:" in call for call in print_calls)
        assert any("Traffic is heavy" in call for call in print_calls)
    
    @patch('builtins.print')
    def test_log_step_with_timestamps(self, mock_print, sample_reasoning_step):
        """Test step logging with timestamps."""
        config = LoggingConfig(include_timestamps=True)
        logger = ConsoleChainOfThoughtLogger(config)
        
        logger.log_step(sample_reasoning_step)
        
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        
        # Should contain timestamp in header
        assert any("(" in call and ")" in call for call in print_calls)
    
    @patch('builtins.print')
    def test_log_step_with_tool_details(self, mock_print, sample_reasoning_step):
        """Test step logging with tool details."""
        config = LoggingConfig(include_tool_details=True)
        logger = ConsoleChainOfThoughtLogger(config)
        
        logger.log_step(sample_reasoning_step)
        
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        
        # Should contain tool results section
        assert any("📊 Tool Results:" in call for call in print_calls)
        assert any("✅ Success" in call for call in print_calls)
        assert any("check_traffic" in call for call in print_calls)
    
    @patch('builtins.print')
    def test_log_step_disabled(self, mock_print, sample_reasoning_step):
        """Test that logging is disabled when console output is off."""
        config = LoggingConfig(enable_console_output=False)
        logger = ConsoleChainOfThoughtLogger(config)
        
        logger.log_step(sample_reasoning_step)
        
        # Should not print anything
        mock_print.assert_not_called()
    
    def test_format_trace(self, sample_reasoning_trace):
        """Test trace formatting."""
        logger = ConsoleChainOfThoughtLogger()
        
        formatted = logger.format_trace(sample_reasoning_trace)
        
        assert "🧠 REASONING TRACE" in formatted
        assert "Driver stuck in traffic" in formatted
        assert "Step 1:" in formatted
        assert "Step 2:" in formatted
        assert "Analyzing traffic situation" in formatted
        assert "notify customer" in formatted
        assert "Duration:" in formatted
    
    def test_get_trace_summary(self, sample_reasoning_trace):
        """Test trace summary generation."""
        logger = ConsoleChainOfThoughtLogger()
        
        summary = logger.get_trace_summary(sample_reasoning_trace)
        
        assert "2 steps" in summary
        assert "check_traffic" in summary
        assert "notify_customer" in summary
        assert "100%" in summary  # Success rate
    
    def test_format_duration(self):
        """Test duration formatting."""
        logger = ConsoleChainOfThoughtLogger()
        
        assert logger._format_duration(timedelta(seconds=30)) == "30s"
        assert logger._format_duration(timedelta(minutes=2, seconds=15)) == "2m 15s"
        assert logger._format_duration(timedelta(hours=1, minutes=30)) == "1h 30m"
    
    def test_summarize_data(self):
        """Test data summarization."""
        logger = ConsoleChainOfThoughtLogger()
        
        # Empty data
        assert logger._summarize_data({}) == "No data"
        
        # Small data
        small_data = {"status": "heavy", "delay": 20}
        result = logger._summarize_data(small_data)
        assert "status" in result and "delay" in result
        
        # Large data
        large_data = {f"field_{i}": f"value_{i}" for i in range(10)}
        result = logger._summarize_data(large_data)
        assert "10 fields" in result


class TestFileChainOfThoughtLogger:
    """Test cases for file chain-of-thought logger."""
    
    @patch('pathlib.Path.mkdir')
    def test_initialization(self, mock_mkdir):
        """Test file logger initialization."""
        config = LoggingConfig(log_directory="test_logs")
        logger = FileChainOfThoughtLogger(config)
        
        assert logger.config.log_directory == "test_logs"
        assert logger.session_file is None
        mock_mkdir.assert_called_once()
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    def test_log_step(self, mock_mkdir, mock_file, sample_reasoning_step):
        """Test step logging to file."""
        logger = FileChainOfThoughtLogger()
        
        logger.log_step(sample_reasoning_step)
        
        # Should have created session file
        assert logger.session_file is not None
        assert "session_" in str(logger.session_file)
        
        # Should have written to file
        mock_file.assert_called()
        
        # Check session data structure
        assert 'steps' in logger.session_data
        assert len(logger.session_data['steps']) == 1
        
        step_data = logger.session_data['steps'][0]
        assert step_data['step_number'] == 1
        assert step_data['thought'] == sample_reasoning_step.thought
        assert 'action' in step_data
        assert step_data['action']['tool_name'] == 'check_traffic'
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    def test_save_trace(self, mock_mkdir, mock_file, sample_reasoning_trace):
        """Test saving complete trace to file."""
        logger = FileChainOfThoughtLogger()
        
        file_path = logger.save_trace(sample_reasoning_trace)
        
        assert "reasoning_trace_" in str(file_path)
        assert str(file_path).endswith(".json")
        
        # Should have written JSON data
        mock_file.assert_called()
        handle = mock_file()
        handle.write.assert_called()
        
        # Check that JSON was written
        written_data = ''.join(call[0][0] for call in handle.write.call_args_list)
        assert '"scenario"' in written_data
        assert '"steps"' in written_data
    
    @patch('builtins.open', new_callable=mock_open, read_data='{"test": "data"}')
    def test_load_trace(self, mock_file):
        """Test loading trace from file."""
        logger = FileChainOfThoughtLogger()
        
        data = logger.load_trace(Path("test_file.json"))
        
        assert data == {"test": "data"}
    
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.glob')
    def test_cleanup_old_logs(self, mock_glob, mock_mkdir):
        """Test cleanup of old log files."""
        config = LoggingConfig(max_log_files=2)
        logger = FileChainOfThoughtLogger(config)
        
        # Mock 5 files with different modification times
        mock_files = []
        for i in range(5):
            mock_file = Mock()
            mock_file.stat.return_value.st_mtime = i  # Different times
            mock_file.unlink = Mock()  # Mock the unlink method
            mock_files.append(mock_file)
        
        # Mock the glob method to return our mock files
        mock_glob.return_value = mock_files
        
        removed_count = logger.cleanup_old_logs()
        
        # Should remove 3 files (keep only 2 newest)
        assert removed_count == 3
        
        # Check that unlink was called on the 3 oldest files
        for i in range(3):  # First 3 files (oldest)
            mock_files[i].unlink.assert_called_once()
    
    def test_serialize_step(self, sample_reasoning_step):
        """Test step serialization."""
        logger = FileChainOfThoughtLogger()
        
        serialized = logger._serialize_step(sample_reasoning_step)
        
        assert serialized['step_number'] == 1
        assert serialized['thought'] == sample_reasoning_step.thought
        assert 'action' in serialized
        assert serialized['action']['tool_name'] == 'check_traffic'
        assert 'tool_results' in serialized
        assert len(serialized['tool_results']) == 1
    
    def test_serialize_trace(self, sample_reasoning_trace):
        """Test trace serialization."""
        logger = FileChainOfThoughtLogger()
        
        serialized = logger._serialize_trace(sample_reasoning_trace)
        
        assert 'scenario' in serialized
        assert 'start_time' in serialized
        assert 'end_time' in serialized
        assert 'steps' in serialized
        assert len(serialized['steps']) == 2
        
        # Check scenario serialization
        scenario = serialized['scenario']
        assert scenario['description'] == sample_reasoning_trace.scenario.description
        assert scenario['scenario_type'] == 'traffic'
        assert scenario['urgency_level'] == 'high'
    
    def test_format_structured_trace(self, sample_reasoning_trace):
        """Test structured trace formatting."""
        config = LoggingConfig(enable_structured_output=True)
        logger = FileChainOfThoughtLogger(config)
        
        formatted = logger.format_trace(sample_reasoning_trace)
        
        # Should be valid JSON
        data = json.loads(formatted)
        assert 'scenario' in data
        assert 'steps' in data
        assert len(data['steps']) == 2


class TestCombinedChainOfThoughtLogger:
    """Test cases for combined chain-of-thought logger."""
    
    @patch('src.reasoning.logger.ConsoleChainOfThoughtLogger')
    @patch('src.reasoning.logger.FileChainOfThoughtLogger')
    def test_initialization(self, mock_file_logger, mock_console_logger):
        """Test combined logger initialization."""
        config = LoggingConfig()
        logger = CombinedChainOfThoughtLogger(config)
        
        mock_console_logger.assert_called_once_with(config)
        mock_file_logger.assert_called_once_with(config)
    
    def test_log_step_calls_both(self, sample_reasoning_step):
        """Test that log_step calls both loggers."""
        logger = CombinedChainOfThoughtLogger()
        logger.console_logger = Mock()
        logger.file_logger = Mock()
        
        logger.log_step(sample_reasoning_step)
        
        logger.console_logger.log_step.assert_called_once_with(sample_reasoning_step)
        logger.file_logger.log_step.assert_called_once_with(sample_reasoning_step)
    
    def test_format_trace_uses_console(self, sample_reasoning_trace):
        """Test that format_trace uses console logger."""
        logger = CombinedChainOfThoughtLogger()
        logger.console_logger = Mock()
        logger.console_logger.format_trace.return_value = "formatted trace"
        
        result = logger.format_trace(sample_reasoning_trace)
        
        assert result == "formatted trace"
        logger.console_logger.format_trace.assert_called_once_with(sample_reasoning_trace)
    
    def test_save_trace_uses_file(self, sample_reasoning_trace):
        """Test that save_trace uses file logger."""
        logger = CombinedChainOfThoughtLogger()
        logger.file_logger = Mock()
        logger.file_logger.save_trace.return_value = Path("test.json")
        
        result = logger.save_trace(sample_reasoning_trace)
        
        assert result == Path("test.json")
        logger.file_logger.save_trace.assert_called_once_with(sample_reasoning_trace, None)


class TestLoggingFactory:
    """Test cases for logging factory function."""
    
    def test_create_combined_logger(self):
        """Test creation of combined logger."""
        config = LoggingConfig(enable_console_output=True, enable_file_logging=True)
        logger = create_chain_of_thought_logger(config)
        
        assert isinstance(logger, CombinedChainOfThoughtLogger)
    
    def test_create_file_only_logger(self):
        """Test creation of file-only logger."""
        config = LoggingConfig(enable_console_output=False, enable_file_logging=True)
        logger = create_chain_of_thought_logger(config)
        
        assert isinstance(logger, FileChainOfThoughtLogger)
    
    def test_create_console_only_logger(self):
        """Test creation of console-only logger."""
        config = LoggingConfig(enable_console_output=True, enable_file_logging=False)
        logger = create_chain_of_thought_logger(config)
        
        assert isinstance(logger, ConsoleChainOfThoughtLogger)
    
    def test_create_default_logger(self):
        """Test creation with default config."""
        logger = create_chain_of_thought_logger()
        
        # Default config has both enabled
        assert isinstance(logger, CombinedChainOfThoughtLogger)


class TestLoggingConfig:
    """Test cases for logging configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LoggingConfig()
        
        assert config.enable_file_logging is True
        assert config.enable_console_output is True
        assert config.enable_structured_output is True
        assert config.include_timestamps is True
        assert config.include_tool_details is True
        assert config.max_log_files == 100
        assert config.log_directory == "logs/reasoning"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = LoggingConfig(
            enable_file_logging=False,
            enable_console_output=False,
            log_directory="custom_logs",
            max_log_files=50
        )
        
        assert config.enable_file_logging is False
        assert config.enable_console_output is False
        assert config.log_directory == "custom_logs"
        assert config.max_log_files == 50