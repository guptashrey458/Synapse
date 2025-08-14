"""
Chain-of-thought logging and visualization for reasoning traces.
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, TextIO
from dataclasses import dataclass
from pathlib import Path

from .interfaces import ChainOfThoughtLogger, ReasoningStep, ReasoningTrace
from ..agent.models import ValidatedReasoningStep, ValidatedReasoningTrace, ToolResult


logger = logging.getLogger(__name__)


@dataclass
class LoggingConfig:
    """Configuration for chain-of-thought logging."""
    enable_file_logging: bool = True
    log_directory: str = "logs/reasoning"
    enable_console_output: bool = True
    enable_structured_output: bool = True
    max_log_files: int = 100
    include_timestamps: bool = True
    include_tool_details: bool = True


class ConsoleChainOfThoughtLogger(ChainOfThoughtLogger):
    """Chain-of-thought logger that outputs to console with formatting."""
    
    def __init__(self, config: Optional[LoggingConfig] = None):
        """
        Initialize the console logger.
        
        Args:
            config: Logging configuration
        """
        self.config = config or LoggingConfig()
        self._step_count = 0
        
        # Console formatting
        self.colors = {
            'thought': '\033[94m',    # Blue
            'action': '\033[92m',     # Green
            'observation': '\033[93m', # Yellow
            'error': '\033[91m',      # Red
            'reset': '\033[0m',       # Reset
            'bold': '\033[1m',        # Bold
            'dim': '\033[2m'          # Dim
        }
    
    def log_step(self, step: ReasoningStep) -> None:
        """
        Log a single reasoning step to console.
        
        Args:
            step: ReasoningStep to log
        """
        if not self.config.enable_console_output:
            return
        
        self._step_count += 1
        
        # Print step header
        timestamp = step.timestamp.strftime("%H:%M:%S") if self.config.include_timestamps else ""
        header = f"{self.colors['bold']}Step {step.step_number}{self.colors['reset']}"
        if timestamp:
            header += f" {self.colors['dim']}({timestamp}){self.colors['reset']}"
        
        print(f"\n{header}")
        print("=" * 50)
        
        # Print thought
        if step.thought:
            print(f"{self.colors['thought']}{self.colors['bold']}💭 Thought:{self.colors['reset']}")
            print(f"   {step.thought}")
        
        # Print action
        if hasattr(step, 'action') and step.action:
            print(f"\n{self.colors['action']}{self.colors['bold']}🔧 Action:{self.colors['reset']}")
            if hasattr(step.action, 'tool_name'):
                print(f"   Tool: {step.action.tool_name}")
                if self.config.include_tool_details and step.action.parameters:
                    print(f"   Parameters: {json.dumps(step.action.parameters, indent=6)}")
            else:
                print(f"   {step.action}")
        
        # Print observation
        if step.observation:
            print(f"\n{self.colors['observation']}{self.colors['bold']}👁️  Observation:{self.colors['reset']}")
            print(f"   {step.observation}")
        
        # Print tool results if available
        if hasattr(step, 'tool_results') and step.tool_results and self.config.include_tool_details:
            print(f"\n{self.colors['dim']}📊 Tool Results:{self.colors['reset']}")
            for result in step.tool_results:
                status_color = self.colors['action'] if result.success else self.colors['error']
                status = "✅ Success" if result.success else "❌ Failed"
                print(f"   {status_color}{status}{self.colors['reset']} - {result.tool_name}")
                if result.error_message:
                    print(f"      Error: {result.error_message}")
                elif result.data:
                    # Show summary of data
                    data_summary = self._summarize_data(result.data)
                    print(f"      Data: {data_summary}")
    
    def format_trace(self, trace: ReasoningTrace) -> str:
        """
        Format complete reasoning trace for display.
        
        Args:
            trace: ReasoningTrace to format
            
        Returns:
            Formatted string representation of the trace
        """
        lines = []
        
        # Header
        lines.append("🧠 REASONING TRACE")
        lines.append("=" * 60)
        lines.append(f"Scenario: {trace.scenario.description}")
        lines.append(f"Started: {trace.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if trace.end_time:
            duration = trace.end_time - trace.start_time
            lines.append(f"Duration: {self._format_duration(duration)}")
        
        lines.append(f"Steps: {len(trace.steps)}")
        lines.append("")
        
        # Steps
        for i, step in enumerate(trace.steps, 1):
            lines.append(f"Step {i}:")
            lines.append("-" * 20)
            
            if step.thought:
                lines.append(f"💭 Thought: {step.thought}")
            
            if hasattr(step, 'action') and step.action:
                if hasattr(step.action, 'tool_name'):
                    lines.append(f"🔧 Action: {step.action.tool_name}")
                    if step.action.parameters:
                        lines.append(f"   Parameters: {json.dumps(step.action.parameters)}")
                else:
                    lines.append(f"🔧 Action: {step.action}")
            
            if step.observation:
                lines.append(f"👁️  Observation: {step.observation}")
            
            # Tool results summary
            if hasattr(step, 'tool_results') and step.tool_results:
                successful = sum(1 for r in step.tool_results if r.success)
                total = len(step.tool_results)
                lines.append(f"📊 Tools: {successful}/{total} successful")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def get_trace_summary(self, trace: ReasoningTrace) -> str:
        """
        Get summary of reasoning trace.
        
        Args:
            trace: ReasoningTrace to summarize
            
        Returns:
            Summary string of key reasoning points
        """
        if not trace.steps:
            return "No reasoning steps recorded"
        
        # Calculate statistics
        total_steps = len(trace.steps)
        total_tools_used = sum(len(getattr(step, 'tool_results', [])) for step in trace.steps)
        successful_tools = sum(
            sum(1 for r in getattr(step, 'tool_results', []) if r.success) 
            for step in trace.steps
        )
        
        duration = ""
        if trace.end_time:
            duration = self._format_duration(trace.end_time - trace.start_time)
        
        # Key insights
        insights = []
        
        # Identify main actions taken
        tools_used = set()
        for step in trace.steps:
            for result in getattr(step, 'tool_results', []):
                if result.success:
                    tools_used.add(result.tool_name)
        
        if tools_used:
            insights.append(f"Used tools: {', '.join(sorted(tools_used))}")
        
        # Success rate
        if total_tools_used > 0:
            success_rate = (successful_tools / total_tools_used) * 100
            insights.append(f"Tool success rate: {success_rate:.0f}%")
        
        summary_parts = [
            f"Reasoning completed in {total_steps} steps"
        ]
        
        if duration:
            summary_parts.append(f"Duration: {duration}")
        
        if insights:
            summary_parts.extend(insights)
        
        return " | ".join(summary_parts)
    
    def _summarize_data(self, data: Dict[str, Any]) -> str:
        """Create a brief summary of tool result data."""
        if not data:
            return "No data"
        
        if len(data) <= 2:
            return json.dumps(data)
        
        # For larger data, show key-value count
        return f"{len(data)} fields"
    
    def _format_duration(self, duration: timedelta) -> str:
        """Format duration for display."""
        total_seconds = int(duration.total_seconds())
        
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}m {seconds}s"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m"


class FileChainOfThoughtLogger(ChainOfThoughtLogger):
    """Chain-of-thought logger that writes to files with structured output."""
    
    def __init__(self, config: Optional[LoggingConfig] = None):
        """
        Initialize the file logger.
        
        Args:
            config: Logging configuration
        """
        self.config = config or LoggingConfig()
        self.log_dir = Path(self.config.log_directory)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Current session file
        self.session_file: Optional[Path] = None
        self.session_data: Dict[str, Any] = {}
        
        logger.info(f"Initialized file logger with directory: {self.log_dir}")
    
    def log_step(self, step: ReasoningStep) -> None:
        """
        Log a single reasoning step to file.
        
        Args:
            step: ReasoningStep to log
        """
        if not self.config.enable_file_logging:
            return
        
        # Initialize session file if needed
        if self.session_file is None:
            self._initialize_session()
        
        # Convert step to serializable format
        step_data = self._serialize_step(step)
        
        # Add to session data
        if 'steps' not in self.session_data:
            self.session_data['steps'] = []
        
        self.session_data['steps'].append(step_data)
        
        # Write updated session data
        self._write_session_data()
    
    def format_trace(self, trace: ReasoningTrace) -> str:
        """
        Format complete reasoning trace for file output.
        
        Args:
            trace: ReasoningTrace to format
            
        Returns:
            Formatted string representation of the trace
        """
        if self.config.enable_structured_output:
            return self._format_structured_trace(trace)
        else:
            return self._format_plain_text_trace(trace)
    
    def get_trace_summary(self, trace: ReasoningTrace) -> str:
        """
        Get summary of reasoning trace for file logging.
        
        Args:
            trace: ReasoningTrace to summarize
            
        Returns:
            Summary string of key reasoning points
        """
        # Use the same summary as console logger
        console_logger = ConsoleChainOfThoughtLogger(self.config)
        return console_logger.get_trace_summary(trace)
    
    def save_trace(self, trace: ReasoningTrace, filename: Optional[str] = None) -> Path:
        """
        Save complete trace to a dedicated file.
        
        Args:
            trace: ReasoningTrace to save
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reasoning_trace_{timestamp}.json"
        
        file_path = self.log_dir / filename
        
        # Serialize trace
        trace_data = self._serialize_trace(trace)
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(trace_data, f, indent=2, default=str)
        
        logger.info(f"Saved reasoning trace to: {file_path}")
        return file_path
    
    def load_trace(self, file_path: Path) -> Dict[str, Any]:
        """
        Load trace data from file.
        
        Args:
            file_path: Path to trace file
            
        Returns:
            Loaded trace data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def cleanup_old_logs(self) -> int:
        """
        Clean up old log files based on max_log_files setting.
        
        Returns:
            Number of files removed
        """
        if self.config.max_log_files <= 0:
            return 0
        
        # Get all log files sorted by modification time
        log_files = list(self.log_dir.glob("*.json"))
        log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Remove excess files
        files_to_remove = log_files[self.config.max_log_files:]
        removed_count = 0
        
        for file_path in files_to_remove:
            try:
                file_path.unlink()
                removed_count += 1
            except OSError as e:
                logger.warning(f"Failed to remove log file {file_path}: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old log files")
        
        return removed_count
    
    def _initialize_session(self) -> None:
        """Initialize a new session file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = self.log_dir / f"session_{timestamp}.json"
        
        self.session_data = {
            'session_id': timestamp,
            'start_time': datetime.now().isoformat(),
            'steps': []
        }
    
    def _write_session_data(self) -> None:
        """Write current session data to file."""
        if self.session_file is None:
            return
        
        try:
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, indent=2, default=str)
        except OSError as e:
            logger.error(f"Failed to write session data: {e}")
    
    def _serialize_step(self, step: ReasoningStep) -> Dict[str, Any]:
        """Serialize a reasoning step for JSON storage."""
        step_data = {
            'step_number': step.step_number,
            'thought': step.thought,
            'timestamp': step.timestamp.isoformat() if step.timestamp else None
        }
        
        if step.observation:
            step_data['observation'] = step.observation
        
        if hasattr(step, 'action') and step.action:
            if hasattr(step.action, 'tool_name'):
                step_data['action'] = {
                    'tool_name': step.action.tool_name,
                    'parameters': step.action.parameters
                }
            else:
                step_data['action'] = str(step.action)
        
        if hasattr(step, 'tool_results') and step.tool_results:
            step_data['tool_results'] = [
                {
                    'tool_name': result.tool_name,
                    'success': result.success,
                    'data': result.data,
                    'execution_time': result.execution_time,
                    'error_message': result.error_message,
                    'timestamp': result.timestamp.isoformat() if result.timestamp else None
                }
                for result in step.tool_results
            ]
        
        return step_data
    
    def _serialize_trace(self, trace: ReasoningTrace) -> Dict[str, Any]:
        """Serialize a complete reasoning trace."""
        trace_data = {
            'scenario': {
                'description': trace.scenario.description,
                'scenario_type': trace.scenario.scenario_type.value,
                'urgency_level': trace.scenario.urgency_level.value,
                'entities': [
                    {
                        'text': entity.text,
                        'entity_type': entity.entity_type.value,
                        'confidence': entity.confidence,
                        'normalized_value': entity.normalized_value
                    }
                    for entity in trace.scenario.entities
                ]
            },
            'start_time': trace.start_time.isoformat(),
            'end_time': trace.end_time.isoformat() if trace.end_time else None,
            'steps': [self._serialize_step(step) for step in trace.steps]
        }
        
        return trace_data
    
    def _format_structured_trace(self, trace: ReasoningTrace) -> str:
        """Format trace as structured JSON."""
        trace_data = self._serialize_trace(trace)
        return json.dumps(trace_data, indent=2, default=str)
    
    def _format_plain_text_trace(self, trace: ReasoningTrace) -> str:
        """Format trace as plain text."""
        # Use console logger formatting
        console_logger = ConsoleChainOfThoughtLogger(self.config)
        return console_logger.format_trace(trace)


class CombinedChainOfThoughtLogger(ChainOfThoughtLogger):
    """Combined logger that outputs to both console and file."""
    
    def __init__(self, config: Optional[LoggingConfig] = None):
        """
        Initialize the combined logger.
        
        Args:
            config: Logging configuration
        """
        self.config = config or LoggingConfig()
        self.console_logger = ConsoleChainOfThoughtLogger(config)
        self.file_logger = FileChainOfThoughtLogger(config)
    
    def log_step(self, step: ReasoningStep) -> None:
        """
        Log a single reasoning step to both console and file.
        
        Args:
            step: ReasoningStep to log
        """
        self.console_logger.log_step(step)
        self.file_logger.log_step(step)
    
    def format_trace(self, trace: ReasoningTrace) -> str:
        """
        Format complete reasoning trace.
        
        Args:
            trace: ReasoningTrace to format
            
        Returns:
            Formatted string representation of the trace
        """
        return self.console_logger.format_trace(trace)
    
    def get_trace_summary(self, trace: ReasoningTrace) -> str:
        """
        Get summary of reasoning trace.
        
        Args:
            trace: ReasoningTrace to summarize
            
        Returns:
            Summary string of key reasoning points
        """
        return self.console_logger.get_trace_summary(trace)
    
    def save_trace(self, trace: ReasoningTrace, filename: Optional[str] = None) -> Path:
        """
        Save complete trace to file.
        
        Args:
            trace: ReasoningTrace to save
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        return self.file_logger.save_trace(trace, filename)
    
    def cleanup_old_logs(self) -> int:
        """
        Clean up old log files.
        
        Returns:
            Number of files removed
        """
        return self.file_logger.cleanup_old_logs()


def create_chain_of_thought_logger(config: Optional[LoggingConfig] = None) -> ChainOfThoughtLogger:
    """
    Factory function to create appropriate chain-of-thought logger.
    
    Args:
        config: Logging configuration
        
    Returns:
        ChainOfThoughtLogger instance
    """
    config = config or LoggingConfig()
    
    if config.enable_console_output and config.enable_file_logging:
        return CombinedChainOfThoughtLogger(config)
    elif config.enable_file_logging:
        return FileChainOfThoughtLogger(config)
    else:
        return ConsoleChainOfThoughtLogger(config)