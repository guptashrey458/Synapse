"""
Error handling and recovery suggestions for CLI.
"""
import click
import sys
import traceback
from datetime import datetime
from typing import Optional, Dict, Any


class CLIErrorHandler:
    """Handles errors and provides recovery suggestions for CLI operations."""
    
    def __init__(self):
        self.error_suggestions = {
            "ConnectionError": [
                "Check your internet connection",
                "Verify LLM provider API endpoints are accessible",
                "Check if API services are experiencing downtime"
            ],
            "AuthenticationError": [
                "Verify your API key is correct",
                "Check if API key has proper permissions",
                "Ensure API key is not expired",
                "Set API key in environment variable or config file"
            ],
            "RateLimitError": [
                "Wait a moment before retrying",
                "Consider upgrading your API plan",
                "Reduce request frequency in configuration"
            ],
            "ValidationError": [
                "Check your input format",
                "Verify all required fields are provided",
                "Review configuration file syntax"
            ],
            "TimeoutError": [
                "Increase timeout values in configuration",
                "Check network connectivity",
                "Try again with a simpler scenario"
            ],
            "ConfigurationError": [
                "Check configuration file syntax",
                "Verify all required configuration fields",
                "Use --config flag to specify config file path",
                "Run 'config-info' command to see current settings"
            ]
        }
    
    def handle_initialization_error(self, error: Exception):
        """Handle errors during application initialization."""
        click.echo("💥 Failed to initialize Autonomous Delivery Coordinator", err=True)
        click.echo(f"Error: {str(error)}", err=True)
        
        error_type = type(error).__name__
        suggestions = self._get_error_suggestions(error_type, error)
        
        if suggestions:
            click.echo("\n🔧 Suggested fixes:", err=True)
            for suggestion in suggestions:
                click.echo(f"   • {suggestion}", err=True)
        
        click.echo("\n💡 For more help:", err=True)
        click.echo("   • Run with --verbose flag for detailed logs", err=True)
        click.echo("   • Check configuration with 'config-info' command", err=True)
        click.echo("   • Verify API keys and network connectivity", err=True)
    
    def handle_runtime_error(self, error: Exception):
        """Handle errors during runtime operations with enhanced formatting."""
        # Format error with visual hierarchy
        click.echo("\n" + "=" * 60, err=True)
        click.echo("💥 RUNTIME ERROR", err=True)
        click.echo("=" * 60, err=True)
        
        error_type = type(error).__name__
        click.echo(f"\n🏷️  Error Type: {click.style(error_type, fg='red', bold=True)}", err=True)
        click.echo(f"📝 Message: {str(error)}", err=True)
        
        # Show error context if available
        self._show_error_context(error)
        
        suggestions = self._get_error_suggestions(error_type, error)
        
        if suggestions:
            click.echo(f"\n🔧 RECOVERY SUGGESTIONS", err=True)
            click.echo("-" * 30, err=True)
            for i, suggestion in enumerate(suggestions[:5], 1):  # Limit to 5 suggestions
                click.echo(f"{i}. {suggestion}", err=True)
        
        # Show quick actions
        self._show_quick_actions(error_type)
        
        # Show traceback in verbose mode
        if click.get_current_context().params.get('verbose', False):
            click.echo(f"\n📋 DETAILED ERROR INFORMATION", err=True)
            click.echo("-" * 40, err=True)
            click.echo(traceback.format_exc(), err=True)
        
        click.echo("=" * 60, err=True)
    
    def handle_scenario_processing_error(self, error: Exception, scenario: str):
        """Handle errors specific to scenario processing."""
        click.echo(f"\n💥 Failed to process scenario: {scenario[:50]}...", err=True)
        click.echo(f"Error: {str(error)}", err=True)
        
        error_type = type(error).__name__
        suggestions = self._get_error_suggestions(error_type, error)
        
        # Add scenario-specific suggestions
        scenario_suggestions = [
            "Try rephrasing the scenario with more specific details",
            "Include key information like addresses, order IDs, or merchant names",
            "Break complex scenarios into simpler parts",
            "Check if the scenario contains any unusual characters or formatting"
        ]
        
        all_suggestions = suggestions + scenario_suggestions
        
        if all_suggestions:
            click.echo("\n🔧 Suggested fixes:", err=True)
            for suggestion in all_suggestions[:5]:  # Limit to 5 suggestions
                click.echo(f"   • {suggestion}", err=True)
    
    def handle_tool_error(self, tool_name: str, error: Exception):
        """Handle errors from tool execution."""
        click.echo(f"\n🔧 Tool Error ({tool_name}): {str(error)}", err=True)
        
        tool_suggestions = [
            f"Check if {tool_name} tool is properly configured",
            "Verify tool parameters are valid",
            "Check network connectivity for external tools",
            "Try processing scenario without this specific tool"
        ]
        
        click.echo("\n🔧 Suggested fixes:", err=True)
        for suggestion in tool_suggestions:
            click.echo(f"   • {suggestion}", err=True)
    
    def handle_llm_error(self, error: Exception):
        """Handle errors from LLM provider."""
        click.echo(f"\n🤖 LLM Provider Error: {str(error)}", err=True)
        
        llm_suggestions = [
            "Check your LLM provider API key",
            "Verify API endpoint is accessible",
            "Check if you have sufficient API credits/quota",
            "Try reducing max_tokens in configuration",
            "Switch to a different LLM provider if available"
        ]
        
        click.echo("\n🔧 Suggested fixes:", err=True)
        for suggestion in llm_suggestions:
            click.echo(f"   • {suggestion}", err=True)
    
    def _get_error_suggestions(self, error_type: str, error: Exception) -> list:
        """Get suggestions based on error type and message."""
        suggestions = []
        
        # Get suggestions based on error type
        if error_type in self.error_suggestions:
            suggestions.extend(self.error_suggestions[error_type])
        
        # Add suggestions based on error message content
        error_message = str(error).lower()
        
        if "api key" in error_message or "authentication" in error_message:
            suggestions.extend(self.error_suggestions.get("AuthenticationError", []))
        elif "timeout" in error_message or "timed out" in error_message:
            suggestions.extend(self.error_suggestions.get("TimeoutError", []))
        elif "connection" in error_message or "network" in error_message:
            suggestions.extend(self.error_suggestions.get("ConnectionError", []))
        elif "rate limit" in error_message or "quota" in error_message:
            suggestions.extend(self.error_suggestions.get("RateLimitError", []))
        elif "config" in error_message or "setting" in error_message:
            suggestions.extend(self.error_suggestions.get("ConfigurationError", []))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions
    
    def format_error_for_json(self, error: Exception) -> Dict[str, Any]:
        """Format error for JSON output."""
        error_type = type(error).__name__
        suggestions = self._get_error_suggestions(error_type, error)
        
        return {
            "error": True,
            "error_type": error_type,
            "error_message": str(error),
            "suggestions": suggestions,
            "timestamp": datetime.now().isoformat()
        }
    
    def show_recovery_options(self, error: Exception) -> str:
        """Show recovery options and get user choice."""
        click.echo("\n🔄 Recovery Options:", err=True)
        click.echo("1. Retry with same input", err=True)
        click.echo("2. Modify input and retry", err=True)
        click.echo("3. Check configuration", err=True)
        click.echo("4. Exit application", err=True)
        
        try:
            choice = click.prompt("Choose option", type=click.Choice(['1', '2', '3', '4']), default='2')
            return choice
        except (KeyboardInterrupt, click.Abort):
            return '4'
    
    def show_configuration_help(self):
        """Show configuration help information."""
        click.echo("\n⚙️  Configuration Help", err=True)
        click.echo("=" * 30, err=True)
        
        click.echo("Configuration file locations (in order of precedence):", err=True)
        click.echo("1. File specified with --config flag", err=True)
        click.echo("2. ./config.json", err=True)
        click.echo("3. ./config/config.json", err=True)
        click.echo("4. ~/.autonomous-delivery-coordinator/config.json", err=True)
        
        click.echo("\nRequired configuration:", err=True)
        click.echo("• LLM provider and API key", err=True)
        click.echo("• Model selection", err=True)
        click.echo("• Tool configurations", err=True)
        
        click.echo("\nExample configuration:", err=True)
        click.echo("""{
  "llm": {
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "your-api-key-here"
  },
  "cli": {
    "verbose": false,
    "output_format": "structured"
  }
}""", err=True)
    
    def validate_environment(self) -> Dict[str, bool]:
        """Validate environment and return status."""
        validation_results = {}
        
        # Check for API keys
        import os
        validation_results['openai_api_key'] = bool(os.getenv('OPENAI_API_KEY'))
        validation_results['anthropic_api_key'] = bool(os.getenv('ANTHROPIC_API_KEY'))
        
        # Check for config file
        config_paths = [
            'config.json',
            'config/config.json',
            os.path.expanduser('~/.autonomous-delivery-coordinator/config.json')
        ]
        validation_results['config_file'] = any(os.path.exists(path) for path in config_paths)
        
        # Check Python version
        validation_results['python_version'] = sys.version_info >= (3, 8)
        
        return validation_results
    
    def show_environment_status(self):
        """Show environment validation status."""
        results = self.validate_environment()
        
        click.echo("\n🔍 Environment Status", err=True)
        click.echo("=" * 30, err=True)
        
        for check, status in results.items():
            status_icon = "✅" if status else "❌"
            check_name = check.replace('_', ' ').title()
            click.echo(f"{status_icon} {check_name}", err=True)
        
        if not all(results.values()):
            click.echo("\n⚠️  Some environment checks failed.", err=True)
            click.echo("This may cause issues during operation.", err=True)
    
    def _show_error_context(self, error: Exception):
        """Show additional context about the error."""
        error_message = str(error).lower()
        
        if "api key" in error_message:
            click.echo(f"🔑 Context: API authentication issue detected", err=True)
        elif "timeout" in error_message:
            click.echo(f"⏱️  Context: Operation timed out", err=True)
        elif "connection" in error_message:
            click.echo(f"🌐 Context: Network connectivity issue", err=True)
        elif "rate limit" in error_message:
            click.echo(f"🚦 Context: API rate limit exceeded", err=True)
        elif "not found" in error_message:
            click.echo(f"🔍 Context: Resource or file not found", err=True)
    
    def _show_quick_actions(self, error_type: str):
        """Show quick action suggestions based on error type."""
        quick_actions = {
            "ConnectionError": [
                "Check internet connection",
                "Verify API endpoints"
            ],
            "AuthenticationError": [
                "Check API key configuration",
                "Verify environment variables"
            ],
            "TimeoutError": [
                "Retry the operation",
                "Increase timeout in config"
            ],
            "ConfigurationError": [
                "Run 'config-info' command",
                "Check config file syntax"
            ]
        }
        
        actions = quick_actions.get(error_type, [])
        if actions:
            click.echo(f"\n⚡ QUICK ACTIONS", err=True)
            click.echo("-" * 20, err=True)
            for action in actions:
                click.echo(f"• {action}", err=True)
    
    def format_error_message_for_display(self, error: Exception, context: Optional[str] = None) -> str:
        """Format error message for consistent display."""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Create formatted error message
        formatted_parts = [
            f"💥 {error_type}",
            f"📝 {error_message}"
        ]
        
        if context:
            formatted_parts.append(f"📍 Context: {context}")
        
        return "\n".join(formatted_parts)
    
    def create_error_report(self, error: Exception, scenario: Optional[str] = None) -> Dict[str, Any]:
        """Create a comprehensive error report."""
        error_type = type(error).__name__
        suggestions = self._get_error_suggestions(error_type, error)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": str(error),
            "suggestions": suggestions,
            "environment_status": self.validate_environment()
        }
        
        if scenario:
            report["scenario"] = scenario[:100] + "..." if len(scenario) > 100 else scenario
        
        return report
    
    def display_error_report(self, report: Dict[str, Any]):
        """Display a formatted error report."""
        click.echo("\n" + "=" * 60, err=True)
        click.echo("📋 ERROR REPORT", err=True)
        click.echo("=" * 60, err=True)
        
        click.echo(f"🕐 Time: {report['timestamp']}", err=True)
        click.echo(f"🏷️  Type: {report['error_type']}", err=True)
        click.echo(f"📝 Message: {report['error_message']}", err=True)
        
        if report.get('scenario'):
            click.echo(f"📋 Scenario: {report['scenario']}", err=True)
        
        if report['suggestions']:
            click.echo(f"\n🔧 Suggestions:", err=True)
            for i, suggestion in enumerate(report['suggestions'], 1):
                click.echo(f"  {i}. {suggestion}", err=True)
        
        # Environment status
        env_status = report['environment_status']
        failed_checks = [k for k, v in env_status.items() if not v]
        if failed_checks:
            click.echo(f"\n⚠️  Environment Issues:", err=True)
            for check in failed_checks:
                check_name = check.replace('_', ' ').title()
                click.echo(f"  ❌ {check_name}", err=True)
        
        click.echo("=" * 60, err=True)