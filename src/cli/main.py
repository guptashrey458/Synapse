"""
Main CLI entry point for the Autonomous Delivery Coordinator.
"""
import click
import sys
import time
import threading
from typing import Optional, Dict, Any
from datetime import datetime

from ..config.settings import load_config, Config
from ..agent.autonomous_agent import AutonomousAgent, AgentConfig
from ..tools.tool_manager import ToolManager
from ..llm.providers import get_llm_provider
from .progress_display import ProgressDisplay, ProgressTracker
from .output_formatter import OutputFormatter
from .interactive_input import InteractiveInput
from .error_handler import CLIErrorHandler


class CLIApplication:
    """Main CLI application class that orchestrates all components."""
    
    def __init__(self, config: Config):
        """Initialize CLI application with configuration."""
        self.config = config
        self.progress_display = ProgressDisplay(verbose=config.cli.verbose)
        self.output_formatter = OutputFormatter(config.cli)
        self.interactive_input = InteractiveInput()
        self.error_handler = CLIErrorHandler()
        
        # Initialize core components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize LLM provider, tool manager, and agent."""
        try:
            # Initialize LLM provider
            self.llm_provider = get_llm_provider(self.config.llm)
            
            # Initialize tool manager
            self.tool_manager = ToolManager()
            
            # Initialize agent
            agent_config = AgentConfig(
                max_reasoning_steps=self.config.reasoning.max_steps,
                reasoning_timeout=300,
                enable_context_tracking=True,
                enable_state_management=True,
                log_reasoning_steps=self.config.cli.verbose
            )
            
            self.agent = AutonomousAgent(
                llm_provider=self.llm_provider,
                tool_manager=self.tool_manager,
                config=agent_config
            )
            
        except Exception as e:
            self.error_handler.handle_initialization_error(e)
            sys.exit(1)
    
    def run_interactive_mode(self):
        """Run the CLI in interactive mode."""
        self.output_formatter.display_welcome()
        
        while True:
            try:
                # Get scenario input from user
                scenario = self.interactive_input.get_scenario_input()
                
                if scenario is None:  # User wants to exit
                    break
                
                # Process the scenario
                self._process_scenario_with_progress(scenario)
                
                # Ask if user wants to continue
                if not self.interactive_input.ask_continue():
                    break
                    
            except KeyboardInterrupt:
                click.echo("\n\n👋 Goodbye!")
                break
            except Exception as e:
                self.error_handler.handle_runtime_error(e)
                if not self.interactive_input.ask_continue_after_error():
                    break
    
    def run_single_scenario(self, scenario: str):
        """Process a single scenario and exit."""
        try:
            self.output_formatter.display_welcome()
            self._process_scenario_with_progress(scenario)
        except Exception as e:
            self.error_handler.handle_runtime_error(e)
            sys.exit(1)
    
    def _process_scenario_with_progress(self, scenario: str):
        """Process scenario with real-time progress display."""
        # Initialize progress tracking
        progress_tracker = ProgressTracker()
        
        # Start progress display in separate thread
        progress_thread = threading.Thread(
            target=self.progress_display.show_processing_progress,
            args=(progress_tracker,),
            daemon=True
        )
        progress_thread.start()
        
        try:
            # Update progress: Starting
            progress_tracker.update_stage("Initializing", "Preparing to process scenario...")
            time.sleep(0.5)  # Brief pause for user experience
            
            # Update progress: Parsing
            progress_tracker.update_stage("Parsing", "Analyzing scenario and extracting entities...")
            
            # Process scenario with agent
            result = self.agent.process_scenario(scenario)
            
            # Update progress: Complete
            progress_tracker.complete()
            
            # Wait for progress thread to finish
            progress_thread.join(timeout=1.0)
            
            # Display results
            self.output_formatter.display_result(result)
            
        except Exception as e:
            progress_tracker.error(str(e))
            progress_thread.join(timeout=1.0)
            raise
    
    def display_agent_status(self):
        """Display current agent status and metrics."""
        state = self.agent.get_current_state()
        metrics = self.agent.get_performance_metrics()
        
        self.output_formatter.display_agent_status(state, metrics)
    
    def display_configuration(self):
        """Display current configuration."""
        self.output_formatter.display_configuration(self.config)


@click.group(invoke_without_command=True)
@click.option('--config', '-c', type=str, help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--scenario', '-s', type=str, help='Disruption scenario to process')
@click.option('--output-format', type=click.Choice(['structured', 'json', 'plain']), 
              help='Output format for results')
@click.option('--show-timing', is_flag=True, help='Show timing information')
@click.option('--show-reasoning', is_flag=True, default=True, help='Show reasoning steps')
@click.pass_context
def main(ctx, config: Optional[str], verbose: bool, scenario: Optional[str],
         output_format: Optional[str], show_timing: bool, show_reasoning: bool):
    """
    Autonomous Delivery Coordinator CLI.
    
    Process delivery disruption scenarios using AI-powered reasoning.
    """
    # Load configuration
    app_config = load_config(config)
    
    # Override config with CLI options
    if verbose:
        app_config.cli.verbose = True
    if output_format:
        app_config.cli.output_format = output_format
    if show_timing:
        app_config.cli.show_timing = True
    if not show_reasoning:
        app_config.cli.show_reasoning = False
    
    # Store config in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['config'] = app_config
    
    # If no subcommand and no scenario, run interactive mode
    if ctx.invoked_subcommand is None:
        app = CLIApplication(app_config)
        
        if scenario:
            app.run_single_scenario(scenario)
        else:
            app.run_interactive_mode()


@main.command()
@click.pass_context
def status(ctx):
    """Display agent status and performance metrics."""
    config = ctx.obj['config']
    app = CLIApplication(config)
    app.display_agent_status()


@main.command()
@click.pass_context
def config_info(ctx):
    """Display current configuration."""
    config = ctx.obj['config']
    app = CLIApplication(config)
    app.display_configuration()


@main.command()
@click.argument('scenario')
@click.pass_context
def process(ctx, scenario: str):
    """Process a specific disruption scenario."""
    config = ctx.obj['config']
    app = CLIApplication(config)
    app.run_single_scenario(scenario)


@main.command()
@click.pass_context
def interactive(ctx):
    """Run in interactive mode."""
    config = ctx.obj['config']
    app = CLIApplication(config)
    app.run_interactive_mode()


if __name__ == '__main__':
    main()