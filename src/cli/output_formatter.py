"""
Output formatting and display for CLI results.
"""
import click
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from ..config.settings import Config, CLIConfig
from ..agent.interfaces import ResolutionResult, ResolutionPlan, PlanStep


class OutputFormatter:
    """Handles formatting and display of CLI output."""
    
    def __init__(self, cli_config: CLIConfig):
        self.config = cli_config
    
    def display_welcome(self):
        """Display welcome message and system information."""
        click.echo("🚚 Autonomous Delivery Coordinator")
        click.echo("=" * 50)
        click.echo("AI-powered delivery disruption resolution system")
        click.echo("")
        
        if self.config.verbose:
            click.echo("💡 Tip: Use Ctrl+C to exit at any time")
            click.echo("📖 For help, use --help flag")
            click.echo("")
    
    def display_result(self, result: ResolutionResult):
        """Display processing result based on configured format."""
        if self.config.output_format == "json":
            self._display_json_result(result)
        elif self.config.output_format == "plain":
            self._display_plain_result(result)
        else:  # structured (default)
            self._display_structured_result(result)
    
    def _display_structured_result(self, result: ResolutionResult):
        """Display result in structured format."""
        click.echo("\n" + "=" * 60)
        click.echo("📋 RESOLUTION RESULT")
        click.echo("=" * 60)
        
        # Scenario summary
        self._display_scenario_summary(result)
        
        # Reasoning trace (if enabled)
        if self.config.show_reasoning and result.reasoning_trace:
            self._display_reasoning_trace(result.reasoning_trace)
        
        # Resolution plan
        self._display_resolution_plan(result.resolution_plan)
        
        # Timing information (if enabled)
        if self.config.show_timing:
            self._display_timing_info(result)
        
        # Success status
        self._display_success_status(result)
    
    def _display_scenario_summary(self, result: ResolutionResult):
        """Display scenario summary."""
        scenario = result.scenario
        click.echo(f"\n📝 Scenario: {scenario.description}")
        click.echo(f"🏷️  Type: {scenario.scenario_type.value.title()}")
        click.echo(f"⚡ Urgency: {scenario.urgency_level.value.title()}")
        
        if scenario.entities:
            click.echo(f"🔍 Entities found: {len(scenario.entities)}")
            if self.config.verbose:
                for entity in scenario.entities[:3]:  # Show first 3
                    click.echo(f"   • {entity.entity_type.value}: {entity.text}")
                if len(scenario.entities) > 3:
                    click.echo(f"   • ... and {len(scenario.entities) - 3} more")
    
    def _display_reasoning_trace(self, trace):
        """Display reasoning trace with enhanced chain-of-thought visualization."""
        click.echo(f"\n🧠 REASONING PROCESS ({len(trace.steps)} steps)")
        click.echo("-" * 40)
        
        # Show reasoning flow diagram if verbose
        if self.config.verbose and len(trace.steps) > 1:
            self._display_reasoning_flow_diagram(trace.steps)
        
        for i, step in enumerate(trace.steps):
            # Add visual separator between steps
            if i > 0:
                click.echo("   │")
            
            # Step header with visual indicator
            step_indicator = "🔄" if step.action else "💭"
            click.echo(f"\n{step_indicator} Step {step.step_number}: {step.thought}")
            
            # Action details
            if step.action:
                if self.config.verbose:
                    click.echo(f"   ├─ 🔧 Action: {step.action.tool_name}")
                    if step.action.parameters:
                        params_str = ", ".join(f"{k}={v}" for k, v in step.action.parameters.items())
                        click.echo(f"   │  Parameters: {params_str}")
                else:
                    click.echo(f"   🔧 Using: {step.action.tool_name}")
            
            # Tool results with enhanced visualization
            if step.tool_results:
                for result in step.tool_results:
                    status = "✅" if result.success else "❌"
                    duration_color = self._get_duration_color(result.execution_time)
                    
                    if self.config.verbose:
                        click.echo(f"   ├─ {status} {result.tool_name} ({duration_color}{result.execution_time:.2f}s{click.style('', reset=True)})")
                        
                        if result.success and result.data:
                            # Show key data points
                            key_data = self._extract_key_data_points(result.data)
                            if key_data:
                                click.echo(f"   │  📊 Key findings: {key_data}")
                        
                        if not result.success and result.error_message:
                            click.echo(f"   │  💥 Error: {click.style(result.error_message, fg='red')}")
                    else:
                        click.echo(f"   {status} {result.tool_name} ({result.execution_time:.2f}s)")
            
            # Observation with enhanced formatting
            if step.observation:
                observation_lines = step.observation.split(';')
                if len(observation_lines) > 1 and self.config.verbose:
                    click.echo(f"   └─ 👁️  Observations:")
                    for obs_line in observation_lines:
                        if obs_line.strip():
                            click.echo(f"      • {obs_line.strip()}")
                else:
                    click.echo(f"   └─ 👁️  {step.observation}")
        
        # Show reasoning summary
        if len(trace.steps) > 2:
            self._display_reasoning_summary(trace)
    
    def _display_resolution_plan(self, plan: ResolutionPlan):
        """Display resolution plan."""
        click.echo(f"\n🎯 RESOLUTION PLAN")
        click.echo("-" * 40)
        
        # Plan overview
        click.echo(f"📊 Success Probability: {plan.success_probability:.1%}")
        click.echo(f"⏱️  Estimated Duration: {self._format_duration(plan.estimated_duration)}")
        click.echo(f"👥 Stakeholders: {', '.join(plan.stakeholders)}")
        
        # Plan steps
        click.echo(f"\n📋 Action Steps ({len(plan.steps)}):")
        for step in plan.steps:
            self._display_plan_step(step)
        
        # Alternatives (if any)
        if plan.alternatives:
            click.echo(f"\n🔄 Alternative Options:")
            for i, alt in enumerate(plan.alternatives, 1):
                click.echo(f"   {i}. {alt}")
    
    def _display_plan_step(self, step: PlanStep):
        """Display a single plan step."""
        click.echo(f"\n   {step.sequence}. {step.action}")
        click.echo(f"      👤 Responsible: {step.responsible_party}")
        click.echo(f"      ⏱️  Time: {self._format_duration(step.estimated_time)}")
        
        if step.dependencies:
            deps = ", ".join(str(d) for d in step.dependencies)
            click.echo(f"      🔗 Depends on: Steps {deps}")
        
        if step.success_criteria:
            click.echo(f"      ✅ Success: {step.success_criteria}")
    
    def _display_timing_info(self, result: ResolutionResult):
        """Display timing information."""
        if result.reasoning_trace and result.reasoning_trace.start_time and result.reasoning_trace.end_time:
            duration = result.reasoning_trace.end_time - result.reasoning_trace.start_time
            click.echo(f"\n⏱️  TIMING INFORMATION")
            click.echo("-" * 40)
            click.echo(f"Total processing time: {duration.total_seconds():.2f}s")
            click.echo(f"Reasoning steps: {len(result.reasoning_trace.steps)}")
            
            if result.reasoning_trace.steps:
                avg_step_time = duration.total_seconds() / len(result.reasoning_trace.steps)
                click.echo(f"Average step time: {avg_step_time:.2f}s")
    
    def _display_success_status(self, result: ResolutionResult):
        """Display success status."""
        click.echo(f"\n{'🎉' if result.success else '💥'} STATUS")
        click.echo("-" * 40)
        
        if result.success:
            click.echo("✅ Scenario processed successfully")
            click.echo("📋 Resolution plan generated")
            click.echo("🚀 Ready for implementation")
        else:
            click.echo("❌ Processing failed")
            if result.error_message:
                click.echo(f"💥 Error: {result.error_message}")
            click.echo("🔧 Please check configuration and try again")
        
        click.echo("=" * 60)
    
    def _display_json_result(self, result: ResolutionResult):
        """Display result in JSON format."""
        result_dict = {
            "success": result.success,
            "scenario": {
                "description": result.scenario.description,
                "type": result.scenario.scenario_type.value,
                "urgency": result.scenario.urgency_level.value,
                "entities": [
                    {
                        "text": e.text,
                        "type": e.entity_type.value,
                        "confidence": e.confidence
                    }
                    for e in result.scenario.entities
                ]
            },
            "reasoning_steps": len(result.reasoning_trace.steps) if result.reasoning_trace else 0,
            "resolution_plan": {
                "steps": [
                    {
                        "sequence": step.sequence,
                        "action": step.action,
                        "responsible_party": step.responsible_party,
                        "estimated_time_minutes": step.estimated_time.total_seconds() / 60,
                        "dependencies": step.dependencies,
                        "success_criteria": step.success_criteria
                    }
                    for step in result.resolution_plan.steps
                ],
                "estimated_duration_minutes": result.resolution_plan.estimated_duration.total_seconds() / 60,
                "success_probability": result.resolution_plan.success_probability,
                "stakeholders": result.resolution_plan.stakeholders,
                "alternatives": result.resolution_plan.alternatives
            }
        }
        
        if result.error_message:
            result_dict["error_message"] = result.error_message
        
        click.echo(json.dumps(result_dict, indent=2))
    
    def _display_plain_result(self, result: ResolutionResult):
        """Display result in plain text format."""
        click.echo(f"Scenario: {result.scenario.description}")
        click.echo(f"Type: {result.scenario.scenario_type.value}")
        click.echo(f"Urgency: {result.scenario.urgency_level.value}")
        click.echo(f"Success: {'Yes' if result.success else 'No'}")
        
        if result.error_message:
            click.echo(f"Error: {result.error_message}")
        
        click.echo(f"Resolution Plan ({len(result.resolution_plan.steps)} steps):")
        for step in result.resolution_plan.steps:
            click.echo(f"  {step.sequence}. {step.action} ({step.responsible_party})")
        
        click.echo(f"Success Probability: {result.resolution_plan.success_probability:.1%}")
        click.echo(f"Estimated Duration: {self._format_duration(result.resolution_plan.estimated_duration)}")
    
    def display_agent_status(self, state: Dict[str, Any], metrics: Dict[str, Any]):
        """Display agent status and metrics."""
        click.echo("🤖 AGENT STATUS")
        click.echo("=" * 40)
        
        click.echo(f"Status: {state['status'].title()}")
        
        if state['current_scenario']:
            click.echo(f"Current Scenario: {state['current_scenario'][:50]}...")
        
        if state['reasoning_steps'] > 0:
            click.echo(f"Reasoning Steps: {state['reasoning_steps']}")
        
        if state['error_message']:
            click.echo(f"Last Error: {state['error_message']}")
        
        # Performance metrics
        if not metrics.get('no_data'):
            click.echo(f"\n📊 PERFORMANCE METRICS")
            click.echo("-" * 40)
            click.echo(f"Total Scenarios: {metrics['total_scenarios_processed']}")
            click.echo(f"Avg Processing Time: {metrics['average_processing_time_seconds']:.1f}s")
            click.echo(f"Avg Reasoning Steps: {metrics['average_reasoning_steps']:.1f}")
            click.echo(f"Avg Success Rate: {metrics['average_success_probability']:.1%}")
    
    def display_configuration(self, config: Config):
        """Display current configuration."""
        click.echo("⚙️  CONFIGURATION")
        click.echo("=" * 40)
        
        click.echo(f"LLM Provider: {config.llm.provider.value}")
        click.echo(f"Model: {config.llm.model}")
        click.echo(f"Max Tokens: {config.llm.max_tokens}")
        click.echo(f"Temperature: {config.llm.temperature}")
        
        click.echo(f"\nReasoning:")
        click.echo(f"  Max Steps: {config.reasoning.max_steps}")
        click.echo(f"  Confidence Threshold: {config.reasoning.confidence_threshold}")
        click.echo(f"  Chain of Thought: {config.reasoning.enable_chain_of_thought}")
        
        click.echo(f"\nCLI:")
        click.echo(f"  Output Format: {config.cli.output_format}")
        click.echo(f"  Show Reasoning: {config.cli.show_reasoning}")
        click.echo(f"  Show Timing: {config.cli.show_timing}")
        click.echo(f"  Verbose: {config.cli.verbose}")
    
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
    
    def _display_reasoning_flow_diagram(self, steps):
        """Display a visual flow diagram of the reasoning process."""
        click.echo("\n📈 Reasoning Flow:")
        
        flow_items = []
        for step in steps:
            if step.action:
                flow_items.append(f"🔧{step.action.tool_name}")
            else:
                flow_items.append("💭Analysis")
        
        # Create flow diagram
        if len(flow_items) <= 5:
            flow_line = " → ".join(flow_items)
            click.echo(f"   {flow_line}")
        else:
            # Multi-line flow for many steps
            for i in range(0, len(flow_items), 3):
                chunk = flow_items[i:i+3]
                flow_line = " → ".join(chunk)
                if i + 3 < len(flow_items):
                    flow_line += " →"
                click.echo(f"   {flow_line}")
        
        click.echo("")
    
    def _get_duration_color(self, duration: float) -> str:
        """Get color code for duration display based on performance."""
        if duration < 1.0:
            return click.style("", fg='green')  # Fast
        elif duration < 3.0:
            return click.style("", fg='yellow')  # Medium
        else:
            return click.style("", fg='red')  # Slow
    
    def _extract_key_data_points(self, data: Dict[str, Any]) -> str:
        """Extract key data points for display."""
        if not data:
            return ""
        
        key_points = []
        
        # Look for common important fields
        important_fields = [
            'status', 'delay_minutes', 'congestion_level', 'eta', 'distance',
            'available', 'prep_time', 'alternative_route', 'success_rate'
        ]
        
        for field in important_fields:
            if field in data:
                value = data[field]
                if isinstance(value, (int, float)):
                    if field.endswith('_minutes') or field == 'prep_time':
                        key_points.append(f"{field.replace('_', ' ')}: {value}min")
                    else:
                        key_points.append(f"{field.replace('_', ' ')}: {value}")
                else:
                    key_points.append(f"{field.replace('_', ' ')}: {value}")
        
        # If no important fields found, show first few fields
        if not key_points:
            for key, value in list(data.items())[:2]:
                if isinstance(value, (str, int, float, bool)):
                    key_points.append(f"{key}: {value}")
        
        return ", ".join(key_points[:3])  # Limit to 3 key points
    
    def _display_reasoning_summary(self, trace):
        """Display a summary of the reasoning process."""
        click.echo(f"\n📋 Reasoning Summary:")
        
        # Count tool usage
        tool_usage = {}
        successful_tools = 0
        failed_tools = 0
        
        for step in trace.steps:
            if step.tool_results:
                for result in step.tool_results:
                    tool_usage[result.tool_name] = tool_usage.get(result.tool_name, 0) + 1
                    if result.success:
                        successful_tools += 1
                    else:
                        failed_tools += 1
        
        if tool_usage:
            tools_used = ", ".join(f"{tool}({count})" for tool, count in tool_usage.items())
            click.echo(f"   🔧 Tools used: {tools_used}")
            click.echo(f"   📊 Success rate: {successful_tools}/{successful_tools + failed_tools} tools")
        
        # Show total reasoning time
        if trace.start_time and trace.end_time:
            total_time = trace.end_time - trace.start_time
            click.echo(f"   ⏱️  Total reasoning time: {total_time.total_seconds():.1f}s")