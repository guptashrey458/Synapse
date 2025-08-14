"""
Main CLI entry point for the Autonomous Delivery Coordinator.
"""
import click
from typing import Optional

from ..config.settings import load_config


@click.command()
@click.option('--config', '-c', type=str, help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--scenario', '-s', type=str, help='Disruption scenario to process')
def main(config: Optional[str], verbose: bool, scenario: Optional[str]):
    """
    Autonomous Delivery Coordinator CLI.
    
    Process delivery disruption scenarios using AI-powered reasoning.
    """
    # Load configuration
    app_config = load_config(config)
    
    if verbose:
        app_config.cli.verbose = True
    
    click.echo("🚚 Autonomous Delivery Coordinator")
    click.echo("=" * 40)
    
    if scenario:
        click.echo(f"Processing scenario: {scenario}")
        # TODO: Implement scenario processing in later tasks
        click.echo("⚠️  Agent implementation not yet available.")
        click.echo("   This will be implemented in subsequent tasks.")
    else:
        click.echo("Interactive mode not yet implemented.")
        click.echo("Use --scenario to provide a disruption scenario.")
        click.echo("\nExample:")
        click.echo("  delivery-coordinator --scenario 'Traffic jam on I-95 affecting delivery to 123 Main St'")
    
    click.echo(f"\nConfiguration loaded from: {config or 'default settings'}")
    click.echo(f"LLM Provider: {app_config.llm.provider.value}")
    click.echo(f"Model: {app_config.llm.model}")


if __name__ == '__main__':
    main()