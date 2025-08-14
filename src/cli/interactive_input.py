"""
Interactive input handling for CLI.
"""
import click
from typing import Optional, List


class InteractiveInput:
    """Handles interactive user input for the CLI."""
    
    def __init__(self):
        self.example_scenarios = [
            "Traffic jam on I-95 is delaying delivery to 123 Main St, Boston",
            "Restaurant 'Pizza Palace' is closed unexpectedly, order #12345 needs alternative",
            "Customer reports wrong address, delivery #67890 going to 456 Oak Ave instead of 456 Oak St",
            "Driver reports damaged packaging for order #11111, customer may dispute",
            "Multiple road closures affecting deliveries in downtown area"
        ]
    
    def get_scenario_input(self) -> Optional[str]:
        """Get scenario input from user with helpful prompts."""
        click.echo("\n" + "=" * 50)
        click.echo("📝 SCENARIO INPUT")
        click.echo("=" * 50)
        
        # Show examples if user wants them
        if click.confirm("Would you like to see example scenarios?", default=False):
            self._show_example_scenarios()
        
        # Get scenario input
        click.echo("\n💬 Please describe the delivery disruption scenario:")
        click.echo("   (Type your scenario description, or 'quit' to exit)")
        
        while True:
            scenario = click.prompt("Scenario", type=str, prompt_suffix="> ").strip()
            
            if scenario.lower() in ['quit', 'exit', 'q']:
                return None
            
            if len(scenario) < 10:
                click.echo("⚠️  Please provide a more detailed scenario description (at least 10 characters)")
                continue
            
            # Confirm scenario
            click.echo(f"\n📋 You entered: {scenario}")
            if click.confirm("Process this scenario?", default=True):
                return scenario
            
            click.echo("Let's try again...")
    
    def _show_example_scenarios(self):
        """Display example scenarios to help user."""
        click.echo("\n💡 Example scenarios:")
        click.echo("-" * 30)
        
        for i, example in enumerate(self.example_scenarios, 1):
            click.echo(f"{i}. {example}")
        
        click.echo("\n🎯 Your scenario should include:")
        click.echo("   • What went wrong (traffic, merchant issue, address problem, etc.)")
        click.echo("   • Specific details (addresses, order IDs, merchant names)")
        click.echo("   • Any relevant context (timing, customer concerns, etc.)")
    
    def ask_continue(self) -> bool:
        """Ask if user wants to process another scenario."""
        click.echo("\n" + "-" * 50)
        return click.confirm("Would you like to process another scenario?", default=True)
    
    def ask_continue_after_error(self) -> bool:
        """Ask if user wants to continue after an error."""
        click.echo("\n💡 You can try again with a different scenario or check your configuration.")
        return click.confirm("Would you like to try again?", default=True)
    
    def get_configuration_choice(self) -> Optional[str]:
        """Get configuration file choice from user."""
        click.echo("\n⚙️  Configuration Options:")
        click.echo("1. Use default configuration")
        click.echo("2. Specify custom configuration file")
        click.echo("3. Show current configuration")
        
        choice = click.prompt("Choose option", type=click.Choice(['1', '2', '3']), default='1')
        
        if choice == '1':
            return None
        elif choice == '2':
            config_path = click.prompt("Configuration file path", type=str)
            return config_path
        else:  # choice == '3'
            return 'show'
    
    def get_output_format_choice(self) -> str:
        """Get output format preference from user."""
        click.echo("\n📄 Output Format Options:")
        click.echo("1. Structured (default) - Human-readable with sections")
        click.echo("2. JSON - Machine-readable format")
        click.echo("3. Plain - Simple text format")
        
        choice = click.prompt("Choose format", type=click.Choice(['1', '2', '3']), default='1')
        
        format_map = {
            '1': 'structured',
            '2': 'json',
            '3': 'plain'
        }
        
        return format_map[choice]
    
    def confirm_processing_options(self) -> dict:
        """Get processing options from user."""
        options = {}
        
        click.echo("\n🔧 Processing Options:")
        
        options['show_reasoning'] = click.confirm(
            "Show reasoning steps?", default=True
        )
        
        options['show_timing'] = click.confirm(
            "Show timing information?", default=False
        )
        
        options['verbose'] = click.confirm(
            "Enable verbose output?", default=False
        )
        
        return options
    
    def select_from_list(self, items: List[str], prompt: str, allow_multiple: bool = False) -> List[str]:
        """Allow user to select from a list of items."""
        if not items:
            return []
        
        click.echo(f"\n{prompt}")
        click.echo("-" * len(prompt))
        
        for i, item in enumerate(items, 1):
            click.echo(f"{i}. {item}")
        
        if allow_multiple:
            click.echo("\nEnter numbers separated by commas (e.g., 1,3,5) or 'all' for all items:")
            selection = click.prompt("Selection", type=str, default="1")
            
            if selection.lower() == 'all':
                return items
            
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                return [items[i] for i in indices if 0 <= i < len(items)]
            except (ValueError, IndexError):
                click.echo("Invalid selection, using first item.")
                return [items[0]]
        else:
            choice = click.prompt("Select item", type=click.IntRange(1, len(items)), default=1)
            return [items[choice - 1]]
    
    def get_multiline_input(self, prompt: str) -> str:
        """Get multiline input from user."""
        click.echo(f"\n{prompt}")
        click.echo("(Enter your text. Press Ctrl+D or type 'END' on a new line to finish)")
        
        lines = []
        try:
            while True:
                line = input()
                if line.strip() == 'END':
                    break
                lines.append(line)
        except EOFError:
            pass
        
        return '\n'.join(lines)
    
    def confirm_with_details(self, prompt: str, details: List[str]) -> bool:
        """Confirm action with detailed information."""
        click.echo(f"\n{prompt}")
        
        if details:
            click.echo("Details:")
            for detail in details:
                click.echo(f"  • {detail}")
        
        return click.confirm("Proceed?", default=True)
    
    def get_priority_input(self) -> str:
        """Get priority level from user."""
        click.echo("\n⚡ Priority Level:")
        click.echo("1. Low - Can wait, not time-sensitive")
        click.echo("2. Medium - Normal priority")
        click.echo("3. High - Urgent, needs immediate attention")
        click.echo("4. Critical - Emergency, customer impact")
        
        choice = click.prompt("Select priority", type=click.Choice(['1', '2', '3', '4']), default='2')
        
        priority_map = {
            '1': 'low',
            '2': 'medium',
            '3': 'high',
            '4': 'critical'
        }
        
        return priority_map[choice]