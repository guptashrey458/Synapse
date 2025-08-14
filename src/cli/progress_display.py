"""
Real-time progress display for CLI operations.
"""
import click
import time
import threading
from typing import Optional
from datetime import datetime, timedelta


class ProgressTracker:
    """Thread-safe progress tracker for CLI operations."""
    
    def __init__(self):
        self.current_stage = "Initializing"
        self.current_message = "Starting up..."
        self.start_time = datetime.now()
        self.is_complete = False
        self.is_error = False
        self.error_message = ""
        self._lock = threading.Lock()
    
    def update_stage(self, stage: str, message: str):
        """Update current processing stage."""
        with self._lock:
            self.current_stage = stage
            self.current_message = message
    
    def complete(self):
        """Mark processing as complete."""
        with self._lock:
            self.is_complete = True
            self.current_stage = "Complete"
            self.current_message = "Processing finished successfully"
    
    def error(self, error_message: str):
        """Mark processing as failed."""
        with self._lock:
            self.is_error = True
            self.error_message = error_message
            self.current_stage = "Error"
            self.current_message = f"Processing failed: {error_message}"
    
    def get_status(self) -> dict:
        """Get current status thread-safely."""
        with self._lock:
            elapsed = datetime.now() - self.start_time
            return {
                "stage": self.current_stage,
                "message": self.current_message,
                "elapsed": elapsed,
                "is_complete": self.is_complete,
                "is_error": self.is_error,
                "error_message": self.error_message
            }


class ProgressDisplay:
    """Handles real-time progress display in the terminal."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.spinner_index = 0
    
    def show_processing_progress(self, tracker: ProgressTracker):
        """Display real-time progress updates."""
        if not self.verbose:
            self._show_simple_progress(tracker)
        else:
            self._show_detailed_progress(tracker)
    
    def _show_simple_progress(self, tracker: ProgressTracker):
        """Show simple spinner progress."""
        click.echo("Processing scenario...", nl=False)
        
        while True:
            status = tracker.get_status()
            
            if status["is_complete"]:
                click.echo(f"\r✅ Processing complete ({status['elapsed'].total_seconds():.1f}s)")
                break
            elif status["is_error"]:
                click.echo(f"\r❌ Processing failed: {status['error_message']}")
                break
            
            # Show spinner
            spinner = self.spinner_chars[self.spinner_index % len(self.spinner_chars)]
            click.echo(f"\r{spinner} Processing scenario...", nl=False)
            self.spinner_index += 1
            
            time.sleep(0.1)
    
    def _show_detailed_progress(self, tracker: ProgressTracker):
        """Show detailed progress with stages."""
        click.echo("\n" + "=" * 50)
        click.echo("🚚 Processing Delivery Disruption Scenario")
        click.echo("=" * 50)
        
        last_stage = ""
        
        while True:
            status = tracker.get_status()
            
            # Show new stage if changed
            if status["stage"] != last_stage:
                if last_stage:  # Not the first stage
                    click.echo("   ✅ Complete")
                
                if not status["is_complete"] and not status["is_error"]:
                    click.echo(f"\n📋 {status['stage']}")
                    click.echo(f"   {status['message']}")
                
                last_stage = status["stage"]
            
            if status["is_complete"]:
                click.echo("   ✅ Complete")
                click.echo(f"\n🎉 Processing completed successfully in {status['elapsed'].total_seconds():.1f}s")
                click.echo("=" * 50)
                break
            elif status["is_error"]:
                click.echo("   ❌ Failed")
                click.echo(f"\n💥 Processing failed: {status['error_message']}")
                click.echo("=" * 50)
                break
            
            # Show spinner for current stage
            if not status["is_complete"] and not status["is_error"]:
                spinner = self.spinner_chars[self.spinner_index % len(self.spinner_chars)]
                elapsed = status['elapsed'].total_seconds()
                click.echo(f"\r   {spinner} {status['message']} ({elapsed:.1f}s)", nl=False)
                self.spinner_index += 1
            
            time.sleep(0.1)
    
    def show_stage_progress(self, stage: str, message: str, duration: Optional[float] = None):
        """Show progress for a specific stage."""
        if self.verbose:
            click.echo(f"\n📋 {stage}")
            click.echo(f"   {message}")
            
            if duration:
                # Show progress bar for known duration
                self._show_progress_bar(duration)
            else:
                # Show spinner for unknown duration
                click.echo("   ⏳ Processing...")
        else:
            click.echo(f"{stage}: {message}")
    
    def _show_progress_bar(self, duration: float):
        """Show a progress bar for a known duration."""
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            progress = min(elapsed / duration, 1.0)
            
            # Create progress bar
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            
            click.echo(f"\r   [{bar}] {progress:.1%}", nl=False)
            
            if progress >= 1.0:
                click.echo("   ✅ Complete")
                break
            
            time.sleep(0.1)
    
    def show_reasoning_step(self, step_number: int, thought: str, action: Optional[str] = None):
        """Show a reasoning step in real-time."""
        if self.verbose:
            click.echo(f"\n🧠 Step {step_number}: {thought}")
            if action:
                click.echo(f"   🔧 Action: {action}")
    
    def show_tool_execution(self, tool_name: str, success: bool, duration: float):
        """Show tool execution result."""
        if self.verbose:
            status = "✅" if success else "❌"
            click.echo(f"   {status} {tool_name} ({duration:.2f}s)")
    
    def show_completion_summary(self, total_time: float, steps: int, tools_used: int):
        """Show completion summary."""
        if self.verbose:
            click.echo(f"\n📊 Summary:")
            click.echo(f"   ⏱️  Total time: {total_time:.1f}s")
            click.echo(f"   🧠 Reasoning steps: {steps}")
            click.echo(f"   🔧 Tools used: {tools_used}")