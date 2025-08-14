"""
Reasoning engine interfaces for ReAct pattern implementation.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from ..agent.interfaces import ReasoningStep, ReasoningTrace, DisruptionScenario, ResolutionPlan
from ..tools.interfaces import ToolResult


@dataclass
class ReasoningContext:
    scenario: DisruptionScenario
    current_step: int
    previous_steps: List[ReasoningStep]
    tool_results: List[ToolResult]
    available_tools: List[str]


@dataclass
class Evaluation:
    confidence: float
    next_action: Optional[str]
    should_continue: bool
    reasoning: str


class ReasoningEngine(ABC):
    """Interface for reasoning engines that implement ReAct pattern."""
    
    @abstractmethod
    def generate_reasoning_step(self, context: ReasoningContext) -> ReasoningStep:
        """
        Generate the next reasoning step based on current context.
        
        Args:
            context: Current reasoning context including scenario and history
            
        Returns:
            ReasoningStep containing thought, action, and observation
        """
        pass
    
    @abstractmethod
    def evaluate_tool_results(self, results: List[ToolResult], context: ReasoningContext) -> Evaluation:
        """
        Evaluate tool results and determine next steps.
        
        Args:
            results: List of tool execution results
            context: Current reasoning context
            
        Returns:
            Evaluation containing confidence and next action recommendations
        """
        pass
    
    @abstractmethod
    def should_continue_reasoning(self, trace: ReasoningTrace) -> bool:
        """
        Determine if more reasoning steps are needed.
        
        Args:
            trace: Current reasoning trace
            
        Returns:
            True if more reasoning is needed, False if ready to generate plan
        """
        pass
    
    @abstractmethod
    def generate_final_plan(self, trace: ReasoningTrace) -> ResolutionPlan:
        """
        Create final resolution plan from reasoning trace.
        
        Args:
            trace: Complete reasoning trace
            
        Returns:
            ResolutionPlan with actionable steps
        """
        pass


class ChainOfThoughtLogger(ABC):
    """Interface for logging and formatting reasoning traces."""
    
    @abstractmethod
    def log_step(self, step: ReasoningStep) -> None:
        """
        Log a single reasoning step.
        
        Args:
            step: ReasoningStep to log
        """
        pass
    
    @abstractmethod
    def format_trace(self, trace: ReasoningTrace) -> str:
        """
        Format complete reasoning trace for display.
        
        Args:
            trace: ReasoningTrace to format
            
        Returns:
            Formatted string representation of the trace
        """
        pass
    
    @abstractmethod
    def get_trace_summary(self, trace: ReasoningTrace) -> str:
        """
        Get summary of reasoning trace.
        
        Args:
            trace: ReasoningTrace to summarize
            
        Returns:
            Summary string of key reasoning points
        """
        pass