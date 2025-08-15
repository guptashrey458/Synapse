"""
Autonomous agent core implementation for delivery coordination.
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .interfaces import Agent, ResolutionResult
from .models import (
    ValidatedDisruptionScenario, EntityExtractor, ValidatedReasoningTrace,
    ValidatedReasoningStep, ToolAction, ToolResult
)
from .scenario_analyzer import ScenarioAnalyzer, ToolRecommendation
from ..reasoning.interfaces import ReasoningEngine, ReasoningContext
from ..reasoning.engine import ReActReasoningEngine
from ..tools.interfaces import ToolManager
from ..llm.interfaces import LLMProvider
from ..config.settings import LLMConfig

# Performance optimization imports (with fallback handling)
try:
    from .cache import ToolResultCache, ScenarioCache, PromptCache
    from .concurrent_executor import ConcurrentToolExecutor, ToolExecution, BatchToolExecutor
    PERFORMANCE_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    PERFORMANCE_OPTIMIZATIONS_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for autonomous agent."""
    max_reasoning_steps: int = 10
    reasoning_timeout: int = 300  # 5 minutes
    enable_context_tracking: bool = True
    enable_state_management: bool = True
    log_reasoning_steps: bool = True
    # Performance optimization options
    enable_caching: bool = False
    concurrent_tools: bool = False
    optimize_prompts: bool = False
    batch_tool_calls: bool = False
    max_concurrent_tools: int = 3


@dataclass
class AgentState:
    """Represents the current state of the agent."""
    current_scenario: Optional[ValidatedDisruptionScenario] = None
    reasoning_trace: Optional[ValidatedReasoningTrace] = None
    processing_start_time: Optional[datetime] = None
    processing_end_time: Optional[datetime] = None
    status: str = "idle"  # idle, processing, completed, error
    error_message: Optional[str] = None
    context_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context_data is None:
            self.context_data = {}


class AutonomousAgent(Agent):
    """
    Main autonomous agent orchestrator that coordinates all components
    to process delivery disruption scenarios.
    """
    
    def __init__(self, llm_provider: LLMProvider, tool_manager: ToolManager,
                 config: Optional[AgentConfig] = None):
        """
        Initialize the autonomous agent.
        
        Args:
            llm_provider: LLM provider for reasoning
            tool_manager: Tool manager for executing actions
            config: Agent configuration
        """
        self.llm_provider = llm_provider
        self.tool_manager = tool_manager
        self.config = config or AgentConfig()
        
        # Initialize reasoning engine
        self.reasoning_engine = ReActReasoningEngine(
            llm_provider=llm_provider,
            tool_manager=tool_manager
        )
        
        # Initialize entity extractor and enhanced scenario analyzer
        self.entity_extractor = EntityExtractor()
        try:
            from .enhanced_scenario_analyzer import EnhancedScenarioAnalyzer
            self.scenario_analyzer = EnhancedScenarioAnalyzer()
            logger.info("Using EnhancedScenarioAnalyzer")
        except ImportError:
            from .scenario_analyzer import ScenarioAnalyzer
            self.scenario_analyzer = ScenarioAnalyzer()
            logger.info("Using standard ScenarioAnalyzer")
        
        # Agent state management
        self.state = AgentState()
        
        # Context tracking
        self.context_history: List[Dict[str, Any]] = []
        
        # Performance optimization components (with fallback handling)
        if PERFORMANCE_OPTIMIZATIONS_AVAILABLE and config:
            self.tool_cache = ToolResultCache() if config.enable_caching else None
            self.scenario_cache = ScenarioCache() if config.enable_caching else None
            self.prompt_cache = PromptCache() if config.optimize_prompts else None
            self.concurrent_executor = (ConcurrentToolExecutor(tool_manager, config.max_concurrent_tools) 
                                      if config.concurrent_tools else None)
            self.batch_executor = BatchToolExecutor(tool_manager) if config.batch_tool_calls else None
        else:
            self.tool_cache = None
            self.scenario_cache = None
            self.prompt_cache = None
            self.concurrent_executor = None
            self.batch_executor = None
        
        logger.info("Initialized AutonomousAgent with reasoning engine and tool manager")
    
    def process_scenario(self, scenario: str) -> ResolutionResult:
        """
        Main entry point for processing disruption scenarios.
        
        Args:
            scenario: Natural language description of the disruption
            
        Returns:
            ResolutionResult containing the reasoning trace and resolution plan
        """
        logger.info(f"Processing scenario: {scenario[:100]}...")
        
        try:
            # Check scenario cache first
            if self.scenario_cache and self.scenario_cache.should_cache(scenario):
                cached_result = self.scenario_cache.get(scenario)
                if cached_result:
                    logger.info("Returning cached scenario result")
                    return cached_result
            
            # Update agent state
            self._update_state("processing", processing_start_time=datetime.now())
            
            # Parse and validate scenario
            validated_scenario = self._parse_scenario(scenario)
            self.state.current_scenario = validated_scenario
            
            # Log scenario acknowledgment
            if self.config.log_reasoning_steps:
                logger.info(f"Acknowledged scenario: {validated_scenario.scenario_type.value} "
                          f"(urgency: {validated_scenario.urgency_level.value})")
            
            # Execute reasoning loop
            reasoning_trace = self._reasoning_loop(validated_scenario)
            self.state.reasoning_trace = reasoning_trace
            
            # Generate final resolution plan
            resolution_plan = self.reasoning_engine.generate_final_plan(reasoning_trace)
            
            # Create successful result
            result = ResolutionResult(
                scenario=validated_scenario,
                reasoning_trace=reasoning_trace,
                resolution_plan=resolution_plan,
                success=True
            )
            
            # Update state
            self._update_state("completed", processing_end_time=datetime.now())
            
            # Track context if enabled
            if self.config.enable_context_tracking:
                self._track_context(validated_scenario, reasoning_trace, resolution_plan)
            
            # Cache the result if caching is enabled
            if self.scenario_cache and self.scenario_cache.should_cache(scenario):
                self.scenario_cache.put(scenario, result)
            
            logger.info(f"Successfully processed scenario in "
                       f"{(self.state.processing_end_time - self.state.processing_start_time).total_seconds():.1f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing scenario: {e}")
            
            # Update state with error
            self._update_state("error", error_message=str(e), processing_end_time=datetime.now())
            
            # Create error result
            return ResolutionResult(
                scenario=self.state.current_scenario or self._create_fallback_scenario(scenario),
                reasoning_trace=self.state.reasoning_trace or self._create_empty_trace(),
                resolution_plan=self._create_fallback_plan(),
                success=False,
                error_message=str(e)
            )
    
    def _reasoning_loop(self, scenario: ValidatedDisruptionScenario) -> ValidatedReasoningTrace:
        """
        Implements the ReAct reasoning pattern loop with intelligent tool selection.
        
        Args:
            scenario: Validated disruption scenario
            
        Returns:
            ValidatedReasoningTrace with complete reasoning process
        """
        logger.debug("Starting reasoning loop with scenario analysis")
        
        # Perform initial scenario analysis
        scenario_analysis = self.scenario_analyzer.analyze_scenario(
            scenario, self.tool_manager.get_available_tools()
        )
        
        # Store analysis in context for future reference
        self.state.context_data["scenario_analysis"] = scenario_analysis
        
        # Initialize reasoning trace
        reasoning_trace = ValidatedReasoningTrace(
            steps=[],
            scenario=scenario,
            start_time=datetime.now()
        )
        
        step_number = 1
        executed_tools = []
        all_tool_results = []
        
        while self._should_continue_reasoning(reasoning_trace):
            try:
                # Use intelligent tool selection
                next_tool = self.scenario_analyzer.select_next_tool(
                    scenario, executed_tools, all_tool_results, 
                    self.tool_manager.get_available_tools()
                )
                
                if next_tool:
                    # Create reasoning step with intelligent tool selection
                    reasoning_step = self._create_intelligent_reasoning_step(
                        step_number, scenario, next_tool, all_tool_results
                    )
                    
                    # Execute the recommended tool
                    if reasoning_step.action:
                        tool_results = self._execute_tool_with_analysis(reasoning_step.action)
                        reasoning_step.tool_results = tool_results
                        all_tool_results.extend(tool_results)
                        
                        # Track executed tools
                        executed_tools.append(reasoning_step.action.tool_name)
                        
                        # Update observation with integrated results
                        integrated_info = self.scenario_analyzer.integrate_tool_results(
                            all_tool_results, scenario
                        )
                        reasoning_step.observation = self._format_integrated_observation(
                            tool_results, integrated_info
                        )
                else:
                    # No more tools recommended, create final reasoning step
                    reasoning_step = ValidatedReasoningStep(
                        step_number=step_number,
                        thought="Analysis complete. Sufficient information gathered for resolution planning.",
                        observation=f"Executed {len(executed_tools)} tools with integrated analysis complete",
                        timestamp=datetime.now()
                    )
                
                # Add step to trace
                reasoning_trace.add_step(reasoning_step)
                
                # Log reasoning step if enabled
                if self.config.log_reasoning_steps:
                    self._log_reasoning_step(reasoning_step)
                
                step_number += 1
                
                # Break if no more tools to execute
                if not next_tool:
                    break
                
                # Small delay to prevent overwhelming APIs
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in reasoning step {step_number}: {e}")
                
                # Add error step
                error_step = ValidatedReasoningStep(
                    step_number=step_number,
                    thought=f"Error in reasoning: {str(e)}",
                    observation="Failed to complete reasoning step",
                    timestamp=datetime.now()
                )
                reasoning_trace.add_step(error_step)
                break
        
        # Complete the trace
        reasoning_trace.complete_trace()
        
        logger.debug(f"Completed intelligent reasoning loop with {len(reasoning_trace.steps)} steps, "
                    f"executed {len(executed_tools)} tools")
        return reasoning_trace
    
    def _should_continue_reasoning(self, trace: ValidatedReasoningTrace) -> bool:
        """
        Determine if more reasoning steps are needed.
        
        Args:
            trace: Current reasoning trace
            
        Returns:
            True if more reasoning is needed
        """
        # Use reasoning engine's logic
        return self.reasoning_engine.should_continue_reasoning(trace)
    
    def _parse_scenario(self, scenario: str) -> ValidatedDisruptionScenario:
        """
        Parse natural language scenario into validated structure.
        
        Args:
            scenario: Natural language description
            
        Returns:
            ValidatedDisruptionScenario with extracted entities and classification
        """
        logger.debug("Parsing scenario and extracting entities")
        
        # Use entity extractor to create validated scenario
        validated_scenario = self.entity_extractor.create_scenario_from_text(scenario)
        
        logger.debug(f"Extracted {len(validated_scenario.entities)} entities, "
                    f"classified as {validated_scenario.scenario_type.value}")
        
        return validated_scenario
    
    def _update_state(self, status: str, **kwargs) -> None:
        """
        Update agent state with new information.
        
        Args:
            status: New status
            **kwargs: Additional state updates
        """
        if not self.config.enable_state_management:
            return
        
        self.state.status = status
        
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
        
        logger.debug(f"Updated agent state: {status}")
    
    def _track_context(self, scenario: ValidatedDisruptionScenario, 
                      trace: ValidatedReasoningTrace, plan) -> None:
        """
        Track context information for future reference.
        
        Args:
            scenario: Processed scenario
            trace: Reasoning trace
            plan: Resolution plan
        """
        if not self.config.enable_context_tracking:
            return
        
        context_entry = {
            "timestamp": datetime.now().isoformat(),
            "scenario_type": scenario.scenario_type.value,
            "urgency_level": scenario.urgency_level.value,
            "entity_count": len(scenario.entities),
            "reasoning_steps": len(trace.steps),
            "processing_time": (trace.end_time - trace.start_time).total_seconds() if trace.end_time else None,
            "plan_steps": len(plan.steps),
            "success_probability": plan.success_probability,
            "tools_used": list(trace.get_tool_usage_summary().keys())
        }
        
        self.context_history.append(context_entry)
        
        # Keep only recent context (last 100 entries)
        if len(self.context_history) > 100:
            self.context_history = self.context_history[-100:]
        
        logger.debug("Tracked context for scenario processing")
    
    def _log_reasoning_step(self, step: ValidatedReasoningStep) -> None:
        """
        Log reasoning step for debugging and transparency.
        
        Args:
            step: Reasoning step to log
        """
        logger.info(f"Step {step.step_number}: {step.thought}")
        
        if step.action:
            logger.info(f"  Action: {step.action.tool_name}({step.action.parameters})")
        
        if step.observation:
            logger.info(f"  Observation: {step.observation}")
        
        if step.tool_results:
            for result in step.tool_results:
                status = "✓" if result.success else "✗"
                logger.info(f"  Tool {result.tool_name}: {status} ({result.execution_time:.2f}s)")
    
    def _create_fallback_scenario(self, description: str) -> ValidatedDisruptionScenario:
        """Create a basic fallback scenario when parsing fails."""
        from .interfaces import ScenarioType, UrgencyLevel
        
        return ValidatedDisruptionScenario(
            description=description,
            entities=[],
            scenario_type=ScenarioType.OTHER,
            urgency_level=UrgencyLevel.MEDIUM
        )
    
    def _create_empty_trace(self) -> ValidatedReasoningTrace:
        """Create an empty reasoning trace for error cases."""
        return ValidatedReasoningTrace(
            steps=[],
            scenario=self._create_fallback_scenario("Unknown scenario"),
            start_time=datetime.now(),
            end_time=datetime.now()
        )
    
    def _create_fallback_plan(self):
        """Create a basic fallback plan when planning fails."""
        from .interfaces import ResolutionPlan, PlanStep
        
        return ResolutionPlan(
            steps=[
                PlanStep(
                    sequence=1,
                    action="Escalate to human operator for manual resolution",
                    responsible_party="Operations team",
                    estimated_time=timedelta(minutes=30),
                    dependencies=[],
                    success_criteria="Human operator acknowledges and takes over"
                )
            ],
            estimated_duration=timedelta(minutes=30),
            success_probability=0.8,
            alternatives=["Contact customer directly for clarification"],
            stakeholders=["Customer", "Operations team"]
        )
    
    # Public methods for state inspection and control
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current agent state information.
        
        Returns:
            Dictionary containing current state
        """
        return {
            "status": self.state.status,
            "current_scenario": (self.state.current_scenario.description 
                               if self.state.current_scenario else None),
            "processing_start_time": (self.state.processing_start_time.isoformat() 
                                    if self.state.processing_start_time else None),
            "processing_end_time": (self.state.processing_end_time.isoformat() 
                                  if self.state.processing_end_time else None),
            "reasoning_steps": (len(self.state.reasoning_trace.steps) 
                              if self.state.reasoning_trace else 0),
            "error_message": self.state.error_message,
            "context_data_keys": list(self.state.context_data.keys()) if self.state.context_data else []
        }
    
    def get_context_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get context history for analysis.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of context history entries
        """
        if limit:
            return self.context_history[-limit:]
        return self.context_history.copy()
    
    def reset_state(self) -> None:
        """Reset agent state to initial conditions."""
        self.state = AgentState()
        logger.info("Reset agent state")
    
    def update_context_data(self, key: str, value: Any) -> None:
        """
        Update context data for the agent.
        
        Args:
            key: Context key
            value: Context value
        """
        if self.state.context_data is None:
            self.state.context_data = {}
        
        self.state.context_data[key] = value
        logger.debug(f"Updated context data: {key}")
    
    def _create_intelligent_reasoning_step(self, step_number: int, 
                                          scenario: ValidatedDisruptionScenario,
                                          tool_recommendation: ToolRecommendation,
                                          previous_results: List[ToolResult]) -> ValidatedReasoningStep:
        """
        Create a reasoning step based on intelligent tool recommendation.
        
        Args:
            step_number: Current step number
            scenario: Current scenario
            tool_recommendation: Recommended tool with reasoning
            previous_results: Previous tool execution results
            
        Returns:
            ValidatedReasoningStep with intelligent reasoning
        """
        # Create thought based on tool recommendation reasoning
        thought = f"Step {step_number}: {tool_recommendation.reasoning}"
        
        # Add context from previous results if available
        if previous_results:
            successful_tools = [r.tool_name for r in previous_results if r.success]
            if successful_tools:
                thought += f" Previous tools ({', '.join(successful_tools)}) provide context for this decision."
        
        # Create tool action with recommended parameters
        action = ToolAction(
            tool_name=tool_recommendation.tool_name,
            parameters=tool_recommendation.suggested_parameters
        )
        
        return ValidatedReasoningStep(
            step_number=step_number,
            thought=thought,
            action=action,
            timestamp=datetime.now()
        )
    
    def _execute_tool_with_analysis(self, action: ToolAction) -> List[ToolResult]:
        """
        Execute tool with enhanced error handling and analysis.
        
        Args:
            action: Tool action to execute
            
        Returns:
            List of tool results
        """
        try:
            result = self.tool_manager.execute_tool(
                tool_name=action.tool_name,
                parameters=action.parameters,
                timeout=30
            )
            return [result]
        except Exception as e:
            logger.error(f"Error executing tool {action.tool_name}: {e}")
            return [ToolResult(
                tool_name=action.tool_name,
                success=False,
                data={},
                execution_time=0.0,
                error_message=str(e)
            )]
    
    def _format_integrated_observation(self, tool_results: List[ToolResult],
                                     integrated_info: Dict[str, Any]) -> str:
        """
        Format observation with integrated analysis results.
        
        Args:
            tool_results: Recent tool results
            integrated_info: Integrated information from scenario analyzer
            
        Returns:
            Formatted observation string
        """
        observations = []
        
        # Add tool execution results
        for result in tool_results:
            if result.success:
                data_summary = self._summarize_tool_data(result.data)
                observations.append(f"{result.tool_name} succeeded: {data_summary}")
            else:
                observations.append(f"{result.tool_name} failed: {result.error_message}")
        
        # Add integrated analysis insights
        if integrated_info.get("key_findings"):
            findings_count = len(integrated_info["key_findings"])
            observations.append(f"Integrated analysis reveals {findings_count} key findings")
        
        if integrated_info.get("action_items"):
            action_count = len(integrated_info["action_items"])
            observations.append(f"{action_count} action items identified")
        
        confidence = integrated_info.get("confidence_score", 0)
        completeness = integrated_info.get("completeness_score", 0)
        observations.append(f"Analysis confidence: {confidence:.1%}, completeness: {completeness:.1%}")
        
        return "; ".join(observations)
    
    def _summarize_tool_data(self, data: Dict[str, Any]) -> str:
        """Summarize tool result data for observation."""
        if not data:
            return "No data returned"
        
        # Create a concise summary of the data
        summary_parts = []
        for key, value in data.items():
            if isinstance(value, (str, int, float, bool)):
                summary_parts.append(f"{key}: {value}")
            elif isinstance(value, list):
                summary_parts.append(f"{key}: {len(value)} items")
            elif isinstance(value, dict):
                summary_parts.append(f"{key}: {len(value)} fields")
        
        return ", ".join(summary_parts[:3])  # Limit to first 3 items
    
    def get_scenario_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Get the current scenario analysis if available.
        
        Returns:
            Scenario analysis dictionary or None
        """
        return self.state.context_data.get("scenario_analysis")
    
    def get_tool_recommendations(self, scenario: ValidatedDisruptionScenario) -> List[Dict[str, Any]]:
        """
        Get tool recommendations for a scenario.
        
        Args:
            scenario: Scenario to analyze
            
        Returns:
            List of tool recommendation dictionaries
        """
        analysis = self.scenario_analyzer.analyze_scenario(
            scenario, self.tool_manager.get_available_tools()
        )
        
        return [
            {
                "tool_name": rec.tool_name,
                "priority": rec.priority.name,
                "confidence": rec.confidence,
                "reasoning": rec.reasoning,
                "suggested_parameters": rec.suggested_parameters,
                "dependencies": rec.dependencies
            }
            for rec in analysis.recommended_tools
        ]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the agent.
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.context_history:
            return {"no_data": True}
        
        # Calculate metrics from context history
        total_scenarios = len(self.context_history)
        avg_processing_time = sum(
            entry.get("processing_time", 0) for entry in self.context_history
            if entry.get("processing_time")
        ) / total_scenarios if total_scenarios > 0 else 0
        
        avg_reasoning_steps = sum(
            entry.get("reasoning_steps", 0) for entry in self.context_history
        ) / total_scenarios if total_scenarios > 0 else 0
        
        avg_plan_steps = sum(
            entry.get("plan_steps", 0) for entry in self.context_history
        ) / total_scenarios if total_scenarios > 0 else 0
        
        avg_success_probability = sum(
            entry.get("success_probability", 0) for entry in self.context_history
        ) / total_scenarios if total_scenarios > 0 else 0
        
        # Scenario type distribution
        scenario_types = {}
        urgency_levels = {}
        
        for entry in self.context_history:
            scenario_type = entry.get("scenario_type", "unknown")
            urgency_level = entry.get("urgency_level", "unknown")
            
            scenario_types[scenario_type] = scenario_types.get(scenario_type, 0) + 1
            urgency_levels[urgency_level] = urgency_levels.get(urgency_level, 0) + 1
        
        # Add tool usage statistics
        tool_usage = {}
        for entry in self.context_history:
            tools_used = entry.get("tools_used", [])
            for tool in tools_used:
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
        
        return {
            "total_scenarios_processed": total_scenarios,
            "average_processing_time_seconds": avg_processing_time,
            "average_reasoning_steps": avg_reasoning_steps,
            "average_plan_steps": avg_plan_steps,
            "average_success_probability": avg_success_probability,
            "scenario_type_distribution": scenario_types,
            "urgency_level_distribution": urgency_levels,
            "tool_usage_distribution": tool_usage
        }