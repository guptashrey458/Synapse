"""
Core reasoning engine implementation using ReAct pattern.
"""
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .interfaces import (
    ReasoningEngine, ChainOfThoughtLogger, ReasoningContext, 
    Evaluation, ReasoningStep, ReasoningTrace, DisruptionScenario, ResolutionPlan
)
from .plan_generator import PlanGenerator
from ..agent.interfaces import PlanStep
from ..agent.models import (
    ValidatedReasoningStep, ValidatedReasoningTrace, ValidatedResolutionPlan,
    ValidatedPlanStep, ToolAction, ToolResult
)
from ..llm.interfaces import LLMProvider, Message, MessageRole, LLMProviderError
from ..llm.templates import PromptTemplateManager, DEFAULT_DELIVERY_EXAMPLES
from ..tools.interfaces import ToolManager


logger = logging.getLogger(__name__)


@dataclass
class ReasoningConfig:
    """Configuration for reasoning engine."""
    max_reasoning_steps: int = 10
    max_tool_calls_per_step: int = 3
    reasoning_timeout: int = 300  # 5 minutes
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    enable_examples: bool = True
    temperature: float = 0.1  # Low temperature for consistent reasoning


class ReActReasoningEngine(ReasoningEngine):
    """ReAct pattern reasoning engine implementation."""
    
    def __init__(self, llm_provider: LLMProvider, tool_manager: ToolManager,
                 config: Optional[ReasoningConfig] = None):
        """
        Initialize the reasoning engine.
        
        Args:
            llm_provider: LLM provider for generating reasoning steps
            tool_manager: Tool manager for executing actions
            config: Configuration for reasoning behavior
        """
        self.llm_provider = llm_provider
        self.tool_manager = tool_manager
        self.config = config or ReasoningConfig()
        self.template_manager = PromptTemplateManager()
        self.plan_generator = PlanGenerator()
        
        # Circuit breaker for infinite loops
        self._consecutive_failures = 0
        self._last_thoughts = []  # Track recent thoughts to detect loops
        self._max_repeated_thoughts = 3
        
        logger.info(f"Initialized ReAct reasoning engine with config: {self.config}")
    
    def generate_reasoning_step(self, context: ReasoningContext) -> ReasoningStep:
        """
        Generate the next reasoning step using ReAct pattern.
        
        Args:
            context: Current reasoning context
            
        Returns:
            ReasoningStep with thought, action, and observation
        """
        try:
            # Check circuit breaker
            if self._should_break_circuit(context):
                return self._create_termination_step(context, "Circuit breaker activated")
            
            # Prepare prompt with current context
            prompt_messages = self._prepare_reasoning_prompt(context)
            
            # Generate response from LLM
            response = self.llm_provider.generate_response(
                messages=prompt_messages,
                tools=self._get_tool_schemas(),
                temperature=self.config.temperature,
                max_tokens=1000
            )
            
            # Parse reasoning step from response
            reasoning_step = self._parse_reasoning_response(response, context)
            
            # Execute tool if action is specified
            if reasoning_step.action:
                tool_results = self._execute_reasoning_action(reasoning_step.action)
                reasoning_step.tool_results = tool_results
                
                # Update observation based on tool results
                reasoning_step.observation = self._format_tool_observations(tool_results)
            
            # Update circuit breaker state
            self._update_circuit_breaker(reasoning_step)
            
            return reasoning_step
            
        except Exception as e:
            logger.error(f"Error generating reasoning step: {e}")
            self._consecutive_failures += 1
            
            return ValidatedReasoningStep(
                step_number=context.current_step,
                thought=f"Error in reasoning: {str(e)}",
                observation="Failed to generate proper reasoning step",
                timestamp=datetime.now()
            )
    
    def evaluate_tool_results(self, results: List[ToolResult], context: ReasoningContext) -> Evaluation:
        """
        Evaluate tool results and determine next steps.
        
        Args:
            results: List of tool execution results
            context: Current reasoning context
            
        Returns:
            Evaluation with confidence and recommendations
        """
        if not results:
            return Evaluation(
                confidence=0.0,
                next_action=None,
                should_continue=True,
                reasoning="No tool results to evaluate"
            )
        
        # Analyze success rate
        successful_results = [r for r in results if r.success]
        success_rate = len(successful_results) / len(results)
        
        # Determine confidence based on success rate and data quality
        confidence = self._calculate_confidence(successful_results, context)
        
        # Determine if we should continue reasoning
        should_continue = self._should_continue_from_results(successful_results, context)
        
        # Suggest next action if needed
        next_action = self._suggest_next_action(successful_results, context) if should_continue else None
        
        reasoning = self._build_evaluation_reasoning(results, success_rate, confidence)
        
        return Evaluation(
            confidence=confidence,
            next_action=next_action,
            should_continue=should_continue,
            reasoning=reasoning
        )
    
    def should_continue_reasoning(self, trace: ReasoningTrace) -> bool:
        """
        Determine if more reasoning steps are needed.
        
        Args:
            trace: Current reasoning trace
            
        Returns:
            True if more reasoning is needed
        """
        # Check maximum steps limit
        if len(trace.steps) >= self.config.max_reasoning_steps:
            logger.info(f"Reached maximum reasoning steps: {self.config.max_reasoning_steps}")
            return False
        
        # Check if we have sufficient information for a plan
        if self._has_sufficient_information(trace):
            logger.info("Sufficient information gathered for plan generation")
            return False
        
        # Check for reasoning loops
        if self._detect_reasoning_loop(trace):
            logger.warning("Detected reasoning loop, terminating")
            return False
        
        # Check timeout
        if trace.end_time is None and trace.start_time:
            elapsed = datetime.now() - trace.start_time
            if elapsed.total_seconds() > self.config.reasoning_timeout:
                logger.warning(f"Reasoning timeout after {elapsed.total_seconds()} seconds")
                return False
        
        return True
    
    def generate_final_plan(self, trace: ReasoningTrace) -> ResolutionPlan:
        """
        Create final resolution plan from reasoning trace.
        
        Args:
            trace: Complete reasoning trace
            
        Returns:
            ResolutionPlan with actionable steps
        """
        try:
            # Use the dedicated plan generator to create the plan
            validated_plan = self.plan_generator.generate_plan(trace)
            
            logger.info(f"Generated resolution plan with {len(validated_plan.steps)} steps")
            
            # Convert ValidatedResolutionPlan to ResolutionPlan for interface compatibility
            return ResolutionPlan(
                steps=[
                    PlanStep(
                        sequence=step.sequence,
                        action=step.action,
                        responsible_party=step.responsible_party,
                        estimated_time=step.estimated_time,
                        dependencies=step.dependencies,
                        success_criteria=step.success_criteria
                    )
                    for step in validated_plan.steps
                ],
                estimated_duration=validated_plan.estimated_duration,
                success_probability=validated_plan.success_probability,
                alternatives=[alt.name for alt in validated_plan.alternatives],
                stakeholders=validated_plan.stakeholders
            )
            
        except Exception as e:
            logger.error(f"Error generating final plan: {e}")
            
            # Create fallback plan
            return self._create_fallback_plan(trace)
    
    def _prepare_reasoning_prompt(self, context: ReasoningContext) -> List[Message]:
        """Prepare prompt messages for reasoning step generation."""
        template = self.template_manager.get_template("react")
        
        # Convert tools to schema format
        available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in self.tool_manager.get_available_tools()
        ]
        
        # Convert previous steps to dict format
        previous_steps = []
        for step in context.previous_steps:
            step_dict = {
                "thought": step.thought,
                "action": step.action,
                "observation": step.observation
            }
            previous_steps.append(step_dict)
        
        # Use examples if enabled
        examples = DEFAULT_DELIVERY_EXAMPLES if self.config.enable_examples else []
        
        return template.format(
            scenario=context.scenario.description,
            available_tools=available_tools,
            previous_steps=previous_steps,
            examples=examples
        )
    
    def _get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get tool schemas for LLM function calling."""
        schemas = []
        for tool in self.tool_manager.get_available_tools():
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            schemas.append(schema)
        return schemas
    
    def _parse_reasoning_response(self, response, context: ReasoningContext) -> ValidatedReasoningStep:
        """Parse LLM response into a reasoning step."""
        content = response.content.strip()
        
        # Extract thought from response
        thought = self._extract_thought(content)
        
        # Check for tool calls
        action = None
        if response.tool_calls:
            # Use the first tool call
            tool_call = response.tool_calls[0]
            function_name = tool_call["function"]["name"]
            try:
                parameters = json.loads(tool_call["function"]["arguments"])
                action = ToolAction(tool_name=function_name, parameters=parameters)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool arguments: {e}")
                action = ToolAction(tool_name=function_name, parameters={})
        
        return ValidatedReasoningStep(
            step_number=context.current_step,
            thought=thought,
            action=action,
            timestamp=datetime.now()
        )
    
    def _extract_thought(self, content: str) -> str:
        """Extract the thought portion from LLM response."""
        # Look for "Thought:" pattern
        lines = content.split('\n')
        thought_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("**Thought:**") or line.startswith("Thought:"):
                thought_lines.append(line.split(":", 1)[1].strip())
            elif line.startswith("**Action:**") or line.startswith("Action:"):
                break
            elif thought_lines and line and not line.startswith("**"):
                thought_lines.append(line)
        
        if thought_lines:
            return " ".join(thought_lines)
        
        # Fallback: use first paragraph
        paragraphs = content.split('\n\n')
        return paragraphs[0] if paragraphs else content[:200]
    
    def _execute_reasoning_action(self, action: ToolAction) -> List[ToolResult]:
        """Execute the action specified in reasoning step."""
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
    
    def _format_tool_observations(self, tool_results: List[ToolResult]) -> str:
        """Format tool results into observation text."""
        if not tool_results:
            return "No tool results available"
        
        observations = []
        for result in tool_results:
            if result.success:
                # Format successful result
                data_summary = self._summarize_tool_data(result.data)
                observations.append(f"{result.tool_name} succeeded: {data_summary}")
            else:
                # Format error result
                observations.append(f"{result.tool_name} failed: {result.error_message}")
        
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
    
    def _should_break_circuit(self, context: ReasoningContext) -> bool:
        """Check if circuit breaker should activate."""
        if not self.config.enable_circuit_breaker:
            return False
        
        return self._consecutive_failures >= self.config.circuit_breaker_threshold
    
    def _create_termination_step(self, context: ReasoningContext, reason: str) -> ValidatedReasoningStep:
        """Create a step that terminates reasoning."""
        return ValidatedReasoningStep(
            step_number=context.current_step,
            thought=f"Terminating reasoning: {reason}",
            observation="Reasoning process terminated",
            timestamp=datetime.now()
        )
    
    def _update_circuit_breaker(self, step: ReasoningStep) -> None:
        """Update circuit breaker state based on reasoning step."""
        # Check for repeated thoughts (potential loop)
        thought_summary = step.thought[:50].lower()
        self._last_thoughts.append(thought_summary)
        
        # Keep only recent thoughts
        if len(self._last_thoughts) > self._max_repeated_thoughts * 2:
            self._last_thoughts = self._last_thoughts[-self._max_repeated_thoughts * 2:]
        
        # Check for repetition
        if len(self._last_thoughts) >= self._max_repeated_thoughts:
            recent_thoughts = self._last_thoughts[-self._max_repeated_thoughts:]
            if len(set(recent_thoughts)) <= 1:  # All thoughts are very similar
                self._consecutive_failures += 1
            else:
                self._consecutive_failures = max(0, self._consecutive_failures - 1)
    
    def _calculate_confidence(self, successful_results: List[ToolResult], 
                            context: ReasoningContext) -> float:
        """Calculate confidence based on tool results and context."""
        if not successful_results:
            return 0.1
        
        # Base confidence from success rate
        base_confidence = 0.5
        
        # Boost confidence based on data quality
        data_quality_score = 0.0
        for result in successful_results:
            if result.data:
                # More data fields = higher confidence
                data_quality_score += min(len(result.data), 5) * 0.1
        
        # Boost confidence based on reasoning progress
        progress_score = min(len(context.previous_steps) * 0.05, 0.3)
        
        total_confidence = min(base_confidence + data_quality_score + progress_score, 1.0)
        return total_confidence
    
    def _should_continue_from_results(self, successful_results: List[ToolResult],
                                    context: ReasoningContext) -> bool:
        """Determine if reasoning should continue based on results."""
        # Continue if we have no successful results
        if not successful_results:
            return len(context.previous_steps) < self.config.max_reasoning_steps
        
        # Check if we have enough information for key scenario aspects
        scenario_type = context.scenario.scenario_type.value
        
        if scenario_type == "traffic":
            return not self._has_traffic_information(successful_results)
        elif scenario_type == "merchant":
            return not self._has_merchant_information(successful_results)
        elif scenario_type == "address":
            return not self._has_address_information(successful_results)
        else:
            # For multi-factor or other scenarios, need more comprehensive info
            return len(context.previous_steps) < 5
    
    def _suggest_next_action(self, successful_results: List[ToolResult],
                           context: ReasoningContext) -> Optional[str]:
        """Suggest the next action based on current results."""
        scenario_type = context.scenario.scenario_type.value
        
        # Get tools that haven't been used yet
        used_tools = {result.tool_name for result in successful_results}
        available_tools = {tool.name for tool in self.tool_manager.get_available_tools()}
        unused_tools = available_tools - used_tools
        
        # Suggest based on scenario type and missing information
        if scenario_type == "traffic" and "check_traffic" not in used_tools:
            return "check_traffic"
        elif scenario_type == "merchant" and "get_merchant_status" not in used_tools:
            return "get_merchant_status"
        elif scenario_type == "address" and "validate_address" not in used_tools:
            return "validate_address"
        elif "notify_customer" not in used_tools:
            return "notify_customer"
        
        return None
    
    def _build_evaluation_reasoning(self, results: List[ToolResult], 
                                  success_rate: float, confidence: float) -> str:
        """Build reasoning text for evaluation."""
        successful = len([r for r in results if r.success])
        total = len(results)
        
        reasoning = f"Evaluated {total} tool results: {successful} successful ({success_rate:.1%}). "
        reasoning += f"Confidence level: {confidence:.1%}. "
        
        if confidence > 0.7:
            reasoning += "High confidence in available information."
        elif confidence > 0.4:
            reasoning += "Moderate confidence, may need additional information."
        else:
            reasoning += "Low confidence, more information gathering needed."
        
        return reasoning
    
    def _has_sufficient_information(self, trace: ReasoningTrace) -> bool:
        """Check if trace contains sufficient information for planning."""
        if len(trace.steps) < 2:
            return False
        
        # Check if we have successful tool results
        successful_tools = set()
        for step in trace.steps:
            for result in step.tool_results:
                if result.success:
                    successful_tools.add(result.tool_name)
        
        # Need at least 2 successful tool calls for most scenarios
        return len(successful_tools) >= 2
    
    def _detect_reasoning_loop(self, trace: ReasoningTrace) -> bool:
        """Detect if reasoning is stuck in a loop."""
        if len(trace.steps) < 4:
            return False
        
        # Check last 4 thoughts for similarity
        recent_thoughts = [step.thought[:50].lower() for step in trace.steps[-4:]]
        unique_thoughts = set(recent_thoughts)
        
        # If all recent thoughts are very similar, we might be in a loop
        return len(unique_thoughts) <= 1
    
    def _has_traffic_information(self, results: List[ToolResult]) -> bool:
        """Check if results contain traffic information."""
        return any(result.tool_name == "check_traffic" and result.success for result in results)
    
    def _has_merchant_information(self, results: List[ToolResult]) -> bool:
        """Check if results contain merchant information."""
        merchant_tools = {"get_merchant_status", "get_nearby_merchants"}
        return any(result.tool_name in merchant_tools and result.success for result in results)
    
    def _has_address_information(self, results: List[ToolResult]) -> bool:
        """Check if results contain address information."""
        address_tools = {"validate_address", "get_address_details"}
        return any(result.tool_name in address_tools and result.success for result in results)
    
    def _prepare_plan_generation_prompt(self, trace: ReasoningTrace) -> List[Message]:
        """Prepare prompt for final plan generation."""
        # Summarize the reasoning process
        reasoning_summary = self._summarize_reasoning_trace(trace)
        
        # Gather all successful tool results
        tool_data = self._gather_tool_data(trace)
        
        system_message = """You are an expert at creating actionable resolution plans for delivery disruptions.

Based on the reasoning process and gathered information, create a comprehensive resolution plan.

Your plan should:
1. Have clear, sequential steps with specific actions
2. Assign responsibility for each step
3. Include realistic time estimates
4. Define success criteria for each step
5. Consider all stakeholders involved
6. Provide alternative approaches if the primary plan fails

Format your response as JSON with this structure:
{
  "summary": "Brief summary of the resolution strategy",
  "steps": [
    {
      "sequence": 1,
      "action": "Specific action to take",
      "responsible_party": "Who should perform this action",
      "estimated_time": "5 minutes",
      "success_criteria": "How to know this step succeeded",
      "dependencies": []
    }
  ],
  "estimated_duration": "30 minutes",
  "success_probability": 0.85,
  "alternatives": ["Alternative approach if primary plan fails"],
  "stakeholders": ["Customer", "Driver", "Restaurant", "Support team"]
}"""
        
        user_message = f"""**Scenario:** {trace.scenario.description}

**Reasoning Summary:**
{reasoning_summary}

**Gathered Information:**
{tool_data}

Please create a comprehensive resolution plan based on this analysis."""
        
        return [
            Message(role=MessageRole.SYSTEM, content=system_message),
            Message(role=MessageRole.USER, content=user_message)
        ]
    
    def _summarize_reasoning_trace(self, trace: ReasoningTrace) -> str:
        """Create a summary of the reasoning process."""
        summary_parts = []
        
        for i, step in enumerate(trace.steps, 1):
            summary_parts.append(f"{i}. {step.thought}")
            if step.observation:
                summary_parts.append(f"   Result: {step.observation}")
        
        return "\n".join(summary_parts)
    
    def _gather_tool_data(self, trace: ReasoningTrace) -> str:
        """Gather and format all tool data from the trace."""
        tool_data_parts = []
        
        for step in trace.steps:
            for result in step.tool_results:
                if result.success and result.data:
                    data_str = json.dumps(result.data, indent=2)
                    tool_data_parts.append(f"**{result.tool_name}:**\n{data_str}")
        
        return "\n\n".join(tool_data_parts) if tool_data_parts else "No tool data available"
    
    def _parse_plan_response(self, response) -> Dict[str, Any]:
        """Parse plan from LLM response."""
        content = response.content.strip()
        
        # Try to extract JSON from response
        try:
            # Handle code blocks
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()
            
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON: {e}")
            raise LLMProviderError(f"Invalid plan format: {e}")
    
    def _create_validated_plan(self, plan_data: Dict[str, Any], 
                             trace: ReasoningTrace) -> ValidatedResolutionPlan:
        """Create validated resolution plan from parsed data."""
        # Convert steps to ValidatedPlanStep objects
        validated_steps = []
        for step_data in plan_data.get("steps", []):
            estimated_time_str = step_data.get("estimated_time", "5 minutes")
            estimated_time = self._parse_time_duration(estimated_time_str)
            
            validated_step = ValidatedPlanStep(
                sequence=step_data.get("sequence", len(validated_steps) + 1),
                action=step_data.get("action", "Unknown action"),
                responsible_party=step_data.get("responsible_party", "System"),
                estimated_time=estimated_time,
                dependencies=step_data.get("dependencies", []),
                success_criteria=step_data.get("success_criteria", "Action completed")
            )
            validated_steps.append(validated_step)
        
        # Parse total duration
        duration_str = plan_data.get("estimated_duration", "30 minutes")
        total_duration = self._parse_time_duration(duration_str)
        
        return ValidatedResolutionPlan(
            steps=validated_steps,
            estimated_duration=total_duration,
            success_probability=plan_data.get("success_probability", 0.8),
            alternatives=plan_data.get("alternatives", []),
            stakeholders=plan_data.get("stakeholders", []),
            created_at=datetime.now()
        )
    
    def _parse_time_duration(self, duration_str: str) -> timedelta:
        """Parse time duration string into timedelta."""
        duration_str = duration_str.lower().strip()
        
        # Extract number and unit
        import re
        match = re.search(r'(\d+)\s*(minute|min|hour|hr|second|sec)', duration_str)
        
        if match:
            value = int(match.group(1))
            unit = match.group(2)
            
            if unit in ['minute', 'min']:
                return timedelta(minutes=value)
            elif unit in ['hour', 'hr']:
                return timedelta(hours=value)
            elif unit in ['second', 'sec']:
                return timedelta(seconds=value)
        
        # Default fallback
        return timedelta(minutes=15)
    
    def _create_fallback_plan(self, trace: ReasoningTrace) -> ValidatedResolutionPlan:
        """Create a basic fallback plan when plan generation fails."""
        fallback_steps = [
            ValidatedPlanStep(
                sequence=1,
                action="Notify customer about the disruption and provide status update",
                responsible_party="Customer service",
                estimated_time=timedelta(minutes=2),
                dependencies=[],
                success_criteria="Customer acknowledges notification"
            ),
            ValidatedPlanStep(
                sequence=2,
                action="Investigate the root cause of the disruption",
                responsible_party="Operations team",
                estimated_time=timedelta(minutes=10),
                dependencies=[1],
                success_criteria="Root cause identified"
            ),
            ValidatedPlanStep(
                sequence=3,
                action="Implement corrective measures based on investigation",
                responsible_party="Operations team",
                estimated_time=timedelta(minutes=15),
                dependencies=[2],
                success_criteria="Corrective measures applied successfully"
            )
        ]
        
        return ValidatedResolutionPlan(
            steps=fallback_steps,
            estimated_duration=timedelta(minutes=30),
            success_probability=0.6,
            alternatives=["Escalate to senior operations team if initial measures fail"],
            stakeholders=["Customer", "Customer service", "Operations team"],
            created_at=datetime.now()
        )