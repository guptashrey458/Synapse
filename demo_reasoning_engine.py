#!/usr/bin/env python3
"""
Demonstration script for the reasoning engine implementation.
"""
import json
from datetime import datetime
from unittest.mock import Mock

from src.reasoning.engine import ReActReasoningEngine, ReasoningConfig
from src.reasoning.logger import ConsoleChainOfThoughtLogger, LoggingConfig
from src.reasoning.interfaces import ReasoningContext
from src.agent.interfaces import ScenarioType, UrgencyLevel, EntityType
from src.agent.models import (
    ValidatedEntity, ValidatedDisruptionScenario, ValidatedReasoningTrace
)
from src.llm.interfaces import LLMResponse, Message, MessageRole, TokenUsage
from src.tools.interfaces import Tool, ToolResult


class DemoTool(Tool):
    """Demo tool for demonstration."""
    
    def __init__(self, name: str, demo_data: dict):
        self.name = name
        self.description = f"Demo tool for {name}"
        self.parameters = {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "Location parameter"}
            },
            "required": ["location"]
        }
        self.demo_data = demo_data
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute demo tool."""
        return ToolResult(
            tool_name=self.name,
            success=True,
            data=self.demo_data,
            execution_time=0.3
        )
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate parameters."""
        return True


def create_demo_llm_provider():
    """Create a demo LLM provider with realistic responses."""
    provider = Mock()
    
    # Define different responses for different steps
    responses = [
        {
            "content": "**Thought:** I need to check the current traffic situation to understand the severity of the delay and find alternative routes.",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "check_traffic",
                    "arguments": '{"location": "Highway 101"}'
                }
            }]
        },
        {
            "content": "**Thought:** The traffic is severely delayed due to an accident. I should notify the customer about the delay and provide an updated ETA.",
            "tool_calls": [{
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "notify_customer",
                    "arguments": '{"message": "Delivery delayed due to traffic accident", "new_eta": "45 minutes"}'
                }
            }]
        },
        {
            "content": "**Thought:** I have sufficient information to create a resolution plan. The customer has been notified and we have traffic data.",
            "tool_calls": None
        }
    ]
    
    call_count = 0
    
    def mock_generate_response(*args, **kwargs):
        nonlocal call_count
        response_data = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        
        return LLMResponse(
            content=response_data["content"],
            messages=[],
            token_usage=TokenUsage(prompt_tokens=150, completion_tokens=75, total_tokens=225),
            model="gpt-4",
            finish_reason="stop",
            response_time=1.2,
            timestamp=datetime.now(),
            tool_calls=response_data["tool_calls"]
        )
    
    provider.generate_response.side_effect = mock_generate_response
    return provider


def create_demo_tool_manager():
    """Create a demo tool manager with realistic tools."""
    manager = Mock()
    
    tools = [
        DemoTool("check_traffic", {
            "status": "heavy",
            "cause": "multi-vehicle accident",
            "estimated_delay": "30-45 minutes",
            "alternative_routes": ["Oak Street", "Pine Avenue"],
            "current_location": "Highway 101, Mile 15"
        }),
        DemoTool("notify_customer", {
            "message_sent": True,
            "customer_response": "Thanks for the update!",
            "notification_method": "SMS",
            "timestamp": datetime.now().isoformat()
        }),
        DemoTool("get_merchant_status", {
            "restaurant_name": "Tony's Pizza",
            "prep_time": "15 minutes",
            "status": "preparing_order",
            "estimated_ready": "7:30 PM"
        })
    ]
    
    manager.get_available_tools.return_value = tools
    
    def mock_execute_tool(tool_name, parameters, **kwargs):
        for tool in tools:
            if tool.name == tool_name:
                return tool.execute(**parameters)
        return ToolResult(
            tool_name=tool_name,
            success=False,
            data={},
            execution_time=0.1,
            error_message=f"Tool {tool_name} not found"
        )
    
    manager.execute_tool.side_effect = mock_execute_tool
    return manager


def create_demo_scenario():
    """Create a realistic demo scenario."""
    entities = [
        ValidatedEntity(
            text="Highway 101",
            entity_type=EntityType.ADDRESS,
            confidence=0.9,
            normalized_value="Highway 101"
        ),
        ValidatedEntity(
            text="John",
            entity_type=EntityType.PERSON,
            confidence=0.8,
            normalized_value="John"
        ),
        ValidatedEntity(
            text="Tony's Pizza",
            entity_type=EntityType.MERCHANT,
            confidence=0.9,
            normalized_value="Tony's Pizza"
        )
    ]
    
    return ValidatedDisruptionScenario(
        description="Driver John is stuck in heavy traffic on Highway 101 due to a multi-vehicle accident. "
                   "He has a pizza delivery from Tony's Pizza that was supposed to arrive at 7:00 PM, "
                   "but it's now 7:15 PM and he's still 20 minutes away from the customer.",
        entities=entities,
        scenario_type=ScenarioType.TRAFFIC,
        urgency_level=UrgencyLevel.HIGH
    )


def demonstrate_reasoning_engine():
    """Demonstrate the reasoning engine in action."""
    print("🚀 Autonomous Delivery Coordinator - Reasoning Engine Demo")
    print("=" * 60)
    
    # Setup components
    llm_provider = create_demo_llm_provider()
    tool_manager = create_demo_tool_manager()
    scenario = create_demo_scenario()
    
    # Configure reasoning engine
    reasoning_config = ReasoningConfig(
        max_reasoning_steps=3,
        enable_examples=False,
        temperature=0.1
    )
    
    # Configure logger for demo
    logging_config = LoggingConfig(
        enable_console_output=True,
        enable_file_logging=False,
        include_timestamps=True,
        include_tool_details=True
    )
    
    # Create reasoning engine and logger
    reasoning_engine = ReActReasoningEngine(llm_provider, tool_manager, reasoning_config)
    logger = ConsoleChainOfThoughtLogger(logging_config)
    
    print(f"📋 Scenario: {scenario.description}")
    print(f"🏷️  Type: {scenario.scenario_type.value.title()}")
    print(f"⚡ Urgency: {scenario.urgency_level.value.title()}")
    print()
    
    # Create reasoning trace
    trace = ValidatedReasoningTrace(
        steps=[],
        scenario=scenario,
        start_time=datetime.now()
    )
    
    # Reasoning loop
    step_count = 0
    print("🧠 Starting Reasoning Process...")
    print()
    
    while reasoning_engine.should_continue_reasoning(trace) and step_count < reasoning_config.max_reasoning_steps:
        step_count += 1
        
        # Create context
        context = ReasoningContext(
            scenario=scenario,
            current_step=step_count,
            previous_steps=trace.steps,
            tool_results=[],
            available_tools=[tool.name for tool in tool_manager.get_available_tools()]
        )
        
        # Generate reasoning step
        step = reasoning_engine.generate_reasoning_step(context)
        
        # Log the step
        logger.log_step(step)
        
        # Add to trace
        trace.add_step(step)
        
        # Small delay for demo effect
        import time
        time.sleep(0.5)
    
    # Complete the trace
    trace.complete_trace()
    
    print("\n" + "=" * 60)
    print("📊 Reasoning Summary")
    print("=" * 60)
    
    summary = logger.get_trace_summary(trace)
    print(summary)
    
    print("\n" + "=" * 60)
    print("📋 Generating Resolution Plan...")
    print("=" * 60)
    
    # Mock plan generation response
    plan_json = {
        "summary": "Reroute driver via alternative route and keep customer informed of delays",
        "steps": [
            {
                "sequence": 1,
                "action": "Reroute driver John via Oak Street to avoid Highway 101 accident",
                "responsible_party": "Dispatch system",
                "estimated_time": "3 minutes",
                "success_criteria": "Driver receives new route and confirms acceptance"
            },
            {
                "sequence": 2,
                "action": "Send updated ETA to customer with explanation of delay",
                "responsible_party": "Customer service system",
                "estimated_time": "1 minute",
                "success_criteria": "Customer receives and acknowledges notification"
            },
            {
                "sequence": 3,
                "action": "Monitor driver progress and provide updates if needed",
                "responsible_party": "Operations team",
                "estimated_time": "25 minutes",
                "success_criteria": "Delivery completed successfully"
            }
        ],
        "estimated_duration": "30 minutes",
        "success_probability": 0.85,
        "alternatives": [
            "If Oak Street also has traffic, consider Pine Avenue route",
            "If delay exceeds 1 hour, offer customer refund and reorder from closer restaurant"
        ],
        "stakeholders": ["Driver John", "Customer", "Tony's Pizza", "Dispatch team", "Customer service"]
    }
    
    # Mock the LLM response for plan generation
    llm_provider.generate_response.return_value = LLMResponse(
        content=f"```json\n{json.dumps(plan_json, indent=2)}\n```",
        messages=[],
        token_usage=TokenUsage(prompt_tokens=300, completion_tokens=200, total_tokens=500),
        model="gpt-4",
        finish_reason="stop",
        response_time=2.0,
        timestamp=datetime.now()
    )
    
    # Generate final plan
    try:
        plan = reasoning_engine.generate_final_plan(trace)
        
        print(f"✅ Resolution Plan Generated Successfully!")
        print(f"📈 Success Probability: {plan.success_probability:.0%}")
        print(f"⏱️  Estimated Duration: {plan.estimated_duration}")
        print(f"👥 Stakeholders: {', '.join(plan.stakeholders)}")
        print()
        
        print("📝 Action Steps:")
        for step in plan.steps:
            print(f"  {step.sequence}. {step.action}")
            print(f"     👤 Responsible: {step.responsible_party}")
            print(f"     ⏰ Time: {step.estimated_time}")
            print(f"     ✓ Success Criteria: {step.success_criteria}")
            print()
        
        if plan.alternatives:
            print("🔄 Alternative Options:")
            for alt in plan.alternatives:
                print(f"  • {alt}")
        
    except Exception as e:
        print(f"❌ Plan generation failed: {e}")
        print("Using fallback plan...")
        
        fallback_plan = reasoning_engine._create_fallback_plan(trace)
        print(f"📋 Fallback plan created with {len(fallback_plan.steps)} steps")
    
    print("\n" + "=" * 60)
    print("🎉 Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_reasoning_engine()