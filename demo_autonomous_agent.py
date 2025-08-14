#!/usr/bin/env python3
"""
Demonstration of the Autonomous Agent Core functionality.
"""
import json
from datetime import datetime
from unittest.mock import Mock

from src.agent.autonomous_agent import AutonomousAgent, AgentConfig
from src.llm.interfaces import LLMResponse, Message, MessageRole, TokenUsage
from src.tools.interfaces import ToolResult


def create_mock_llm_provider():
    """Create a mock LLM provider for demonstration."""
    provider = Mock()
    provider.generate_response.return_value = LLMResponse(
        content="Analysis complete. I have gathered sufficient information to create a resolution plan.",
        messages=[],
        token_usage=TokenUsage(100, 50, 150),
        model="gpt-4",
        finish_reason="stop",
        response_time=1.2,
        timestamp=datetime.now()
    )
    return provider


def create_mock_tool_manager():
    """Create a mock tool manager with realistic tools and responses."""
    manager = Mock()
    
    # Mock available tools
    tools = []
    tool_configs = [
        ("check_traffic", "Check traffic conditions for delivery route"),
        ("get_merchant_status", "Get current merchant availability and prep time"),
        ("notify_customer", "Send notification to customer about delivery status"),
        ("validate_address", "Validate and correct delivery address"),
        ("re_route_driver", "Calculate alternative route for driver"),
        ("get_nearby_merchants", "Find alternative merchants near delivery location")
    ]
    
    for name, desc in tool_configs:
        tool = Mock()
        tool.name = name
        tool.description = desc
        tool.parameters = {
            "location": {"type": "string", "description": "Location or address"},
            "message": {"type": "string", "description": "Message content"}
        }
        tools.append(tool)
    
    manager.get_available_tools.return_value = tools
    
    # Mock realistic tool execution results
    def mock_execute_tool(tool_name, parameters, **kwargs):
        if tool_name == "check_traffic":
            return ToolResult(
                tool_name="check_traffic",
                success=True,
                data={
                    "status": "heavy_traffic",
                    "delay_minutes": 15,
                    "alternative_routes": ["Route via Oak Street", "Route via Pine Avenue"],
                    "current_location": parameters.get("location", "unknown")
                },
                execution_time=0.8
            )
        elif tool_name == "get_merchant_status":
            return ToolResult(
                tool_name="get_merchant_status",
                success=True,
                data={
                    "available": True,
                    "prep_time_minutes": 12,
                    "capacity_status": "busy",
                    "estimated_ready_time": "15:45"
                },
                execution_time=0.5
            )
        elif tool_name == "notify_customer":
            return ToolResult(
                tool_name="notify_customer",
                success=True,
                data={
                    "message_sent": True,
                    "customer_id": "CUST789",
                    "notification_method": "SMS",
                    "delivery_id": parameters.get("delivery_id", "unknown")
                },
                execution_time=0.3
            )
        elif tool_name == "validate_address":
            return ToolResult(
                tool_name="validate_address",
                success=True,
                data={
                    "valid": True,
                    "confidence": 0.95,
                    "corrected_address": parameters.get("address", "123 Main St"),
                    "geocoded": True
                },
                execution_time=0.6
            )
        else:
            return ToolResult(
                tool_name=tool_name,
                success=True,
                data={"result": "success", "action_taken": f"Executed {tool_name}"},
                execution_time=0.4
            )
    
    manager.execute_tool.side_effect = mock_execute_tool
    return manager


def demonstrate_autonomous_agent():
    """Demonstrate the autonomous agent capabilities."""
    print("🤖 Autonomous Delivery Coordinator Agent Demo")
    print("=" * 50)
    
    # Create agent with mocked dependencies
    llm_provider = create_mock_llm_provider()
    tool_manager = create_mock_tool_manager()
    
    config = AgentConfig(
        max_reasoning_steps=5,
        reasoning_timeout=60,
        enable_context_tracking=True,
        enable_state_management=True,
        log_reasoning_steps=True
    )
    
    agent = AutonomousAgent(llm_provider, tool_manager, config)
    
    # Test scenarios
    scenarios = [
        {
            "title": "Traffic Disruption Scenario",
            "description": "Driver is stuck in heavy traffic on Highway 101 for delivery DEL123456 to Pizza Palace at 456 Oak Street"
        },
        {
            "title": "Merchant Unavailability Scenario", 
            "description": "Burger King on Main Street is closed and cannot prepare delivery DEL789012 for customer John Smith"
        },
        {
            "title": "Address Validation Scenario",
            "description": "Wrong address provided for delivery DEL555555 - customer says they live at 789 Pine Avenue not 789 Pine Street"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['title']}")
        print("-" * 40)
        print(f"Description: {scenario['description']}")
        
        # Process scenario
        result = agent.process_scenario(scenario['description'])
        
        # Display results
        print(f"\n✅ Processing Result: {'SUCCESS' if result.success else 'FAILED'}")
        
        if result.success:
            # Scenario analysis
            print(f"📊 Scenario Type: {result.scenario.scenario_type.value}")
            print(f"🚨 Urgency Level: {result.scenario.urgency_level.value}")
            print(f"🏷️  Entities Found: {len(result.scenario.entities)}")
            
            # Entity details
            if result.scenario.entities:
                print("   Entities:")
                for entity in result.scenario.entities:
                    print(f"   - {entity.entity_type.value}: {entity.text} (confidence: {entity.confidence:.1%})")
            
            # Reasoning trace
            print(f"🧠 Reasoning Steps: {len(result.reasoning_trace.steps)}")
            for step in result.reasoning_trace.steps:
                print(f"   Step {step.step_number}: {step.thought[:80]}...")
                if step.action:
                    print(f"   → Action: {step.action.tool_name}")
                if step.tool_results:
                    successful = len([r for r in step.tool_results if r.success])
                    print(f"   → Tools executed: {successful}/{len(step.tool_results)} successful")
            
            # Resolution plan
            print(f"📋 Resolution Plan: {len(result.resolution_plan.steps)} steps")
            print(f"⏱️  Estimated Duration: {result.resolution_plan.estimated_duration}")
            print(f"🎯 Success Probability: {result.resolution_plan.success_probability:.1%}")
            print(f"👥 Stakeholders: {', '.join(result.resolution_plan.stakeholders)}")
            
            # Scenario analysis details
            scenario_analysis = agent.get_scenario_analysis()
            if scenario_analysis:
                print(f"🔍 Scenario Complexity: {scenario_analysis.scenario_complexity}")
                print(f"📈 Recommended Tools: {len(scenario_analysis.recommended_tools)}")
                print(f"⏰ Est. Resolution Time: {scenario_analysis.estimated_resolution_time} minutes")
        
        else:
            print(f"❌ Error: {result.error_message}")
        
        print()
    
    # Display performance metrics
    print("📊 Agent Performance Metrics")
    print("-" * 30)
    metrics = agent.get_performance_metrics()
    
    if not metrics.get("no_data"):
        print(f"Total Scenarios Processed: {metrics['total_scenarios_processed']}")
        print(f"Average Processing Time: {metrics['average_processing_time_seconds']:.2f}s")
        print(f"Average Reasoning Steps: {metrics['average_reasoning_steps']:.1f}")
        print(f"Average Success Probability: {metrics['average_success_probability']:.1%}")
        
        print("\nScenario Type Distribution:")
        for scenario_type, count in metrics['scenario_type_distribution'].items():
            print(f"  {scenario_type}: {count}")
        
        print("\nTool Usage Distribution:")
        for tool, count in metrics.get('tool_usage_distribution', {}).items():
            print(f"  {tool}: {count}")
    
    # Display current agent state
    print("\n🔧 Agent State")
    print("-" * 15)
    state = agent.get_current_state()
    print(f"Status: {state['status']}")
    print(f"Context Data Keys: {state['context_data_keys']}")
    
    print("\n✨ Demo Complete!")
    print("The Autonomous Agent successfully demonstrated:")
    print("• Intelligent scenario parsing and entity extraction")
    print("• Scenario-based tool selection and prioritization") 
    print("• Reasoning loop with tool integration")
    print("• Resolution plan generation")
    print("• Performance tracking and metrics")
    print("• State management and context tracking")


if __name__ == "__main__":
    demonstrate_autonomous_agent()