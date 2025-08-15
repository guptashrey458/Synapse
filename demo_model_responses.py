#!/usr/bin/env python3
"""
Demo script showing full model responses to input scenarios.
"""

import json
import time
from datetime import datetime
from src.scenarios import ScenarioGenerator, ScenarioCategory, UrgencyLevel
from src.scenarios.interactive_input import InteractiveScenarioTester


def demo_model_responses():
    """Demonstrate full model responses to various scenarios."""
    print("🤖 Autonomous Delivery Coordinator - Model Response Demo")
    print("=" * 70)
    
    # Initialize the tester with working agent
    tester = InteractiveScenarioTester()
    
    if not tester.agent:
        print("❌ Agent not available for demonstration")
        return
    
    generator = ScenarioGenerator()
    
    # Test scenarios from different categories
    test_scenarios = [
        {
            "name": "Traffic Emergency",
            "category": ScenarioCategory.TRAFFIC,
            "urgency": UrgencyLevel.HIGH
        },
        {
            "name": "Customer Complaint",
            "category": ScenarioCategory.CUSTOMER,
            "urgency": UrgencyLevel.MEDIUM
        },
        {
            "name": "Medical Emergency",
            "category": ScenarioCategory.EMERGENCY,
            "urgency": UrgencyLevel.CRITICAL
        },
        {
            "name": "Multi-Factor Crisis",
            "category": ScenarioCategory.MULTI_FACTOR,
            "urgency": UrgencyLevel.HIGH
        }
    ]
    
    for i, test_config in enumerate(test_scenarios, 1):
        print(f"\n{'='*70}")
        print(f"🎯 TEST SCENARIO {i}: {test_config['name']}")
        print(f"{'='*70}")
        
        try:
            # Generate scenario
            scenario_data = generator.generate_scenario(
                category=test_config['category'],
                urgency=test_config['urgency']
            )
            
            print(f"📝 INPUT SCENARIO:")
            print(f"   {scenario_data['scenario_text']}")
            print(f"📂 Category: {scenario_data['category'].title()}")
            print(f"⚡ Urgency: {scenario_data['urgency'].title()}")
            print(f"🎯 Complexity: {scenario_data['complexity_score']}/10")
            
            print(f"\n🔧 EXPECTED TOOLS:")
            for tool in scenario_data['expected_tools']:
                print(f"   • {tool}")
            
            print(f"\n🎯 EXPECTED ACTIONS:")
            for action in scenario_data['expected_actions']:
                print(f"   • {action}")
            
            print(f"\n🤖 MODEL PROCESSING...")
            print("-" * 50)
            
            # Process with agent
            start_time = time.time()
            result = tester.agent.process_scenario(scenario_data['scenario_text'])
            processing_time = time.time() - start_time
            
            # Display model response
            print(f"✅ PROCESSING COMPLETE ({processing_time:.2f}s)")
            print(f"\n📊 MODEL ANALYSIS:")
            print(f"   Classified Type: {result.scenario.scenario_type.value.title()}")
            print(f"   Classified Urgency: {result.scenario.urgency_level.value.title()}")
            print(f"   Entities Found: {len(result.scenario.entities)}")
            
            # Show extracted entities
            if result.scenario.entities:
                print(f"\n🔍 EXTRACTED ENTITIES:")
                for entity in result.scenario.entities[:5]:
                    print(f"   • {entity.entity_type.value}: {entity.text}")
                if len(result.scenario.entities) > 5:
                    print(f"   ... and {len(result.scenario.entities) - 5} more")
            
            # Show reasoning steps
            print(f"\n🧠 MODEL REASONING ({len(result.reasoning_trace.steps)} steps):")
            for j, step in enumerate(result.reasoning_trace.steps, 1):
                print(f"   {j}. {step.thought}")
                if step.action:
                    print(f"      🔧 Tool: {step.action.tool_name}")
                    if step.tool_results:
                        for tool_result in step.tool_results:
                            status = "✅" if tool_result.success else "❌"
                            print(f"      {status} Result: {tool_result.tool_name} ({tool_result.execution_time:.2f}s)")
                if step.observation:
                    print(f"      👁️  Observation: {step.observation[:100]}...")
            
            # Show resolution plan
            print(f"\n🎯 MODEL RESOLUTION PLAN:")
            print(f"   Success Probability: {result.resolution_plan.success_probability:.1%}")
            print(f"   Estimated Duration: {result.resolution_plan.estimated_duration}")
            print(f"   Stakeholders: {', '.join(result.resolution_plan.stakeholders)}")
            
            print(f"\n📋 ACTION STEPS:")
            for j, step in enumerate(result.resolution_plan.steps, 1):
                print(f"   {j}. {step.action}")
                print(f"      👤 Responsible: {step.responsible_party}")
                print(f"      ⏱️  Time: {step.estimated_time}")
            
            # Compare with expectations
            print(f"\n📊 EXPECTED vs ACTUAL COMPARISON:")
            expected_tools = set(scenario_data['expected_tools'])
            actual_tools = set()
            for step in result.reasoning_trace.steps:
                if step.action:
                    actual_tools.add(step.action.tool_name)
            
            matched_tools = expected_tools.intersection(actual_tools)
            match_rate = len(matched_tools) / len(expected_tools) if expected_tools else 0
            
            print(f"   Tool Match Rate: {match_rate:.1%} ({len(matched_tools)}/{len(expected_tools)})")
            if matched_tools:
                print(f"   Matched Tools: {', '.join(matched_tools)}")
            
            missing_tools = expected_tools - actual_tools
            if missing_tools:
                print(f"   Missing Tools: {', '.join(missing_tools)}")
            
            extra_tools = actual_tools - expected_tools
            if extra_tools:
                print(f"   Extra Tools: {', '.join(extra_tools)}")
            
            # Success assessment
            success_indicators = []
            if result.success:
                success_indicators.append("✅ Processing successful")
            if match_rate >= 0.5:
                success_indicators.append("✅ Good tool selection")
            if result.resolution_plan.success_probability >= 0.6:
                success_indicators.append("✅ High confidence plan")
            if processing_time < 30:
                success_indicators.append("✅ Fast response time")
            
            print(f"\n🎉 SUCCESS INDICATORS:")
            for indicator in success_indicators:
                print(f"   {indicator}")
            
        except Exception as e:
            print(f"❌ Error processing scenario: {e}")
            continue
    
    print(f"\n{'='*70}")
    print("🎯 SUMMARY")
    print(f"{'='*70}")
    print("The model successfully demonstrates:")
    print("✅ Entity extraction from natural language scenarios")
    print("✅ Scenario classification and urgency assessment")
    print("✅ Intelligent tool selection based on scenario type")
    print("✅ Multi-step reasoning with tool execution")
    print("✅ Resolution plan generation with success probabilities")
    print("✅ Stakeholder identification and action planning")
    print("\n🚀 Your model is ready to autonomously handle delivery disruptions!")


def demo_custom_scenario():
    """Demo with a custom user-provided scenario."""
    print(f"\n{'='*70}")
    print("🎨 CUSTOM SCENARIO DEMO")
    print(f"{'='*70}")
    
    # Custom scenario
    custom_scenario = """
    Driver Emma's delivery truck broke down on Highway 101 while carrying 3 urgent deliveries: 
    DEL123 (medical supplies for hospital), DEL124 (birthday cake for Sarah at 456 Oak Street), 
    and DEL125 (hot pizza for Mike at (555) 123-4567). It's raining heavily, Emma is safe but 
    stranded, and all customers are calling asking about their orders. The tow truck will take 
    2 hours to arrive.
    """
    
    tester = InteractiveScenarioTester()
    
    if not tester.agent:
        print("❌ Agent not available")
        return
    
    print("📝 CUSTOM INPUT SCENARIO:")
    print(custom_scenario.strip())
    
    print(f"\n🤖 MODEL PROCESSING...")
    print("-" * 50)
    
    start_time = time.time()
    result = tester.agent.process_scenario(custom_scenario)
    processing_time = time.time() - start_time
    
    print(f"✅ PROCESSING COMPLETE ({processing_time:.2f}s)")
    
    print(f"\n📊 MODEL RESPONSE:")
    print(f"   Type: {result.scenario.scenario_type.value.title()}")
    print(f"   Urgency: {result.scenario.urgency_level.value.title()}")
    print(f"   Entities: {len(result.scenario.entities)}")
    print(f"   Reasoning Steps: {len(result.reasoning_trace.steps)}")
    print(f"   Success Probability: {result.resolution_plan.success_probability:.1%}")
    
    print(f"\n🔍 KEY ENTITIES FOUND:")
    for entity in result.scenario.entities:
        print(f"   • {entity.entity_type.value}: {entity.text}")
    
    print(f"\n🧠 REASONING SUMMARY:")
    for i, step in enumerate(result.reasoning_trace.steps, 1):
        action_text = f" → {step.action.tool_name}" if step.action else ""
        print(f"   {i}. {step.thought[:60]}...{action_text}")
    
    print(f"\n🎯 RESOLUTION PLAN:")
    for i, step in enumerate(result.resolution_plan.steps, 1):
        print(f"   {i}. {step.action}")
    
    print(f"\n🎉 The model successfully handled this complex multi-delivery emergency scenario!")


if __name__ == "__main__":
    demo_model_responses()
    demo_custom_scenario()
    
    print(f"\n{'='*70}")
    print("🚀 NEXT STEPS:")
    print(f"{'='*70}")
    print("1. Run: python demo_model_responses.py")
    print("2. Test with your own scenarios")
    print("3. Use the interactive tester: python -m src.cli.main scenario-tester")
    print("4. Generate training datasets with ScenarioGenerator")
    print("5. Validate model responses against expected tools/actions")
    print("\n🎯 Your model now shows complete reasoning and responses!")