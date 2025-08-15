#!/usr/bin/env python3
"""
Simple demo showing complete model response to a scenario.
"""

from src.scenarios import ScenarioGenerator, ScenarioCategory, UrgencyLevel
from src.scenarios.interactive_input import InteractiveScenarioTester


def show_complete_model_response():
    """Show a complete model response to demonstrate the system."""
    print("🤖 COMPLETE MODEL RESPONSE DEMONSTRATION")
    print("=" * 60)
    
    # Initialize tester
    tester = InteractiveScenarioTester()
    
    if not tester.agent:
        print("❌ Agent not available")
        return
    
    # Generate a specific scenario
    generator = ScenarioGenerator()
    
    try:
        scenario_data = generator.generate_scenario(
            category=ScenarioCategory.CUSTOMER,
            urgency=UrgencyLevel.HIGH
        )
    except:
        # Fallback to a manual scenario
        scenario_data = {
            "scenario_text": "Customer Sarah at 456 Oak Street received delivery DEL789 with cold pizza from Tony's Pizza. She's demanding a full refund and threatening to leave bad reviews. Driver Mike says the food was hot when delivered.",
            "category": "customer",
            "urgency": "high",
            "complexity_score": 6,
            "expected_tools": ["collect_evidence", "notify_customer", "issue_instant_refund"],
            "expected_actions": ["investigate_complaint", "provide_resolution", "prevent_escalation"]
        }
    
    print("📝 INPUT SCENARIO:")
    print(f"   {scenario_data['scenario_text']}")
    print(f"\n📊 SCENARIO METADATA:")
    print(f"   Category: {scenario_data['category'].title()}")
    print(f"   Urgency: {scenario_data['urgency'].title()}")
    print(f"   Complexity: {scenario_data['complexity_score']}/10")
    
    print(f"\n🔧 EXPECTED MODEL BEHAVIOR:")
    print(f"   Tools: {', '.join(scenario_data['expected_tools'])}")
    print(f"   Actions: {', '.join(scenario_data['expected_actions'])}")
    
    print(f"\n🤖 ACTUAL MODEL RESPONSE:")
    print("-" * 50)
    
    # Process with the agent
    import time
    start_time = time.time()
    
    try:
        result = tester.agent.process_scenario(scenario_data['scenario_text'])
        processing_time = time.time() - start_time
        
        print(f"✅ Processing completed in {processing_time:.2f}s")
        
        print(f"\n📊 MODEL CLASSIFICATION:")
        print(f"   Detected Type: {result.scenario.scenario_type.value.title()}")
        print(f"   Detected Urgency: {result.scenario.urgency_level.value.title()}")
        print(f"   Entities Extracted: {len(result.scenario.entities)}")
        
        print(f"\n🔍 ENTITIES FOUND:")
        for entity in result.scenario.entities:
            print(f"   • {entity.entity_type.value}: '{entity.text}' (confidence: {entity.confidence:.2f})")
        
        print(f"\n🧠 MODEL REASONING PROCESS:")
        for i, step in enumerate(result.reasoning_trace.steps, 1):
            print(f"   Step {i}: {step.thought}")
            if step.action:
                print(f"           🔧 Tool Used: {step.action.tool_name}")
                print(f"           📋 Parameters: {step.action.parameters}")
                
                if step.tool_results:
                    for tool_result in step.tool_results:
                        status = "✅ Success" if tool_result.success else "❌ Failed"
                        print(f"           {status}: {tool_result.execution_time:.2f}s")
                        if tool_result.success and tool_result.data:
                            # Show key data points
                            key_data = []
                            for key, value in list(tool_result.data.items())[:3]:
                                key_data.append(f"{key}: {value}")
                            print(f"           📊 Data: {', '.join(key_data)}")
            
            if step.observation:
                print(f"           👁️  Observation: {step.observation}")
            print()
        
        print(f"🎯 MODEL RESOLUTION PLAN:")
        print(f"   Success Probability: {result.resolution_plan.success_probability:.1%}")
        print(f"   Estimated Duration: {result.resolution_plan.estimated_duration}")
        print(f"   Stakeholders: {', '.join(result.resolution_plan.stakeholders)}")
        
        print(f"\n📋 RECOMMENDED ACTIONS:")
        for i, step in enumerate(result.resolution_plan.steps, 1):
            print(f"   {i}. {step.action}")
            print(f"      👤 Responsible: {step.responsible_party}")
            print(f"      ⏱️  Time: {step.estimated_time}")
            print(f"      ✅ Success Criteria: {step.success_criteria}")
        
        if result.resolution_plan.alternatives:
            print(f"\n🔄 ALTERNATIVE OPTIONS:")
            for alt in result.resolution_plan.alternatives:
                print(f"   • {alt}")
        
        # Validation against expectations
        print(f"\n📊 VALIDATION RESULTS:")
        expected_tools = set(scenario_data['expected_tools'])
        actual_tools = set()
        for step in result.reasoning_trace.steps:
            if step.action:
                actual_tools.add(step.action.tool_name)
        
        matched_tools = expected_tools.intersection(actual_tools)
        match_rate = len(matched_tools) / len(expected_tools) if expected_tools else 0
        
        print(f"   Tool Selection Accuracy: {match_rate:.1%}")
        print(f"   Expected Tools: {', '.join(expected_tools)}")
        print(f"   Actual Tools: {', '.join(actual_tools)}")
        
        if match_rate >= 0.7:
            print("   ✅ Excellent tool selection!")
        elif match_rate >= 0.5:
            print("   ✅ Good tool selection!")
        else:
            print("   ⚠️  Tool selection could be improved")
        
        print(f"\n🎉 OVERALL ASSESSMENT:")
        if result.success and match_rate >= 0.5 and result.resolution_plan.success_probability >= 0.6:
            print("   ✅ EXCELLENT: Model performed very well!")
        elif result.success and match_rate >= 0.3:
            print("   ✅ GOOD: Model performed adequately!")
        else:
            print("   ⚠️  NEEDS IMPROVEMENT: Model could perform better")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    show_complete_model_response()
    
    print(f"\n{'='*60}")
    print("🚀 TRY IT YOURSELF:")
    print(f"{'='*60}")
    print("1. Run: python simple_demo.py")
    print("2. Modify the scenario in the script")
    print("3. See how the model responds differently")
    print("4. Test with your own scenarios!")
    print("\n🎯 The model shows complete reasoning from input to resolution!")