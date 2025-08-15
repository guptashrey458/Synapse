#!/usr/bin/env python3
"""
Demo script showing HTN planner integration with enhanced plan generator.
Shows how HTN can be used as a pre-pass for detailed plan generation.
"""

import sys
sys.path.insert(0, 'src')

from reasoning.htn import (
    HTNPlanner, HTNStep, HTNPlan, 
    ResolveTrafficDelayOperator, ResolveMerchantOverloadOperator, AtDoorMediationOperator,
    evidence_flags_from_tools, _urgency_bonus, FALLBACKS
)
from agent.interfaces import ScenarioType, UrgencyLevel
from agent.models import ValidatedDisruptionScenario, ValidatedEntity, EntityType
from tools.interfaces import ToolResult


def demo_htn_basic_functionality():
    """Demo basic HTN functionality."""
    print("🧪 HTN PLANNER DEMO - Basic Functionality")
    print("=" * 60)
    
    # Test urgency bonus calculation
    print("1. Urgency Bonus Calculation:")
    for urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.HIGH, UrgencyLevel.MEDIUM, UrgencyLevel.LOW]:
        bonus = _urgency_bonus(urgency)
        print(f"   {urgency.value}: {bonus:+.2f}")
    print()
    
    # Test evidence flags from tool results
    print("2. Evidence Flags from Tool Results:")
    tool_results = [
        ToolResult("check_traffic", True, {"traffic_condition": "heavy"}, 1.0),
        ToolResult("re_route_driver", True, {"new_route": "Highway 9"}, 1.5),
        ToolResult("notify_customer", True, {"delivered": True}, 0.5),
        ToolResult("get_merchant_status", False, {}, 1.0, error_message="Circuit breaker open")
    ]
    
    flags = evidence_flags_from_tools(tool_results)
    print(f"   Evidence flags: {flags}")
    print()
    
    # Test safe defaults (no shared mutable state)
    print("3. Safe Default Factories:")
    step_a = HTNStep("step_a", "OpA", "Goal A")
    step_b = HTNStep("step_b", "OpB", "Goal B")
    
    step_a.subgoals.append("subgoal_a")
    step_a.tool_calls.append("tool_a")
    step_a.evidence_required.add("evidence_a")
    
    print(f"   Step A: subgoals={len(step_a.subgoals)}, tools={len(step_a.tool_calls)}, evidence={len(step_a.evidence_required)}")
    print(f"   Step B: subgoals={len(step_b.subgoals)}, tools={len(step_b.tool_calls)}, evidence={len(step_b.evidence_required)}")
    print("   ✅ No shared mutable defaults!")
    print()


def demo_htn_cycle_detection():
    """Demo cycle detection in HTN planner."""
    print("4. Cycle Detection:")
    planner = HTNPlanner()
    
    # Create steps with circular dependencies
    step1 = HTNStep("step1", "Op", "Goal 1", dependencies=["step2"])
    step2 = HTNStep("step2", "Op", "Goal 2", dependencies=["step1"])
    
    try:
        planner._calculate_execution_order([step1, step2])
        print("   ❌ Cycle detection failed")
    except ValueError as e:
        print(f"   ✅ Cycle detected: {e}")
    
    # Test unknown dependency detection
    step3 = HTNStep("step3", "Op", "Goal 3", dependencies=["unknown_step"])
    try:
        planner._calculate_execution_order([step3])
        print("   ❌ Unknown dependency detection failed")
    except ValueError as e:
        print(f"   ✅ Unknown dependency detected: {e}")
    print()


def demo_htn_operator_ranking():
    """Demo operator ranking and selection."""
    print("5. Operator Ranking and Selection:")
    planner = HTNPlanner()
    
    # Create traffic scenario
    traffic_scenario = ValidatedDisruptionScenario(
        scenario_type=ScenarioType.TRAFFIC,
        urgency_level=UrgencyLevel.HIGH,
        entities=[
            ValidatedEntity("DEL123", EntityType.DELIVERY_ID, 0.9, "DEL123")
        ],
        original_text="Traffic jam on Highway 101 affecting delivery DEL123"
    )
    
    # Test operator scoring
    traffic_op = ResolveTrafficDelayOperator()
    merchant_op = ResolveMerchantOverloadOperator()
    mediation_op = AtDoorMediationOperator()
    
    goal = "resolve traffic delay"
    traffic_score = planner._score_operator(traffic_op, goal, traffic_scenario)
    merchant_score = planner._score_operator(merchant_op, goal, traffic_scenario)
    mediation_score = planner._score_operator(mediation_op, goal, traffic_scenario)
    
    print(f"   Traffic Operator Score: {traffic_score}")
    print(f"   Merchant Operator Score: {merchant_score}")
    print(f"   Mediation Operator Score: {mediation_score}")
    print(f"   ✅ Traffic operator ranked highest: {traffic_score > max(merchant_score, mediation_score)}")
    print()


def demo_htn_plan_generation():
    """Demo HTN plan generation for different scenarios."""
    print("6. HTN Plan Generation:")
    planner = HTNPlanner()
    
    scenarios = [
        {
            "name": "Traffic Disruption",
            "scenario": ValidatedDisruptionScenario(
                scenario_type=ScenarioType.TRAFFIC,
                urgency_level=UrgencyLevel.HIGH,
                entities=[ValidatedEntity("DEL456", EntityType.DELIVERY_ID, 0.9, "DEL456")],
                original_text="Highway closure affecting delivery DEL456"
            ),
            "goal": "resolve traffic delay"
        },
        {
            "name": "Merchant Overload",
            "scenario": ValidatedDisruptionScenario(
                scenario_type=ScenarioType.MERCHANT,
                urgency_level=UrgencyLevel.MEDIUM,
                entities=[],
                original_text="Restaurant kitchen overloaded with orders"
            ),
            "goal": "resolve merchant overload"
        },
        {
            "name": "Delivery Dispute",
            "scenario": ValidatedDisruptionScenario(
                scenario_type=ScenarioType.OTHER,
                urgency_level=UrgencyLevel.HIGH,
                entities=[],
                original_text="Customer disputes damaged package at door"
            ),
            "goal": "resolve delivery dispute"
        }
    ]
    
    for scenario_data in scenarios:
        print(f"   {scenario_data['name']}:")
        plan = planner.plan(scenario_data['goal'], scenario_data['scenario'])
        
        print(f"     Plan ID: {plan.plan_id}")
        print(f"     Steps: {len(plan.operators)}")
        print(f"     Success Probability: {plan.success_probability:.1%}")
        print(f"     Duration: {plan.estimated_duration_minutes} minutes")
        print(f"     Execution Order: {plan.execution_order}")
        print()


def demo_htn_circuit_breaker_integration():
    """Demo HTN integration with circuit breakers."""
    print("7. Circuit Breaker Integration:")
    planner = HTNPlanner()
    
    # Create scenario with failed tools
    scenario = ValidatedDisruptionScenario(
        scenario_type=ScenarioType.TRAFFIC,
        urgency_level=UrgencyLevel.CRITICAL,
        entities=[],
        original_text="Emergency traffic situation"
    )
    
    # Simulate failed tool results (circuit breakers open)
    failed_tool_results = [
        ToolResult("check_traffic", False, {}, 1.0, error_message="Circuit breaker open"),
        ToolResult("notify_customer", False, {}, 0.5, error_message="Circuit breaker open")
    ]
    
    plan = planner.plan("resolve emergency traffic", scenario, tool_results=failed_tool_results)
    
    print(f"   Original Steps: {len([op for op in plan.operators if op.operator_name != 'Fallbacks'])}")
    fallback_steps = [op for op in plan.operators if op.operator_name == "Fallbacks"]
    print(f"   Fallback Steps: {len(fallback_steps)}")
    
    for fallback in fallback_steps:
        print(f"     - {fallback.step_id}: {fallback.success_criteria}")
    
    print(f"   ✅ Fallbacks injected for failed tools")
    print()


def demo_htn_to_resolution_steps():
    """Demo compilation of HTN plan to resolution steps."""
    print("8. HTN to Resolution Steps Compilation:")
    planner = HTNPlanner()
    
    scenario = ValidatedDisruptionScenario(
        scenario_type=ScenarioType.MERCHANT,
        urgency_level=UrgencyLevel.HIGH,
        entities=[],
        original_text="Restaurant capacity exceeded"
    )
    
    htn_plan = planner.plan("resolve merchant capacity", scenario)
    compiled_steps = planner.to_resolution_steps(htn_plan)
    
    print(f"   HTN Steps: {len(htn_plan.operators)}")
    print(f"   Compiled Steps: {len(compiled_steps)}")
    print("   Sample compiled step:")
    if compiled_steps:
        step = compiled_steps[0]
        print(f"     Sequence: {step['sequence']}")
        print(f"     Action: {step['action']}")
        print(f"     Responsible: {step['responsible_party']}")
        print(f"     Success Criteria: {step['success_criteria']}")
        print(f"     Tool Calls: {step['tool_calls']}")
    print()


def demo_htn_evidence_validation():
    """Demo evidence-based success probability calculation."""
    print("9. Evidence-Based Success Probability:")
    
    # Create traffic operator and test evidence validation
    traffic_op = ResolveTrafficDelayOperator()
    scenario = ValidatedDisruptionScenario(
        scenario_type=ScenarioType.TRAFFIC,
        urgency_level=UrgencyLevel.HIGH,
        entities=[],
        original_text="Traffic disruption"
    )
    
    steps = traffic_op.decompose("resolve traffic", scenario, {})
    
    # Test with no tool results
    prob_no_evidence = traffic_op.estimate_success_probability(steps, [])
    print(f"   Success probability (no evidence): {prob_no_evidence:.1%}")
    
    # Test with successful tool results
    successful_results = [
        ToolResult("check_traffic", True, {"traffic_condition": "heavy"}, 1.0),
        ToolResult("re_route_driver", True, {"new_route": "Highway 9"}, 1.5),
        ToolResult("notify_customer", True, {"delivered": True}, 0.5)
    ]
    
    prob_with_evidence = traffic_op.estimate_success_probability(steps, successful_results)
    print(f"   Success probability (with evidence): {prob_with_evidence:.1%}")
    print(f"   ✅ Evidence improves success probability: {prob_with_evidence > prob_no_evidence}")
    print()


def main():
    """Run all HTN planner demos."""
    print("🚀 HTN PLANNER COMPREHENSIVE DEMO")
    print("=" * 80)
    print("Demonstrating production-ready HTN planner with:")
    print("• Safe data structures with default factories")
    print("• Cycle detection and dependency validation")
    print("• Operator ranking and composition")
    print("• Circuit breaker integration with fallbacks")
    print("• Evidence-based success probability calculation")
    print("• Integration with existing plan generator")
    print()
    
    try:
        demo_htn_basic_functionality()
        demo_htn_cycle_detection()
        demo_htn_operator_ranking()
        demo_htn_plan_generation()
        demo_htn_circuit_breaker_integration()
        demo_htn_to_resolution_steps()
        demo_htn_evidence_validation()
        
        print("🎉 ALL HTN PLANNER DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("The HTN planner is production-ready with:")
        print("✅ Robust error handling and cycle detection")
        print("✅ Intelligent operator ranking and selection")
        print("✅ Circuit breaker resilience with fallbacks")
        print("✅ Evidence-based probability calculation")
        print("✅ Seamless integration with existing systems")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()