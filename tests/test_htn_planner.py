"""
Unit tests for HTN (Hierarchical Task Network) planner.
Tests cycle detection, operator ranking, evidence mapping, and integration.
"""
import pytest
from unittest.mock import Mock

from src.reasoning.htn import (
    HTNPlanner, HTNStep, HTNPlan, OperatorStatus,
    ResolveTrafficDelayOperator, ResolveMerchantOverloadOperator, AtDoorMediationOperator,
    evidence_flags_from_tools, _urgency_bonus
)
from src.agent.interfaces import ScenarioType, UrgencyLevel
from src.agent.models import ValidatedDisruptionScenario, ValidatedEntity, EntityType
from src.tools.interfaces import ToolResult


class TestHTNPlanner:
    """Test cases for HTN planner functionality."""
    
    @pytest.fixture
    def planner(self):
        """Create HTN planner instance."""
        return HTNPlanner()
    
    @pytest.fixture
    def traffic_scenario(self):
        """Create traffic disruption scenario."""
        return ValidatedDisruptionScenario(
            scenario_type=ScenarioType.TRAFFIC,
            urgency_level=UrgencyLevel.HIGH,
            entities=[
                ValidatedEntity(
                    text="DEL123",
                    entity_type=EntityType.DELIVERY_ID,
                    confidence=0.9,
                    normalized_value="DEL123"
                )
            ],
            original_text="Traffic jam on Highway 101 affecting delivery DEL123"
        )
    
    @pytest.fixture
    def merchant_scenario(self):
        """Create merchant overload scenario."""
        return ValidatedDisruptionScenario(
            scenario_type=ScenarioType.MERCHANT,
            urgency_level=UrgencyLevel.MEDIUM,
            entities=[],
            original_text="Restaurant kitchen overloaded with orders"
        )
    
    def test_cycle_detection(self, planner):
        """Test that cycle detection raises ValueError."""
        # Create steps with circular dependencies
        step1 = HTNStep(
            step_id="step1",
            operator_name="TestOp",
            goal="Goal 1",
            dependencies=["step2"]
        )
        step2 = HTNStep(
            step_id="step2", 
            operator_name="TestOp",
            goal="Goal 2",
            dependencies=["step1"]
        )
        
        with pytest.raises(ValueError, match="Cycle detected"):
            planner._calculate_execution_order([step1, step2])
    
    def test_unknown_dependency_detection(self, planner):
        """Test that unknown dependencies raise ValueError."""
        step = HTNStep(
            step_id="step1",
            operator_name="TestOp", 
            goal="Goal 1",
            dependencies=["unknown_step"]
        )
        
        with pytest.raises(ValueError, match="Unknown dependencies"):
            planner._calculate_execution_order([step])
    
    def test_operator_ranking_traffic_scenario(self, planner, traffic_scenario):
        """Test that traffic operator gets highest score for traffic scenarios."""
        goal = "resolve traffic delay"
        
        traffic_op = ResolveTrafficDelayOperator()
        merchant_op = ResolveMerchantOverloadOperator()
        mediation_op = AtDoorMediationOperator()
        
        traffic_score = planner._score_operator(traffic_op, goal, traffic_scenario)
        merchant_score = planner._score_operator(merchant_op, goal, traffic_scenario)
        mediation_score = planner._score_operator(mediation_op, goal, traffic_scenario)
        
        assert traffic_score > merchant_score
        assert traffic_score > mediation_score
    
    def test_operator_ranking_merchant_scenario(self, planner, merchant_scenario):
        """Test that merchant operator gets highest score for merchant scenarios."""
        goal = "resolve merchant overload"
        
        traffic_op = ResolveTrafficDelayOperator()
        merchant_op = ResolveMerchantOverloadOperator()
        mediation_op = AtDoorMediationOperator()
        
        traffic_score = planner._score_operator(traffic_op, goal, merchant_scenario)
        merchant_score = planner._score_operator(merchant_op, goal, merchant_scenario)
        mediation_score = planner._score_operator(mediation_op, goal, merchant_scenario)
        
        assert merchant_score > traffic_score
        assert merchant_score > mediation_score
    
    def test_evidence_mapping_from_tools(self):
        """Test that tool results produce expected evidence flags."""
        tool_results = [
            ToolResult(
                tool_name="check_traffic",
                success=True,
                data={"traffic_condition": "heavy"},
                execution_time=1.0
            ),
            ToolResult(
                tool_name="re_route_driver", 
                success=True,
                data={"new_route": "Highway 9"},
                execution_time=1.5
            ),
            ToolResult(
                tool_name="notify_customer",
                success=True,
                data={"delivered": True},
                execution_time=0.5
            )
        ]
        
        flags = evidence_flags_from_tools(tool_results)
        
        assert "traffic_condition_verified" in flags
        assert "route_changed" in flags
        assert "customer_notified" in flags
    
    def test_evidence_mapping_failed_tools(self):
        """Test that failed tools don't produce evidence flags."""
        tool_results = [
            ToolResult(
                tool_name="check_traffic",
                success=False,
                data={},
                execution_time=1.0,
                error_message="Tool failed"
            )
        ]
        
        flags = evidence_flags_from_tools(tool_results)
        assert "traffic_condition_verified" not in flags
    
    def test_urgency_bonus_calculation(self):
        """Test urgency bonus calculation."""
        assert _urgency_bonus(UrgencyLevel.CRITICAL) == 0.03
        assert _urgency_bonus(UrgencyLevel.HIGH) == 0.02
        assert _urgency_bonus(UrgencyLevel.MEDIUM) == 0.0
        assert _urgency_bonus(UrgencyLevel.LOW) == -0.01
    
    def test_fallback_injection(self, planner):
        """Test that fallback steps are injected for open breakers."""
        steps = [
            HTNStep(
                step_id="traffic_check",
                operator_name="TrafficOp",
                goal="Check traffic",
                tool_calls=["check_traffic"]
            )
        ]
        
        open_breakers = ["check_traffic"]
        planner.inject_fallback_steps(steps, open_breakers)
        
        # Should have original step + fallback step
        assert len(steps) == 2
        fallback_step = steps[1]
        assert fallback_step.step_id == "fallback_check_traffic"
        assert fallback_step.operator_name == "Fallbacks"
        assert "Use cached traffic" in fallback_step.success_criteria
    
    def test_plan_generation_traffic(self, planner, traffic_scenario):
        """Test HTN plan generation for traffic scenario."""
        goal = "resolve traffic delay"
        plan = planner.plan(goal, traffic_scenario)
        
        assert plan.root_goal == goal
        assert len(plan.operators) >= 3  # Should have traffic resolution steps
        assert plan.success_probability > 0.3
        assert plan.estimated_duration_minutes > 0
        assert len(plan.execution_order) == len(plan.operators)
    
    def test_plan_generation_merchant(self, planner, merchant_scenario):
        """Test HTN plan generation for merchant scenario."""
        goal = "resolve merchant overload"
        plan = planner.plan(goal, merchant_scenario)
        
        assert plan.root_goal == goal
        assert len(plan.operators) >= 3  # Should have merchant resolution steps
        assert plan.success_probability > 0.3
        assert any("merchant" in op.goal.lower() for op in plan.operators)
    
    def test_plan_with_tool_results(self, planner, traffic_scenario):
        """Test HTN planning with existing tool results."""
        goal = "resolve traffic delay"
        tool_results = [
            ToolResult(
                tool_name="check_traffic",
                success=True,
                data={"traffic_condition": "heavy"},
                execution_time=1.0
            )
        ]
        
        plan = planner.plan(goal, traffic_scenario, tool_results=tool_results)
        
        # Success probability should be higher with successful tool results
        assert plan.success_probability > 0.5
    
    def test_plan_with_circuit_breakers(self, planner, traffic_scenario):
        """Test HTN planning with circuit breaker failures."""
        goal = "resolve traffic delay"
        tool_results = [
            ToolResult(
                tool_name="check_traffic",
                success=False,
                data={},
                execution_time=1.0,
                error_message="Circuit breaker open"
            )
        ]
        
        plan = planner.plan(goal, traffic_scenario, tool_results=tool_results)
        
        # Should have fallback steps injected
        fallback_steps = [op for op in plan.operators if op.operator_name == "Fallbacks"]
        assert len(fallback_steps) > 0
    
    def test_to_resolution_steps_compilation(self, planner, traffic_scenario):
        """Test compilation of HTN plan to resolution steps."""
        goal = "resolve traffic delay"
        plan = planner.plan(goal, traffic_scenario)
        
        compiled_steps = planner.to_resolution_steps(plan)
        
        assert len(compiled_steps) == len(plan.operators)
        for i, step in enumerate(compiled_steps):
            assert step["sequence"] == i + 1
            assert "action" in step
            assert "responsible_party" in step
            assert "success_criteria" in step
            assert "tool_calls" in step
    
    def test_no_applicable_operators_fallback(self, planner):
        """Test fallback plan when no operators can handle the goal."""
        # Create scenario that no operator can handle
        unknown_scenario = ValidatedDisruptionScenario(
            scenario_type=ScenarioType.OTHER,
            urgency_level=UrgencyLevel.LOW,
            entities=[],
            original_text="Unknown disruption type"
        )
        
        goal = "handle unknown disruption"
        plan = planner.plan(goal, unknown_scenario)
        
        assert plan.plan_id == "htn_fallback"
        assert len(plan.operators) == 1
        assert plan.operators[0].operator_name == "GenericFallback"
    
    def test_execution_order_with_dependencies(self, planner):
        """Test that execution order respects dependencies."""
        steps = [
            HTNStep(step_id="step3", operator_name="Op", goal="Goal 3", dependencies=["step1", "step2"]),
            HTNStep(step_id="step1", operator_name="Op", goal="Goal 1", dependencies=[]),
            HTNStep(step_id="step2", operator_name="Op", goal="Goal 2", dependencies=["step1"])
        ]
        
        order = planner._calculate_execution_order(steps)
        
        # step1 should come first, step2 should come before step3
        assert order.index("step1") < order.index("step2")
        assert order.index("step2") < order.index("step3")
        assert order == ["step1", "step2", "step3"]
    
    def test_operator_composition(self, planner):
        """Test that multiple applicable operators can be composed."""
        # Create scenario that multiple operators can handle
        complex_scenario = ValidatedDisruptionScenario(
            scenario_type=ScenarioType.MULTI_FACTOR,
            urgency_level=UrgencyLevel.HIGH,
            entities=[],
            original_text="Traffic jam and restaurant dispute affecting delivery"
        )
        
        goal = "resolve complex disruption with traffic and dispute"
        plan = planner.plan(goal, complex_scenario)
        
        # Should potentially compose multiple operators
        operator_names = {op.operator_name for op in plan.operators}
        assert len(operator_names) >= 1  # At least one operator should apply
    
    def test_htn_step_safe_defaults(self):
        """Test that HTNStep uses safe default factories."""
        step1 = HTNStep(step_id="test1", operator_name="Op", goal="Goal")
        step2 = HTNStep(step_id="test2", operator_name="Op", goal="Goal")
        
        # Modify one step's lists
        step1.subgoals.append("subgoal1")
        step1.tool_calls.append("tool1")
        step1.dependencies.append("dep1")
        step1.evidence_required.add("evidence1")
        
        # Other step should not be affected (no shared mutable defaults)
        assert len(step2.subgoals) == 0
        assert len(step2.tool_calls) == 0
        assert len(step2.dependencies) == 0
        assert len(step2.evidence_required) == 0
    
    def test_htn_plan_safe_defaults(self):
        """Test that HTNPlan uses safe default factories."""
        plan1 = HTNPlan(plan_id="test1", root_goal="Goal1")
        plan2 = HTNPlan(plan_id="test2", root_goal="Goal2")
        
        # Modify one plan's lists
        plan1.operators.append(HTNStep(step_id="step1", operator_name="Op", goal="Goal"))
        plan1.execution_order.append("step1")
        
        # Other plan should not be affected
        assert len(plan2.operators) == 0
        assert len(plan2.execution_order) == 0


class TestHTNOperators:
    """Test individual HTN operators."""
    
    def test_traffic_operator_can_handle(self):
        """Test traffic operator goal matching."""
        op = ResolveTrafficDelayOperator()
        traffic_scenario = ValidatedDisruptionScenario(
            scenario_type=ScenarioType.TRAFFIC,
            urgency_level=UrgencyLevel.MEDIUM,
            entities=[],
            original_text="Traffic delay"
        )
        
        assert op.can_handle("resolve traffic delay", traffic_scenario)
        assert op.can_handle("handle route disruption", traffic_scenario)
        assert not op.can_handle("resolve merchant issue", 
                                ValidatedDisruptionScenario(
                                    scenario_type=ScenarioType.MERCHANT,
                                    urgency_level=UrgencyLevel.MEDIUM,
                                    entities=[],
                                    original_text="Merchant issue"
                                ))
    
    def test_merchant_operator_decomposition(self):
        """Test merchant operator step decomposition."""
        op = ResolveMerchantOverloadOperator()
        scenario = ValidatedDisruptionScenario(
            scenario_type=ScenarioType.MERCHANT,
            urgency_level=UrgencyLevel.HIGH,
            entities=[],
            original_text="Restaurant overloaded"
        )
        
        steps = op.decompose("resolve merchant overload", scenario, {})
        
        assert len(steps) == 3
        assert steps[0].step_id == "merchant_assess"
        assert steps[1].step_id == "merchant_alternatives"
        assert steps[2].step_id == "merchant_coordinate"
        
        # Check dependencies
        assert len(steps[0].dependencies) == 0
        assert "merchant_assess" in steps[1].dependencies
        assert "merchant_alternatives" in steps[2].dependencies
    
    def test_mediation_operator_evidence_requirements(self):
        """Test mediation operator evidence requirements."""
        op = AtDoorMediationOperator()
        scenario = ValidatedDisruptionScenario(
            scenario_type=ScenarioType.OTHER,
            urgency_level=UrgencyLevel.HIGH,
            entities=[],
            original_text="Delivery dispute at door"
        )
        
        steps = op.decompose("resolve delivery dispute", scenario, {})
        
        # Check evidence requirements
        evidence_step = next(s for s in steps if s.step_id == "mediation_evidence")
        assert "evidence_collected" in evidence_step.evidence_required
        assert "fault_determined" in evidence_step.evidence_required
        
        resolve_step = next(s for s in steps if s.step_id == "mediation_resolve")
        assert "resolution_applied" in resolve_step.evidence_required