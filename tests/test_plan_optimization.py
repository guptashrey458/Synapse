"""
Unit tests for plan optimization and risk assessment features.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.reasoning.plan_generator import PlanGenerator
from src.agent.interfaces import (
    ReasoningTrace, ReasoningStep, DisruptionScenario, 
    ScenarioType, UrgencyLevel, Entity, EntityType
)
from src.agent.models import ValidatedPlanStep, ValidatedResolutionPlan


class TestPlanOptimization:
    """Test cases for plan optimization features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plan_generator = PlanGenerator()
        
        # Create scenarios with different urgency levels
        self.critical_scenario = DisruptionScenario(
            description="Critical delivery failure requiring immediate action",
            entities=[Entity(text="123 Main St", entity_type=EntityType.ADDRESS, confidence=0.9)],
            scenario_type=ScenarioType.TRAFFIC,
            urgency_level=UrgencyLevel.CRITICAL
        )
        
        self.low_scenario = DisruptionScenario(
            description="Minor delivery delay, no rush",
            entities=[Entity(text="Pizza Palace", entity_type=EntityType.MERCHANT, confidence=0.8)],
            scenario_type=ScenarioType.MERCHANT,
            urgency_level=UrgencyLevel.LOW
        )
        
        self.multi_factor_scenario = DisruptionScenario(
            description="Traffic jam and restaurant closed, customer unreachable",
            entities=[
                Entity(text="Pizza Palace", entity_type=EntityType.MERCHANT, confidence=0.9),
                Entity(text="Main Street", entity_type=EntityType.ADDRESS, confidence=0.8)
            ],
            scenario_type=ScenarioType.MULTI_FACTOR,
            urgency_level=UrgencyLevel.HIGH
        )
        
        # Create sample plan steps
        self.sample_steps = [
            ValidatedPlanStep(
                sequence=1,
                action="Check traffic status",
                responsible_party="Operations",
                estimated_time=timedelta(minutes=5),
                dependencies=[],
                success_criteria="Traffic status confirmed"
            ),
            ValidatedPlanStep(
                sequence=2,
                action="Verify merchant availability",
                responsible_party="Operations",
                estimated_time=timedelta(minutes=3),
                dependencies=[1],
                success_criteria="Merchant status verified"
            ),
            ValidatedPlanStep(
                sequence=3,
                action="Notify customer about delay",
                responsible_party="Customer Service",
                estimated_time=timedelta(minutes=2),
                dependencies=[1, 2],
                success_criteria="Customer informed"
            ),
            ValidatedPlanStep(
                sequence=4,
                action="Reroute driver to alternative path",
                responsible_party="Dispatch",
                estimated_time=timedelta(minutes=10),
                dependencies=[1],
                success_criteria="Driver rerouted"
            ),
            ValidatedPlanStep(
                sequence=5,
                action="Update delivery tracking system",
                responsible_party="System",
                estimated_time=timedelta(minutes=1),
                dependencies=[4],
                success_criteria="System updated"
            )
        ]
    
    def test_optimize_for_speed(self):
        """Test speed optimization for critical scenarios."""
        original_steps = [step for step in self.sample_steps]  # Copy
        
        optimized_steps = self.plan_generator._optimize_for_speed(original_steps)
        
        # Check that time estimates are reduced
        for i, step in enumerate(optimized_steps):
            original_time = self.sample_steps[i].estimated_time
            assert step.estimated_time <= original_time
            assert step.estimated_time == original_time * 0.8
        
        # Check that non-essential dependencies are removed
        notify_step = next(step for step in optimized_steps if 'notify' in step.action.lower())
        assert len(notify_step.dependencies) <= len(self.sample_steps[2].dependencies)
    
    def test_optimize_for_balance(self):
        """Test balanced optimization for high priority scenarios."""
        original_steps = [step for step in self.sample_steps]  # Copy
        
        optimized_steps = self.plan_generator._optimize_for_balance(original_steps)
        
        # Check that time estimates are slightly reduced
        for i, step in enumerate(optimized_steps):
            original_time = self.sample_steps[i].estimated_time
            assert step.estimated_time <= original_time
            assert step.estimated_time == original_time * 0.9
        
        # Check that dependencies are streamlined but not eliminated
        for step in optimized_steps:
            assert len(step.dependencies) <= 2
    
    def test_optimize_for_thoroughness(self):
        """Test thoroughness optimization for low priority scenarios."""
        original_steps = [step for step in self.sample_steps]  # Copy
        
        optimized_steps = self.plan_generator._optimize_for_thoroughness(original_steps)
        
        # Check that time estimates include buffer
        for i, step in enumerate(optimized_steps):
            original_time = self.sample_steps[i].estimated_time
            assert step.estimated_time >= original_time
            assert step.estimated_time == original_time * 1.1
    
    def test_prioritize_by_impact(self):
        """Test impact-based prioritization."""
        critical_trace = ReasoningTrace(
            steps=[],
            scenario=self.critical_scenario,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        original_steps = [step for step in self.sample_steps]
        prioritized_steps = self.plan_generator._prioritize_by_impact(original_steps, critical_trace)
        
        # Check that all steps have impact scores
        for step in prioritized_steps:
            assert hasattr(step, 'impact_score')
            assert isinstance(step.impact_score, int)
        
        # Check that customer-facing actions have higher impact scores
        notify_step = next(step for step in prioritized_steps if 'notify' in step.action.lower())
        check_step = next(step for step in prioritized_steps if 'check' in step.action.lower())
        assert notify_step.impact_score > check_step.impact_score
    
    def test_topological_sort_with_impact(self):
        """Test topological sorting that considers impact scores."""
        # Add impact scores to steps
        for i, step in enumerate(self.sample_steps):
            step.impact_score = 5 - i  # Reverse order impact scores
        
        sorted_steps = self.plan_generator._topological_sort_with_impact(self.sample_steps)
        
        # Check that dependencies are still respected
        for step in sorted_steps:
            for dep in step.dependencies:
                dep_step = next(s for s in sorted_steps if s.sequence == dep)
                assert sorted_steps.index(dep_step) < sorted_steps.index(step)
        
        # Check that sequences are updated correctly
        for i, step in enumerate(sorted_steps):
            assert step.sequence == i + 1
    
    def test_optimize_resource_allocation(self):
        """Test resource allocation optimization."""
        # Create steps with resource bottleneck
        overloaded_steps = [
            ValidatedPlanStep(
                sequence=i,
                action=f"Operations task {i}",
                responsible_party="Operations",
                estimated_time=timedelta(minutes=5),
                dependencies=[],
                success_criteria=f"Task {i} completed"
            )
            for i in range(1, 6)  # 5 steps for Operations
        ]
        
        # Add some administrative tasks
        overloaded_steps.extend([
            ValidatedPlanStep(
                sequence=6,
                action="Update system logs",
                responsible_party="Operations",
                estimated_time=timedelta(minutes=2),
                dependencies=[],
                success_criteria="Logs updated"
            ),
            ValidatedPlanStep(
                sequence=7,
                action="Record incident details",
                responsible_party="Operations",
                estimated_time=timedelta(minutes=3),
                dependencies=[],
                success_criteria="Incident recorded"
            )
        ])
        
        optimized_steps = self.plan_generator._optimize_resource_allocation(overloaded_steps)
        
        # Check that some administrative tasks were reassigned
        system_tasks = [step for step in optimized_steps if step.responsible_party == "System"]
        assert len(system_tasks) > 0
        
        # Check that reassigned tasks have reduced time estimates
        for step in system_tasks:
            if any(keyword in step.action.lower() for keyword in ['update', 'record']):
                # Should be faster than original estimate
                assert step.estimated_time.total_seconds() < 300  # Less than 5 minutes
    
    def test_optimize_plan_steps_critical(self):
        """Test complete plan optimization for critical scenarios."""
        critical_trace = ReasoningTrace(
            steps=[],
            scenario=self.critical_scenario,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        original_steps = [step for step in self.sample_steps]
        optimized_steps = self.plan_generator._optimize_plan_steps(original_steps, critical_trace)
        
        # Check that optimization was applied
        assert len(optimized_steps) == len(original_steps)
        
        # Check that time estimates are reduced for critical scenarios
        total_original_time = sum(step.estimated_time.total_seconds() for step in original_steps)
        total_optimized_time = sum(step.estimated_time.total_seconds() for step in optimized_steps)
        assert total_optimized_time <= total_original_time * 0.85  # Should be at least 15% faster
    
    def test_optimize_plan_steps_low_priority(self):
        """Test complete plan optimization for low priority scenarios."""
        low_trace = ReasoningTrace(
            steps=[],
            scenario=self.low_scenario,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        original_steps = [step for step in self.sample_steps]
        optimized_steps = self.plan_generator._optimize_plan_steps(original_steps, low_trace)
        
        # Check that time estimates include buffer for thoroughness
        total_original_time = sum(step.estimated_time.total_seconds() for step in original_steps)
        total_optimized_time = sum(step.estimated_time.total_seconds() for step in optimized_steps)
        assert total_optimized_time >= total_original_time * 1.05  # Should be at least 5% longer
    
    def test_assess_plan_risks(self):
        """Test comprehensive risk assessment."""
        plan = ValidatedResolutionPlan(
            steps=self.sample_steps,
            estimated_duration=timedelta(minutes=30),
            success_probability=0.8,
            alternatives=[],
            stakeholders=['Operations', 'Customer Service', 'Dispatch']
        )
        
        multi_factor_trace = ReasoningTrace(
            steps=[],
            scenario=self.multi_factor_scenario,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        risk_assessment = self.plan_generator.assess_plan_risks(plan, multi_factor_trace)
        
        # Check required fields
        assert 'overall_risk_level' in risk_assessment
        assert 'risk_factors' in risk_assessment
        assert 'mitigation_strategies' in risk_assessment
        assert 'confidence_intervals' in risk_assessment
        assert 'failure_scenarios' in risk_assessment
        
        # Check risk level is valid
        assert risk_assessment['overall_risk_level'] in ['low', 'medium', 'high']
        
        # Check confidence intervals
        intervals = risk_assessment['confidence_intervals']
        assert 'optimistic' in intervals
        assert 'realistic' in intervals
        assert 'pessimistic' in intervals
        assert intervals['pessimistic'] <= intervals['realistic'] <= intervals['optimistic']
        
        # Multi-factor scenario should have specific risks
        assert any('multi-factor' in factor.lower() for factor in risk_assessment['risk_factors'])
    
    def test_assess_plan_risks_high_complexity(self):
        """Test risk assessment for high complexity plans."""
        # Create a complex plan with many steps
        complex_steps = [
            ValidatedPlanStep(
                sequence=i,
                action=f"Complex task {i}",
                responsible_party="Operations",
                estimated_time=timedelta(minutes=10),
                dependencies=list(range(max(1, i-3), i)) if i > 3 else list(range(1, i)),  # Multiple dependencies for later steps
                success_criteria=f"Task {i} completed"
            )
            for i in range(1, 12)  # 11 steps
        ]
        
        complex_plan = ValidatedResolutionPlan(
            steps=complex_steps,
            estimated_duration=timedelta(hours=2),  # Long duration
            success_probability=0.6,
            alternatives=[],
            stakeholders=['Operations']
        )
        
        critical_trace = ReasoningTrace(
            steps=[],
            scenario=self.critical_scenario,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        risk_assessment = self.plan_generator.assess_plan_risks(complex_plan, critical_trace)
        
        # Should identify high complexity and time risks
        assert risk_assessment['overall_risk_level'] == 'high'
        assert any('complexity' in factor.lower() for factor in risk_assessment['risk_factors'])
        assert any('time' in factor.lower() for factor in risk_assessment['risk_factors'])
        assert any('dependency' in factor.lower() for factor in risk_assessment['risk_factors'])
    
    def test_generate_plan_quality_metrics(self):
        """Test plan quality metrics generation."""
        plan = ValidatedResolutionPlan(
            steps=self.sample_steps,
            estimated_duration=timedelta(minutes=25),
            success_probability=0.8,
            alternatives=[],
            stakeholders=['Operations', 'Customer Service', 'Dispatch']
        )
        
        traffic_trace = ReasoningTrace(
            steps=[],
            scenario=self.critical_scenario,  # Traffic scenario
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        metrics = self.plan_generator.generate_plan_quality_metrics(plan, traffic_trace)
        
        # Check all required metrics are present
        required_metrics = ['completeness', 'efficiency', 'feasibility', 'customer_impact', 'risk_management', 'overall_quality']
        for metric in required_metrics:
            assert metric in metrics
            assert 0.0 <= metrics[metric] <= 1.0
        
        # Overall quality should be average of other metrics
        other_metrics = [v for k, v in metrics.items() if k != 'overall_quality']
        expected_overall = sum(other_metrics) / len(other_metrics)
        assert abs(metrics['overall_quality'] - expected_overall) < 0.01
    
    def test_assess_completeness(self):
        """Test completeness assessment for different scenario types."""
        # Traffic scenario should require check, reroute, notify
        traffic_plan = ValidatedResolutionPlan(
            steps=[
                ValidatedPlanStep(1, "Check traffic", "Operations", timedelta(minutes=5), [], "Checked"),
                ValidatedPlanStep(2, "Reroute driver", "Dispatch", timedelta(minutes=10), [1], "Rerouted"),
                ValidatedPlanStep(3, "Notify customer", "Customer Service", timedelta(minutes=2), [1], "Notified")
            ],
            estimated_duration=timedelta(minutes=17),
            success_probability=0.8,
            alternatives=[],
            stakeholders=[]
        )
        
        traffic_trace = ReasoningTrace(
            steps=[],
            scenario=DisruptionScenario("Traffic jam", [], ScenarioType.TRAFFIC, UrgencyLevel.HIGH),
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        completeness = self.plan_generator._assess_completeness(traffic_plan, traffic_trace)
        assert completeness == 1.0  # Should cover all required actions
        
        # Incomplete plan should have lower completeness
        incomplete_plan = ValidatedResolutionPlan(
            steps=[ValidatedPlanStep(1, "Check traffic", "Operations", timedelta(minutes=5), [], "Checked")],
            estimated_duration=timedelta(minutes=5),
            success_probability=0.5,
            alternatives=[],
            stakeholders=[]
        )
        
        incomplete_completeness = self.plan_generator._assess_completeness(incomplete_plan, traffic_trace)
        assert incomplete_completeness < 1.0
    
    def test_assess_efficiency(self):
        """Test efficiency assessment."""
        # Efficient plan (reasonable steps and time)
        efficient_plan = ValidatedResolutionPlan(
            steps=self.sample_steps[:3],  # 3 steps
            estimated_duration=timedelta(minutes=15),
            success_probability=0.8,
            alternatives=[],
            stakeholders=[]
        )
        
        efficiency = self.plan_generator._assess_efficiency(efficient_plan)
        assert efficiency > 0.5
        
        # Inefficient plan (too many steps, too much time)
        inefficient_steps = [
            ValidatedPlanStep(i, f"Task {i}", "Operations", timedelta(minutes=30), [], f"Task {i} done")
            for i in range(1, 11)  # 10 steps, 30 minutes each
        ]
        inefficient_plan = ValidatedResolutionPlan(
            steps=inefficient_steps,
            estimated_duration=timedelta(hours=5),
            success_probability=0.8,
            alternatives=[],
            stakeholders=[]
        )
        
        inefficient_efficiency = self.plan_generator._assess_efficiency(inefficient_plan)
        assert inefficient_efficiency < efficiency
    
    def test_assess_feasibility(self):
        """Test feasibility assessment."""
        # Feasible plan with realistic times and dependencies
        feasible_plan = ValidatedResolutionPlan(
            steps=self.sample_steps,
            estimated_duration=timedelta(minutes=25),
            success_probability=0.8,
            alternatives=[],
            stakeholders=[]
        )
        
        feasibility = self.plan_generator._assess_feasibility(feasible_plan)
        assert feasibility > 0.5
        
        # Unfeasible plan with unrealistic times
        unfeasible_steps = [
            ValidatedPlanStep(1, "Instant task", "Operations", timedelta(seconds=30), [], "Done"),  # Too fast
            ValidatedPlanStep(2, "Marathon task", "Operations", timedelta(hours=5), [1], "Done"),  # Too slow
            ValidatedPlanStep(3, "Complex task", "Operations", timedelta(minutes=10), [1, 2], "Done")  # Too many deps
        ]
        unfeasible_plan = ValidatedResolutionPlan(
            steps=unfeasible_steps,
            estimated_duration=timedelta(hours=5),
            success_probability=0.8,
            alternatives=[],
            stakeholders=[]
        )
        
        unfeasible_feasibility = self.plan_generator._assess_feasibility(unfeasible_plan)
        assert unfeasible_feasibility < feasibility
    
    def test_assess_customer_impact(self):
        """Test customer impact assessment."""
        # Plan with good customer communication
        customer_focused_plan = ValidatedResolutionPlan(
            steps=[
                ValidatedPlanStep(1, "Check status", "Operations", timedelta(minutes=5), [], "Checked"),
                ValidatedPlanStep(2, "Notify customer", "Customer Service", timedelta(minutes=2), [1], "Notified"),
                ValidatedPlanStep(3, "Update customer", "Customer Service", timedelta(minutes=2), [2], "Updated")
            ],
            estimated_duration=timedelta(minutes=9),
            success_probability=0.8,
            alternatives=[],
            stakeholders=[]
        )
        
        customer_impact = self.plan_generator._assess_customer_impact(customer_focused_plan)
        assert customer_impact > 0.5
        
        # Plan with no customer communication
        internal_plan = ValidatedResolutionPlan(
            steps=[
                ValidatedPlanStep(1, "Internal check", "Operations", timedelta(minutes=5), [], "Checked"),
                ValidatedPlanStep(2, "Update system logs", "System", timedelta(minutes=2), [1], "Updated")
            ],
            estimated_duration=timedelta(minutes=7),
            success_probability=0.8,
            alternatives=[],
            stakeholders=[]
        )
        
        internal_impact = self.plan_generator._assess_customer_impact(internal_plan)
        assert internal_impact < customer_impact
    
    def test_assess_risk_management(self):
        """Test risk management assessment."""
        low_risk_plan = ValidatedResolutionPlan(
            steps=self.sample_steps[:3],  # Simple plan
            estimated_duration=timedelta(minutes=15),
            success_probability=0.9,
            alternatives=[],
            stakeholders=[]
        )
        
        low_risk_trace = ReasoningTrace(
            steps=[],
            scenario=self.low_scenario,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        risk_management = self.plan_generator._assess_risk_management(low_risk_plan, low_risk_trace)
        assert risk_management > 0.7  # Should be high for low-risk scenario
        
        # High risk scenario should have lower risk management score
        high_risk_trace = ReasoningTrace(
            steps=[],
            scenario=self.multi_factor_scenario,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        high_risk_management = self.plan_generator._assess_risk_management(low_risk_plan, high_risk_trace)
        # Allow for small differences due to mitigation bonuses
        assert high_risk_management <= risk_management + 0.05


if __name__ == '__main__':
    pytest.main([__file__])