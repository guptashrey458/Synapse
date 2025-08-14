"""
Unit tests for the plan generation system.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.reasoning.plan_generator import PlanGenerator
from src.agent.interfaces import (
    ReasoningTrace, ReasoningStep, DisruptionScenario, 
    ScenarioType, UrgencyLevel, Entity, EntityType
)
from src.agent.models import ValidatedPlanStep, ValidatedResolutionPlan, AlternativePlan


class TestPlanGenerator:
    """Test cases for PlanGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plan_generator = PlanGenerator()
        
        # Create a sample scenario
        self.sample_scenario = DisruptionScenario(
            description="Traffic jam on Main Street blocking delivery to 123 Oak Ave",
            entities=[
                Entity(text="123 Oak Ave", entity_type=EntityType.ADDRESS, confidence=0.9),
                Entity(text="Main Street", entity_type=EntityType.ADDRESS, confidence=0.8)
            ],
            scenario_type=ScenarioType.TRAFFIC,
            urgency_level=UrgencyLevel.HIGH
        )
        
        # Create sample reasoning steps
        self.sample_steps = [
            ReasoningStep(
                step_number=1,
                thought="I need to check the current traffic situation on Main Street",
                action="check_traffic",
                observation="Traffic is heavily congested due to construction",
                timestamp=datetime.now()
            ),
            ReasoningStep(
                step_number=2,
                thought="I should reroute the driver to avoid the traffic jam",
                action="reroute_driver",
                observation="Alternative route found via Elm Street",
                timestamp=datetime.now()
            ),
            ReasoningStep(
                step_number=3,
                thought="I need to notify the customer about the delay",
                action="notify_customer",
                observation="Customer notified about 15-minute delay",
                timestamp=datetime.now()
            )
        ]
        
        # Create sample reasoning trace
        self.sample_trace = ReasoningTrace(
            steps=self.sample_steps,
            scenario=self.sample_scenario,
            start_time=datetime.now() - timedelta(minutes=5),
            end_time=datetime.now()
        )
    
    def test_plan_generator_initialization(self):
        """Test that PlanGenerator initializes correctly."""
        generator = PlanGenerator()
        
        assert isinstance(generator.action_stakeholders, dict)
        assert isinstance(generator.action_time_estimates, dict)
        assert 'notify' in generator.action_stakeholders
        assert 'check' in generator.action_time_estimates
    
    def test_generate_plan_basic(self):
        """Test basic plan generation from reasoning trace."""
        plan = self.plan_generator.generate_plan(self.sample_trace)
        
        assert isinstance(plan, ValidatedResolutionPlan)
        assert len(plan.steps) > 0
        assert isinstance(plan.estimated_duration, timedelta)
        assert 0.0 <= plan.success_probability <= 1.0
        assert len(plan.stakeholders) > 0
        assert isinstance(plan.alternatives, list)
    
    def test_extract_actions_from_trace(self):
        """Test action extraction from reasoning trace."""
        actions = self.plan_generator._extract_actions_from_trace(self.sample_trace)
        
        assert isinstance(actions, list)
        assert len(actions) > 0
        
        # Check that actions have required fields
        for action in actions:
            assert 'verb' in action
            assert 'details' in action
            assert 'step_number' in action
            assert 'priority' in action
    
    def test_parse_step_for_actions(self):
        """Test parsing individual steps for actions."""
        step = ReasoningStep(
            step_number=1,
            thought="I should notify the customer about the delay",
            action=None,
            observation="Customer needs to be informed",
            timestamp=datetime.now()
        )
        
        actions = self.plan_generator._parse_step_for_actions(step)
        
        assert isinstance(actions, list)
        # Should find "notify" action
        notify_actions = [a for a in actions if a['verb'] == 'notify']
        assert len(notify_actions) > 0
    
    def test_determine_action_priority(self):
        """Test action priority determination."""
        # High priority action
        priority1 = self.plan_generator._determine_action_priority('notify', 'customer about delay')
        assert priority1 == 1
        
        # Medium priority action
        priority2 = self.plan_generator._determine_action_priority('check', 'traffic status')
        assert priority2 == 2
        
        # Urgent details should increase priority
        priority3 = self.plan_generator._determine_action_priority('update', 'urgent system status')
        assert priority3 == 1
    
    def test_deduplicate_actions(self):
        """Test action deduplication."""
        actions = [
            {'verb': 'notify', 'details': 'customer about delay', 'step_number': 1, 'priority': 1},
            {'verb': 'notify', 'details': 'customer about delay', 'step_number': 2, 'priority': 2},
            {'verb': 'check', 'details': 'traffic status', 'step_number': 1, 'priority': 2}
        ]
        
        deduplicated = self.plan_generator._deduplicate_actions(actions)
        
        assert len(deduplicated) == 2  # Should remove one duplicate
        # Should keep the higher priority version
        notify_action = next(a for a in deduplicated if a['verb'] == 'notify')
        assert notify_action['priority'] == 1
    
    def test_create_plan_steps(self):
        """Test creation of structured plan steps."""
        actions = [
            {'verb': 'check', 'details': 'traffic status', 'step_number': 1, 'priority': 2},
            {'verb': 'notify', 'details': 'customer about delay', 'step_number': 2, 'priority': 1}
        ]
        
        plan_steps = self.plan_generator._create_plan_steps(actions, self.sample_trace)
        
        assert len(plan_steps) == 2
        assert all(isinstance(step, ValidatedPlanStep) for step in plan_steps)
        
        # Check step sequencing
        assert plan_steps[0].sequence == 1
        assert plan_steps[1].sequence == 2
        
        # Check that all required fields are present
        for step in plan_steps:
            assert step.action
            assert step.responsible_party
            assert isinstance(step.estimated_time, timedelta)
            assert isinstance(step.dependencies, list)
            assert step.success_criteria
    
    def test_format_action_description(self):
        """Test action description formatting."""
        action = {'verb': 'notify', 'details': 'customer about the delay'}
        
        description = self.plan_generator._format_action_description(action)
        
        assert description == "Notify customer about the delay"
        assert description[0].isupper()  # Should be capitalized
    
    def test_determine_responsible_party(self):
        """Test responsible party determination."""
        assert self.plan_generator._determine_responsible_party('notify') == 'Customer Service'
        assert self.plan_generator._determine_responsible_party('reroute') == 'Dispatch'
        assert self.plan_generator._determine_responsible_party('check') == 'Operations'
        assert self.plan_generator._determine_responsible_party('unknown_action') == 'Operations'
    
    def test_estimate_action_time(self):
        """Test action time estimation."""
        # Basic time estimate
        time1 = self.plan_generator._estimate_action_time('notify', 'customer')
        assert isinstance(time1, timedelta)
        assert time1.total_seconds() > 0
        
        # Complex action should take longer
        time2 = self.plan_generator._estimate_action_time('investigate', 'complex issue')
        assert time2 > time1
        
        # Quick action should be faster
        time3 = self.plan_generator._estimate_action_time('update', 'quick status change')
        assert time3 < time1
    
    def test_determine_dependencies(self):
        """Test dependency determination."""
        previous_actions = [
            {'verb': 'check', 'details': 'status', 'step_number': 1, 'priority': 2},
            {'verb': 'verify', 'details': 'information', 'step_number': 2, 'priority': 2}
        ]
        current_action = {'verb': 'notify', 'details': 'customer', 'step_number': 3, 'priority': 1}
        
        dependencies = self.plan_generator._determine_dependencies(3, previous_actions, current_action)
        
        assert isinstance(dependencies, list)
        # Notify should depend on check and verify
        assert 1 in dependencies  # check step
        assert 2 in dependencies  # verify step
    
    def test_create_success_criteria(self):
        """Test success criteria creation."""
        action = {'verb': 'notify', 'details': 'customer about delay'}
        
        criteria = self.plan_generator._create_success_criteria(action)
        
        assert isinstance(criteria, str)
        assert len(criteria) > 0
        assert 'customer' in criteria.lower() or 'contact' in criteria.lower()
    
    def test_calculate_total_duration(self):
        """Test total duration calculation."""
        steps = [
            ValidatedPlanStep(
                sequence=1,
                action="Check traffic",
                responsible_party="Operations",
                estimated_time=timedelta(minutes=5),
                dependencies=[],
                success_criteria="Traffic status confirmed"
            ),
            ValidatedPlanStep(
                sequence=2,
                action="Notify customer",
                responsible_party="Customer Service",
                estimated_time=timedelta(minutes=2),
                dependencies=[1],
                success_criteria="Customer informed"
            )
        ]
        
        duration = self.plan_generator._calculate_total_duration(steps)
        
        assert isinstance(duration, timedelta)
        assert duration >= timedelta(minutes=7)  # Should be at least sum of sequential steps
    
    def test_estimate_success_probability(self):
        """Test success probability estimation."""
        simple_steps = [
            ValidatedPlanStep(
                sequence=1,
                action="Simple action",
                responsible_party="Operations",
                estimated_time=timedelta(minutes=5),
                dependencies=[],
                success_criteria="Action completed"
            )
        ]
        
        probability = self.plan_generator._estimate_success_probability(self.sample_trace, simple_steps)
        
        assert 0.0 <= probability <= 1.0
        
        # Test with critical urgency (should lower probability)
        critical_scenario = DisruptionScenario(
            description="Critical issue",
            entities=[],
            scenario_type=ScenarioType.OTHER,
            urgency_level=UrgencyLevel.CRITICAL
        )
        critical_trace = ReasoningTrace(
            steps=self.sample_steps,
            scenario=critical_scenario,
            start_time=datetime.now(),
            end_time=None
        )
        
        critical_probability = self.plan_generator._estimate_success_probability(critical_trace, simple_steps)
        assert critical_probability < probability
    
    def test_identify_stakeholders(self):
        """Test stakeholder identification."""
        steps = [
            ValidatedPlanStep(
                sequence=1,
                action="Check status",
                responsible_party="Operations",
                estimated_time=timedelta(minutes=5),
                dependencies=[],
                success_criteria="Status confirmed"
            ),
            ValidatedPlanStep(
                sequence=2,
                action="Notify customer",
                responsible_party="Customer Service",
                estimated_time=timedelta(minutes=2),
                dependencies=[1],
                success_criteria="Customer informed"
            )
        ]
        
        stakeholders = self.plan_generator._identify_stakeholders(steps)
        
        assert isinstance(stakeholders, list)
        assert 'Customer' in stakeholders  # Always included
        assert 'Operations' in stakeholders
        assert 'Customer Service' in stakeholders
        assert len(set(stakeholders)) == len(stakeholders)  # No duplicates
    
    def test_generate_alternatives(self):
        """Test alternative plan generation."""
        primary_steps = [
            ValidatedPlanStep(
                sequence=1,
                action="Check traffic",
                responsible_party="Operations",
                estimated_time=timedelta(minutes=5),
                dependencies=[],
                success_criteria="Traffic confirmed"
            ),
            ValidatedPlanStep(
                sequence=2,
                action="Verify route",
                responsible_party="Operations",
                estimated_time=timedelta(minutes=3),
                dependencies=[1],
                success_criteria="Route verified"
            ),
            ValidatedPlanStep(
                sequence=3,
                action="Notify customer",
                responsible_party="Customer Service",
                estimated_time=timedelta(minutes=2),
                dependencies=[2],
                success_criteria="Customer informed"
            ),
            ValidatedPlanStep(
                sequence=4,
                action="Reroute driver",
                responsible_party="Dispatch",
                estimated_time=timedelta(minutes=10),
                dependencies=[2],
                success_criteria="Driver rerouted"
            )
        ]
        
        alternatives = self.plan_generator._generate_alternatives(self.sample_trace, primary_steps)
        
        assert isinstance(alternatives, list)
        assert len(alternatives) >= 1
        
        for alt in alternatives:
            assert isinstance(alt, AlternativePlan)
            assert alt.name
            assert alt.description
            assert len(alt.steps) > 0
            assert isinstance(alt.estimated_duration, timedelta)
            assert 0.0 <= alt.success_probability <= 1.0
            assert isinstance(alt.trade_offs, list)
    
    def test_create_fast_alternative(self):
        """Test fast alternative plan creation."""
        primary_steps = [
            ValidatedPlanStep(
                sequence=1,
                action="Check traffic status",
                responsible_party="Operations",
                estimated_time=timedelta(minutes=5),
                dependencies=[],
                success_criteria="Traffic confirmed"
            ),
            ValidatedPlanStep(
                sequence=2,
                action="Notify customer about delay",
                responsible_party="Customer Service",
                estimated_time=timedelta(minutes=2),
                dependencies=[1],
                success_criteria="Customer informed"
            ),
            ValidatedPlanStep(
                sequence=3,
                action="Update internal logs",
                responsible_party="System",
                estimated_time=timedelta(minutes=1),
                dependencies=[],
                success_criteria="Logs updated"
            )
        ]
        
        fast_alt = self.plan_generator._create_fast_alternative(primary_steps)
        
        assert isinstance(fast_alt, AlternativePlan)
        assert fast_alt.name == "Fast Resolution"
        assert len(fast_alt.steps) <= len(primary_steps)  # Should have fewer steps
        assert fast_alt.success_probability < 0.8  # Should be lower than typical
        assert len(fast_alt.trade_offs) > 0
    
    def test_create_thorough_alternative(self):
        """Test thorough alternative plan creation."""
        primary_steps = [
            ValidatedPlanStep(
                sequence=1,
                action="Notify customer",
                responsible_party="Customer Service",
                estimated_time=timedelta(minutes=2),
                dependencies=[],
                success_criteria="Customer informed"
            )
        ]
        
        thorough_alt = self.plan_generator._create_thorough_alternative(self.sample_trace, primary_steps)
        
        assert isinstance(thorough_alt, AlternativePlan)
        assert thorough_alt.name == "Comprehensive Resolution"
        assert len(thorough_alt.steps) > len(primary_steps)  # Should have more steps
        assert thorough_alt.success_probability > 0.8  # Should be higher
        assert len(thorough_alt.trade_offs) > 0
    
    def test_empty_trace_handling(self):
        """Test handling of empty or minimal reasoning traces."""
        empty_trace = ReasoningTrace(
            steps=[],
            scenario=self.sample_scenario,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        # Should handle empty trace gracefully
        plan = self.plan_generator.generate_plan(empty_trace)
        assert isinstance(plan, ValidatedResolutionPlan)
        # May have empty steps but should still be valid
    
    def test_complex_scenario_handling(self):
        """Test handling of complex multi-factor scenarios."""
        complex_scenario = DisruptionScenario(
            description="Traffic jam and restaurant closed, customer unreachable",
            entities=[
                Entity(text="Pizza Palace", entity_type=EntityType.MERCHANT, confidence=0.9),
                Entity(text="555-1234", entity_type=EntityType.PHONE_NUMBER, confidence=0.8),
                Entity(text="Main Street", entity_type=EntityType.ADDRESS, confidence=0.7)
            ],
            scenario_type=ScenarioType.MULTI_FACTOR,
            urgency_level=UrgencyLevel.CRITICAL
        )
        
        complex_steps = [
            ReasoningStep(
                step_number=1,
                thought="Multiple issues detected - traffic and merchant problems",
                action="analyze_situation",
                observation="Both traffic and merchant availability are issues",
                timestamp=datetime.now()
            ),
            ReasoningStep(
                step_number=2,
                thought="Need to check alternative merchants and routes",
                action="find_alternatives",
                observation="Found alternative merchant and route",
                timestamp=datetime.now()
            ),
            ReasoningStep(
                step_number=3,
                thought="Must notify customer about changes and get approval",
                action="contact_customer",
                observation="Customer contacted and approved changes",
                timestamp=datetime.now()
            )
        ]
        
        complex_trace = ReasoningTrace(
            steps=complex_steps,
            scenario=complex_scenario,
            start_time=datetime.now() - timedelta(minutes=10),
            end_time=datetime.now()
        )
        
        plan = self.plan_generator.generate_plan(complex_trace)
        
        assert isinstance(plan, ValidatedResolutionPlan)
        assert len(plan.steps) > 0
        # Complex scenarios should have lower success probability
        assert plan.success_probability < 0.8
        assert len(plan.stakeholders) >= 2  # Multiple parties involved


if __name__ == '__main__':
    pytest.main([__file__])