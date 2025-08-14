"""
Plan generation system that converts reasoning traces to actionable resolution plans.
"""
from datetime import timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
import re

from ..agent.interfaces import ReasoningTrace, ResolutionPlan, PlanStep
from ..agent.models import ValidatedReasoningTrace, ValidatedResolutionPlan, ValidatedPlanStep, AlternativePlan
from ..tools.interfaces import ToolResult


class PlanGenerator:
    """
    Converts reasoning traces to actionable multi-step resolution plans.
    
    This class analyzes the reasoning trace to identify key actions, stakeholders,
    dependencies, and timing to create structured resolution plans with optimization
    and risk assessment capabilities.
    """
    
    def __init__(self):
        """Initialize the plan generator with stakeholder and action mappings."""
        # Mapping of action types to responsible parties
        self.action_stakeholders = {
            'notify': 'Customer Service',
            'contact': 'Customer Service', 
            'call': 'Customer Service',
            'message': 'Customer Service',
            'reroute': 'Dispatch',
            're_route': 'Dispatch',
            'redirect': 'Dispatch',
            'assign': 'Dispatch',
            'check': 'Operations',
            'verify': 'Operations',
            'investigate': 'Operations',
            'collect': 'Operations',
            'refund': 'Finance',
            'compensate': 'Finance',
            'credit': 'Finance',
            'update': 'System',
            'log': 'System',
            'record': 'System'
        }
        
        # Time estimates for different action types (in minutes)
        self.action_time_estimates = {
            'notify': 2,
            'contact': 5,
            'call': 10,
            'message': 2,
            'reroute': 15,
            're_route': 15,
            'redirect': 15,
            'assign': 10,
            'check': 5,
            'verify': 8,
            'investigate': 20,
            'collect': 15,
            'refund': 5,
            'compensate': 10,
            'credit': 5,
            'update': 3,
            'log': 2,
            'record': 2
        }
    
    def generate_plan(self, trace: ReasoningTrace) -> ValidatedResolutionPlan:
        """
        Generate a comprehensive resolution plan from a reasoning trace.
        
        Args:
            trace: Complete reasoning trace with all reasoning steps
            
        Returns:
            ValidatedResolutionPlan with actionable steps, timing, and stakeholders
        """
        # Extract actions from reasoning trace
        actions = self._extract_actions_from_trace(trace)
        
        # Create plan steps with dependencies and timing
        plan_steps = self._create_plan_steps(actions, trace)
        
        # Ensure we have at least one step for validation
        if not plan_steps:
            fallback_step = ValidatedPlanStep(
                sequence=1,
                action="Review scenario and determine appropriate action",
                responsible_party="Operations",
                estimated_time=timedelta(minutes=5),
                dependencies=[],
                success_criteria="Scenario reviewed and action plan determined"
            )
            plan_steps = [fallback_step]
        
        # Calculate total duration and success probability
        estimated_duration = self._calculate_total_duration(plan_steps)
        success_probability = self._estimate_success_probability(trace, plan_steps)
        
        # Identify all stakeholders
        stakeholders = self._identify_stakeholders(plan_steps)
        
        # Generate alternative plans
        alternatives = self._generate_alternatives(trace, plan_steps)
        
        # Optimize the plan based on urgency and impact
        optimized_steps = self._optimize_plan_steps(plan_steps, trace)
        
        # Recalculate duration and probability after optimization
        optimized_duration = self._calculate_total_duration(optimized_steps)
        optimized_probability = self._estimate_success_probability(trace, optimized_steps)
        
        return ValidatedResolutionPlan(
            steps=optimized_steps,
            estimated_duration=optimized_duration,
            success_probability=optimized_probability,
            alternatives=alternatives,
            stakeholders=stakeholders
        )
    
    def _extract_actions_from_trace(self, trace: ReasoningTrace) -> List[Dict]:
        """
        Extract actionable items from the reasoning trace.
        
        Args:
            trace: Reasoning trace to analyze
            
        Returns:
            List of action dictionaries with extracted information
        """
        actions = []
        
        for step in trace.steps:
            # Look for action indicators in thoughts and observations
            step_actions = self._parse_step_for_actions(step)
            actions.extend(step_actions)
        
        # Remove duplicates and merge similar actions
        actions = self._deduplicate_actions(actions)
        
        return actions
    
    def _parse_step_for_actions(self, step) -> List[Dict]:
        """
        Parse a single reasoning step for actionable items.
        
        Args:
            step: Individual reasoning step
            
        Returns:
            List of actions found in this step
        """
        actions = []
        text = f"{step.thought} {step.observation or ''}"
        
        # Common action patterns
        action_patterns = [
            r'(?:should|need to|must|will)\s+(\w+)\s+([^.!?]+)',
            r'(\w+)\s+(?:the|customer|driver|merchant)\s+([^.!?]+)',
            r'(?:action|step|next):\s*(\w+)\s+([^.!?]+)',
            r'(?:plan|strategy):\s*(\w+)\s+([^.!?]+)'
        ]
        
        for pattern in action_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                action_verb = match.group(1).lower()
                action_details = match.group(2).strip()
                
                # Skip non-actionable verbs
                if action_verb in ['is', 'are', 'was', 'were', 'have', 'has', 'had', 'can', 'could', 'would', 'should']:
                    continue
                
                actions.append({
                    'verb': action_verb,
                    'details': action_details,
                    'step_number': step.step_number,
                    'priority': self._determine_action_priority(action_verb, action_details)
                })
        
        return actions
    
    def _determine_action_priority(self, verb: str, details: str) -> int:
        """
        Determine action priority based on verb and details.
        
        Args:
            verb: Action verb
            details: Action details
            
        Returns:
            Priority score (1=highest, 5=lowest)
        """
        high_priority_verbs = ['notify', 'contact', 'call', 'refund', 'reroute']
        medium_priority_verbs = ['check', 'verify', 'assign']
        
        # Check for urgent indicators first
        if 'urgent' in details.lower() or 'immediate' in details.lower():
            return 1
        elif verb in high_priority_verbs:
            return 1
        elif verb in medium_priority_verbs:
            return 2
        elif 'customer' in details.lower():
            return 2
        else:
            return 3
    
    def _deduplicate_actions(self, actions: List[Dict]) -> List[Dict]:
        """
        Remove duplicate actions and merge similar ones.
        
        Args:
            actions: List of raw actions
            
        Returns:
            Deduplicated list of actions
        """
        seen_actions = {}
        deduplicated = []
        
        for action in actions:
            key = f"{action['verb']}_{action['details'][:20]}"
            
            if key not in seen_actions:
                seen_actions[key] = action
                deduplicated.append(action)
            else:
                # Merge with existing action (keep higher priority)
                existing = seen_actions[key]
                if action['priority'] < existing['priority']:
                    existing['priority'] = action['priority']
        
        # Sort by priority and step number
        deduplicated.sort(key=lambda x: (x['priority'], x['step_number']))
        
        return deduplicated
    
    def _create_plan_steps(self, actions: List[Dict], trace: ReasoningTrace) -> List[ValidatedPlanStep]:
        """
        Convert actions into structured plan steps with dependencies.
        
        Args:
            actions: List of extracted actions
            trace: Original reasoning trace for context
            
        Returns:
            List of validated plan steps
        """
        plan_steps = []
        
        for i, action in enumerate(actions):
            sequence = i + 1
            
            # Create action description
            action_description = self._format_action_description(action)
            
            # Determine responsible party
            responsible_party = self._determine_responsible_party(action['verb'])
            
            # Estimate time
            estimated_time = self._estimate_action_time(action['verb'], action['details'])
            
            # Determine dependencies
            dependencies = self._determine_dependencies(sequence, actions[:i], action)
            
            # Create success criteria
            success_criteria = self._create_success_criteria(action)
            
            plan_step = ValidatedPlanStep(
                sequence=sequence,
                action=action_description,
                responsible_party=responsible_party,
                estimated_time=estimated_time,
                dependencies=dependencies,
                success_criteria=success_criteria
            )
            
            plan_steps.append(plan_step)
        
        return plan_steps
    
    def _format_action_description(self, action: Dict) -> str:
        """
        Format action into a clear, actionable description.
        
        Args:
            action: Action dictionary
            
        Returns:
            Formatted action description
        """
        verb = action['verb'].capitalize()
        details = action['details'].strip()
        
        # Clean up the details
        details = re.sub(r'\s+', ' ', details)
        details = details.rstrip('.,!?')
        
        return f"{verb} {details}"
    
    def _determine_responsible_party(self, verb: str) -> str:
        """
        Determine who should be responsible for the action.
        
        Args:
            verb: Action verb
            
        Returns:
            Responsible party name
        """
        return self.action_stakeholders.get(verb.lower(), 'Operations')
    
    def _estimate_action_time(self, verb: str, details: str) -> timedelta:
        """
        Estimate time required for an action.
        
        Args:
            verb: Action verb
            details: Action details
            
        Returns:
            Estimated time as timedelta
        """
        base_time = self.action_time_estimates.get(verb.lower(), 10)
        
        # Adjust based on complexity indicators
        if 'complex' in details.lower() or 'investigate' in details.lower():
            base_time *= 2
        elif 'quick' in details.lower() or 'simple' in details.lower():
            base_time *= 0.5
        
        return timedelta(minutes=base_time)
    
    def _determine_dependencies(self, current_sequence: int, previous_actions: List[Dict], current_action: Dict) -> List[int]:
        """
        Determine which previous steps this action depends on.
        
        Args:
            current_sequence: Current step sequence number
            previous_actions: List of previous actions
            current_action: Current action being processed
            
        Returns:
            List of step sequence numbers this step depends on
        """
        dependencies = []
        current_verb = current_action['verb'].lower()
        
        # Define dependency rules
        dependency_rules = {
            'notify': ['check', 'verify', 'investigate'],
            'refund': ['collect', 'investigate', 'verify'],
            'reroute': ['check', 'verify'],
            'update': ['check', 'verify', 'investigate']
        }
        
        if current_verb in dependency_rules:
            required_verbs = dependency_rules[current_verb]
            
            for i, prev_action in enumerate(previous_actions):
                if prev_action['verb'].lower() in required_verbs:
                    dependencies.append(i + 1)  # Step sequences are 1-based
        
        return dependencies
    
    def _create_success_criteria(self, action: Dict) -> str:
        """
        Create success criteria for an action.
        
        Args:
            action: Action dictionary
            
        Returns:
            Success criteria description
        """
        verb = action['verb'].lower()
        
        criteria_templates = {
            'notify': 'Customer has been successfully contacted and informed',
            'contact': 'Communication established and information conveyed',
            'call': 'Phone call completed and issue discussed',
            'reroute': 'New route assigned and driver notified',
            'check': 'Information verified and status confirmed',
            'verify': 'Data accuracy confirmed and documented',
            'investigate': 'Root cause identified and documented',
            'refund': 'Refund processed and customer notified',
            'update': 'System records updated with current status'
        }
        
        return criteria_templates.get(verb, f'{action["verb"].capitalize()} action completed successfully')
    
    def _calculate_total_duration(self, plan_steps: List[ValidatedPlanStep]) -> timedelta:
        """
        Calculate total estimated duration considering dependencies.
        
        Args:
            plan_steps: List of plan steps
            
        Returns:
            Total estimated duration
        """
        if not plan_steps:
            return timedelta(0)
        
        # Build dependency graph
        step_times = {}
        for step in plan_steps:
            step_times[step.sequence] = step.estimated_time
        
        # Calculate critical path
        max_duration = timedelta(0)
        
        for step in plan_steps:
            # Calculate earliest start time based on dependencies
            earliest_start = timedelta(0)
            for dep in step.dependencies:
                if dep in step_times:
                    earliest_start = max(earliest_start, step_times[dep])
            
            # Total time for this path
            total_time = earliest_start + step.estimated_time
            max_duration = max(max_duration, total_time)
        
        return max_duration
    
    def _estimate_success_probability(self, trace: ReasoningTrace, plan_steps: List[ValidatedPlanStep]) -> float:
        """
        Estimate the probability of plan success.
        
        Args:
            trace: Reasoning trace
            plan_steps: Generated plan steps
            
        Returns:
            Success probability between 0.0 and 1.0
        """
        base_probability = 0.8  # Start with 80% base probability
        
        # Adjust based on scenario complexity
        scenario = trace.scenario
        
        if scenario.urgency_level.value == 'critical':
            base_probability -= 0.1
        elif scenario.urgency_level.value == 'low':
            base_probability += 0.1
        
        if scenario.scenario_type.value == 'multi_factor':
            base_probability -= 0.15
        
        # Adjust based on plan complexity
        if len(plan_steps) > 8:
            base_probability -= 0.1
        elif len(plan_steps) < 4:
            base_probability += 0.05
        
        # Adjust based on reasoning quality (number of steps and tool usage)
        if len(trace.steps) > 10:  # Thorough reasoning
            base_probability += 0.05
        elif len(trace.steps) < 3:  # Insufficient reasoning
            base_probability -= 0.1
        
        # Ensure probability stays within bounds
        return max(0.1, min(0.95, base_probability))
    
    def _identify_stakeholders(self, plan_steps: List[ValidatedPlanStep]) -> List[str]:
        """
        Identify all stakeholders involved in the plan.
        
        Args:
            plan_steps: List of plan steps
            
        Returns:
            List of unique stakeholder names
        """
        stakeholders = set()
        
        for step in plan_steps:
            stakeholders.add(step.responsible_party)
        
        # Always include customer as stakeholder
        stakeholders.add('Customer')
        
        return sorted(list(stakeholders))
    
    def _generate_alternatives(self, trace: ReasoningTrace, primary_steps: List[ValidatedPlanStep]) -> List[AlternativePlan]:
        """
        Generate alternative resolution plans.
        
        Args:
            trace: Reasoning trace
            primary_steps: Primary plan steps
            
        Returns:
            List of alternative plans
        """
        alternatives = []
        
        # Generate a faster but potentially less thorough alternative
        if len(primary_steps) > 3:
            fast_alternative = self._create_fast_alternative(primary_steps)
            alternatives.append(fast_alternative)
        
        # Generate a more thorough but slower alternative
        thorough_alternative = self._create_thorough_alternative(trace, primary_steps)
        alternatives.append(thorough_alternative)
        
        return alternatives
    
    def _create_fast_alternative(self, primary_steps: List[ValidatedPlanStep]) -> AlternativePlan:
        """
        Create a faster alternative plan by combining or skipping steps.
        
        Args:
            primary_steps: Primary plan steps
            
        Returns:
            Alternative plan focused on speed
        """
        # Keep only high-priority actions
        essential_steps = []
        sequence = 1
        
        for step in primary_steps:
            # Keep customer-facing actions and critical operations
            if any(keyword in step.action.lower() for keyword in ['notify', 'contact', 'refund', 'reroute']):
                fast_step = ValidatedPlanStep(
                    sequence=sequence,
                    action=step.action,
                    responsible_party=step.responsible_party,
                    estimated_time=step.estimated_time * 0.7,  # 30% faster
                    dependencies=[],  # Remove dependencies for speed
                    success_criteria=step.success_criteria
                )
                essential_steps.append(fast_step)
                sequence += 1
        
        total_duration = sum((step.estimated_time for step in essential_steps), timedelta(0))
        
        # Ensure minimum duration for validation
        if total_duration.total_seconds() <= 0:
            total_duration = timedelta(minutes=2)  # Minimum 2 minutes for fast resolution
        
        return AlternativePlan(
            name="Fast Resolution",
            description="Prioritizes immediate customer impact resolution with minimal verification steps",
            steps=essential_steps,
            estimated_duration=total_duration,
            success_probability=0.7,  # Lower due to reduced verification
            trade_offs=[
                "Reduced verification may lead to incomplete resolution",
                "Faster customer response but higher risk of follow-up issues",
                "May require additional steps if initial resolution fails"
            ]
        )
    
    def _create_thorough_alternative(self, trace: ReasoningTrace, primary_steps: List[ValidatedPlanStep]) -> AlternativePlan:
        """
        Create a more thorough alternative plan with additional verification.
        
        Args:
            trace: Reasoning trace
            primary_steps: Primary plan steps
            
        Returns:
            Alternative plan focused on thoroughness
        """
        thorough_steps = []
        sequence = 1
        
        # Add verification steps before key actions
        for step in primary_steps:
            # Add verification step before customer-facing actions
            if any(keyword in step.action.lower() for keyword in ['notify', 'contact', 'refund']):
                verify_step = ValidatedPlanStep(
                    sequence=sequence,
                    action=f"Verify information before {step.action.lower()}",
                    responsible_party="Operations",
                    estimated_time=timedelta(minutes=5),
                    dependencies=[],
                    success_criteria="All relevant information verified and documented"
                )
                thorough_steps.append(verify_step)
                sequence += 1
            
            # Add the original step with updated sequence and dependencies
            thorough_step = ValidatedPlanStep(
                sequence=sequence,
                action=step.action,
                responsible_party=step.responsible_party,
                estimated_time=step.estimated_time * 1.2,  # 20% longer for thoroughness
                dependencies=[sequence - 1] if thorough_steps else [],
                success_criteria=step.success_criteria
            )
            thorough_steps.append(thorough_step)
            sequence += 1
        
        # Add follow-up verification step only if there are previous steps
        if thorough_steps:
            followup_step = ValidatedPlanStep(
                sequence=sequence,
                action="Follow up with customer to confirm resolution satisfaction",
                responsible_party="Customer Service",
                estimated_time=timedelta(minutes=10),
                dependencies=[sequence - 1],
                success_criteria="Customer confirms issue is fully resolved"
            )
            thorough_steps.append(followup_step)
        
        total_duration = sum((step.estimated_time for step in thorough_steps), timedelta(0))
        
        # Ensure minimum duration for validation
        if total_duration.total_seconds() <= 0:
            total_duration = timedelta(minutes=5)  # Minimum 5 minutes
        
        return AlternativePlan(
            name="Comprehensive Resolution",
            description="Includes additional verification and follow-up steps to ensure complete resolution",
            steps=thorough_steps,
            estimated_duration=total_duration,
            success_probability=0.9,  # Higher due to additional verification
            trade_offs=[
                "Longer resolution time may impact customer satisfaction",
                "Higher resource utilization across multiple departments",
                "More thorough but potentially over-engineered for simple issues"
            ]
        )
    
    def _optimize_plan_steps(self, plan_steps: List[ValidatedPlanStep], trace: ReasoningTrace) -> List[ValidatedPlanStep]:
        """
        Optimize plan steps based on urgency, impact, and resource efficiency.
        
        Args:
            plan_steps: Original plan steps
            trace: Reasoning trace for context
            
        Returns:
            Optimized list of plan steps
        """
        if not plan_steps:
            return plan_steps
        
        # Create a copy to avoid modifying original
        optimized_steps = [step for step in plan_steps]
        
        # Apply optimization strategies based on scenario urgency
        urgency = trace.scenario.urgency_level
        
        if urgency.value == 'critical':
            optimized_steps = self._optimize_for_speed(optimized_steps)
        elif urgency.value == 'high':
            optimized_steps = self._optimize_for_balance(optimized_steps)
        else:
            optimized_steps = self._optimize_for_thoroughness(optimized_steps)
        
        # Apply impact-based prioritization
        optimized_steps = self._prioritize_by_impact(optimized_steps, trace)
        
        # Optimize resource allocation
        optimized_steps = self._optimize_resource_allocation(optimized_steps)
        
        return optimized_steps
    
    def _optimize_for_speed(self, steps: List[ValidatedPlanStep]) -> List[ValidatedPlanStep]:
        """
        Optimize plan for maximum speed in critical situations.
        
        Args:
            steps: Original plan steps
            
        Returns:
            Speed-optimized plan steps
        """
        # Create copies to avoid modifying originals
        optimized_steps = []
        for step in steps:
            new_step = ValidatedPlanStep(
                sequence=step.sequence,
                action=step.action,
                responsible_party=step.responsible_party,
                estimated_time=step.estimated_time * 0.8,  # Reduce time by 20%
                dependencies=step.dependencies.copy(),
                success_criteria=step.success_criteria,
                status=step.status,
                actual_duration=step.actual_duration,
                notes=step.notes
            )
            optimized_steps.append(new_step)
        
        steps = optimized_steps
        
        # Remove non-essential dependencies to enable parallel execution
        essential_dependencies = {}
        for step in steps:
            # Keep only direct dependencies for critical actions
            if any(keyword in step.action.lower() for keyword in ['notify', 'refund', 'reroute']):
                # Keep dependencies that are verification steps
                essential_deps = []
                for dep in step.dependencies:
                    dep_step = next((s for s in steps if s.sequence == dep), None)
                    if dep_step and any(keyword in dep_step.action.lower() for keyword in ['check', 'verify']):
                        essential_deps.append(dep)
                step.dependencies = essential_deps
            else:
                # For non-critical actions, remove most dependencies
                step.dependencies = []
        
        return steps
    
    def _optimize_for_balance(self, steps: List[ValidatedPlanStep]) -> List[ValidatedPlanStep]:
        """
        Optimize plan for balanced speed and thoroughness.
        
        Args:
            steps: Original plan steps
            
        Returns:
            Balanced plan steps
        """
        # Create copies to avoid modifying originals
        optimized_steps = []
        for step in steps:
            new_step = ValidatedPlanStep(
                sequence=step.sequence,
                action=step.action,
                responsible_party=step.responsible_party,
                estimated_time=step.estimated_time * 0.9,  # Slight time reduction (10%)
                dependencies=step.dependencies.copy(),
                success_criteria=step.success_criteria,
                status=step.status,
                actual_duration=step.actual_duration,
                notes=step.notes
            )
            optimized_steps.append(new_step)
        
        steps = optimized_steps
        
        # Keep essential dependencies but streamline others
        for step in steps:
            if len(step.dependencies) > 2:
                # Keep only the most recent dependencies
                step.dependencies = step.dependencies[-2:]
        
        return steps
    
    def _optimize_for_thoroughness(self, steps: List[ValidatedPlanStep]) -> List[ValidatedPlanStep]:
        """
        Optimize plan for maximum thoroughness and verification.
        
        Args:
            steps: Original plan steps
            
        Returns:
            Thoroughness-optimized plan steps
        """
        # Create copies to avoid modifying originals
        optimized_steps = []
        for step in steps:
            new_step = ValidatedPlanStep(
                sequence=step.sequence,
                action=step.action,
                responsible_party=step.responsible_party,
                estimated_time=step.estimated_time * 1.1,  # Add buffer time (10%)
                dependencies=step.dependencies.copy(),
                success_criteria=step.success_criteria,
                status=step.status,
                actual_duration=step.actual_duration,
                notes=step.notes
            )
            optimized_steps.append(new_step)
        
        steps = optimized_steps
        
        # Ensure all dependencies are maintained
        # (No changes needed as original dependencies are already comprehensive)
        
        return steps
    
    def _prioritize_by_impact(self, steps: List[ValidatedPlanStep], trace: ReasoningTrace) -> List[ValidatedPlanStep]:
        """
        Reorder steps based on their impact on resolution success.
        
        Args:
            steps: Plan steps to prioritize
            trace: Reasoning trace for context
            
        Returns:
            Impact-prioritized plan steps
        """
        # Define impact scores for different action types
        impact_scores = {
            'notify': 9,    # High customer impact
            'contact': 9,   # High customer impact
            'refund': 8,    # High customer satisfaction impact
            'reroute': 7,   # High operational impact
            'check': 6,     # Medium operational impact
            'verify': 6,    # Medium operational impact
            'update': 4,    # Low operational impact
            'log': 2        # Low administrative impact
        }
        
        # Calculate impact score for each step
        for step in steps:
            action_verb = step.action.split()[0].lower()
            base_impact = impact_scores.get(action_verb, 5)  # Default medium impact
            
            # Boost impact for customer-facing actions
            if 'customer' in step.action.lower():
                base_impact += 2
            
            # Boost impact for time-sensitive actions
            if any(keyword in step.action.lower() for keyword in ['urgent', 'immediate', 'asap']):
                base_impact += 1
            
            step.impact_score = base_impact
        
        # Sort by impact score (descending) while respecting dependencies
        return self._topological_sort_with_impact(steps)
    
    def _topological_sort_with_impact(self, steps: List[ValidatedPlanStep]) -> List[ValidatedPlanStep]:
        """
        Sort steps topologically while considering impact scores.
        
        Args:
            steps: Steps to sort
            
        Returns:
            Sorted steps respecting dependencies and impact
        """
        # Create adjacency list for dependencies
        graph = {step.sequence: [] for step in steps}
        in_degree = {step.sequence: 0 for step in steps}
        step_map = {step.sequence: step for step in steps}
        
        # Build dependency graph
        for step in steps:
            for dep in step.dependencies:
                if dep in graph:
                    graph[dep].append(step.sequence)
                    in_degree[step.sequence] += 1
        
        # Modified topological sort considering impact
        result = []
        available = [seq for seq, degree in in_degree.items() if degree == 0]
        
        while available:
            # Sort available steps by impact score (descending)
            available.sort(key=lambda seq: getattr(step_map[seq], 'impact_score', 5), reverse=True)
            
            # Take the highest impact available step
            current = available.pop(0)
            result.append(step_map[current])
            
            # Update dependencies
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    available.append(neighbor)
        
        # Update sequence numbers to match new order
        for i, step in enumerate(result):
            step.sequence = i + 1
            # Update dependencies to match new sequence numbers
            new_dependencies = []
            for dep_seq in step.dependencies:
                dep_step = step_map[dep_seq]
                new_dep_seq = next(i + 1 for i, s in enumerate(result) if s == dep_step)
                new_dependencies.append(new_dep_seq)
            step.dependencies = new_dependencies
        
        return result
    
    def _optimize_resource_allocation(self, steps: List[ValidatedPlanStep]) -> List[ValidatedPlanStep]:
        """
        Optimize resource allocation across steps to minimize bottlenecks.
        
        Args:
            steps: Plan steps to optimize
            
        Returns:
            Resource-optimized plan steps
        """
        # Count resource usage by responsible party
        resource_usage = {}
        for step in steps:
            party = step.responsible_party
            if party not in resource_usage:
                resource_usage[party] = []
            resource_usage[party].append(step)
        
        # Identify overloaded resources
        overloaded_threshold = 3  # More than 3 steps per party
        overloaded_parties = {party: steps_list for party, steps_list in resource_usage.items() 
                             if len(steps_list) > overloaded_threshold}
        
        # Redistribute some tasks if possible
        for party, party_steps in overloaded_parties.items():
            # Find steps that could be reassigned
            reassignable_steps = [step for step in party_steps 
                                if any(keyword in step.action.lower() for keyword in ['update', 'log', 'record'])]
            
            # Reassign some administrative tasks to 'System'
            for step in reassignable_steps[:2]:  # Reassign up to 2 steps
                if step.responsible_party != 'System':
                    step.responsible_party = 'System'
                    # Reduce time estimate for automated tasks
                    step.estimated_time = step.estimated_time * 0.5
        
        return steps
    
    def assess_plan_risks(self, plan: ValidatedResolutionPlan, trace: ReasoningTrace) -> Dict[str, Any]:
        """
        Assess risks associated with the resolution plan.
        
        Args:
            plan: Resolution plan to assess
            trace: Original reasoning trace
            
        Returns:
            Risk assessment dictionary
        """
        risks = {
            'overall_risk_level': 'medium',
            'risk_factors': [],
            'mitigation_strategies': [],
            'confidence_intervals': {},
            'failure_scenarios': []
        }
        
        # Assess complexity risk
        if len(plan.steps) > 8:
            risks['risk_factors'].append('High plan complexity with many steps')
            risks['mitigation_strategies'].append('Consider breaking into phases')
        
        # Assess dependency risk
        max_dependencies = max(len(step.dependencies) for step in plan.steps) if plan.steps else 0
        if max_dependencies > 2:
            risks['risk_factors'].append('Complex dependency chains')
            risks['mitigation_strategies'].append('Identify critical path and monitor closely')
        
        # Assess resource risk
        resource_counts = {}
        for step in plan.steps:
            party = step.responsible_party
            resource_counts[party] = resource_counts.get(party, 0) + 1
        
        max_resource_load = max(resource_counts.values()) if resource_counts else 0
        if max_resource_load > 4:
            overloaded_party = max(resource_counts, key=resource_counts.get)
            risks['risk_factors'].append(f'Resource bottleneck: {overloaded_party} has {max_resource_load} tasks')
            risks['mitigation_strategies'].append('Consider task redistribution or additional resources')
        
        # Assess time risk
        total_time = plan.estimated_duration.total_seconds() / 60  # Convert to minutes
        if total_time > 60:  # More than 1 hour
            risks['risk_factors'].append('Extended resolution time may impact customer satisfaction')
            risks['mitigation_strategies'].append('Provide regular customer updates')
        
        # Assess scenario-specific risks
        scenario_type = trace.scenario.scenario_type.value
        urgency = trace.scenario.urgency_level.value
        
        if scenario_type == 'multi_factor':
            risks['risk_factors'].append('Multi-factor scenario increases complexity')
            risks['mitigation_strategies'].append('Assign dedicated coordinator for multi-factor issues')
        
        if urgency == 'critical':
            risks['risk_factors'].append('Critical urgency increases pressure and error risk')
            risks['mitigation_strategies'].append('Implement additional verification checkpoints')
        
        # Calculate overall risk level
        risk_score = len(risks['risk_factors'])
        if risk_score >= 4:
            risks['overall_risk_level'] = 'high'
        elif risk_score <= 1:
            risks['overall_risk_level'] = 'low'
        
        # Add confidence intervals
        base_probability = plan.success_probability
        risks['confidence_intervals'] = {
            'optimistic': min(0.95, base_probability + 0.1),
            'realistic': base_probability,
            'pessimistic': max(0.1, base_probability - 0.15)
        }
        
        # Identify potential failure scenarios
        if any('notify' in step.action.lower() for step in plan.steps):
            risks['failure_scenarios'].append('Customer communication failure')
        
        if any('reroute' in step.action.lower() for step in plan.steps):
            risks['failure_scenarios'].append('Alternative route unavailable')
        
        if any('refund' in step.action.lower() for step in plan.steps):
            risks['failure_scenarios'].append('Payment processing issues')
        
        return risks
    
    def generate_plan_quality_metrics(self, plan: ValidatedResolutionPlan, trace: ReasoningTrace) -> Dict[str, float]:
        """
        Generate quality metrics for the resolution plan.
        
        Args:
            plan: Resolution plan to evaluate
            trace: Original reasoning trace
            
        Returns:
            Dictionary of quality metrics (0.0 to 1.0 scale)
        """
        metrics = {}
        
        # Completeness: How well the plan addresses the scenario
        metrics['completeness'] = self._assess_completeness(plan, trace)
        
        # Efficiency: Time and resource optimization
        metrics['efficiency'] = self._assess_efficiency(plan)
        
        # Feasibility: How realistic the plan is to execute
        metrics['feasibility'] = self._assess_feasibility(plan)
        
        # Customer Impact: How well the plan serves customer needs
        metrics['customer_impact'] = self._assess_customer_impact(plan)
        
        # Risk Management: How well risks are mitigated
        metrics['risk_management'] = self._assess_risk_management(plan, trace)
        
        # Overall Quality Score
        metrics['overall_quality'] = sum(metrics.values()) / len(metrics)
        
        return metrics
    
    def _assess_completeness(self, plan: ValidatedResolutionPlan, trace: ReasoningTrace) -> float:
        """Assess how completely the plan addresses the scenario."""
        scenario_type = trace.scenario.scenario_type.value
        required_actions = {
            'traffic': ['check', 'reroute', 'notify'],
            'merchant': ['check', 'contact', 'notify'],
            'address': ['verify', 'contact', 'update'],
            'multi_factor': ['check', 'verify', 'notify', 'contact']
        }
        
        expected_actions = required_actions.get(scenario_type, ['check', 'notify'])
        plan_actions = [step.action.lower() for step in plan.steps]
        
        covered_actions = sum(1 for action in expected_actions 
                            if any(action in plan_action for plan_action in plan_actions))
        
        return covered_actions / len(expected_actions) if expected_actions else 1.0
    
    def _assess_efficiency(self, plan: ValidatedResolutionPlan) -> float:
        """Assess the efficiency of the plan."""
        if not plan.steps:
            return 0.0
        
        # Base efficiency on step count and time
        step_efficiency = max(0.0, 1.0 - (len(plan.steps) - 3) * 0.1)  # Penalty for too many steps
        time_efficiency = max(0.0, 1.0 - (plan.estimated_duration.total_seconds() / 3600 - 0.5) * 0.2)  # Penalty for > 30 min
        
        return (step_efficiency + time_efficiency) / 2
    
    def _assess_feasibility(self, plan: ValidatedResolutionPlan) -> float:
        """Assess how feasible the plan is to execute."""
        if not plan.steps:
            return 0.0
        
        # Check for realistic time estimates
        unrealistic_steps = sum(1 for step in plan.steps 
                              if step.estimated_time.total_seconds() < 60 or step.estimated_time.total_seconds() > 3600)
        time_feasibility = 1.0 - (unrealistic_steps / len(plan.steps))
        
        # Check for reasonable dependencies
        max_deps = max(len(step.dependencies) for step in plan.steps)
        dependency_feasibility = max(0.0, 1.0 - (max_deps - 2) * 0.2)
        
        return (time_feasibility + dependency_feasibility) / 2
    
    def _assess_customer_impact(self, plan: ValidatedResolutionPlan) -> float:
        """Assess how well the plan serves customer needs."""
        if not plan.steps:
            return 0.0
        
        customer_facing_steps = sum(1 for step in plan.steps 
                                  if any(keyword in step.action.lower() 
                                       for keyword in ['notify', 'contact', 'refund']))
        
        # More specific customer communication gets higher score
        update_steps = sum(1 for step in plan.steps 
                          if 'update' in step.action.lower() and 'customer' in step.action.lower())
        
        total_customer_steps = customer_facing_steps + update_steps
        
        # Higher score for plans with appropriate customer communication
        customer_ratio = total_customer_steps / len(plan.steps)
        
        # Cap at 1.0, optimal around 33% customer-facing actions
        return min(1.0, customer_ratio * 3)
    
    def _assess_risk_management(self, plan: ValidatedResolutionPlan, trace: ReasoningTrace) -> float:
        """Assess how well the plan manages risks."""
        risk_assessment = self.assess_plan_risks(plan, trace)
        risk_level = risk_assessment['overall_risk_level']
        
        risk_scores = {'low': 0.9, 'medium': 0.7, 'high': 0.4}
        base_score = risk_scores.get(risk_level, 0.5)
        
        # Bonus for having mitigation strategies
        mitigation_bonus = min(0.1, len(risk_assessment['mitigation_strategies']) * 0.02)
        
        return min(1.0, base_score + mitigation_bonus)