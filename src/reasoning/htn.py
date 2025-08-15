"""
Hierarchical Task Network (HTN) Planner for Delivery Coordination

This module implements an HTN planner that decomposes high-level goals into
executable action sequences using domain-specific operators and methods.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from ..agent.models import ValidatedDisruptionScenario, ScenarioType, UrgencyLevel
from ..tools.interfaces import ToolResult

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Types of tasks in the HTN hierarchy."""
    PRIMITIVE = "primitive"
    COMPOUND = "compound"
    
class OperatorType(Enum):
    """Types of HTN operators."""
    TRAFFIC_COORDINATION = "traffic_coordination"
    MERCHANT_MANAGEMENT = "merchant_management"
    CUSTOMER_SERVICE = "customer_service"
    MEDIATION_FLOW = "mediation_flow"
    EMERGENCY_RESPONSE = "emergency_response"

@dataclass
class HTNTask:
    """Represents a task in the HTN hierarchy."""
    name: str
    task_type: TaskType
    parameters: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    estimated_duration: Optional[timedelta] = None
    priority: int = 1  # 1=low, 5=critical
    
    def __hash__(self):
        return hash((self.name, str(sorted(self.parameters.items()))))

@dataclass
class HTNMethod:
    """Represents a method for decomposing compound tasks."""
    name: str
    task_name: str  # The compound task this method decomposes
    preconditions: List[str] = field(default_factory=list)
    subtasks: List[HTNTask] = field(default_factory=list)
    ordering_constraints: List[Tuple[str, str]] = field(default_factory=list)  # (before, after)
    success_probability: float = 0.8
    
@dataclass
class HTNOperator:
    """Represents a primitive operator that can be executed."""
    name: str
    parameters: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    tool_name: Optional[str] = None
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    success_probability: float = 0.9
    operator_type: OperatorType = OperatorType.TRAFFIC_COORDINATION

@dataclass
class HTNPlan:
    """Represents a complete HTN plan."""
    tasks: List[HTNTask] = field(default_factory=list)
    dependencies: List[Tuple[str, str]] = field(default_factory=list)  # (prerequisite, dependent)
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    success_probability: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HTNState:
    """Represents the current state of the world for HTN planning."""
    facts: Set[str] = field(default_factory=set)
    variables: Dict[str, Any] = field(default_factory=dict)
    tool_results: List[ToolResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def add_fact(self, fact: str):
        """Add a fact to the current state."""
        self.facts.add(fact)
        
    def remove_fact(self, fact: str):
        """Remove a fact from the current state."""
        self.facts.discard(fact)
        
    def has_fact(self, fact: str) -> bool:
        """Check if a fact exists in the current state."""
        return fact in self.facts
        
    def update_from_tool_results(self, tool_results: List[ToolResult]):
        """Update state based on tool execution results."""
        self.tool_results = tool_results
        
        # Extract facts from successful tool results
        for result in tool_results:
            if result.success:
                # Add success facts
                self.add_fact(f"tool_{result.tool_name}_succeeded")
                
                # Extract specific facts based on tool type
                if result.tool_name == "get_traffic_status":
                    if result.data and "severity" in result.data:
                        severity = result.data["severity"]
                        self.add_fact(f"traffic_severity_{severity}")
                        
                elif result.tool_name == "get_merchant_status":
                    if result.data and "status" in result.data:
                        status = result.data["status"]
                        self.add_fact(f"merchant_status_{status}")
                        
                elif result.tool_name == "collect_evidence":
                    self.add_fact("evidence_collected")
                    if result.data and "evidence_count" in result.data:
                        count = result.data["evidence_count"]
                        if count > 3:
                            self.add_fact("sufficient_evidence")
            else:
                # Add failure facts
                self.add_fact(f"tool_{result.tool_name}_failed")

class HTNPlanner:
    """Hierarchical Task Network planner for delivery coordination."""
    
    def __init__(self):
        self.operators: Dict[str, HTNOperator] = {}
        self.methods: Dict[str, List[HTNMethod]] = {}  # task_name -> methods
        self.state = HTNState()
        
        logger.info("HTN Planner initialized with domain knowledge")
        self._initialize_domain()
    
    def _initialize_domain(self):
        """Initialize the HTN domain with operators and methods."""
        self._create_traffic_operators()
        self._create_merchant_operators()
        self._create_customer_operators()
        self._create_mediation_operators()
        self._create_emergency_operators()
        self._create_methods()
    
    def _create_traffic_operators(self):
        """Create operators for traffic-related tasks."""
        operators = [
            HTNOperator(
                name="assess_traffic_situation",
                parameters=["location", "severity"],
                preconditions=[],
                effects=["traffic_assessed", "situation_known"],
                tool_name="get_traffic_status",
                estimated_duration=timedelta(minutes=2),
                success_probability=0.95,
                operator_type=OperatorType.TRAFFIC_COORDINATION
            ),
            HTNOperator(
                name="calculate_alternative_route",
                parameters=["origin", "destination", "avoid_areas"],
                preconditions=["traffic_assessed"],
                effects=["alternative_route_calculated", "eta_updated"],
                tool_name="calculate_route",
                estimated_duration=timedelta(minutes=3),
                success_probability=0.9,
                operator_type=OperatorType.TRAFFIC_COORDINATION
            ),
            HTNOperator(
                name="notify_customer_delay",
                parameters=["customer_id", "delay_estimate", "reason"],
                preconditions=["situation_known"],
                effects=["customer_notified", "expectations_managed"],
                tool_name="notify_customer",
                estimated_duration=timedelta(minutes=1),
                success_probability=0.95,
                operator_type=OperatorType.CUSTOMER_SERVICE
            ),
            HTNOperator(
                name="coordinate_driver_reroute",
                parameters=["driver_id", "new_route"],
                preconditions=["alternative_route_calculated"],
                effects=["driver_rerouted", "delivery_optimized"],
                tool_name="coordinate_replacement",
                estimated_duration=timedelta(minutes=4),
                success_probability=0.85,
                operator_type=OperatorType.TRAFFIC_COORDINATION
            )
        ]
        
        for op in operators:
            self.operators[op.name] = op
    
    def _create_merchant_operators(self):
        """Create operators for merchant-related tasks."""
        operators = [
            HTNOperator(
                name="assess_merchant_capacity",
                parameters=["merchant_id"],
                preconditions=[],
                effects=["merchant_assessed", "capacity_known"],
                tool_name="get_merchant_status",
                estimated_duration=timedelta(minutes=2),
                success_probability=0.9,
                operator_type=OperatorType.MERCHANT_MANAGEMENT
            ),
            HTNOperator(
                name="find_alternative_merchants",
                parameters=["location", "cuisine_type", "capacity_needed"],
                preconditions=["merchant_assessed"],
                effects=["alternatives_identified", "backup_options_available"],
                tool_name="get_nearby_merchants",
                estimated_duration=timedelta(minutes=3),
                success_probability=0.8,
                operator_type=OperatorType.MERCHANT_MANAGEMENT
            ),
            HTNOperator(
                name="coordinate_merchant_replacement",
                parameters=["original_merchant", "replacement_merchant", "orders"],
                preconditions=["alternatives_identified"],
                effects=["merchant_replaced", "orders_transferred"],
                tool_name="coordinate_replacement",
                estimated_duration=timedelta(minutes=8),
                success_probability=0.75,
                operator_type=OperatorType.MERCHANT_MANAGEMENT
            )
        ]
        
        for op in operators:
            self.operators[op.name] = op
    
    def _create_customer_operators(self):
        """Create operators for customer service tasks."""
        operators = [
            HTNOperator(
                name="collect_customer_feedback",
                parameters=["customer_id", "incident_id"],
                preconditions=[],
                effects=["feedback_collected", "customer_voice_heard"],
                tool_name="collect_evidence",
                estimated_duration=timedelta(minutes=5),
                success_probability=0.85,
                operator_type=OperatorType.CUSTOMER_SERVICE
            ),
            HTNOperator(
                name="provide_instant_resolution",
                parameters=["customer_id", "resolution_type", "amount"],
                preconditions=["feedback_collected"],
                effects=["customer_satisfied", "resolution_provided"],
                tool_name="issue_instant_refund",
                estimated_duration=timedelta(minutes=2),
                success_probability=0.95,
                operator_type=OperatorType.CUSTOMER_SERVICE
            )
        ]
        
        for op in operators:
            self.operators[op.name] = op
    
    def _create_mediation_operators(self):
        """Create operators for dispute mediation."""
        operators = [
            HTNOperator(
                name="initiate_mediation_process",
                parameters=["order_id", "dispute_type"],
                preconditions=[],
                effects=["mediation_initiated", "process_started"],
                tool_name="initiate_mediation_flow",
                estimated_duration=timedelta(minutes=3),
                success_probability=0.9,
                operator_type=OperatorType.MEDIATION_FLOW
            ),
            HTNOperator(
                name="collect_dispute_evidence",
                parameters=["order_id", "parties"],
                preconditions=["mediation_initiated"],
                effects=["evidence_collected", "facts_gathered"],
                tool_name="collect_evidence",
                estimated_duration=timedelta(minutes=7),
                success_probability=0.8,
                operator_type=OperatorType.MEDIATION_FLOW
            ),
            HTNOperator(
                name="analyze_evidence_objectively",
                parameters=["evidence_items", "dispute_type"],
                preconditions=["evidence_collected"],
                effects=["evidence_analyzed", "fault_determined"],
                tool_name="analyze_evidence",
                estimated_duration=timedelta(minutes=5),
                success_probability=0.85,
                operator_type=OperatorType.MEDIATION_FLOW
            ),
            HTNOperator(
                name="execute_mediated_resolution",
                parameters=["resolution_type", "parties", "compensation"],
                preconditions=["evidence_analyzed"],
                effects=["dispute_resolved", "parties_satisfied"],
                tool_name="issue_instant_refund",
                estimated_duration=timedelta(minutes=3),
                success_probability=0.9,
                operator_type=OperatorType.MEDIATION_FLOW
            )
        ]
        
        for op in operators:
            self.operators[op.name] = op
    
    def _create_emergency_operators(self):
        """Create operators for emergency response."""
        operators = [
            HTNOperator(
                name="activate_emergency_protocols",
                parameters=["incident_type", "severity"],
                preconditions=[],
                effects=["emergency_activated", "protocols_engaged"],
                tool_name="escalate_to_support",
                estimated_duration=timedelta(minutes=2),
                success_probability=0.95,
                operator_type=OperatorType.EMERGENCY_RESPONSE
            ),
            HTNOperator(
                name="coordinate_multi_stakeholder_response",
                parameters=["stakeholders", "incident_id"],
                preconditions=["emergency_activated"],
                effects=["stakeholders_coordinated", "unified_response"],
                tool_name="coordinate_replacement",
                estimated_duration=timedelta(minutes=10),
                success_probability=0.8,
                operator_type=OperatorType.EMERGENCY_RESPONSE
            )
        ]
        
        for op in operators:
            self.operators[op.name] = op
    
    def _create_methods(self):
        """Create methods for decomposing compound tasks."""
        
        # Traffic disruption handling method
        traffic_method = HTNMethod(
            name="handle_traffic_disruption",
            task_name="resolve_traffic_disruption",
            preconditions=[],
            subtasks=[
                HTNTask("assess_traffic_situation", TaskType.PRIMITIVE, 
                       {"location": "incident_location", "severity": "unknown"}),
                HTNTask("calculate_alternative_route", TaskType.PRIMITIVE,
                       {"origin": "current_location", "destination": "delivery_address"}),
                HTNTask("notify_customer_delay", TaskType.PRIMITIVE,
                       {"reason": "traffic_incident"}),
                HTNTask("coordinate_driver_reroute", TaskType.PRIMITIVE,
                       {"new_route": "alternative_route"})
            ],
            ordering_constraints=[
                ("assess_traffic_situation", "calculate_alternative_route"),
                ("assess_traffic_situation", "notify_customer_delay"),
                ("calculate_alternative_route", "coordinate_driver_reroute")
            ],
            success_probability=0.85
        )
        
        # Merchant disruption handling method
        merchant_method = HTNMethod(
            name="handle_merchant_disruption",
            task_name="resolve_merchant_disruption",
            preconditions=[],
            subtasks=[
                HTNTask("assess_merchant_capacity", TaskType.PRIMITIVE,
                       {"merchant_id": "affected_merchant"}),
                HTNTask("find_alternative_merchants", TaskType.PRIMITIVE,
                       {"location": "merchant_area", "capacity_needed": "current_orders"}),
                HTNTask("coordinate_merchant_replacement", TaskType.PRIMITIVE,
                       {"orders": "affected_orders"}),
                HTNTask("notify_customer_delay", TaskType.PRIMITIVE,
                       {"reason": "merchant_issue"})
            ],
            ordering_constraints=[
                ("assess_merchant_capacity", "find_alternative_merchants"),
                ("find_alternative_merchants", "coordinate_merchant_replacement"),
                ("assess_merchant_capacity", "notify_customer_delay")
            ],
            success_probability=0.75
        )
        
        # Dispute mediation method
        mediation_method = HTNMethod(
            name="handle_dispute_mediation",
            task_name="resolve_customer_dispute",
            preconditions=[],
            subtasks=[
                HTNTask("initiate_mediation_process", TaskType.PRIMITIVE,
                       {"dispute_type": "customer_complaint"}),
                HTNTask("collect_dispute_evidence", TaskType.PRIMITIVE,
                       {"parties": "customer_merchant"}),
                HTNTask("analyze_evidence_objectively", TaskType.PRIMITIVE,
                       {"dispute_type": "service_quality"}),
                HTNTask("execute_mediated_resolution", TaskType.PRIMITIVE,
                       {"resolution_type": "fair_compensation"})
            ],
            ordering_constraints=[
                ("initiate_mediation_process", "collect_dispute_evidence"),
                ("collect_dispute_evidence", "analyze_evidence_objectively"),
                ("analyze_evidence_objectively", "execute_mediated_resolution")
            ],
            success_probability=0.8
        )
        
        # Emergency response method
        emergency_method = HTNMethod(
            name="handle_emergency_situation",
            task_name="resolve_emergency",
            preconditions=[],
            subtasks=[
                HTNTask("activate_emergency_protocols", TaskType.PRIMITIVE,
                       {"incident_type": "critical_disruption"}),
                HTNTask("coordinate_multi_stakeholder_response", TaskType.PRIMITIVE,
                       {"stakeholders": "all_affected_parties"}),
                HTNTask("assess_merchant_capacity", TaskType.PRIMITIVE,
                       {"merchant_id": "emergency_affected"}),
                HTNTask("find_alternative_merchants", TaskType.PRIMITIVE,
                       {"location": "emergency_area", "capacity_needed": "high"})
            ],
            ordering_constraints=[
                ("activate_emergency_protocols", "coordinate_multi_stakeholder_response"),
                ("activate_emergency_protocols", "assess_merchant_capacity"),
                ("assess_merchant_capacity", "find_alternative_merchants")
            ],
            success_probability=0.7
        )
        
        # Store methods
        self.methods["resolve_traffic_disruption"] = [traffic_method]
        self.methods["resolve_merchant_disruption"] = [merchant_method]
        self.methods["resolve_customer_dispute"] = [mediation_method]
        self.methods["resolve_emergency"] = [emergency_method]
    
    def plan(self, scenario: ValidatedDisruptionScenario, 
             tool_results: Optional[List[ToolResult]] = None) -> Optional[HTNPlan]:
        """Generate an HTN plan for the given scenario."""
        
        # Update state with tool results
        if tool_results:
            self.state.update_from_tool_results(tool_results)
        
        # Determine the main goal based on scenario type
        main_goal = self._determine_main_goal(scenario)
        if not main_goal:
            logger.warning(f"No HTN goal determined for scenario type: {scenario.scenario_type}")
            return None
        
        # Create initial task
        initial_task = HTNTask(main_goal, TaskType.COMPOUND, 
                              {"scenario": scenario, "urgency": scenario.urgency_level})
        
        # Plan using HTN decomposition
        try:
            plan = self._decompose_task(initial_task, scenario)
            if plan:
                # Enhance plan with tool result insights
                self._enhance_plan_with_tool_insights(plan, tool_results or [])
                logger.info(f"HTN plan generated with {len(plan.tasks)} tasks")
            return plan
        except Exception as e:
            logger.error(f"HTN planning failed: {e}")
            return None
    
    def _determine_main_goal(self, scenario: ValidatedDisruptionScenario) -> Optional[str]:
        """Determine the main HTN goal based on scenario characteristics."""
        
        # Check for emergency indicators
        emergency_keywords = ["fire", "accident", "emergency", "critical", "evacuation"]
        if (scenario.urgency_level == UrgencyLevel.CRITICAL or 
            any(keyword in scenario.description.lower() for keyword in emergency_keywords)):
            return "resolve_emergency"
        
        # Check scenario type
        if scenario.scenario_type == ScenarioType.TRAFFIC:
            return "resolve_traffic_disruption"
        elif scenario.scenario_type == ScenarioType.MERCHANT:
            return "resolve_merchant_disruption"
        elif scenario.scenario_type == ScenarioType.MULTI_FACTOR:
            # For multi-factor, choose based on primary issue
            if "traffic" in scenario.description.lower():
                return "resolve_traffic_disruption"
            elif "merchant" in scenario.description.lower() or "restaurant" in scenario.description.lower():
                return "resolve_merchant_disruption"
            else:
                return "resolve_emergency"
        
        # Check for dispute/mediation keywords
        dispute_keywords = ["dispute", "complaint", "wrong", "cold", "late", "refund", "mediation"]
        if any(keyword in scenario.description.lower() for keyword in dispute_keywords):
            return "resolve_customer_dispute"
        
        # Default fallback
        return "resolve_traffic_disruption"
    
    def _decompose_task(self, task: HTNTask, scenario: ValidatedDisruptionScenario) -> Optional[HTNPlan]:
        """Decompose a compound task into primitive tasks."""
        
        if task.task_type == TaskType.PRIMITIVE:
            # Base case: primitive task
            return HTNPlan(tasks=[task])
        
        # Find applicable methods for this compound task
        applicable_methods = self._find_applicable_methods(task.name)
        if not applicable_methods:
            logger.warning(f"No applicable methods found for task: {task.name}")
            return None
        
        # Select the best method based on current state and scenario
        best_method = self._select_best_method(applicable_methods, scenario)
        if not best_method:
            logger.warning(f"No suitable method selected for task: {task.name}")
            return None
        
        # Decompose using the selected method
        plan = HTNPlan()
        plan.tasks = []
        plan.dependencies = []
        
        # Add subtasks to plan
        for subtask in best_method.subtasks:
            if subtask.task_type == TaskType.PRIMITIVE:
                plan.tasks.append(subtask)
            else:
                # Recursively decompose compound subtasks
                subplan = self._decompose_task(subtask, scenario)
                if subplan:
                    plan.tasks.extend(subplan.tasks)
                    plan.dependencies.extend(subplan.dependencies)
        
        # Add ordering constraints as dependencies
        for before, after in best_method.ordering_constraints:
            # Find tasks with matching names
            before_tasks = [t for t in plan.tasks if before in t.name]
            after_tasks = [t for t in plan.tasks if after in t.name]
            
            for bt in before_tasks:
                for at in after_tasks:
                    plan.dependencies.append((bt.name, at.name))
        
        # Calculate plan metrics
        plan.success_probability = best_method.success_probability
        plan.estimated_duration = sum(
            (t.estimated_duration or timedelta(minutes=5) for t in plan.tasks),
            timedelta()
        )
        
        # Add metadata
        plan.metadata = {
            "method_used": best_method.name,
            "scenario_type": scenario.scenario_type.value,
            "urgency_level": scenario.urgency_level.value,
            "decomposition_depth": 1
        }
        
        return plan
    
    def _find_applicable_methods(self, task_name: str) -> List[HTNMethod]:
        """Find methods that can decompose the given task."""
        return self.methods.get(task_name, [])
    
    def _select_best_method(self, methods: List[HTNMethod], 
                           scenario: ValidatedDisruptionScenario) -> Optional[HTNMethod]:
        """Select the best method based on current state and scenario."""
        
        if not methods:
            return None
        
        # Score methods based on various factors
        scored_methods = []
        
        for method in methods:
            score = 0.0
            
            # Base score from success probability
            score += method.success_probability * 100
            
            # Urgency bonus
            if scenario.urgency_level == UrgencyLevel.CRITICAL:
                score += 20
            elif scenario.urgency_level == UrgencyLevel.HIGH:
                score += 10
            
            # State-based scoring
            satisfied_preconditions = sum(
                1 for precond in method.preconditions 
                if self.state.has_fact(precond)
            )
            if method.preconditions:
                score += (satisfied_preconditions / len(method.preconditions)) * 30
            else:
                score += 30  # No preconditions is good
            
            # Tool availability bonus
            available_tools = sum(
                1 for subtask in method.subtasks
                if subtask.task_type == TaskType.PRIMITIVE and
                self._is_tool_available(subtask.name)
            )
            if method.subtasks:
                score += (available_tools / len(method.subtasks)) * 20
            
            scored_methods.append((method, score))
        
        # Return method with highest score
        scored_methods.sort(key=lambda x: x[1], reverse=True)
        return scored_methods[0][0]
    
    def _is_tool_available(self, task_name: str) -> bool:
        """Check if the tool for a primitive task is available."""
        operator = self.operators.get(task_name)
        if not operator or not operator.tool_name:
            return False
        
        # Check if tool has failed recently
        tool_failed_fact = f"tool_{operator.tool_name}_failed"
        return not self.state.has_fact(tool_failed_fact)
    
    def _enhance_plan_with_tool_insights(self, plan: HTNPlan, tool_results: List[ToolResult]):
        """Enhance the plan based on tool execution results."""
        
        # Adjust success probability based on tool results
        successful_tools = sum(1 for r in tool_results if r.success)
        total_tools = len(tool_results)
        
        if total_tools > 0:
            tool_success_rate = successful_tools / total_tools
            # Blend with base probability
            plan.success_probability = (plan.success_probability * 0.7 + tool_success_rate * 0.3)
        
        # Add insights to metadata
        plan.metadata["tool_insights"] = {
            "successful_tools": successful_tools,
            "total_tools": total_tools,
            "tool_success_rate": successful_tools / total_tools if total_tools > 0 else 1.0,
            "key_facts": list(self.state.facts)[:10]  # Top 10 facts
        }
        
        # Adjust task priorities based on tool results
        for task in plan.tasks:
            operator = self.operators.get(task.name)
            if operator and operator.tool_name:
                # Check if this tool type has been successful
                tool_success_fact = f"tool_{operator.tool_name}_succeeded"
                if self.state.has_fact(tool_success_fact):
                    task.priority = min(5, task.priority + 1)
    
    def to_resolution_steps(self, plan: HTNPlan) -> List[Dict[str, Any]]:
        """Convert HTN plan to resolution steps format."""
        
        if not plan or not plan.tasks:
            return []
        
        steps = []
        
        # Topological sort to respect dependencies
        sorted_tasks = self._topological_sort(plan.tasks, plan.dependencies)
        
        for i, task in enumerate(sorted_tasks, 1):
            operator = self.operators.get(task.name)
            
            # Determine responsible party based on operator type
            responsible_party = "System"
            if operator:
                if operator.operator_type == OperatorType.CUSTOMER_SERVICE:
                    responsible_party = "Customer Service"
                elif operator.operator_type == OperatorType.MERCHANT_MANAGEMENT:
                    responsible_party = "Merchant Relations"
                elif operator.operator_type == OperatorType.TRAFFIC_COORDINATION:
                    responsible_party = "Operations"
                elif operator.operator_type == OperatorType.EMERGENCY_RESPONSE:
                    responsible_party = "Emergency Response"
                elif operator.operator_type == OperatorType.MEDIATION_FLOW:
                    responsible_party = "Mediation Team"
            
            # Create step
            step = {
                "action": self._humanize_task_name(task.name),
                "responsible_party": responsible_party,
                "timeframe": self._format_duration(
                    operator.estimated_duration if operator else timedelta(minutes=5)
                ),
                "success_criteria": f"Task '{task.name}' completed successfully",
                "dependencies": self._get_task_dependencies(task.name, plan.dependencies),
                "specific_instructions": self._generate_task_instructions(task, operator),
                "priority": task.priority
            }
            
            steps.append(step)
        
        return steps
    
    def _topological_sort(self, tasks: List[HTNTask], 
                         dependencies: List[Tuple[str, str]]) -> List[HTNTask]:
        """Sort tasks topologically based on dependencies."""
        
        # Create adjacency list and in-degree count
        graph = {task.name: [] for task in tasks}
        in_degree = {task.name: 0 for task in tasks}
        task_map = {task.name: task for task in tasks}
        
        # Build graph
        for before, after in dependencies:
            if before in graph and after in graph:
                graph[before].append(after)
                in_degree[after] += 1
        
        # Kahn's algorithm with cycle detection
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []
        processed = 0
        
        while queue:
            current = queue.pop(0)
            result.append(task_map[current])
            processed += 1
            
            # Reduce in-degree of neighbors
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for cycles
        if processed != len(tasks):
            logger.warning("Cycle detected in task dependencies, using original order")
            return tasks
        
        return result
    
    def _humanize_task_name(self, task_name: str) -> str:
        """Convert technical task name to human-readable action."""
        
        humanized = {
            "assess_traffic_situation": "Assess current traffic conditions and impact",
            "calculate_alternative_route": "Calculate optimal alternative route",
            "notify_customer_delay": "Notify customer about delay and provide updates",
            "coordinate_driver_reroute": "Coordinate driver rerouting to alternative path",
            "assess_merchant_capacity": "Evaluate merchant capacity and availability",
            "find_alternative_merchants": "Identify suitable alternative merchants",
            "coordinate_merchant_replacement": "Coordinate order transfer to alternative merchant",
            "collect_customer_feedback": "Collect detailed customer feedback and evidence",
            "provide_instant_resolution": "Provide immediate resolution to customer",
            "initiate_mediation_process": "Initiate formal mediation process",
            "collect_dispute_evidence": "Collect evidence from all parties involved",
            "analyze_evidence_objectively": "Analyze evidence objectively to determine fault",
            "execute_mediated_resolution": "Execute fair resolution based on evidence",
            "activate_emergency_protocols": "Activate emergency response protocols",
            "coordinate_multi_stakeholder_response": "Coordinate response across all stakeholders"
        }
        
        return humanized.get(task_name, task_name.replace("_", " ").title())
    
    def _format_duration(self, duration: timedelta) -> str:
        """Format duration as human-readable timeframe."""
        
        total_minutes = int(duration.total_seconds() / 60)
        
        if total_minutes < 5:
            return "1-3 minutes"
        elif total_minutes < 10:
            return "5-8 minutes"
        elif total_minutes < 20:
            return "10-15 minutes"
        elif total_minutes < 45:
            return "20-30 minutes"
        else:
            return "30+ minutes"
    
    def _get_task_dependencies(self, task_name: str, 
                              dependencies: List[Tuple[str, str]]) -> List[str]:
        """Get the dependencies for a specific task."""
        
        deps = []
        for before, after in dependencies:
            if after == task_name:
                deps.append(self._humanize_task_name(before))
        
        return deps
    
    def _generate_task_instructions(self, task: HTNTask, 
                                   operator: Optional[HTNOperator]) -> str:
        """Generate specific instructions for a task."""
        
        if not operator:
            return f"Execute {task.name} according to standard procedures"
        
        instructions = []
        
        # Add precondition checks
        if operator.preconditions:
            instructions.append(f"Verify: {', '.join(operator.preconditions)}")
        
        # Add tool-specific instructions
        if operator.tool_name:
            instructions.append(f"Use {operator.tool_name} tool for execution")
        
        # Add parameter guidance
        if task.parameters:
            param_str = ", ".join(f"{k}={v}" for k, v in task.parameters.items())
            instructions.append(f"Parameters: {param_str}")
        
        # Add success criteria
        if operator.effects:
            instructions.append(f"Expected outcomes: {', '.join(operator.effects)}")
        
        return "; ".join(instructions) if instructions else "Follow standard operating procedures"

    def get_operator_ranking(self, scenario: ValidatedDisruptionScenario) -> List[Tuple[str, float]]:
        """Get ranked list of operators suitable for the scenario."""
        
        rankings = []
        
        for name, operator in self.operators.items():
            score = 0.0
            
            # Base score from success probability
            score += operator.success_probability * 100
            
            # Scenario type matching
            if scenario.scenario_type == ScenarioType.TRAFFIC:
                if operator.operator_type == OperatorType.TRAFFIC_COORDINATION:
                    score += 50
                elif operator.operator_type == OperatorType.CUSTOMER_SERVICE:
                    score += 20
            elif scenario.scenario_type == ScenarioType.MERCHANT:
                if operator.operator_type == OperatorType.MERCHANT_MANAGEMENT:
                    score += 50
                elif operator.operator_type == OperatorType.CUSTOMER_SERVICE:
                    score += 30
            elif scenario.scenario_type == ScenarioType.MULTI_FACTOR:
                if operator.operator_type == OperatorType.EMERGENCY_RESPONSE:
                    score += 40
                else:
                    score += 25  # All operators somewhat relevant
            
            # Urgency matching
            if scenario.urgency_level == UrgencyLevel.CRITICAL:
                if operator.operator_type == OperatorType.EMERGENCY_RESPONSE:
                    score += 30
                score += 10  # General urgency bonus
            elif scenario.urgency_level == UrgencyLevel.HIGH:
                score += 5
            
            # State compatibility
            satisfied_preconditions = sum(
                1 for precond in operator.preconditions 
                if self.state.has_fact(precond)
            )
            if operator.preconditions:
                score += (satisfied_preconditions / len(operator.preconditions)) * 20
            else:
                score += 20  # No preconditions is good
            
            rankings.append((name, score))
        
        # Sort by score descending
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings