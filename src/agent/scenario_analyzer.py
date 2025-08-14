"""
Scenario analysis and intelligent tool selection for autonomous agent.
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .interfaces import ScenarioType, UrgencyLevel, EntityType
from .models import ValidatedDisruptionScenario, ValidatedEntity
from ..tools.interfaces import Tool, ToolResult


logger = logging.getLogger(__name__)


class ToolPriority(Enum):
    """Priority levels for tool selection."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class ToolRecommendation:
    """Represents a tool recommendation with priority and reasoning."""
    tool_name: str
    priority: ToolPriority
    confidence: float
    reasoning: str
    suggested_parameters: Dict[str, Any]
    dependencies: List[str] = None  # Tools that should be executed first
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class ScenarioAnalysis:
    """Results of scenario analysis."""
    scenario_complexity: str  # simple, moderate, complex
    key_entities: List[ValidatedEntity]
    primary_issues: List[str]
    stakeholders: List[str]
    urgency_factors: List[str]
    recommended_tools: List[ToolRecommendation]
    execution_strategy: str
    estimated_resolution_time: int  # minutes


class ScenarioAnalyzer:
    """
    Analyzes disruption scenarios and provides intelligent tool selection
    and execution ordering recommendations.
    """
    
    def __init__(self):
        """Initialize the scenario analyzer."""
        # Tool selection rules based on scenario types and entities
        self.tool_selection_rules = self._initialize_tool_rules()
        
        # Tool execution dependencies
        self.tool_dependencies = {
            "notify_customer": [],  # Can be executed anytime
            "check_traffic": [],  # Independent
            "get_merchant_status": [],  # Independent
            "validate_address": [],  # Independent
            "re_route_driver": ["check_traffic"],  # Needs traffic info first
            "get_nearby_merchants": ["get_merchant_status"],  # Check current merchant first
            "collect_evidence": [],  # Independent
            "issue_instant_refund": ["collect_evidence"],  # Need evidence first
        }
        
        logger.info("Initialized ScenarioAnalyzer with tool selection rules")
    
    def analyze_scenario(self, scenario: ValidatedDisruptionScenario, 
                        available_tools: List[Tool]) -> ScenarioAnalysis:
        """
        Perform comprehensive scenario analysis.
        
        Args:
            scenario: Validated disruption scenario
            available_tools: List of available tools
            
        Returns:
            ScenarioAnalysis with recommendations
        """
        logger.debug(f"Analyzing scenario: {scenario.scenario_type.value}")
        
        # Analyze scenario complexity
        complexity = self._assess_complexity(scenario)
        
        # Identify key entities and issues
        key_entities = self._identify_key_entities(scenario)
        primary_issues = self._identify_primary_issues(scenario)
        
        # Identify stakeholders
        stakeholders = self._identify_stakeholders(scenario)
        
        # Analyze urgency factors
        urgency_factors = self._analyze_urgency_factors(scenario)
        
        # Generate tool recommendations
        recommended_tools = self._recommend_tools(scenario, available_tools)
        
        # Determine execution strategy
        execution_strategy = self._determine_execution_strategy(scenario, recommended_tools)
        
        # Estimate resolution time
        estimated_time = self._estimate_resolution_time(scenario, recommended_tools)
        
        analysis = ScenarioAnalysis(
            scenario_complexity=complexity,
            key_entities=key_entities,
            primary_issues=primary_issues,
            stakeholders=stakeholders,
            urgency_factors=urgency_factors,
            recommended_tools=recommended_tools,
            execution_strategy=execution_strategy,
            estimated_resolution_time=estimated_time
        )
        
        logger.debug(f"Scenario analysis complete: {complexity} complexity, "
                    f"{len(recommended_tools)} tools recommended")
        
        return analysis
    
    def prioritize_tools(self, tools: List[ToolRecommendation], 
                        scenario: ValidatedDisruptionScenario) -> List[ToolRecommendation]:
        """
        Prioritize tools based on scenario context and dependencies.
        
        Args:
            tools: List of tool recommendations
            scenario: Scenario context
            
        Returns:
            Prioritized list of tool recommendations
        """
        # Sort by priority first, then by confidence
        prioritized = sorted(tools, key=lambda t: (t.priority.value, -t.confidence))
        
        # Adjust order based on dependencies
        ordered_tools = self._resolve_dependencies(prioritized)
        
        # Apply urgency-based adjustments
        if scenario.urgency_level in [UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]:
            ordered_tools = self._apply_urgency_adjustments(ordered_tools, scenario)
        
        logger.debug(f"Prioritized {len(ordered_tools)} tools for execution")
        return ordered_tools
    
    def select_next_tool(self, scenario: ValidatedDisruptionScenario,
                        executed_tools: List[str],
                        tool_results: List[ToolResult],
                        available_tools: List[Tool]) -> Optional[ToolRecommendation]:
        """
        Select the next best tool to execute based on current context.
        
        Args:
            scenario: Current scenario
            executed_tools: List of already executed tool names
            tool_results: Results from executed tools
            available_tools: Available tools
            
        Returns:
            Next tool recommendation or None if no more tools needed
        """
        # Get fresh analysis based on current state
        analysis = self.analyze_scenario(scenario, available_tools)
        
        # Filter out already executed tools
        remaining_tools = [
            tool for tool in analysis.recommended_tools
            if tool.tool_name not in executed_tools
        ]
        
        if not remaining_tools:
            return None
        
        # Update recommendations based on tool results
        updated_tools = self._update_recommendations_from_results(
            remaining_tools, tool_results, scenario
        )
        
        # Prioritize remaining tools
        prioritized = self.prioritize_tools(updated_tools, scenario)
        
        # Check dependencies
        for tool in prioritized:
            if self._are_dependencies_satisfied(tool, executed_tools):
                logger.debug(f"Selected next tool: {tool.tool_name} (priority: {tool.priority.value})")
                return tool
        
        # If no tool has satisfied dependencies, return the highest priority one
        # (this might indicate a dependency issue that needs to be resolved)
        if prioritized:
            logger.warning(f"No tool dependencies satisfied, selecting highest priority: {prioritized[0].tool_name}")
            return prioritized[0]
        
        return None
    
    def integrate_tool_results(self, results: List[ToolResult], 
                             scenario: ValidatedDisruptionScenario) -> Dict[str, Any]:
        """
        Integrate information from multiple tool results.
        
        Args:
            results: List of tool execution results
            scenario: Current scenario
            
        Returns:
            Integrated information dictionary
        """
        integrated_info = {
            "successful_tools": [],
            "failed_tools": [],
            "key_findings": {},
            "action_items": [],
            "confidence_score": 0.0,
            "completeness_score": 0.0
        }
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        integrated_info["successful_tools"] = [r.tool_name for r in successful_results]
        integrated_info["failed_tools"] = [r.tool_name for r in failed_results]
        
        # Extract key findings from successful results
        for result in successful_results:
            key_findings = self._extract_key_findings(result, scenario)
            integrated_info["key_findings"][result.tool_name] = key_findings
        
        # Generate action items based on findings
        integrated_info["action_items"] = self._generate_action_items(
            successful_results, scenario
        )
        
        # Calculate confidence and completeness scores
        integrated_info["confidence_score"] = self._calculate_confidence_score(
            successful_results, scenario
        )
        integrated_info["completeness_score"] = self._calculate_completeness_score(
            successful_results, scenario
        )
        
        logger.debug(f"Integrated {len(results)} tool results with "
                    f"{integrated_info['confidence_score']:.2f} confidence")
        
        return integrated_info
    
    def _initialize_tool_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize tool selection rules."""
        return {
            ScenarioType.TRAFFIC.value: {
                "primary_tools": ["check_traffic", "re_route_driver", "notify_customer"],
                "secondary_tools": ["get_merchant_status"],
                "priority_factors": ["location", "delivery_time", "traffic_severity"]
            },
            ScenarioType.MERCHANT.value: {
                "primary_tools": ["get_merchant_status", "get_nearby_merchants", "notify_customer"],
                "secondary_tools": ["check_traffic"],
                "priority_factors": ["merchant_availability", "prep_time", "alternatives"]
            },
            ScenarioType.ADDRESS.value: {
                "primary_tools": ["validate_address", "notify_customer"],
                "secondary_tools": ["check_traffic", "get_merchant_status"],
                "priority_factors": ["address_accuracy", "customer_contact", "location_verification"]
            },
            ScenarioType.MULTI_FACTOR.value: {
                "primary_tools": ["check_traffic", "get_merchant_status", "validate_address", "notify_customer"],
                "secondary_tools": ["re_route_driver", "get_nearby_merchants"],
                "priority_factors": ["urgency", "impact", "complexity"]
            },
            ScenarioType.OTHER.value: {
                "primary_tools": ["notify_customer"],
                "secondary_tools": ["collect_evidence", "issue_instant_refund"],
                "priority_factors": ["customer_impact", "resolution_complexity"]
            }
        }
    
    def _assess_complexity(self, scenario: ValidatedDisruptionScenario) -> str:
        """Assess scenario complexity."""
        complexity_score = 0
        
        # Factor in scenario type
        if scenario.scenario_type == ScenarioType.MULTI_FACTOR:
            complexity_score += 3
        elif scenario.scenario_type in [ScenarioType.TRAFFIC, ScenarioType.MERCHANT]:
            complexity_score += 2
        else:
            complexity_score += 1
        
        # Factor in number of entities
        complexity_score += min(len(scenario.entities), 3)
        
        # Factor in urgency
        if scenario.urgency_level == UrgencyLevel.CRITICAL:
            complexity_score += 2
        elif scenario.urgency_level == UrgencyLevel.HIGH:
            complexity_score += 1
        
        # Factor in description length (more details = more complex)
        if len(scenario.description) > 200:
            complexity_score += 1
        
        if complexity_score <= 3:
            return "simple"
        elif complexity_score <= 6:
            return "moderate"
        else:
            return "complex"
    
    def _identify_key_entities(self, scenario: ValidatedDisruptionScenario) -> List[ValidatedEntity]:
        """Identify the most important entities in the scenario."""
        # Sort entities by confidence and importance
        entity_importance = {
            EntityType.DELIVERY_ID: 5,
            EntityType.ADDRESS: 4,
            EntityType.MERCHANT: 4,
            EntityType.PERSON: 3,
            EntityType.PHONE_NUMBER: 2,
            EntityType.TIME: 3
        }
        
        scored_entities = []
        for entity in scenario.entities:
            importance = entity_importance.get(entity.entity_type, 1)
            score = entity.confidence * importance
            scored_entities.append((score, entity))
        
        # Return top entities (up to 5)
        scored_entities.sort(key=lambda x: x[0], reverse=True)
        return [entity for _, entity in scored_entities[:5]]
    
    def _identify_primary_issues(self, scenario: ValidatedDisruptionScenario) -> List[str]:
        """Identify primary issues from scenario description."""
        description_lower = scenario.description.lower()
        issues = []
        
        # Traffic-related issues
        traffic_keywords = ["traffic", "jam", "accident", "construction", "road closed", "delay"]
        if any(keyword in description_lower for keyword in traffic_keywords):
            issues.append("traffic_disruption")
        
        # Merchant-related issues
        merchant_keywords = ["closed", "busy", "overloaded", "unavailable", "prep time"]
        if any(keyword in description_lower for keyword in merchant_keywords):
            issues.append("merchant_unavailability")
        
        # Address-related issues
        address_keywords = ["wrong address", "incorrect", "missing", "can't find"]
        if any(keyword in description_lower for keyword in address_keywords):
            issues.append("address_issue")
        
        # Customer-related issues
        customer_keywords = ["complaint", "angry", "refund", "cancel"]
        if any(keyword in description_lower for keyword in customer_keywords):
            issues.append("customer_dissatisfaction")
        
        # Driver-related issues
        driver_keywords = ["driver", "stuck", "lost", "vehicle"]
        if any(keyword in description_lower for keyword in driver_keywords):
            issues.append("driver_issue")
        
        return issues if issues else ["general_disruption"]
    
    def _identify_stakeholders(self, scenario: ValidatedDisruptionScenario) -> List[str]:
        """Identify stakeholders involved in the scenario."""
        stakeholders = ["customer"]  # Always include customer
        
        # Add stakeholders based on entities and scenario type
        if scenario.has_entity_type(EntityType.MERCHANT):
            stakeholders.append("merchant")
        
        if scenario.has_entity_type(EntityType.DELIVERY_ID):
            stakeholders.append("driver")
        
        # Add stakeholders based on scenario type
        if scenario.scenario_type == ScenarioType.TRAFFIC:
            stakeholders.extend(["driver", "dispatch"])
        elif scenario.scenario_type == ScenarioType.MERCHANT:
            stakeholders.extend(["merchant", "kitchen_staff"])
        elif scenario.scenario_type == ScenarioType.ADDRESS:
            stakeholders.extend(["driver", "customer_service"])
        
        # Always include operations team for complex scenarios
        if scenario.urgency_level in [UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]:
            stakeholders.append("operations_team")
        
        return list(set(stakeholders))  # Remove duplicates
    
    def _analyze_urgency_factors(self, scenario: ValidatedDisruptionScenario) -> List[str]:
        """Analyze factors contributing to urgency."""
        factors = []
        description_lower = scenario.description.lower()
        
        # Time-sensitive keywords
        if any(word in description_lower for word in ["urgent", "asap", "immediately", "now"]):
            factors.append("explicit_urgency")
        
        # Customer impact keywords
        if any(word in description_lower for word in ["angry", "complaint", "frustrated"]):
            factors.append("customer_dissatisfaction")
        
        # Delivery time factors
        if any(word in description_lower for word in ["late", "delayed", "overdue"]):
            factors.append("time_delay")
        
        # Safety factors
        if any(word in description_lower for word in ["accident", "emergency", "stuck"]):
            factors.append("safety_concern")
        
        # Business impact
        if scenario.urgency_level in [UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]:
            factors.append("high_business_impact")
        
        return factors
    
    def _recommend_tools(self, scenario: ValidatedDisruptionScenario, 
                        available_tools: List[Tool]) -> List[ToolRecommendation]:
        """Generate tool recommendations based on scenario analysis."""
        recommendations = []
        available_tool_names = {tool.name for tool in available_tools}
        
        # Get rules for scenario type
        rules = self.tool_selection_rules.get(scenario.scenario_type.value, 
                                            self.tool_selection_rules[ScenarioType.OTHER.value])
        
        # Recommend primary tools
        for tool_name in rules["primary_tools"]:
            if tool_name in available_tool_names:
                recommendation = self._create_tool_recommendation(
                    tool_name, scenario, ToolPriority.HIGH
                )
                recommendations.append(recommendation)
        
        # Recommend secondary tools based on context
        for tool_name in rules["secondary_tools"]:
            if tool_name in available_tool_names:
                # Lower priority for secondary tools
                recommendation = self._create_tool_recommendation(
                    tool_name, scenario, ToolPriority.MEDIUM
                )
                recommendations.append(recommendation)
        
        # Add scenario-specific recommendations
        specific_recommendations = self._get_scenario_specific_recommendations(
            scenario, available_tool_names
        )
        recommendations.extend(specific_recommendations)
        
        return recommendations
    
    def _create_tool_recommendation(self, tool_name: str, 
                                  scenario: ValidatedDisruptionScenario,
                                  base_priority: ToolPriority) -> ToolRecommendation:
        """Create a tool recommendation with appropriate parameters."""
        # Generate suggested parameters based on scenario entities
        suggested_params = self._generate_tool_parameters(tool_name, scenario)
        
        # Calculate confidence based on scenario match
        confidence = self._calculate_tool_confidence(tool_name, scenario)
        
        # Adjust priority based on urgency
        priority = self._adjust_priority_for_urgency(base_priority, scenario.urgency_level)
        
        # Generate reasoning
        reasoning = self._generate_tool_reasoning(tool_name, scenario)
        
        # Get dependencies
        dependencies = self.tool_dependencies.get(tool_name, [])
        
        return ToolRecommendation(
            tool_name=tool_name,
            priority=priority,
            confidence=confidence,
            reasoning=reasoning,
            suggested_parameters=suggested_params,
            dependencies=dependencies
        )
    
    def _generate_tool_parameters(self, tool_name: str, 
                                scenario: ValidatedDisruptionScenario) -> Dict[str, Any]:
        """Generate suggested parameters for a tool based on scenario."""
        params = {}
        
        if tool_name == "check_traffic":
            address_entity = scenario.get_primary_address()
            if address_entity:
                params["location"] = address_entity.normalized_value or address_entity.text
            else:
                params["location"] = "delivery location"
        
        elif tool_name == "get_merchant_status":
            merchant_entity = scenario.get_primary_merchant()
            if merchant_entity:
                params["merchant_name"] = merchant_entity.normalized_value or merchant_entity.text
            else:
                params["merchant_name"] = "restaurant"
        
        elif tool_name == "validate_address":
            address_entity = scenario.get_primary_address()
            if address_entity:
                params["address"] = address_entity.normalized_value or address_entity.text
        
        elif tool_name == "notify_customer":
            delivery_ids = scenario.get_delivery_ids()
            if delivery_ids:
                params["delivery_id"] = delivery_ids[0]
            params["message_type"] = self._determine_notification_type(scenario)
        
        return params
    
    def _calculate_tool_confidence(self, tool_name: str, 
                                 scenario: ValidatedDisruptionScenario) -> float:
        """Calculate confidence score for tool recommendation."""
        base_confidence = 0.7
        
        # Boost confidence based on scenario type match
        scenario_type = scenario.scenario_type.value
        if scenario_type == "traffic" and tool_name in ["check_traffic", "re_route_driver"]:
            base_confidence += 0.2
        elif scenario_type == "merchant" and tool_name in ["get_merchant_status", "get_nearby_merchants"]:
            base_confidence += 0.2
        elif scenario_type == "address" and tool_name in ["validate_address"]:
            base_confidence += 0.2
        
        # Boost confidence based on available entities
        if tool_name == "check_traffic" and scenario.has_entity_type(EntityType.ADDRESS):
            base_confidence += 0.1
        elif tool_name == "get_merchant_status" and scenario.has_entity_type(EntityType.MERCHANT):
            base_confidence += 0.1
        elif tool_name == "notify_customer" and scenario.has_entity_type(EntityType.DELIVERY_ID):
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _adjust_priority_for_urgency(self, base_priority: ToolPriority, 
                                   urgency: UrgencyLevel) -> ToolPriority:
        """Adjust tool priority based on scenario urgency."""
        if urgency == UrgencyLevel.CRITICAL:
            # Upgrade priority by one level
            if base_priority == ToolPriority.HIGH:
                return ToolPriority.CRITICAL
            elif base_priority == ToolPriority.MEDIUM:
                return ToolPriority.HIGH
            elif base_priority == ToolPriority.LOW:
                return ToolPriority.MEDIUM
        
        return base_priority
    
    def _generate_tool_reasoning(self, tool_name: str, 
                               scenario: ValidatedDisruptionScenario) -> str:
        """Generate reasoning for why a tool is recommended."""
        scenario_type = scenario.scenario_type.value
        
        reasoning_templates = {
            "check_traffic": f"Traffic information needed for {scenario_type} scenario to assess delays and routing options",
            "get_merchant_status": f"Merchant status check required for {scenario_type} scenario to determine availability",
            "validate_address": f"Address validation needed for {scenario_type} scenario to ensure accurate delivery location",
            "notify_customer": f"Customer notification essential for {scenario_type} scenario to maintain communication",
            "re_route_driver": f"Route optimization may be needed for {scenario_type} scenario to avoid delays",
            "get_nearby_merchants": f"Alternative merchant options may be needed for {scenario_type} scenario",
            "collect_evidence": f"Evidence collection may be required for {scenario_type} scenario resolution",
            "issue_instant_refund": f"Refund processing may be appropriate for {scenario_type} scenario resolution"
        }
        
        return reasoning_templates.get(tool_name, f"Tool may be useful for {scenario_type} scenario resolution")
    
    def _get_scenario_specific_recommendations(self, scenario: ValidatedDisruptionScenario,
                                             available_tools: set) -> List[ToolRecommendation]:
        """Get additional recommendations specific to scenario characteristics."""
        recommendations = []
        
        # If customer dissatisfaction is detected, prioritize customer service tools
        if any(word in scenario.description.lower() for word in ["angry", "complaint", "frustrated"]):
            if "collect_evidence" in available_tools:
                rec = self._create_tool_recommendation("collect_evidence", scenario, ToolPriority.HIGH)
                recommendations.append(rec)
            if "issue_instant_refund" in available_tools:
                rec = self._create_tool_recommendation("issue_instant_refund", scenario, ToolPriority.MEDIUM)
                recommendations.append(rec)
        
        # If multiple delivery IDs mentioned, might need batch operations
        delivery_ids = scenario.get_delivery_ids()
        if len(delivery_ids) > 1:
            # Boost priority for notification tools
            for rec in recommendations:
                if rec.tool_name == "notify_customer":
                    rec.priority = ToolPriority.CRITICAL
                    rec.reasoning += " (multiple deliveries affected)"
        
        return recommendations
    
    def _determine_execution_strategy(self, scenario: ValidatedDisruptionScenario,
                                    tools: List[ToolRecommendation]) -> str:
        """Determine the best execution strategy for tools."""
        if scenario.urgency_level == UrgencyLevel.CRITICAL:
            return "parallel_urgent"
        elif len(tools) <= 2:
            return "sequential_simple"
        elif scenario.scenario_type == ScenarioType.MULTI_FACTOR:
            return "parallel_complex"
        else:
            return "sequential_standard"
    
    def _estimate_resolution_time(self, scenario: ValidatedDisruptionScenario,
                                tools: List[ToolRecommendation]) -> int:
        """Estimate resolution time in minutes."""
        base_time = 5  # Base resolution time
        
        # Add time based on complexity
        if scenario.scenario_type == ScenarioType.MULTI_FACTOR:
            base_time += 15
        elif scenario.scenario_type in [ScenarioType.TRAFFIC, ScenarioType.MERCHANT]:
            base_time += 10
        else:
            base_time += 5
        
        # Add time based on number of tools
        base_time += len(tools) * 2
        
        # Adjust for urgency (urgent scenarios get more resources)
        if scenario.urgency_level == UrgencyLevel.CRITICAL:
            base_time = int(base_time * 0.7)  # 30% faster with more resources
        elif scenario.urgency_level == UrgencyLevel.HIGH:
            base_time = int(base_time * 0.85)  # 15% faster
        
        return max(base_time, 5)  # Minimum 5 minutes
    
    def _resolve_dependencies(self, tools: List[ToolRecommendation]) -> List[ToolRecommendation]:
        """Resolve tool dependencies and order tools appropriately."""
        ordered = []
        remaining = tools.copy()
        
        while remaining:
            # Find tools with no unresolved dependencies
            ready_tools = []
            for tool in remaining:
                if self._are_dependencies_satisfied(tool, [t.tool_name for t in ordered]):
                    ready_tools.append(tool)
            
            if not ready_tools:
                # If no tools are ready, there might be a circular dependency
                # Just take the highest priority remaining tool
                ready_tools = [min(remaining, key=lambda t: t.priority.value)]
            
            # Add the highest priority ready tool
            next_tool = min(ready_tools, key=lambda t: (t.priority.value, -t.confidence))
            ordered.append(next_tool)
            remaining.remove(next_tool)
        
        return ordered
    
    def _are_dependencies_satisfied(self, tool: ToolRecommendation, 
                                  executed_tools: List[str]) -> bool:
        """Check if tool dependencies are satisfied."""
        return all(dep in executed_tools for dep in tool.dependencies)
    
    def _apply_urgency_adjustments(self, tools: List[ToolRecommendation],
                                 scenario: ValidatedDisruptionScenario) -> List[ToolRecommendation]:
        """Apply urgency-based adjustments to tool ordering."""
        if scenario.urgency_level == UrgencyLevel.CRITICAL:
            # Move customer notification to the front for critical scenarios
            customer_tools = [t for t in tools if t.tool_name == "notify_customer"]
            other_tools = [t for t in tools if t.tool_name != "notify_customer"]
            return customer_tools + other_tools
        
        return tools
    
    def _update_recommendations_from_results(self, tools: List[ToolRecommendation],
                                           results: List[ToolResult],
                                           scenario: ValidatedDisruptionScenario) -> List[ToolRecommendation]:
        """Update tool recommendations based on previous results."""
        updated_tools = []
        
        for tool in tools:
            updated_tool = tool
            
            # Adjust recommendations based on previous results
            for result in results:
                if result.success and result.data:
                    updated_tool = self._adjust_tool_from_result(updated_tool, result, scenario)
            
            updated_tools.append(updated_tool)
        
        return updated_tools
    
    def _adjust_tool_from_result(self, tool: ToolRecommendation, 
                               result: ToolResult,
                               scenario: ValidatedDisruptionScenario) -> ToolRecommendation:
        """Adjust tool recommendation based on a specific result."""
        # Example adjustments based on tool results
        if result.tool_name == "check_traffic" and result.data.get("status") == "heavy_traffic":
            if tool.tool_name == "re_route_driver":
                tool.priority = ToolPriority.CRITICAL
                tool.confidence = min(tool.confidence + 0.2, 1.0)
                tool.reasoning += " (heavy traffic detected)"
        
        elif result.tool_name == "get_merchant_status" and not result.data.get("available", True):
            if tool.tool_name == "get_nearby_merchants":
                tool.priority = ToolPriority.HIGH
                tool.confidence = min(tool.confidence + 0.3, 1.0)
                tool.reasoning += " (primary merchant unavailable)"
        
        return tool
    
    def _extract_key_findings(self, result: ToolResult, 
                            scenario: ValidatedDisruptionScenario) -> Dict[str, Any]:
        """Extract key findings from a tool result."""
        findings = {"status": "success" if result.success else "failed"}
        
        if result.success and result.data:
            # Extract relevant information based on tool type
            if result.tool_name == "check_traffic":
                findings.update({
                    "traffic_status": result.data.get("status"),
                    "delay_minutes": result.data.get("delay_minutes"),
                    "alternative_routes": result.data.get("alternative_routes", [])
                })
            elif result.tool_name == "get_merchant_status":
                findings.update({
                    "merchant_available": result.data.get("available"),
                    "prep_time_minutes": result.data.get("prep_time_minutes"),
                    "capacity_status": result.data.get("capacity_status")
                })
            elif result.tool_name == "validate_address":
                findings.update({
                    "address_valid": result.data.get("valid"),
                    "corrected_address": result.data.get("corrected_address"),
                    "confidence": result.data.get("confidence")
                })
        
        return findings
    
    def _generate_action_items(self, results: List[ToolResult],
                             scenario: ValidatedDisruptionScenario) -> List[str]:
        """Generate action items based on tool results."""
        action_items = []
        
        for result in results:
            if result.success and result.data:
                if result.tool_name == "check_traffic" and result.data.get("delay_minutes", 0) > 10:
                    action_items.append("Consider re-routing driver due to significant traffic delay")
                
                elif result.tool_name == "get_merchant_status" and not result.data.get("available", True):
                    action_items.append("Find alternative merchant or reschedule delivery")
                
                elif result.tool_name == "validate_address" and not result.data.get("valid", True):
                    action_items.append("Contact customer to verify correct delivery address")
        
        return action_items
    
    def _calculate_confidence_score(self, results: List[ToolResult],
                                  scenario: ValidatedDisruptionScenario) -> float:
        """Calculate overall confidence score based on tool results."""
        if not results:
            return 0.0
        
        success_rate = len([r for r in results if r.success]) / len(results)
        
        # Boost confidence if we have results for scenario-critical tools
        critical_tools = self._get_critical_tools_for_scenario(scenario)
        critical_results = [r for r in results if r.tool_name in critical_tools and r.success]
        critical_coverage = len(critical_results) / len(critical_tools) if critical_tools else 1.0
        
        return (success_rate * 0.6) + (critical_coverage * 0.4)
    
    def _calculate_completeness_score(self, results: List[ToolResult],
                                    scenario: ValidatedDisruptionScenario) -> float:
        """Calculate completeness score based on information gathered."""
        # Get recommended tools for this scenario type
        rules = self.tool_selection_rules.get(scenario.scenario_type.value,
                                            self.tool_selection_rules[ScenarioType.OTHER.value])
        
        primary_tools = set(rules["primary_tools"])
        executed_primary = set(r.tool_name for r in results if r.success and r.tool_name in primary_tools)
        
        primary_coverage = len(executed_primary) / len(primary_tools) if primary_tools else 1.0
        
        # Factor in data quality
        data_quality = sum(1 for r in results if r.success and r.data) / len(results) if results else 0
        
        return (primary_coverage * 0.7) + (data_quality * 0.3)
    
    def _get_critical_tools_for_scenario(self, scenario: ValidatedDisruptionScenario) -> List[str]:
        """Get critical tools for a scenario type."""
        rules = self.tool_selection_rules.get(scenario.scenario_type.value,
                                            self.tool_selection_rules[ScenarioType.OTHER.value])
        return rules.get("primary_tools", [])
    
    def _determine_notification_type(self, scenario: ValidatedDisruptionScenario) -> str:
        """Determine appropriate notification type for customer."""
        if scenario.urgency_level == UrgencyLevel.CRITICAL:
            return "urgent_update"
        elif scenario.scenario_type == ScenarioType.TRAFFIC:
            return "delay_notification"
        elif scenario.scenario_type == ScenarioType.MERCHANT:
            return "preparation_update"
        else:
            return "status_update"