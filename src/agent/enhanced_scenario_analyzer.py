"""
Enhanced scenario analysis with improved classification and tool selection.
Addresses the issues identified in test results.
"""
import logging
import re
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

from .interfaces import ScenarioType, UrgencyLevel, EntityType
from .models import ValidatedDisruptionScenario, ValidatedEntity
from .scenario_analyzer import ScenarioAnalyzer, ToolRecommendation, ToolPriority
from ..tools.interfaces import Tool, ToolResult


logger = logging.getLogger(__name__)


@dataclass
class ClassificationConfidence:
    """Represents classification confidence with reasoning."""
    scenario_type: ScenarioType
    urgency_level: UrgencyLevel
    confidence_score: float
    reasoning: List[str]
    alternative_classifications: List[tuple]  # (type, urgency, confidence)


@dataclass
class MandatoryToolCheck:
    """Defines mandatory tools for specific scenario types."""
    scenario_types: List[ScenarioType]
    urgency_levels: List[UrgencyLevel]
    mandatory_tools: List[str]
    reasoning: str


class EnhancedScenarioAnalyzer(ScenarioAnalyzer):
    """
    Enhanced scenario analyzer with improved classification accuracy
    and mandatory tool enforcement.
    """
    
    def __init__(self):
        """Initialize enhanced analyzer with improved rules."""
        super().__init__()
        
        # Enhanced classification patterns
        self.classification_patterns = self._initialize_classification_patterns()
        
        # Mandatory tool rules for critical scenarios
        self.mandatory_tool_rules = self._initialize_mandatory_tools()
        
        # Multi-factor scenario detection
        self.multi_factor_indicators = self._initialize_multi_factor_indicators()
        
        # Urgency escalation keywords
        self.urgency_keywords = self._initialize_urgency_keywords()
        
        logger.info("Initialized EnhancedScenarioAnalyzer with improved classification")
    
    def _initialize_classification_patterns(self) -> Dict[ScenarioType, List[Dict]]:
        """Initialize enhanced classification patterns."""
        return {
            ScenarioType.TRAFFIC: [
                {
                    "keywords": ["traffic", "road", "highway", "route", "congestion", "accident", "construction"],
                    "patterns": [r"traffic.*delay", r"road.*closed", r"highway.*blocked", r"route.*unavailable"],
                    "weight": 0.8
                },
                {
                    "keywords": ["bridge", "tunnel", "intersection", "detour"],
                    "patterns": [r"bridge.*closure", r"tunnel.*blocked", r"intersection.*blocked"],
                    "weight": 0.7
                }
            ],
            ScenarioType.MERCHANT: [
                {
                    "keywords": ["restaurant", "merchant", "store", "kitchen", "staff", "preparation"],
                    "patterns": [r"restaurant.*closed", r"merchant.*unavailable", r"kitchen.*problem"],
                    "weight": 0.8
                },
                {
                    "keywords": ["overloaded", "busy", "prep time", "queue", "orders"],
                    "patterns": [r"prep.*time.*increased", r"orders.*backed.*up"],
                    "weight": 0.7
                }
            ],
            ScenarioType.OTHER: [
                {
                    "keywords": ["customer", "complaint", "dissatisfied", "angry", "upset"],
                    "patterns": [r"customer.*complain", r"customer.*angry", r"customer.*upset"],
                    "weight": 0.8
                },
                {
                    "keywords": ["wrong order", "missing items", "cold food", "late delivery"],
                    "patterns": [r"wrong.*order", r"missing.*item", r"food.*cold"],
                    "weight": 0.7
                }
            ],
            ScenarioType.MULTI_FACTOR: [
                {
                    "keywords": ["multiple", "several", "both", "also", "additionally", "meanwhile"],
                    "patterns": [r"multiple.*deliveries", r"several.*issues", r"both.*problems"],
                    "weight": 0.9
                },
                {
                    "keywords": ["crisis", "emergency", "urgent", "critical", "medical"],
                    "patterns": [r"emergency.*delivery", r"medical.*supplies", r"critical.*situation"],
                    "weight": 0.95
                }
            ]
        }
    
    def _initialize_mandatory_tools(self) -> List[MandatoryToolCheck]:
        """Initialize mandatory tool requirements."""
        return [
            MandatoryToolCheck(
                scenario_types=[ScenarioType.MULTI_FACTOR],
                urgency_levels=[UrgencyLevel.CRITICAL, UrgencyLevel.HIGH],
                mandatory_tools=["escalate_to_support"],
                reasoning="Critical multi-factor scenarios require support escalation"
            ),
            MandatoryToolCheck(
                scenario_types=[ScenarioType.OTHER],
                urgency_levels=[UrgencyLevel.HIGH, UrgencyLevel.CRITICAL],
                mandatory_tools=["escalate_to_support", "notify_customer"],
                reasoning="High-urgency customer issues require escalation and communication"
            ),
            MandatoryToolCheck(
                scenario_types=[ScenarioType.MULTI_FACTOR],
                urgency_levels=[UrgencyLevel.MEDIUM, UrgencyLevel.HIGH, UrgencyLevel.CRITICAL],
                mandatory_tools=["validate_address"],
                reasoning="Multi-factor scenarios often involve address validation"
            ),
            MandatoryToolCheck(
                scenario_types=[ScenarioType.TRAFFIC],
                urgency_levels=[UrgencyLevel.MEDIUM, UrgencyLevel.HIGH, UrgencyLevel.CRITICAL],
                mandatory_tools=["check_traffic", "re_route_driver"],
                reasoning="Traffic scenarios require traffic checking and re-routing"
            )
        ]
    
    def _initialize_multi_factor_indicators(self) -> List[str]:
        """Initialize indicators for multi-factor scenarios."""
        return [
            "multiple deliveries", "several orders", "both problems", "also affected",
            "meanwhile", "additionally", "at the same time", "simultaneously",
            "emergency", "crisis", "urgent", "critical", "medical supplies",
            "time-sensitive", "life-threatening", "hospital", "patient"
        ]
    
    def _initialize_urgency_keywords(self) -> Dict[UrgencyLevel, List[str]]:
        """Initialize urgency classification keywords."""
        return {
            UrgencyLevel.CRITICAL: [
                "emergency", "urgent", "critical", "life-threatening", "medical",
                "hospital", "patient", "ambulance", "crisis", "immediate"
            ],
            UrgencyLevel.HIGH: [
                "important", "priority", "asap", "quickly", "soon", "frustrated",
                "angry", "multiple complaints", "escalated", "supervisor"
            ],
            UrgencyLevel.MEDIUM: [
                "delayed", "concerned", "waiting", "expected", "scheduled",
                "moderate", "standard", "regular", "normal"
            ],
            UrgencyLevel.LOW: [
                "minor", "small", "slight", "eventually", "when possible",
                "low priority", "routine", "maintenance"
            ]
        }
    
    def classify_scenario_enhanced(self, scenario_text: str, 
                                 entities: List[ValidatedEntity]) -> ClassificationConfidence:
        """
        Enhanced scenario classification with confidence scoring.
        
        Args:
            scenario_text: Raw scenario text
            entities: Extracted entities
            
        Returns:
            ClassificationConfidence with detailed reasoning
        """
        scenario_lower = scenario_text.lower()
        
        # Calculate scores for each scenario type
        type_scores = {}
        type_reasoning = {}
        
        for scenario_type, patterns in self.classification_patterns.items():
            score = 0.0
            reasoning = []
            
            for pattern_group in patterns:
                # Check keywords
                keyword_matches = sum(1 for keyword in pattern_group["keywords"] 
                                    if keyword in scenario_lower)
                if keyword_matches > 0:
                    keyword_score = (keyword_matches / len(pattern_group["keywords"])) * pattern_group["weight"]
                    score += keyword_score
                    reasoning.append(f"Found {keyword_matches} keywords: {[k for k in pattern_group['keywords'] if k in scenario_lower]}")
                
                # Check regex patterns
                pattern_matches = sum(1 for pattern in pattern_group["patterns"] 
                                    if re.search(pattern, scenario_lower))
                if pattern_matches > 0:
                    pattern_score = (pattern_matches / len(pattern_group["patterns"])) * pattern_group["weight"]
                    score += pattern_score
                    reasoning.append(f"Matched {pattern_matches} patterns")
            
            type_scores[scenario_type] = score
            type_reasoning[scenario_type] = reasoning
        
        # Check for multi-factor indicators
        multi_factor_score = 0.0
        multi_factor_reasons = []
        for indicator in self.multi_factor_indicators:
            if indicator in scenario_lower:
                multi_factor_score += 0.3
                multi_factor_reasons.append(f"Multi-factor indicator: '{indicator}'")
        
        if multi_factor_score > 0:
            type_scores[ScenarioType.MULTI_FACTOR] = max(
                type_scores.get(ScenarioType.MULTI_FACTOR, 0), 
                multi_factor_score
            )
            type_reasoning[ScenarioType.MULTI_FACTOR].extend(multi_factor_reasons)
        
        # Determine best classification
        best_type = max(type_scores.keys(), key=lambda k: type_scores[k])
        best_score = type_scores[best_type]
        
        # Classify urgency
        urgency_scores = {}
        urgency_reasoning = []
        
        for urgency_level, keywords in self.urgency_keywords.items():
            urgency_score = sum(1 for keyword in keywords if keyword in scenario_lower)
            urgency_scores[urgency_level] = urgency_score
            if urgency_score > 0:
                urgency_reasoning.append(f"{urgency_level.value}: {urgency_score} keywords")
        
        # Special urgency escalation rules
        if any(indicator in scenario_lower for indicator in ["emergency", "critical", "medical", "urgent"]):
            urgency_scores[UrgencyLevel.CRITICAL] += 2
            urgency_reasoning.append("Emergency keywords detected - escalating to CRITICAL")
        
        if best_type == ScenarioType.MULTI_FACTOR and best_score > 0.5:
            urgency_scores[UrgencyLevel.HIGH] += 1
            urgency_reasoning.append("Multi-factor scenario - escalating urgency")
        
        best_urgency = max(urgency_scores.keys(), key=lambda k: urgency_scores[k])
        
        # Create alternative classifications
        alternatives = []
        for scenario_type, score in sorted(type_scores.items(), key=lambda x: x[1], reverse=True)[1:3]:
            for urgency_level in [UrgencyLevel.HIGH, UrgencyLevel.MEDIUM]:
                alternatives.append((scenario_type, urgency_level, score * 0.8))
        
        # Combine reasoning
        all_reasoning = type_reasoning[best_type] + urgency_reasoning
        
        return ClassificationConfidence(
            scenario_type=best_type,
            urgency_level=best_urgency,
            confidence_score=min(best_score, 1.0),
            reasoning=all_reasoning,
            alternative_classifications=alternatives[:2]
        )
    
    def get_mandatory_tools(self, scenario_type: ScenarioType, 
                          urgency_level: UrgencyLevel) -> List[str]:
        """
        Get mandatory tools for a scenario type and urgency level.
        
        Args:
            scenario_type: Classified scenario type
            urgency_level: Classified urgency level
            
        Returns:
            List of mandatory tool names
        """
        mandatory_tools = set()
        
        for rule in self.mandatory_tool_rules:
            if (scenario_type in rule.scenario_types and 
                urgency_level in rule.urgency_levels):
                mandatory_tools.update(rule.mandatory_tools)
                logger.debug(f"Added mandatory tools {rule.mandatory_tools}: {rule.reasoning}")
        
        return list(mandatory_tools)
    
    def _recommend_tools_enhanced(self, scenario: ValidatedDisruptionScenario, 
                                available_tools: List[Tool]) -> List[ToolRecommendation]:
        """
        Enhanced tool recommendation with mandatory tool enforcement.
        
        Args:
            scenario: Validated scenario
            available_tools: Available tools
            
        Returns:
            List of tool recommendations including mandatory tools
        """
        # Get base recommendations from parent class
        base_recommendations = super()._recommend_tools(scenario, available_tools)
        
        # Get mandatory tools
        mandatory_tool_names = self.get_mandatory_tools(
            scenario.scenario_type, scenario.urgency_level
        )
        
        # Create set of recommended tool names
        recommended_names = {rec.tool_name for rec in base_recommendations}
        
        # Add missing mandatory tools
        available_tool_names = {tool.name for tool in available_tools}
        
        for mandatory_tool in mandatory_tool_names:
            if (mandatory_tool not in recommended_names and 
                mandatory_tool in available_tool_names):
                
                # Create high-priority recommendation for mandatory tool
                mandatory_rec = ToolRecommendation(
                    tool_name=mandatory_tool,
                    priority=ToolPriority.CRITICAL,
                    confidence=0.95,
                    reasoning=f"Mandatory tool for {scenario.scenario_type.value} scenarios with {scenario.urgency_level.value} urgency",
                    suggested_parameters=self._get_default_parameters(mandatory_tool, scenario)
                )
                base_recommendations.append(mandatory_rec)
                logger.info(f"Added mandatory tool: {mandatory_tool}")
        
        return base_recommendations
    
    def _get_default_parameters(self, tool_name: str, 
                              scenario: ValidatedDisruptionScenario) -> Dict[str, Any]:
        """Get default parameters for a tool based on scenario context."""
        # Extract relevant entities for parameter suggestions
        delivery_ids = [e.text for e in scenario.entities if e.entity_type == EntityType.DELIVERY_ID]
        addresses = [e.text for e in scenario.entities if e.entity_type == EntityType.ADDRESS]
        persons = [e.text for e in scenario.entities if e.entity_type == EntityType.PERSON]
        
        parameter_defaults = {
            "escalate_to_support": {
                "issue_type": scenario.scenario_type.value,
                "urgency": scenario.urgency_level.value,
                "delivery_id": delivery_ids[0] if delivery_ids else None
            },
            "notify_customer": {
                "delivery_id": delivery_ids[0] if delivery_ids else None,
                "customer_name": persons[0] if persons else None,
                "urgency": scenario.urgency_level.value
            },
            "validate_address": {
                "address": addresses[0] if addresses else None
            },
            "check_traffic": {
                "location": addresses[0] if addresses else "current_route"
            },
            "re_route_driver": {
                "destination": addresses[0] if addresses else None,
                "delivery_id": delivery_ids[0] if delivery_ids else None
            }
        }
        
        return parameter_defaults.get(tool_name, {})
    
    def create_specific_resolution_plan(self, scenario: ValidatedDisruptionScenario,
                                      tool_results: List[ToolResult]) -> Dict[str, Any]:
        """
        Create specific resolution plan based on scenario type and results.
        
        Args:
            scenario: Validated scenario
            tool_results: Results from executed tools
            
        Returns:
            Specific resolution plan with concrete steps
        """
        plan_templates = {
            ScenarioType.TRAFFIC: self._create_traffic_resolution_plan,
            ScenarioType.MERCHANT: self._create_merchant_resolution_plan,
            ScenarioType.OTHER: self._create_customer_resolution_plan,
            ScenarioType.MULTI_FACTOR: self._create_multi_factor_resolution_plan
        }
        
        plan_creator = plan_templates.get(scenario.scenario_type, self._create_generic_resolution_plan)
        return plan_creator(scenario, tool_results)
    
    def _create_traffic_resolution_plan(self, scenario: ValidatedDisruptionScenario,
                                      tool_results: List[ToolResult]) -> Dict[str, Any]:
        """Create specific plan for traffic scenarios."""
        steps = []
        
        # Analyze traffic results
        traffic_results = [r for r in tool_results if r.tool_name == "check_traffic"]
        if traffic_results and traffic_results[0].success:
            traffic_data = traffic_results[0].data
            if traffic_data.get("estimated_delay_minutes", 0) > 15:
                steps.append({
                    "action": "Implement alternative routing",
                    "timeframe": "Immediate (0-5 minutes)",
                    "responsible": "System + Driver",
                    "details": f"Route around {traffic_data.get('incident_type', 'traffic issue')}"
                })
        
        steps.extend([
            {
                "action": "Notify affected customers",
                "timeframe": "Within 10 minutes",
                "responsible": "Customer Service System",
                "details": "Send proactive delay notifications with updated ETAs"
            },
            {
                "action": "Monitor traffic conditions",
                "timeframe": "Continuous",
                "responsible": "System",
                "details": "Real-time traffic monitoring for further route adjustments"
            }
        ])
        
        return {
            "plan_type": "Traffic Disruption Resolution",
            "steps": steps,
            "estimated_duration": "15-30 minutes",
            "success_probability": 0.85,
            "contingency": "If alternative routes also blocked, consider delivery postponement"
        }
    
    def _create_multi_factor_resolution_plan(self, scenario: ValidatedDisruptionScenario,
                                           tool_results: List[ToolResult]) -> Dict[str, Any]:
        """Create specific plan for multi-factor scenarios."""
        steps = []
        
        # Prioritize based on urgency
        if scenario.urgency_level == UrgencyLevel.CRITICAL:
            steps.append({
                "action": "Immediate escalation to support",
                "timeframe": "0-2 minutes",
                "responsible": "System",
                "details": "Critical situation requires human oversight"
            })
        
        # Check for medical/emergency deliveries
        scenario_text = getattr(scenario, 'original_text', '').lower()
        if any(keyword in scenario_text for keyword in ["medical", "hospital", "emergency", "patient"]):
            steps.insert(0, {
                "action": "Prioritize medical deliveries",
                "timeframe": "Immediate",
                "responsible": "System + Operations",
                "details": "Medical supplies take absolute priority over other deliveries"
            })
        
        steps.extend([
            {
                "action": "Coordinate multiple delivery resolution",
                "timeframe": "5-15 minutes",
                "responsible": "Operations Team",
                "details": "Optimize delivery sequence based on criticality and location"
            },
            {
                "action": "Comprehensive customer communication",
                "timeframe": "Within 20 minutes",
                "responsible": "Customer Service",
                "details": "Individual updates for each affected delivery with specific timelines"
            }
        ])
        
        return {
            "plan_type": "Multi-Factor Crisis Resolution",
            "steps": steps,
            "estimated_duration": "30-60 minutes",
            "success_probability": 0.75,
            "contingency": "Activate emergency protocols if medical deliveries cannot be completed"
        }
    
    def _create_customer_resolution_plan(self, scenario: ValidatedDisruptionScenario,
                                       tool_results: List[ToolResult]) -> Dict[str, Any]:
        """Create specific plan for customer scenarios."""
        steps = [
            {
                "action": "Acknowledge customer concern",
                "timeframe": "Immediate (0-2 minutes)",
                "responsible": "Customer Service",
                "details": "Immediate response to show we're addressing the issue"
            },
            {
                "action": "Investigate delivery issue",
                "timeframe": "2-10 minutes",
                "responsible": "Operations Team",
                "details": "Gather evidence and determine root cause"
            },
            {
                "action": "Provide resolution and compensation",
                "timeframe": "10-20 minutes",
                "responsible": "Customer Service Manager",
                "details": "Offer appropriate remedy (refund, redelivery, credit)"
            }
        ]
        
        return {
            "plan_type": "Customer Issue Resolution",
            "steps": steps,
            "estimated_duration": "20-30 minutes",
            "success_probability": 0.90,
            "contingency": "Escalate to senior management if customer remains unsatisfied"
        }
    
    def _create_merchant_resolution_plan(self, scenario: ValidatedDisruptionScenario,
                                       tool_results: List[ToolResult]) -> Dict[str, Any]:
        """Create specific plan for merchant scenarios."""
        steps = [
            {
                "action": "Assess merchant capacity",
                "timeframe": "2-5 minutes",
                "responsible": "System",
                "details": "Determine current merchant status and recovery timeline"
            },
            {
                "action": "Find alternative merchants",
                "timeframe": "5-10 minutes",
                "responsible": "System + Operations",
                "details": "Identify nearby merchants with capacity for displaced orders"
            },
            {
                "action": "Customer notification and options",
                "timeframe": "10-15 minutes",
                "responsible": "Customer Service",
                "details": "Offer wait time, alternative merchant, or refund options"
            }
        ]
        
        return {
            "plan_type": "Merchant Disruption Resolution",
            "steps": steps,
            "estimated_duration": "15-25 minutes",
            "success_probability": 0.80,
            "contingency": "If no alternative merchants available, offer full refund and future discount"
        }
    
    def _create_generic_resolution_plan(self, scenario: ValidatedDisruptionScenario,
                                      tool_results: List[ToolResult]) -> Dict[str, Any]:
        """Create generic resolution plan for unclassified scenarios."""
        return {
            "plan_type": "Generic Disruption Resolution",
            "steps": [
                {
                    "action": "Analyze situation",
                    "timeframe": "5-10 minutes",
                    "responsible": "System + Operations",
                    "details": "Gather information and assess impact"
                },
                {
                    "action": "Implement appropriate response",
                    "timeframe": "10-20 minutes",
                    "responsible": "Operations Team",
                    "details": "Execute resolution based on analysis"
                }
            ],
            "estimated_duration": "20-30 minutes",
            "success_probability": 0.70,
            "contingency": "Escalate to human oversight if automated resolution fails"
        }