"""
Scenario generator for creating diverse delivery disruption cases.
"""
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ScenarioCategory(Enum):
    """Categories of delivery disruption scenarios."""
    TRAFFIC = "traffic"
    MERCHANT = "merchant"
    CUSTOMER = "customer"
    ADDRESS = "address"
    WEATHER = "weather"
    VEHICLE = "vehicle"
    EMERGENCY = "emergency"
    MULTI_FACTOR = "multi_factor"


class UrgencyLevel(Enum):
    """Urgency levels for scenarios."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ScenarioTemplate:
    """Template for generating scenarios."""
    category: ScenarioCategory
    urgency: UrgencyLevel
    template: str
    variables: Dict[str, List[str]]
    expected_tools: List[str]
    expected_actions: List[str]
    complexity_score: int  # 1-10


class ScenarioGenerator:
    """Generates diverse delivery disruption scenarios for testing and training."""
    
    def __init__(self):
        """Initialize scenario generator with templates."""
        self.templates = self._initialize_templates()
        self.delivery_ids = self._generate_delivery_ids()
        self.customer_names = ["John", "Sarah", "Mike", "Lisa", "Tom", "Emma", "David", "Maria", "Alex", "Jennifer"]
        self.driver_names = ["Carlos", "Emma", "Frank", "Lisa", "Mike", "Sarah", "Tom", "Alex", "Maria", "David"]
        self.addresses = [
            "123 Main Street", "456 Oak Avenue", "789 Pine Road", "321 Elm Street",
            "654 Maple Drive", "987 Cedar Lane", "147 Birch Way", "258 Ash Court",
            "369 Willow Street", "741 Spruce Avenue"
        ]
        self.restaurants = [
            "Tony's Pizza", "Mario's Kitchen", "Burger Palace", "Taco Bell", "Subway",
            "Pizza Hut", "McDonald's", "KFC", "Domino's", "Papa John's"
        ]
        self.phone_numbers = [
            "(555) 123-4567", "(555) 234-5678", "(555) 345-6789", "(555) 456-7890",
            "(555) 567-8901", "(555) 678-9012", "(555) 789-0123", "(555) 890-1234"
        ]
    
    def _initialize_templates(self) -> List[ScenarioTemplate]:
        """Initialize scenario templates for different disruption types."""
        return [
            # TRAFFIC SCENARIOS
            ScenarioTemplate(
                category=ScenarioCategory.TRAFFIC,
                urgency=UrgencyLevel.LOW,
                template="Minor traffic backup on {road} is causing a {delay_minutes}-minute delay for delivery {delivery_id} to customer {customer_name}.",
                variables={
                    "road": ["Route 1", "Highway 101", "Main Street", "Oak Avenue", "Pine Road"],
                    "delay_minutes": ["10", "15", "20"],
                },
                expected_tools=["check_traffic", "notify_customer", "re_route_driver"],
                expected_actions=["assess_delay", "communicate_with_customer", "find_alternative_route"],
                complexity_score=3
            ),
            ScenarioTemplate(
                category=ScenarioCategory.TRAFFIC,
                urgency=UrgencyLevel.HIGH,
                template="Major accident on {highway} has completely blocked traffic. Driver {driver_name} is stuck with delivery {delivery_id} for customer {customer_name} at {address}. Estimated delay is {delay_hours} hours.",
                variables={
                    "highway": ["Highway 95", "Interstate 80", "Route 101", "Highway 1"],
                    "delay_hours": ["2", "3", "4"],
                },
                expected_tools=["check_traffic", "re_route_driver", "notify_customer", "escalate_to_support"],
                expected_actions=["find_emergency_route", "proactive_communication", "consider_alternative_delivery"],
                complexity_score=7
            ),
            
            # MERCHANT SCENARIOS
            ScenarioTemplate(
                category=ScenarioCategory.MERCHANT,
                urgency=UrgencyLevel.MEDIUM,
                template="Restaurant {restaurant} is running {delay_minutes} minutes behind schedule due to {issue}. Delivery {delivery_id} for customer {customer_name} is affected.",
                variables={
                    "issue": ["kitchen equipment failure", "staff shortage", "high order volume", "ingredient shortage"],
                    "delay_minutes": ["30", "45", "60"],
                },
                expected_tools=["get_merchant_status", "notify_customer", "get_nearby_merchants"],
                expected_actions=["assess_merchant_capacity", "manage_expectations", "find_alternatives"],
                complexity_score=4
            ),
            ScenarioTemplate(
                category=ScenarioCategory.MERCHANT,
                urgency=UrgencyLevel.CRITICAL,
                template="EMERGENCY: {restaurant} has been evacuated due to {emergency_type}. {affected_orders} active deliveries are affected including {delivery_id}. Customers are calling asking about their orders.",
                variables={
                    "emergency_type": ["kitchen fire", "gas leak", "health department closure"],
                    "affected_orders": ["5", "8", "12", "15"],
                },
                expected_tools=["get_merchant_status", "get_nearby_merchants", "notify_customer", "escalate_to_support"],
                expected_actions=["emergency_response", "mass_communication", "coordinate_alternatives"],
                complexity_score=9
            ),
            
            # CUSTOMER SCENARIOS
            ScenarioTemplate(
                category=ScenarioCategory.CUSTOMER,
                urgency=UrgencyLevel.MEDIUM,
                template="Customer {customer_name} at {address} is complaining about {complaint_type} for delivery {delivery_id}. They are demanding {resolution_demand}.",
                variables={
                    "complaint_type": ["cold food", "wrong order", "missing items", "damaged packaging", "late delivery"],
                    "resolution_demand": ["full refund", "replacement order", "partial refund", "compensation credit"],
                },
                expected_tools=["collect_evidence", "notify_customer", "issue_instant_refund", "coordinate_replacement"],
                expected_actions=["investigate_complaint", "provide_resolution", "prevent_escalation"],
                complexity_score=5
            ),
            ScenarioTemplate(
                category=ScenarioCategory.CUSTOMER,
                urgency=UrgencyLevel.HIGH,
                template="High-value customer {customer_name} (VIP status, {order_count}+ orders) received delivery {delivery_id} with {issue}. They are threatening to {threat_action} and contact corporate.",
                variables={
                    "order_count": ["50", "75", "100"],
                    "issue": ["severely damaged food", "completely wrong order", "2-hour delay", "rude driver behavior"],
                    "threat_action": ["leave negative reviews", "cancel subscription", "post on social media"],
                },
                expected_tools=["collect_evidence", "issue_instant_refund", "coordinate_replacement", "escalate_to_support"],
                expected_actions=["priority_handling", "immediate_resolution", "retention_strategy"],
                complexity_score=8
            ),
            
            # ADDRESS SCENARIOS
            ScenarioTemplate(
                category=ScenarioCategory.ADDRESS,
                urgency=UrgencyLevel.MEDIUM,
                template="Driver {driver_name} cannot locate the delivery address for {delivery_id}. Customer {customer_name} provided address '{address}' but {address_issue}. Customer phone {phone} goes to voicemail.",
                variables={
                    "address_issue": ["building doesn't exist", "apartment number is invalid", "address is incomplete", "GPS shows empty lot"],
                },
                expected_tools=["validate_address", "notify_customer", "get_delivery_status"],
                expected_actions=["verify_address", "contact_customer", "coordinate_correction"],
                complexity_score=6
            ),
            
            # EMERGENCY SCENARIOS
            ScenarioTemplate(
                category=ScenarioCategory.EMERGENCY,
                urgency=UrgencyLevel.CRITICAL,
                template="URGENT: {emergency_type} delivery {delivery_id} carrying {medical_item} for {medical_condition} patient is {obstruction}. Patient's family at {address} is calling frantically. {authority} says delivery is critical within {time_limit} minutes.",
                variables={
                    "emergency_type": ["Medical", "Emergency"],
                    "medical_item": ["insulin", "heart medication", "EpiPen", "prescription medication"],
                    "medical_condition": ["diabetic", "cardiac", "allergic", "chronic"],
                    "obstruction": ["stuck due to bridge collapse", "blocked by emergency vehicles", "delayed by road closure"],
                    "authority": ["Hospital dispatch", "Doctor's office", "Emergency services"],
                    "time_limit": ["15", "20", "30"],
                },
                expected_tools=["check_traffic", "re_route_driver", "notify_customer", "escalate_to_support"],
                expected_actions=["emergency_routing", "priority_escalation", "coordinate_with_authorities"],
                complexity_score=10
            ),
            
            # MULTI-FACTOR SCENARIOS
            ScenarioTemplate(
                category=ScenarioCategory.MULTI_FACTOR,
                urgency=UrgencyLevel.HIGH,
                template="Perfect storm scenario: {traffic_issue} is causing city-wide delays. Simultaneously, {restaurant} {merchant_issue}. Driver {driver_name} was heading there for pickup of {delivery_id} for customer {customer_name} at {address}, but {address_problem}. Customer's phone {phone} {phone_issue}. It's {time_context} and customers are getting frustrated.",
                variables={
                    "traffic_issue": ["Highway 95 closure due to major accident", "Bridge collapse on Route 9", "Emergency road closures"],
                    "merchant_issue": ["had a kitchen fire and is temporarily closed", "is overwhelmed with orders", "lost power and cannot operate"],
                    "address_problem": ["the address appears to be incorrect", "the building is under construction", "access is blocked"],
                    "phone_issue": ["goes straight to voicemail", "is disconnected", "keeps ringing with no answer"],
                    "time_context": ["dinner rush hour", "lunch peak time", "late evening"],
                },
                expected_tools=["check_traffic", "get_merchant_status", "validate_address", "notify_customer", "escalate_to_support"],
                expected_actions=["multi_factor_analysis", "crisis_coordination", "comprehensive_resolution"],
                complexity_score=10
            ),
            
            # WEATHER SCENARIOS
            ScenarioTemplate(
                category=ScenarioCategory.WEATHER,
                urgency=UrgencyLevel.HIGH,
                template="Severe {weather_type} warning issued for the area. Driver {driver_name} with delivery {delivery_id} is {weather_impact}. Customer {customer_name} at {address} is {customer_concern}. Weather service predicts {duration} of dangerous conditions.",
                variables={
                    "weather_type": ["thunderstorm", "snowstorm", "ice storm", "tornado"],
                    "weather_impact": ["unable to continue safely", "stuck in dangerous conditions", "experiencing vehicle problems"],
                    "customer_concern": ["worried about driver safety", "requesting delivery cancellation", "asking for updates"],
                    "duration": ["2-3 hours", "4-6 hours", "overnight"],
                },
                expected_tools=["check_traffic", "notify_customer", "get_delivery_status", "escalate_to_support"],
                expected_actions=["safety_assessment", "weather_contingency", "customer_communication"],
                complexity_score=7
            ),
            
            # VEHICLE SCENARIOS
            ScenarioTemplate(
                category=ScenarioCategory.VEHICLE,
                urgency=UrgencyLevel.HIGH,
                template="Driver {driver_name}'s vehicle {vehicle_issue} while en route to deliver {delivery_id} to customer {customer_name} at {address}. Driver is {driver_status}. Customer has been waiting {wait_time} and is {customer_mood}.",
                variables={
                    "vehicle_issue": ["broke down", "got a flat tire", "ran out of gas", "was in a minor accident"],
                    "driver_status": ["safe but stranded", "waiting for roadside assistance", "looking for alternative transport"],
                    "wait_time": ["45 minutes", "1 hour", "90 minutes"],
                    "customer_mood": ["getting impatient", "calling repeatedly", "threatening to cancel"],
                },
                expected_tools=["get_delivery_status", "notify_customer", "coordinate_replacement", "escalate_to_support"],
                expected_actions=["driver_assistance", "delivery_reassignment", "customer_retention"],
                complexity_score=6
            )
        ]
    
    def _generate_delivery_ids(self) -> List[str]:
        """Generate realistic delivery IDs."""
        prefixes = ["DEL", "ORD", "DLV", "TXN"]
        return [f"{prefix}{random.randint(100, 999)}" for prefix in prefixes for _ in range(25)]
    
    def generate_scenario(self, category: Optional[ScenarioCategory] = None, 
                         urgency: Optional[UrgencyLevel] = None,
                         complexity_min: int = 1, complexity_max: int = 10) -> Dict[str, Any]:
        """
        Generate a random scenario based on specified criteria.
        
        Args:
            category: Specific category to generate (optional)
            urgency: Specific urgency level (optional)
            complexity_min: Minimum complexity score
            complexity_max: Maximum complexity score
            
        Returns:
            Dictionary containing scenario details
        """
        # Filter templates based on criteria
        available_templates = self.templates
        
        if category:
            available_templates = [t for t in available_templates if t.category == category]
        
        if urgency:
            available_templates = [t for t in available_templates if t.urgency == urgency]
        
        available_templates = [t for t in available_templates 
                             if complexity_min <= t.complexity_score <= complexity_max]
        
        if not available_templates:
            raise ValueError("No templates match the specified criteria")
        
        # Select random template
        template = random.choice(available_templates)
        
        # Generate variable values
        variables = {}
        for var_name, options in template.variables.items():
            variables[var_name] = random.choice(options)
        
        # Add common variables
        variables.update({
            "delivery_id": random.choice(self.delivery_ids),
            "customer_name": random.choice(self.customer_names),
            "driver_name": random.choice(self.driver_names),
            "address": random.choice(self.addresses),
            "restaurant": random.choice(self.restaurants),
            "phone": random.choice(self.phone_numbers)
        })
        
        # Generate scenario text
        scenario_text = template.template.format(**variables)
        
        return {
            "scenario_text": scenario_text,
            "category": template.category.value,
            "urgency": template.urgency.value,
            "complexity_score": template.complexity_score,
            "expected_tools": template.expected_tools,
            "expected_actions": template.expected_actions,
            "variables": variables,
            "template_id": self.templates.index(template),
            "generated_at": datetime.now().isoformat()
        }
    
    def generate_scenario_batch(self, count: int, **kwargs) -> List[Dict[str, Any]]:
        """Generate multiple scenarios."""
        return [self.generate_scenario(**kwargs) for _ in range(count)]
    
    def generate_training_dataset(self, scenarios_per_category: int = 10) -> List[Dict[str, Any]]:
        """Generate a comprehensive training dataset with scenarios from all categories."""
        dataset = []
        
        for category in ScenarioCategory:
            try:
                scenarios = self.generate_scenario_batch(
                    scenarios_per_category, 
                    category=category
                )
                dataset.extend(scenarios)
            except ValueError:
                # Skip categories with no templates
                continue
        
        return dataset
    
    def get_scenario_statistics(self) -> Dict[str, Any]:
        """Get statistics about available scenario templates."""
        stats = {
            "total_templates": len(self.templates),
            "categories": {},
            "urgency_levels": {},
            "complexity_distribution": {},
            "average_complexity": 0
        }
        
        # Category distribution
        for template in self.templates:
            category = template.category.value
            urgency = template.urgency.value
            complexity = template.complexity_score
            
            stats["categories"][category] = stats["categories"].get(category, 0) + 1
            stats["urgency_levels"][urgency] = stats["urgency_levels"].get(urgency, 0) + 1
            stats["complexity_distribution"][complexity] = stats["complexity_distribution"].get(complexity, 0) + 1
        
        # Average complexity
        total_complexity = sum(t.complexity_score for t in self.templates)
        stats["average_complexity"] = total_complexity / len(self.templates)
        
        return stats


def create_custom_scenario(description: str, category: str, urgency: str, 
                          expected_tools: List[str], expected_actions: List[str]) -> Dict[str, Any]:
    """Create a custom scenario from user input."""
    return {
        "scenario_text": description,
        "category": category,
        "urgency": urgency,
        "complexity_score": 5,  # Default complexity
        "expected_tools": expected_tools,
        "expected_actions": expected_actions,
        "variables": {},
        "template_id": -1,  # Custom scenario
        "generated_at": datetime.now().isoformat()
    }