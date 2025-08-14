"""
Traffic and routing tools for delivery coordination.
"""
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .interfaces import Tool, ToolResult


@dataclass
class TrafficCondition:
    """Represents traffic conditions on a route."""
    severity: str  # "light", "moderate", "heavy", "blocked"
    delay_minutes: int
    cause: str
    affected_area: str
    estimated_clearance: Optional[datetime] = None


@dataclass
class RouteInfo:
    """Information about a delivery route."""
    origin: str
    destination: str
    distance_miles: float
    estimated_time_minutes: int
    traffic_conditions: List[TrafficCondition]
    alternative_available: bool


class CheckTrafficTool(Tool):
    """Tool for checking traffic conditions on delivery routes."""
    
    def __init__(self):
        super().__init__(
            name="check_traffic",
            description="Check current traffic conditions for a delivery route",
            parameters={
                "origin": {"type": "string", "description": "Starting address or location"},
                "destination": {"type": "string", "description": "Delivery destination address"},
                "delivery_id": {"type": "string", "description": "Delivery ID for tracking", "required": False}
            }
        )
        
        # Predefined traffic scenarios for realistic simulation
        self._traffic_scenarios = [
            {
                "severity": "light",
                "delay_range": (0, 5),
                "causes": ["Normal traffic flow", "Light congestion"],
                "probability": 0.4
            },
            {
                "severity": "moderate", 
                "delay_range": (5, 15),
                "causes": ["Rush hour traffic", "Construction zone", "School zone"],
                "probability": 0.3
            },
            {
                "severity": "heavy",
                "delay_range": (15, 45),
                "causes": ["Accident ahead", "Road construction", "Weather conditions", "Event traffic"],
                "probability": 0.2
            },
            {
                "severity": "blocked",
                "delay_range": (45, 120),
                "causes": ["Major accident", "Road closure", "Emergency situation", "Severe weather"],
                "probability": 0.1
            }
        ]
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate required parameters for traffic check."""
        required = ["origin", "destination"]
        return all(param in kwargs and kwargs[param] for param in required)
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute traffic check with realistic simulation."""
        start_time = time.time()
        
        if not self.validate_parameters(**kwargs):
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                execution_time=time.time() - start_time,
                error_message="Missing required parameters: origin and destination"
            )
        
        try:
            # Simulate API call delay
            time.sleep(random.uniform(0.1, 0.5))
            
            origin = kwargs["origin"]
            destination = kwargs["destination"]
            delivery_id = kwargs.get("delivery_id", "unknown")
            
            # Generate realistic route info
            route_info = self._generate_route_info(origin, destination)
            
            # Determine if alternative routes are available
            alternative_available = random.choice([True, False]) if route_info.traffic_conditions else True
            
            result_data = {
                "delivery_id": delivery_id,
                "route": {
                    "origin": origin,
                    "destination": destination,
                    "distance_miles": route_info.distance_miles,
                    "base_time_minutes": route_info.estimated_time_minutes,
                    "current_conditions": [
                        {
                            "severity": condition.severity,
                            "delay_minutes": condition.delay_minutes,
                            "cause": condition.cause,
                            "affected_area": condition.affected_area,
                            "estimated_clearance": condition.estimated_clearance.isoformat() if condition.estimated_clearance else None
                        }
                        for condition in route_info.traffic_conditions
                    ],
                    "total_delay_minutes": sum(c.delay_minutes for c in route_info.traffic_conditions),
                    "estimated_arrival_delay": sum(c.delay_minutes for c in route_info.traffic_conditions),
                    "alternative_routes_available": alternative_available
                },
                "recommendations": self._generate_recommendations(route_info, alternative_available)
            }
            
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=result_data,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                execution_time=time.time() - start_time,
                error_message=f"Traffic check failed: {str(e)}"
            )
    
    def _generate_route_info(self, origin: str, destination: str) -> RouteInfo:
        """Generate realistic route information with traffic conditions."""
        # Simulate route distance and base time
        distance = random.uniform(2.0, 25.0)  # 2-25 miles
        base_time = int(distance * random.uniform(2, 4))  # 2-4 minutes per mile
        
        # Generate traffic conditions based on probability
        conditions = []
        for scenario in self._traffic_scenarios:
            if random.random() < scenario["probability"]:
                delay = random.randint(*scenario["delay_range"])
                cause = random.choice(scenario["causes"])
                
                # Generate estimated clearance time for blocked/heavy traffic
                clearance_time = None
                if scenario["severity"] in ["heavy", "blocked"]:
                    clearance_time = datetime.now() + timedelta(minutes=random.randint(30, 180))
                
                conditions.append(TrafficCondition(
                    severity=scenario["severity"],
                    delay_minutes=delay,
                    cause=cause,
                    affected_area=f"Between {origin} and {destination}",
                    estimated_clearance=clearance_time
                ))
        
        return RouteInfo(
            origin=origin,
            destination=destination,
            distance_miles=round(distance, 1),
            estimated_time_minutes=base_time,
            traffic_conditions=conditions,
            alternative_available=len(conditions) > 0
        )
    
    def _generate_recommendations(self, route_info: RouteInfo, alternative_available: bool) -> List[str]:
        """Generate actionable recommendations based on traffic conditions."""
        recommendations = []
        
        total_delay = sum(c.delay_minutes for c in route_info.traffic_conditions)
        
        if total_delay == 0:
            recommendations.append("Route is clear - proceed as planned")
        elif total_delay < 10:
            recommendations.append("Minor delays expected - consider notifying customer")
        elif total_delay < 30:
            recommendations.append("Moderate delays - recommend customer notification and ETA update")
            if alternative_available:
                recommendations.append("Consider alternative route to reduce delay")
        else:
            recommendations.append("Significant delays detected - immediate action required")
            recommendations.append("Notify customer immediately with updated ETA")
            if alternative_available:
                recommendations.append("Strongly recommend re-routing driver")
            
            # Check for blocked conditions
            blocked_conditions = [c for c in route_info.traffic_conditions if c.severity == "blocked"]
            if blocked_conditions:
                recommendations.append("Route may be impassable - re-routing essential")
        
        return recommendations


class ReRouteDriverTool(Tool):
    """Tool for re-routing drivers to alternative routes."""
    
    def __init__(self):
        super().__init__(
            name="re_route_driver",
            description="Calculate and assign alternative route for driver",
            parameters={
                "driver_id": {"type": "string", "description": "Driver identifier"},
                "current_location": {"type": "string", "description": "Driver's current location"},
                "destination": {"type": "string", "description": "Final delivery destination"},
                "avoid_areas": {"type": "array", "description": "Areas to avoid in new route", "required": False},
                "priority": {"type": "string", "description": "Route priority: fastest, shortest, or safest", "required": False}
            }
        )
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate required parameters for re-routing."""
        required = ["driver_id", "current_location", "destination"]
        return all(param in kwargs and kwargs[param] for param in required)
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute driver re-routing with alternative route calculation."""
        start_time = time.time()
        
        if not self.validate_parameters(**kwargs):
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                execution_time=time.time() - start_time,
                error_message="Missing required parameters: driver_id, current_location, destination"
            )
        
        try:
            # Simulate API processing time
            time.sleep(random.uniform(0.2, 0.8))
            
            driver_id = kwargs["driver_id"]
            current_location = kwargs["current_location"]
            destination = kwargs["destination"]
            avoid_areas = kwargs.get("avoid_areas", [])
            priority = kwargs.get("priority", "fastest")
            
            # Generate alternative route options
            alternative_routes = self._generate_alternative_routes(
                current_location, destination, avoid_areas, priority
            )
            
            # Select best route based on priority
            selected_route = self._select_best_route(alternative_routes, priority)
            
            result_data = {
                "driver_id": driver_id,
                "re_routing_successful": True,
                "original_destination": destination,
                "new_route": {
                    "route_id": f"ALT_{random.randint(1000, 9999)}",
                    "waypoints": selected_route["waypoints"],
                    "total_distance_miles": selected_route["distance"],
                    "estimated_time_minutes": selected_route["time"],
                    "route_type": selected_route["type"],
                    "traffic_free": selected_route["traffic_free"]
                },
                "alternatives_considered": len(alternative_routes),
                "avoided_areas": avoid_areas,
                "eta_improvement_minutes": random.randint(5, 30),
                "driver_notification_sent": True,
                "gps_updated": True
            }
            
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=result_data,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                execution_time=time.time() - start_time,
                error_message=f"Re-routing failed: {str(e)}"
            )
    
    def _generate_alternative_routes(self, origin: str, destination: str, 
                                   avoid_areas: List[str], priority: str) -> List[Dict[str, Any]]:
        """Generate multiple alternative route options."""
        routes = []
        
        # Generate 2-4 alternative routes
        num_routes = random.randint(2, 4)
        
        for i in range(num_routes):
            # Simulate different route characteristics
            base_distance = random.uniform(3.0, 20.0)
            base_time = int(base_distance * random.uniform(2.5, 4.5))
            
            # Adjust based on route type
            route_types = ["highway", "surface_streets", "mixed", "scenic"]
            route_type = random.choice(route_types)
            
            if route_type == "highway":
                distance_modifier = 1.1
                time_modifier = 0.8
            elif route_type == "surface_streets":
                distance_modifier = 0.9
                time_modifier = 1.2
            elif route_type == "scenic":
                distance_modifier = 1.3
                time_modifier = 1.4
            else:  # mixed
                distance_modifier = 1.0
                time_modifier = 1.0
            
            routes.append({
                "route_id": f"ALT_{i+1}",
                "type": route_type,
                "distance": round(base_distance * distance_modifier, 1),
                "time": int(base_time * time_modifier),
                "waypoints": self._generate_waypoints(origin, destination, route_type),
                "traffic_free": random.choice([True, False]),
                "toll_roads": random.choice([True, False]) if route_type == "highway" else False
            })
        
        return routes
    
    def _generate_waypoints(self, origin: str, destination: str, route_type: str) -> List[str]:
        """Generate realistic waypoints for a route."""
        waypoints = [origin]
        
        # Add intermediate waypoints based on route type
        if route_type == "highway":
            waypoints.extend([
                f"Highway entrance near {origin}",
                f"Highway exit for {destination}"
            ])
        elif route_type == "surface_streets":
            waypoints.extend([
                f"Main St from {origin}",
                f"Downtown area",
                f"Residential area near {destination}"
            ])
        elif route_type == "scenic":
            waypoints.extend([
                f"Scenic route via parkway",
                f"Overlook point",
                f"Winding road to {destination}"
            ])
        else:  # mixed
            waypoints.extend([
                f"Local roads from {origin}",
                f"Highway segment",
                f"Surface streets to {destination}"
            ])
        
        waypoints.append(destination)
        return waypoints
    
    def _select_best_route(self, routes: List[Dict[str, Any]], priority: str) -> Dict[str, Any]:
        """Select the best route based on specified priority."""
        if priority == "shortest":
            return min(routes, key=lambda r: r["distance"])
        elif priority == "safest":
            # Prefer surface streets and traffic-free routes
            safe_routes = [r for r in routes if r["traffic_free"] and r["type"] != "highway"]
            return safe_routes[0] if safe_routes else routes[0]
        else:  # fastest (default)
            return min(routes, key=lambda r: r["time"])