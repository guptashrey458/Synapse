"""
Unit tests for traffic and routing tools.
"""
import pytest
from unittest.mock import patch
from datetime import datetime

from src.tools.traffic_tools import CheckTrafficTool, ReRouteDriverTool, TrafficCondition, RouteInfo


class TestCheckTrafficTool:
    """Test cases for CheckTrafficTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = CheckTrafficTool()
    
    def test_tool_initialization(self):
        """Test tool is properly initialized."""
        assert self.tool.name == "check_traffic"
        assert "Check current traffic conditions" in self.tool.description
        assert "origin" in self.tool.parameters
        assert "destination" in self.tool.parameters
    
    def test_validate_parameters_success(self):
        """Test parameter validation with valid inputs."""
        valid_params = {
            "origin": "123 Main St",
            "destination": "456 Oak Ave",
            "delivery_id": "DEL123"
        }
        assert self.tool.validate_parameters(**valid_params) is True
    
    def test_validate_parameters_missing_required(self):
        """Test parameter validation with missing required fields."""
        # Missing destination
        invalid_params = {"origin": "123 Main St"}
        assert self.tool.validate_parameters(**invalid_params) is False
        
        # Missing origin
        invalid_params = {"destination": "456 Oak Ave"}
        assert self.tool.validate_parameters(**invalid_params) is False
        
        # Empty values
        invalid_params = {"origin": "", "destination": "456 Oak Ave"}
        assert self.tool.validate_parameters(**invalid_params) is False
    
    def test_execute_success(self):
        """Test successful traffic check execution."""
        params = {
            "origin": "Downtown Plaza",
            "destination": "Suburban Mall",
            "delivery_id": "DEL456"
        }
        
        result = self.tool.execute(**params)
        
        assert result.success is True
        assert result.tool_name == "check_traffic"
        assert result.execution_time > 0
        assert result.error_message is None
        
        # Check result data structure
        data = result.data
        assert "delivery_id" in data
        assert "route" in data
        assert "recommendations" in data
        
        route = data["route"]
        assert route["origin"] == params["origin"]
        assert route["destination"] == params["destination"]
        assert "distance_miles" in route
        assert "base_time_minutes" in route
        assert "current_conditions" in route
        assert "total_delay_minutes" in route
        assert isinstance(route["alternative_routes_available"], bool)
    
    def test_execute_parameter_validation_failure(self):
        """Test execution with invalid parameters."""
        invalid_params = {"origin": "123 Main St"}  # Missing destination
        
        result = self.tool.execute(**invalid_params)
        
        assert result.success is False
        assert "Missing required parameters" in result.error_message
        assert result.data == {}
    
    @patch('random.uniform')
    @patch('random.randint')
    @patch('random.choice')
    def test_generate_route_info_deterministic(self, mock_choice, mock_randint, mock_uniform):
        """Test route info generation with mocked randomness."""
        # Mock random values for predictable testing
        mock_uniform.side_effect = [10.0, 3.0, 0.5, 0.1]  # distance, time_per_mile, probability checks
        mock_randint.return_value = 15  # delay minutes
        mock_choice.return_value = "Construction zone"  # cause
        
        route_info = self.tool._generate_route_info("Origin", "Destination")
        
        assert isinstance(route_info, RouteInfo)
        assert route_info.origin == "Origin"
        assert route_info.destination == "Destination"
        assert route_info.distance_miles > 0
        assert route_info.estimated_time_minutes > 0
    
    def test_generate_recommendations_no_delay(self):
        """Test recommendations for routes with no delays."""
        route_info = RouteInfo(
            origin="A", destination="B", distance_miles=5.0,
            estimated_time_minutes=15, traffic_conditions=[],
            alternative_available=False
        )
        
        recommendations = self.tool._generate_recommendations(route_info, False)
        
        assert len(recommendations) > 0
        assert any("clear" in rec.lower() for rec in recommendations)
    
    def test_generate_recommendations_heavy_delay(self):
        """Test recommendations for routes with heavy delays."""
        heavy_condition = TrafficCondition(
            severity="heavy", delay_minutes=45, cause="Major accident",
            affected_area="Highway 101"
        )
        
        route_info = RouteInfo(
            origin="A", destination="B", distance_miles=10.0,
            estimated_time_minutes=25, traffic_conditions=[heavy_condition],
            alternative_available=True
        )
        
        recommendations = self.tool._generate_recommendations(route_info, True)
        
        assert len(recommendations) > 0
        assert any("significant" in rec.lower() or "immediate" in rec.lower() for rec in recommendations)
        assert any("notify customer" in rec.lower() for rec in recommendations)
        assert any("re-routing" in rec.lower() for rec in recommendations)
    
    def test_generate_recommendations_blocked_route(self):
        """Test recommendations for blocked routes."""
        blocked_condition = TrafficCondition(
            severity="blocked", delay_minutes=90, cause="Road closure",
            affected_area="Main Street Bridge"
        )
        
        route_info = RouteInfo(
            origin="A", destination="B", distance_miles=8.0,
            estimated_time_minutes=20, traffic_conditions=[blocked_condition],
            alternative_available=True
        )
        
        recommendations = self.tool._generate_recommendations(route_info, True)
        
        assert any("impassable" in rec.lower() or "essential" in rec.lower() for rec in recommendations)


class TestReRouteDriverTool:
    """Test cases for ReRouteDriverTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = ReRouteDriverTool()
    
    def test_tool_initialization(self):
        """Test tool is properly initialized."""
        assert self.tool.name == "re_route_driver"
        assert "alternative route" in self.tool.description
        assert "driver_id" in self.tool.parameters
        assert "current_location" in self.tool.parameters
        assert "destination" in self.tool.parameters
    
    def test_validate_parameters_success(self):
        """Test parameter validation with valid inputs."""
        valid_params = {
            "driver_id": "DRV123",
            "current_location": "123 Main St",
            "destination": "456 Oak Ave",
            "avoid_areas": ["Construction Zone"],
            "priority": "fastest"
        }
        assert self.tool.validate_parameters(**valid_params) is True
    
    def test_validate_parameters_missing_required(self):
        """Test parameter validation with missing required fields."""
        # Missing driver_id
        invalid_params = {
            "current_location": "123 Main St",
            "destination": "456 Oak Ave"
        }
        assert self.tool.validate_parameters(**invalid_params) is False
        
        # Missing current_location
        invalid_params = {
            "driver_id": "DRV123",
            "destination": "456 Oak Ave"
        }
        assert self.tool.validate_parameters(**invalid_params) is False
    
    def test_execute_success(self):
        """Test successful re-routing execution."""
        params = {
            "driver_id": "DRV789",
            "current_location": "City Center",
            "destination": "Airport Terminal",
            "avoid_areas": ["Highway 101"],
            "priority": "fastest"
        }
        
        result = self.tool.execute(**params)
        
        assert result.success is True
        assert result.tool_name == "re_route_driver"
        assert result.execution_time > 0
        assert result.error_message is None
        
        # Check result data structure
        data = result.data
        assert data["driver_id"] == params["driver_id"]
        assert data["re_routing_successful"] is True
        assert "new_route" in data
        assert "alternatives_considered" in data
        
        new_route = data["new_route"]
        assert "route_id" in new_route
        assert "waypoints" in new_route
        assert "total_distance_miles" in new_route
        assert "estimated_time_minutes" in new_route
        assert isinstance(new_route["waypoints"], list)
        assert len(new_route["waypoints"]) >= 2  # At least origin and destination
    
    def test_execute_parameter_validation_failure(self):
        """Test execution with invalid parameters."""
        invalid_params = {"driver_id": "DRV123"}  # Missing required fields
        
        result = self.tool.execute(**invalid_params)
        
        assert result.success is False
        assert "Missing required parameters" in result.error_message
        assert result.data == {}
    
    def test_generate_alternative_routes(self):
        """Test alternative route generation."""
        routes = self.tool._generate_alternative_routes(
            "Origin", "Destination", ["Avoid Area"], "fastest"
        )
        
        assert isinstance(routes, list)
        assert len(routes) >= 2  # Should generate at least 2 alternatives
        
        for route in routes:
            assert "route_id" in route
            assert "type" in route
            assert "distance" in route
            assert "time" in route
            assert "waypoints" in route
            assert isinstance(route["waypoints"], list)
            assert len(route["waypoints"]) >= 2
    
    def test_generate_waypoints_different_types(self):
        """Test waypoint generation for different route types."""
        route_types = ["highway", "surface_streets", "mixed", "scenic"]
        
        for route_type in route_types:
            waypoints = self.tool._generate_waypoints("Start", "End", route_type)
            
            assert isinstance(waypoints, list)
            assert len(waypoints) >= 2
            assert waypoints[0] == "Start"
            assert waypoints[-1] == "End"
    
    def test_select_best_route_fastest(self):
        """Test route selection with fastest priority."""
        routes = [
            {"route_id": "1", "time": 30, "distance": 10.0, "traffic_free": True, "type": "highway"},
            {"route_id": "2", "time": 20, "distance": 12.0, "traffic_free": False, "type": "surface_streets"},
            {"route_id": "3", "time": 25, "distance": 8.0, "traffic_free": True, "type": "mixed"}
        ]
        
        best_route = self.tool._select_best_route(routes, "fastest")
        assert best_route["route_id"] == "2"  # Shortest time
    
    def test_select_best_route_shortest(self):
        """Test route selection with shortest priority."""
        routes = [
            {"route_id": "1", "time": 30, "distance": 10.0, "traffic_free": True, "type": "highway"},
            {"route_id": "2", "time": 20, "distance": 12.0, "traffic_free": False, "type": "surface_streets"},
            {"route_id": "3", "time": 25, "distance": 8.0, "traffic_free": True, "type": "mixed"}
        ]
        
        best_route = self.tool._select_best_route(routes, "shortest")
        assert best_route["route_id"] == "3"  # Shortest distance
    
    def test_select_best_route_safest(self):
        """Test route selection with safest priority."""
        routes = [
            {"route_id": "1", "time": 30, "distance": 10.0, "traffic_free": True, "type": "highway"},
            {"route_id": "2", "time": 20, "distance": 12.0, "traffic_free": True, "type": "surface_streets"},
            {"route_id": "3", "time": 25, "distance": 8.0, "traffic_free": False, "type": "mixed"}
        ]
        
        best_route = self.tool._select_best_route(routes, "safest")
        assert best_route["route_id"] == "2"  # Traffic-free surface streets


class TestTrafficCondition:
    """Test cases for TrafficCondition data class."""
    
    def test_traffic_condition_creation(self):
        """Test TrafficCondition creation and attributes."""
        condition = TrafficCondition(
            severity="moderate",
            delay_minutes=15,
            cause="Construction",
            affected_area="Main St"
        )
        
        assert condition.severity == "moderate"
        assert condition.delay_minutes == 15
        assert condition.cause == "Construction"
        assert condition.affected_area == "Main St"
        assert condition.estimated_clearance is None
    
    def test_traffic_condition_with_clearance(self):
        """Test TrafficCondition with estimated clearance time."""
        clearance_time = datetime.now()
        condition = TrafficCondition(
            severity="heavy",
            delay_minutes=45,
            cause="Accident",
            affected_area="Highway 101",
            estimated_clearance=clearance_time
        )
        
        assert condition.estimated_clearance == clearance_time


class TestRouteInfo:
    """Test cases for RouteInfo data class."""
    
    def test_route_info_creation(self):
        """Test RouteInfo creation and attributes."""
        conditions = [
            TrafficCondition("light", 5, "Normal traffic", "Route segment 1")
        ]
        
        route = RouteInfo(
            origin="Start",
            destination="End",
            distance_miles=10.5,
            estimated_time_minutes=25,
            traffic_conditions=conditions,
            alternative_available=True
        )
        
        assert route.origin == "Start"
        assert route.destination == "End"
        assert route.distance_miles == 10.5
        assert route.estimated_time_minutes == 25
        assert len(route.traffic_conditions) == 1
        assert route.alternative_available is True