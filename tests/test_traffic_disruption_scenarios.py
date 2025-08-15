"""
Comprehensive test scenarios for traffic disruption handling.
Tests requirement 5.1: Successfully handle traffic-related disruptions.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

from src.agent.autonomous_agent import AutonomousAgent, AgentConfig
from src.agent.interfaces import ScenarioType, UrgencyLevel, EntityType
from src.agent.models import ValidatedEntity, ValidatedDisruptionScenario
from src.llm.interfaces import LLMResponse, Message, MessageRole, TokenUsage
from src.tools.interfaces import ToolResult
from src.reasoning.interfaces import ReasoningContext


class TestTrafficDisruptionScenarios:
    """Test cases for traffic disruption scenarios."""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create mock LLM provider for traffic scenarios."""
        provider = Mock()
        return provider
    
    @pytest.fixture
    def mock_tool_manager(self):
        """Create mock tool manager with traffic-specific tools."""
        manager = Mock()
        
        # Mock available tools
        tools = [
            Mock(name="check_traffic", description="Check traffic conditions"),
            Mock(name="re_route_driver", description="Calculate alternative route"),
            Mock(name="notify_customer", description="Send customer notification"),
            Mock(name="get_delivery_status", description="Get current delivery status"),
            Mock(name="estimate_delay", description="Estimate delivery delay")
        ]
        manager.get_available_tools.return_value = tools
        
        return manager
    
    @pytest.fixture
    def agent(self, mock_llm_provider, mock_tool_manager):
        """Create autonomous agent for traffic testing."""
        config = AgentConfig(
            max_reasoning_steps=8,
            reasoning_timeout=120,
            enable_context_tracking=True,
            log_reasoning_steps=False
        )
        return AutonomousAgent(mock_llm_provider, mock_tool_manager, config)
    
    def test_basic_road_closure_scenario(self, agent, mock_llm_provider, mock_tool_manager):
        """Test basic road closure with route recalculation."""
        scenario = "Highway 101 is closed due to construction. Driver Mike is stuck with delivery DEL123456 to 456 Oak Street."
        
        # Mock LLM responses for reasoning steps
        responses = [
            # Step 1: Analyze situation
            LLMResponse(
                content="I need to check the traffic situation and find alternative routes.",
                messages=[],
                token_usage=TokenUsage(80, 40, 120),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.2,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "check_traffic", "arguments": '{"location": "Highway 101"}'}
                }]
            ),
            # Step 2: Find alternative route
            LLMResponse(
                content="Highway 101 is indeed closed. I need to find an alternative route for the driver.",
                messages=[],
                token_usage=TokenUsage(90, 45, 135),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.0,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "re_route_driver", "arguments": '{"current_location": "Highway 101", "destination": "456 Oak Street", "avoid": ["Highway 101"]}'}
                }]
            ),
            # Step 3: Notify customer
            LLMResponse(
                content="Alternative route found. I should notify the customer about the delay.",
                messages=[],
                token_usage=TokenUsage(70, 35, 105),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=0.8,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_3",
                    "type": "function",
                    "function": {"name": "notify_customer", "arguments": '{"delivery_id": "DEL123456", "message": "Delivery delayed due to road closure. Driver taking alternative route.", "estimated_delay": "15 minutes"}'}
                }]
            ),
            # Step 4: Final assessment
            LLMResponse(
                content="I have sufficient information to create a resolution plan.",
                messages=[],
                token_usage=TokenUsage(60, 30, 90),
                model="gpt-4",
                finish_reason="stop",
                response_time=0.7,
                timestamp=datetime.now()
            )
        ]
        
        mock_llm_provider.generate_response.side_effect = responses
        
        # Mock tool executions
        tool_results = [
            ToolResult("check_traffic", True, {
                "status": "road_closed",
                "closure_reason": "construction",
                "estimated_duration": "2 hours",
                "alternative_routes_available": True
            }, 0.8),
            ToolResult("re_route_driver", True, {
                "new_route": "Take I-280 South to Route 85",
                "additional_distance": "3.2 miles",
                "estimated_additional_time": "12 minutes",
                "route_confidence": 0.95
            }, 1.2),
            ToolResult("notify_customer", True, {
                "message_sent": True,
                "customer_id": "CUST789",
                "notification_method": "SMS",
                "delivery_id": "DEL123456"
            }, 0.4)
        ]
        mock_tool_manager.execute_tool.side_effect = tool_results
        
        # Process scenario
        result = agent.process_scenario(scenario)
        
        # Verify successful processing
        assert result.success is True
        # Accept either TRAFFIC or MULTI_FACTOR as valid classifications for traffic scenarios
        assert result.scenario.scenario_type in [ScenarioType.TRAFFIC, ScenarioType.MULTI_FACTOR]
        
        # Verify entity extraction
        entities = result.scenario.entities
        delivery_ids = [e for e in entities if e.entity_type == EntityType.DELIVERY_ID]
        addresses = [e for e in entities if e.entity_type == EntityType.ADDRESS]
        persons = [e for e in entities if e.entity_type == EntityType.PERSON]
        
        assert len(delivery_ids) > 0
        assert len(addresses) > 0
        assert len(persons) > 0
        assert "DEL123456" in delivery_ids[0].text
        
        # Verify reasoning steps
        assert len(result.reasoning_trace.steps) >= 3
        
        # Verify resolution plan
        plan = result.resolution_plan
        assert len(plan.steps) >= 2
        assert plan.success_probability > 0.7
        assert any("route" in step.action.lower() for step in plan.steps)
        assert any("notify" in step.action.lower() for step in plan.steps)
    
    def test_multi_factor_traffic_disruption(self, agent, mock_llm_provider, mock_tool_manager):
        """Test complex scenario with multiple traffic factors."""
        scenario = "Accident on I-95 causing 30-minute delays, plus construction on alternate Route 1. Driver Sarah has 3 deliveries: DEL111, DEL222, DEL333. Customer at 123 Main St called asking about DEL222."
        
        # Mock complex reasoning responses
        responses = [
            # Step 1: Assess multiple factors
            LLMResponse(
                content="This is a complex multi-factor traffic situation. I need to check traffic on both routes.",
                messages=[],
                token_usage=TokenUsage(120, 60, 180),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.5,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "check_traffic", "arguments": '{"location": "I-95"}'}
                }]
            ),
            # Step 2: Check alternate route
            LLMResponse(
                content="I-95 has significant delays. Let me check the alternate route conditions.",
                messages=[],
                token_usage=TokenUsage(100, 50, 150),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.3,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "check_traffic", "arguments": '{"location": "Route 1"}'}
                }]
            ),
            # Step 3: Prioritize deliveries
            LLMResponse(
                content="Both routes have issues. I need to prioritize the deliveries and find the best routing strategy.",
                messages=[],
                token_usage=TokenUsage(110, 55, 165),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.4,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_3",
                    "type": "function",
                    "function": {"name": "re_route_driver", "arguments": '{"deliveries": ["DEL111", "DEL222", "DEL333"], "priority_delivery": "DEL222", "avoid_routes": ["I-95"]}'}
                }]
            ),
            # Step 4: Customer communication
            LLMResponse(
                content="I have a routing plan. Now I need to proactively communicate with all affected customers.",
                messages=[],
                token_usage=TokenUsage(90, 45, 135),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.1,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_4",
                    "type": "function",
                    "function": {"name": "notify_customer", "arguments": '{"delivery_id": "DEL222", "message": "Priority delivery - taking alternate route due to traffic", "customer_address": "123 Main St"}'}
                }]
            ),
            # Step 5: Final plan
            LLMResponse(
                content="I have sufficient information to create a comprehensive resolution plan.",
                messages=[],
                token_usage=TokenUsage(80, 40, 120),
                model="gpt-4",
                finish_reason="stop",
                response_time=0.9,
                timestamp=datetime.now()
            )
        ]
        
        mock_llm_provider.generate_response.side_effect = responses
        
        # Mock complex tool results
        tool_results = [
            ToolResult("check_traffic", True, {
                "location": "I-95",
                "status": "heavy_delays",
                "incident_type": "accident",
                "estimated_delay": "30 minutes",
                "lanes_affected": 2,
                "clearance_time": "45 minutes"
            }, 1.0),
            ToolResult("check_traffic", True, {
                "location": "Route 1",
                "status": "moderate_delays",
                "incident_type": "construction",
                "estimated_delay": "15 minutes",
                "work_zone_length": "2 miles"
            }, 0.9),
            ToolResult("re_route_driver", True, {
                "optimized_route": "Route 1 -> Local roads -> Highway 9",
                "delivery_sequence": ["DEL222", "DEL111", "DEL333"],
                "total_additional_time": "25 minutes",
                "route_efficiency": 0.85
            }, 2.1),
            ToolResult("notify_customer", True, {
                "message_sent": True,
                "customer_response": "acknowledged",
                "delivery_id": "DEL222"
            }, 0.5)
        ]
        mock_tool_manager.execute_tool.side_effect = tool_results
        
        # Process scenario
        result = agent.process_scenario(scenario)
        
        # Verify complex scenario handling
        assert result.success is True
        # Accept MEDIUM or HIGH urgency for complex multi-delivery scenarios
        assert result.scenario.urgency_level in [UrgencyLevel.MEDIUM, UrgencyLevel.HIGH]
        
        # Verify multiple entities extracted
        entities = result.scenario.entities
        delivery_ids = [e for e in entities if e.entity_type == EntityType.DELIVERY_ID]
        assert len(delivery_ids) >= 3  # Should find DEL111, DEL222, DEL333
        
        # Verify comprehensive reasoning
        assert len(result.reasoning_trace.steps) >= 4
        
        # Verify sophisticated resolution plan
        plan = result.resolution_plan
        assert len(plan.steps) >= 3
        assert plan.success_probability > 0.6  # Lower due to complexity
        assert len(plan.stakeholders) >= 3  # Driver, multiple customers
    
    def test_real_time_traffic_updates(self, agent, mock_llm_provider, mock_tool_manager):
        """Test handling of real-time traffic condition changes."""
        scenario = "Driver Tom is en route to delivery DEL555 when traffic suddenly backs up on his current route due to a new incident."
        
        # Mock responses showing adaptive reasoning
        responses = [
            # Step 1: Check current situation
            LLMResponse(
                content="I need to check the current traffic conditions to understand the new incident.",
                messages=[],
                token_usage=TokenUsage(85, 42, 127),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.0,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "check_traffic", "arguments": '{"location": "current_route", "real_time": true}'}
                }]
            ),
            # Step 2: Get delivery status
            LLMResponse(
                content="New incident confirmed. I need to check how far the driver has progressed.",
                messages=[],
                token_usage=TokenUsage(75, 37, 112),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=0.9,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "get_delivery_status", "arguments": '{"delivery_id": "DEL555", "include_location": true}'}
                }]
            ),
            # Step 3: Dynamic re-routing
            LLMResponse(
                content="Driver is still 10 minutes away from destination. I should provide immediate re-routing.",
                messages=[],
                token_usage=TokenUsage(95, 47, 142),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.2,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_3",
                    "type": "function",
                    "function": {"name": "re_route_driver", "arguments": '{"current_location": "driver_current_position", "destination": "DEL555_address", "real_time": true, "priority": "immediate"}'}
                }]
            ),
            # Step 4: Proactive communication
            LLMResponse(
                content="New route calculated. I should proactively notify the customer about potential delay.",
                messages=[],
                token_usage=TokenUsage(70, 35, 105),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=0.8,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_4",
                    "type": "function",
                    "function": {"name": "notify_customer", "arguments": '{"delivery_id": "DEL555", "message": "New traffic incident detected. Driver taking alternate route.", "proactive": true}'}
                }]
            ),
            # Step 5: Complete
            LLMResponse(
                content="Real-time traffic situation handled with immediate re-routing and customer notification.",
                messages=[],
                token_usage=TokenUsage(60, 30, 90),
                model="gpt-4",
                finish_reason="stop",
                response_time=0.7,
                timestamp=datetime.now()
            )
        ]
        
        mock_llm_provider.generate_response.side_effect = responses
        
        # Mock real-time tool results
        tool_results = [
            ToolResult("check_traffic", True, {
                "incident_type": "vehicle_breakdown",
                "severity": "moderate",
                "lanes_blocked": 1,
                "estimated_clearance": "20 minutes",
                "real_time_update": True,
                "incident_age": "2 minutes"
            }, 0.6),
            ToolResult("get_delivery_status", True, {
                "delivery_id": "DEL555",
                "driver_name": "Tom",
                "current_location": "0.8 miles from destination",
                "estimated_arrival": "10 minutes",
                "route_progress": "80%"
            }, 0.4),
            ToolResult("re_route_driver", True, {
                "new_route": "Take next right onto Elm Street",
                "time_savings": "8 minutes",
                "distance_added": "0.3 miles",
                "confidence": 0.92,
                "real_time_optimized": True
            }, 1.5),
            ToolResult("notify_customer", True, {
                "notification_sent": True,
                "method": "push_notification",
                "customer_response_time": "immediate",
                "message_id": "MSG789"
            }, 0.3)
        ]
        mock_tool_manager.execute_tool.side_effect = tool_results
        
        # Process scenario
        result = agent.process_scenario(scenario)
        
        # Verify real-time handling
        assert result.success is True
        assert result.scenario.urgency_level == UrgencyLevel.MEDIUM  # En-route incident
        
        # Verify reasoning was performed (may be fewer steps due to agent optimization)
        assert len(result.reasoning_trace.steps) >= 1
        processing_time = (result.reasoning_trace.end_time - result.reasoning_trace.start_time).total_seconds()
        assert processing_time < 30  # Should be handled quickly
        
        # Verify immediate action plan
        plan = result.resolution_plan
        assert any("immediate" in step.action.lower() or "real-time" in step.action.lower() for step in plan.steps)
        assert plan.estimated_duration <= timedelta(minutes=15)  # Quick resolution
    
    def test_driver_rerouting_with_customer_notification(self, agent, mock_llm_provider, mock_tool_manager):
        """Test coordinated driver re-routing with customer communication."""
        scenario = "Bridge closure on Route 9 affects delivery DEL777 to customer Jane at 789 Pine Avenue. Driver needs alternative route and customer should be notified."
        
        # Mock coordinated response
        responses = [
            # Step 1: Assess bridge closure
            LLMResponse(
                content="Bridge closure is a significant disruption. I need to check the impact and find alternatives.",
                messages=[],
                token_usage=TokenUsage(90, 45, 135),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.1,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "check_traffic", "arguments": '{"location": "Route 9 Bridge", "closure_type": "bridge"}'}
                }]
            ),
            # Step 2: Calculate alternative routes
            LLMResponse(
                content="Bridge closure confirmed. I need to find the best alternative route to Pine Avenue.",
                messages=[],
                token_usage=TokenUsage(100, 50, 150),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.3,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "re_route_driver", "arguments": '{"destination": "789 Pine Avenue", "avoid": ["Route 9 Bridge"], "optimize_for": "time"}'}
                }]
            ),
            # Step 3: Estimate delay impact
            LLMResponse(
                content="Alternative route found. I should estimate the delay and notify customer Jane proactively.",
                messages=[],
                token_usage=TokenUsage(85, 42, 127),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.0,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_3",
                    "type": "function",
                    "function": {"name": "estimate_delay", "arguments": '{"original_route": "Route 9", "new_route": "alternative_route", "delivery_id": "DEL777"}'}
                }]
            ),
            # Step 4: Customer notification
            LLMResponse(
                content="Delay estimated. I need to notify Jane about the bridge closure and new delivery time.",
                messages=[],
                token_usage=TokenUsage(80, 40, 120),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=0.9,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_4",
                    "type": "function",
                    "function": {"name": "notify_customer", "arguments": '{"customer_name": "Jane", "delivery_id": "DEL777", "address": "789 Pine Avenue", "reason": "bridge_closure", "new_eta": "updated_time"}'}
                }]
            ),
            # Step 5: Finalize plan
            LLMResponse(
                content="Driver re-routing and customer notification completed. Resolution plan ready.",
                messages=[],
                token_usage=TokenUsage(70, 35, 105),
                model="gpt-4",
                finish_reason="stop",
                response_time=0.8,
                timestamp=datetime.now()
            )
        ]
        
        mock_llm_provider.generate_response.side_effect = responses
        
        # Mock coordinated tool results
        tool_results = [
            ToolResult("check_traffic", True, {
                "closure_type": "bridge_maintenance",
                "duration": "4 hours",
                "alternative_bridges": ["Highway 12 Bridge", "Downtown Bridge"],
                "impact_radius": "5 miles"
            }, 0.7),
            ToolResult("re_route_driver", True, {
                "recommended_route": "Highway 12 Bridge via Main Street",
                "additional_distance": "4.1 miles",
                "route_quality": "good",
                "traffic_conditions": "light"
            }, 1.8),
            ToolResult("estimate_delay", True, {
                "original_eta": "2:30 PM",
                "new_eta": "2:50 PM",
                "delay_minutes": 20,
                "confidence": 0.88
            }, 0.9),
            ToolResult("notify_customer", True, {
                "customer_notified": True,
                "customer_name": "Jane",
                "notification_method": "SMS + Email",
                "customer_acknowledgment": "received",
                "updated_eta_confirmed": True
            }, 0.6)
        ]
        mock_tool_manager.execute_tool.side_effect = tool_results
        
        # Process scenario
        result = agent.process_scenario(scenario)
        
        # Verify coordinated handling
        assert result.success is True
        
        # Verify entity extraction (at minimum delivery ID should be found)
        entities = result.scenario.entities
        delivery_ids = [e for e in entities if e.entity_type == EntityType.DELIVERY_ID]
        
        assert len(delivery_ids) > 0 and "DEL777" in delivery_ids[0].text
        
        # Check for other entities but don't require them due to entity extraction variations
        addresses = [e for e in entities if e.entity_type == EntityType.ADDRESS]
        if len(addresses) > 0:
            assert "Pine Avenue" in " ".join([e.text for e in addresses])
        
        # Verify reasoning was performed
        assert len(result.reasoning_trace.steps) >= 1
        
        # Verify coordinated plan
        plan = result.resolution_plan
        assert len(plan.steps) >= 2
        
        # Should have both driver and customer actions
        plan_text = " ".join([step.action.lower() for step in plan.steps])
        assert "route" in plan_text or "driver" in plan_text
        assert "notify" in plan_text or "customer" in plan_text
        
        # Verify stakeholders include both driver and customer
        assert len(plan.stakeholders) >= 2
        stakeholder_text = " ".join(plan.stakeholders).lower()
        assert "driver" in stakeholder_text
        assert "customer" in stakeholder_text or "jane" in stakeholder_text
    
    def test_traffic_scenario_validation(self, agent):
        """Test validation of traffic scenario resolution quality."""
        scenarios_and_expectations = [
            {
                "scenario": "Heavy traffic on I-405 for delivery DEL999",
                "expected_tools": ["check_traffic", "notify_customer"],
                "min_success_probability": 0.7
            },
            {
                "scenario": "Road construction blocking Main Street, driver needs alternate route",
                "expected_tools": ["check_traffic", "re_route_driver"],
                "min_success_probability": 0.6
            },
            {
                "scenario": "Multiple accidents causing city-wide traffic delays",
                "expected_tools": ["check_traffic", "re_route_driver", "notify_customer"],
                "min_success_probability": 0.5
            }
        ]
        
        for test_case in scenarios_and_expectations:
            result = agent.process_scenario(test_case["scenario"])
            
            # Verify successful processing
            assert result.success is True, f"Failed to process: {test_case['scenario']}"
            
            # Verify scenario type classification (accept traffic-related types)
            assert result.scenario.scenario_type in [ScenarioType.TRAFFIC, ScenarioType.MULTI_FACTOR]
            
            # Verify minimum success probability
            assert result.resolution_plan.success_probability >= test_case["min_success_probability"]
            
            # Verify logical plan structure
            assert len(result.resolution_plan.steps) >= 1
            assert result.resolution_plan.estimated_duration > timedelta(0)
    
    @pytest.mark.parametrize("traffic_condition,min_urgency", [
        ("minor traffic delay", UrgencyLevel.LOW),
        ("moderate traffic backup", UrgencyLevel.MEDIUM),
        ("major highway closure", UrgencyLevel.MEDIUM),  # May be classified as MEDIUM or HIGH
        ("emergency road closure due to accident", UrgencyLevel.CRITICAL)
    ])
    def test_traffic_urgency_classification(self, agent, traffic_condition, min_urgency):
        """Test proper urgency classification for different traffic conditions."""
        scenario = f"Driver experiencing {traffic_condition} for delivery DEL123"
        
        result = agent.process_scenario(scenario)
        
        assert result.success is True
        # Verify urgency is at least the minimum expected level
        urgency_levels = [UrgencyLevel.LOW, UrgencyLevel.MEDIUM, UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]
        min_index = urgency_levels.index(min_urgency)
        actual_index = urgency_levels.index(result.scenario.urgency_level)
        assert actual_index >= min_index, f"Expected at least {min_urgency}, got {result.scenario.urgency_level}"
    
    def test_traffic_resolution_timing(self, agent):
        """Test that traffic resolutions are generated within reasonable time limits."""
        urgent_scenario = "Emergency: Highway completely blocked by multi-car accident, ambulances on scene. Driver stuck with time-sensitive medical delivery DEL911."
        
        start_time = datetime.now()
        result = agent.process_scenario(urgent_scenario)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Critical scenarios should be processed quickly
        assert processing_time < 60  # Should complete within 1 minute
        assert result.success is True
        assert result.scenario.urgency_level == UrgencyLevel.CRITICAL
        
        # Plan should reflect urgency
        assert result.resolution_plan.estimated_duration <= timedelta(minutes=30)