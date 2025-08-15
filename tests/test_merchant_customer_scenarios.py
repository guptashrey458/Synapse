"""
Comprehensive test scenarios for merchant and customer disruption handling.
Tests requirements 5.2 and 5.4: Successfully handle merchant availability issues and complex multi-factor disruptions.
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


class TestMerchantCustomerScenarios:
    """Test cases for merchant and customer disruption scenarios."""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create mock LLM provider for merchant/customer scenarios."""
        provider = Mock()
        return provider
    
    @pytest.fixture
    def mock_tool_manager(self):
        """Create mock tool manager with merchant and customer tools."""
        manager = Mock()
        
        # Mock available tools
        tools = [
            Mock(name="get_merchant_status", description="Check merchant availability and prep times"),
            Mock(name="get_nearby_merchants", description="Find alternative merchants"),
            Mock(name="notify_customer", description="Send customer notifications"),
            Mock(name="collect_evidence", description="Collect evidence for disputes"),
            Mock(name="issue_instant_refund", description="Process immediate refunds"),
            Mock(name="escalate_to_support", description="Escalate complex issues"),
            Mock(name="coordinate_replacement", description="Coordinate replacement orders")
        ]
        manager.get_available_tools.return_value = tools
        
        return manager
    
    @pytest.fixture
    def agent(self, mock_llm_provider, mock_tool_manager):
        """Create autonomous agent for merchant/customer testing."""
        config = AgentConfig(
            max_reasoning_steps=10,
            reasoning_timeout=180,
            enable_context_tracking=True,
            log_reasoning_steps=False
        )
        return AutonomousAgent(mock_llm_provider, mock_tool_manager, config)
    
    def test_overloaded_restaurant_scenario(self, agent, mock_llm_provider, mock_tool_manager):
        """Test overloaded restaurant with proactive customer communication."""
        scenario = "Pizza Palace is overwhelmed with orders and running 45 minutes behind. Customer Maria at (555) 123-4567 has been waiting for delivery DEL888 for over an hour."
        
        # Mock LLM responses for overloaded restaurant handling
        responses = [
            # Step 1: Check merchant status
            LLMResponse(
                content="I need to check the current status of Pizza Palace to understand the delay situation.",
                messages=[],
                token_usage=TokenUsage(95, 47, 142),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.2,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_merchant_status", "arguments": '{"merchant_name": "Pizza Palace", "include_queue_info": true}'}
                }]
            ),
            # Step 2: Assess alternatives
            LLMResponse(
                content="Pizza Palace is severely overloaded. I should check for nearby alternatives for future orders and address the current delay.",
                messages=[],
                token_usage=TokenUsage(110, 55, 165),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.4,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "get_nearby_merchants", "arguments": '{"location": "Pizza Palace", "cuisine_type": "pizza", "radius": "3 miles"}'}
                }]
            ),
            # Step 3: Proactive customer communication
            LLMResponse(
                content="I found alternatives but the current order is already in progress. I need to proactively communicate with Maria about the delay and offer options.",
                messages=[],
                token_usage=TokenUsage(120, 60, 180),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.5,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_3",
                    "type": "function",
                    "function": {"name": "notify_customer", "arguments": '{"customer_name": "Maria", "phone": "(555) 123-4567", "delivery_id": "DEL888", "message": "Restaurant experiencing high volume. Offering compensation for delay.", "offer_options": true}'}
                }]
            ),
            # Step 4: Coordinate compensation
            LLMResponse(
                content="Customer notified. I should coordinate appropriate compensation for the extended delay.",
                messages=[],
                token_usage=TokenUsage(85, 42, 127),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.0,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_4",
                    "type": "function",
                    "function": {"name": "issue_instant_refund", "arguments": '{"delivery_id": "DEL888", "refund_type": "partial", "reason": "excessive_delay", "amount": "delivery_fee_plus_tip"}'}
                }]
            ),
            # Step 5: Final coordination
            LLMResponse(
                content="I have addressed the immediate customer concern and provided compensation. Creating comprehensive resolution plan.",
                messages=[],
                token_usage=TokenUsage(75, 37, 112),
                model="gpt-4",
                finish_reason="stop",
                response_time=0.9,
                timestamp=datetime.now()
            )
        ]
        
        mock_llm_provider.generate_response.side_effect = responses
        
        # Mock tool results for overloaded restaurant
        tool_results = [
            ToolResult("get_merchant_status", True, {
                "merchant_name": "Pizza Palace",
                "status": "overloaded",
                "current_prep_time": "45 minutes",
                "normal_prep_time": "15 minutes",
                "orders_in_queue": 23,
                "staff_available": 3,
                "estimated_recovery_time": "2 hours"
            }, 1.1),
            ToolResult("get_nearby_merchants", True, {
                "alternatives": [
                    {"name": "Tony's Pizza", "distance": "1.2 miles", "prep_time": "20 minutes", "rating": 4.5},
                    {"name": "Mario's Italian", "distance": "2.1 miles", "prep_time": "25 minutes", "rating": 4.3}
                ],
                "total_found": 2
            }, 1.3),
            ToolResult("notify_customer", True, {
                "customer_contacted": True,
                "customer_name": "Maria",
                "response": "understanding",
                "preferred_option": "wait_with_compensation",
                "contact_method": "phone_call"
            }, 0.8),
            ToolResult("issue_instant_refund", True, {
                "refund_processed": True,
                "refund_amount": "$8.50",
                "refund_type": "delivery_fee_and_tip",
                "processing_time": "immediate",
                "customer_notification_sent": True
            }, 0.6)
        ]
        mock_tool_manager.execute_tool.side_effect = tool_results
        
        # Process scenario
        result = agent.process_scenario(scenario)
        
        # Verify successful processing
        assert result.success is True
        assert result.scenario.scenario_type == ScenarioType.MERCHANT
        # Accept MEDIUM or HIGH urgency for customer waiting over an hour
        assert result.scenario.urgency_level in [UrgencyLevel.MEDIUM, UrgencyLevel.HIGH]
        
        # Verify entity extraction
        entities = result.scenario.entities
        merchants = [e for e in entities if e.entity_type == EntityType.MERCHANT]
        persons = [e for e in entities if e.entity_type == EntityType.PERSON]
        phones = [e for e in entities if e.entity_type == EntityType.PHONE_NUMBER]
        delivery_ids = [e for e in entities if e.entity_type == EntityType.DELIVERY_ID]
        
        assert len(merchants) > 0 and "Pizza Palace" in merchants[0].text
        assert len(persons) > 0 and "Maria" in persons[0].text
        assert len(phones) > 0 and "555" in phones[0].text
        assert len(delivery_ids) > 0 and "DEL888" in delivery_ids[0].text
        
        # Verify reasoning was performed
        assert len(result.reasoning_trace.steps) >= 1
        
        # Verify proactive resolution plan
        plan = result.resolution_plan
        assert len(plan.steps) >= 3
        
        # Should include customer communication and compensation
        plan_text = " ".join([step.action.lower() for step in plan.steps])
        assert "customer" in plan_text or "notify" in plan_text
        assert "compensation" in plan_text or "refund" in plan_text
        
        # Should have high success probability due to proactive handling
        assert plan.success_probability > 0.8
    
    def test_damaged_packaging_dispute_resolution(self, agent, mock_llm_provider, mock_tool_manager):
        """Test damaged packaging dispute with evidence collection."""
        scenario = "Customer John received delivery DEL999 with damaged packaging and spilled food. He's requesting a full refund and threatening to leave a bad review. Driver claims packaging was intact when delivered."
        
        # Mock responses for dispute resolution
        responses = [
            # Step 1: Collect evidence from customer
            LLMResponse(
                content="This is a dispute situation requiring evidence collection. I need to gather information from the customer first.",
                messages=[],
                token_usage=TokenUsage(100, 50, 150),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.3,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "collect_evidence", "arguments": '{"delivery_id": "DEL999", "evidence_type": "customer_photos", "customer_name": "John"}'}
                }]
            ),
            # Step 2: Collect evidence from driver
            LLMResponse(
                content="Customer evidence collected. Now I need to get the driver's perspective and any delivery photos.",
                messages=[],
                token_usage=TokenUsage(90, 45, 135),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.1,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "collect_evidence", "arguments": '{"delivery_id": "DEL999", "evidence_type": "driver_photos", "include_delivery_confirmation": true}'}
                }]
            ),
            # Step 3: Assess merchant packaging
            LLMResponse(
                content="I have evidence from both sides. Let me check the merchant's packaging standards to understand the root cause.",
                messages=[],
                token_usage=TokenUsage(85, 42, 127),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.0,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_3",
                    "type": "function",
                    "function": {"name": "get_merchant_status", "arguments": '{"delivery_id": "DEL999", "check_packaging_standards": true, "recent_packaging_issues": true}'}
                }]
            ),
            # Step 4: Immediate resolution
            LLMResponse(
                content="Evidence suggests packaging failure at merchant level. I should provide immediate resolution to prevent negative review.",
                messages=[],
                token_usage=TokenUsage(110, 55, 165),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.4,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_4",
                    "type": "function",
                    "function": {"name": "issue_instant_refund", "arguments": '{"delivery_id": "DEL999", "customer_name": "John", "refund_type": "full", "reason": "packaging_failure", "include_credit": true}'}
                }]
            ),
            # Step 5: Coordinate replacement
            LLMResponse(
                content="Refund processed. I should coordinate a replacement order to fully resolve the situation.",
                messages=[],
                token_usage=TokenUsage(95, 47, 142),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.2,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_5",
                    "type": "function",
                    "function": {"name": "coordinate_replacement", "arguments": '{"original_delivery": "DEL999", "customer_name": "John", "priority": "high", "special_packaging": true}'}
                }]
            ),
            # Step 6: Final resolution
            LLMResponse(
                content="Comprehensive dispute resolution completed with evidence-based decision and proactive customer service.",
                messages=[],
                token_usage=TokenUsage(70, 35, 105),
                model="gpt-4",
                finish_reason="stop",
                response_time=0.8,
                timestamp=datetime.now()
            )
        ]
        
        mock_llm_provider.generate_response.side_effect = responses
        
        # Mock evidence collection and dispute resolution results
        tool_results = [
            ToolResult("collect_evidence", True, {
                "evidence_type": "customer_photos",
                "photos_received": 3,
                "damage_severity": "moderate",
                "customer_description": "sauce leaked through container",
                "timestamp": "2024-01-15 19:30:00"
            }, 0.9),
            ToolResult("collect_evidence", True, {
                "evidence_type": "driver_photos",
                "delivery_photo_available": True,
                "packaging_condition": "appeared_intact",
                "driver_notes": "customer seemed satisfied at delivery",
                "delivery_timestamp": "2024-01-15 19:25:00"
            }, 0.7),
            ToolResult("get_merchant_status", True, {
                "merchant_packaging_score": 3.2,
                "recent_packaging_complaints": 5,
                "packaging_standards": "below_average",
                "recommended_action": "merchant_training_needed"
            }, 1.0),
            ToolResult("issue_instant_refund", True, {
                "refund_amount": "$24.50",
                "refund_type": "full_order_plus_credit",
                "additional_credit": "$10.00",
                "customer_satisfaction_expected": "high",
                "processing_time": "immediate"
            }, 0.5),
            ToolResult("coordinate_replacement", True, {
                "replacement_order_id": "DEL999R",
                "estimated_delivery": "45 minutes",
                "special_packaging_applied": True,
                "merchant_notified": True,
                "priority_status": "expedited"
            }, 1.5)
        ]
        mock_tool_manager.execute_tool.side_effect = tool_results
        
        # Process scenario
        result = agent.process_scenario(scenario)
        
        # Verify dispute resolution success
        assert result.success is True
        assert result.scenario.scenario_type in [ScenarioType.CUSTOMER, ScenarioType.MULTI_FACTOR]
        assert result.scenario.urgency_level == UrgencyLevel.HIGH  # Threat of bad review
        
        # Verify entity extraction
        entities = result.scenario.entities
        persons = [e for e in entities if e.entity_type == EntityType.PERSON]
        delivery_ids = [e for e in entities if e.entity_type == EntityType.DELIVERY_ID]
        
        assert len(persons) > 0 and "John" in persons[0].text
        assert len(delivery_ids) > 0 and "DEL999" in delivery_ids[0].text
        
        # Verify reasoning was performed
        assert len(result.reasoning_trace.steps) >= 1
        
        # Verify comprehensive resolution plan
        plan = result.resolution_plan
        assert len(plan.steps) >= 4
        
        # Should include evidence collection, refund, and replacement
        plan_text = " ".join([step.action.lower() for step in plan.steps])
        assert "evidence" in plan_text or "collect" in plan_text
        assert "refund" in plan_text
        assert "replacement" in plan_text or "coordinate" in plan_text
        
        # Should have high success probability due to comprehensive handling
        assert plan.success_probability > 0.85
    
    def test_complex_multi_stakeholder_coordination(self, agent, mock_llm_provider, mock_tool_manager):
        """Test complex scenario requiring coordination between multiple stakeholders."""
        scenario = "Restaurant fire at Burger King caused evacuation during peak hours. 15 active deliveries affected including DEL001, DEL002, DEL003. Customers are calling asking about their orders. Drivers are waiting outside. Health department is investigating."
        
        # Mock responses for complex multi-stakeholder scenario
        responses = [
            # Step 1: Assess emergency situation
            LLMResponse(
                content="This is an emergency situation affecting multiple stakeholders. I need to assess the merchant status first.",
                messages=[],
                token_usage=TokenUsage(120, 60, 180),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.5,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_merchant_status", "arguments": '{"merchant_name": "Burger King", "emergency_status": true, "affected_orders": ["DEL001", "DEL002", "DEL003"]}'}
                }]
            ),
            # Step 2: Find alternative merchants
            LLMResponse(
                content="Restaurant is closed due to fire emergency. I need to find alternatives for affected customers.",
                messages=[],
                token_usage=TokenUsage(110, 55, 165),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.4,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "get_nearby_merchants", "arguments": '{"location": "Burger King", "cuisine_type": "fast_food", "radius": "5 miles", "capacity_check": true, "emergency_priority": true}'}
                }]
            ),
            # Step 3: Mass customer notification
            LLMResponse(
                content="Alternative merchants identified. I need to notify all affected customers about the emergency and their options.",
                messages=[],
                token_usage=TokenUsage(130, 65, 195),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.6,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_3",
                    "type": "function",
                    "function": {"name": "notify_customer", "arguments": '{"delivery_ids": ["DEL001", "DEL002", "DEL003"], "message_type": "emergency_notification", "offer_alternatives": true, "offer_refund": true}'}
                }]
            ),
            # Step 4: Coordinate driver reassignment
            LLMResponse(
                content="Customers notified. I need to coordinate with drivers who are waiting and reassign them appropriately.",
                messages=[],
                token_usage=TokenUsage(100, 50, 150),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.3,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_4",
                    "type": "function",
                    "function": {"name": "coordinate_replacement", "arguments": '{"emergency_reassignment": true, "affected_drivers": "multiple", "alternative_merchants": "nearby_options", "priority": "critical"}'}
                }]
            ),
            # Step 5: Escalate to support
            LLMResponse(
                content="Immediate actions taken. This situation requires escalation to support team for ongoing coordination.",
                messages=[],
                token_usage=TokenUsage(90, 45, 135),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.1,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_5",
                    "type": "function",
                    "function": {"name": "escalate_to_support", "arguments": '{"incident_type": "merchant_emergency", "severity": "critical", "affected_orders": 15, "requires_ongoing_coordination": true}'}
                }]
            ),
            # Step 6: Final coordination
            LLMResponse(
                content="Emergency response plan activated with multi-stakeholder coordination and support escalation.",
                messages=[],
                token_usage=TokenUsage(80, 40, 120),
                model="gpt-4",
                finish_reason="stop",
                response_time=1.0,
                timestamp=datetime.now()
            )
        ]
        
        mock_llm_provider.generate_response.side_effect = responses
        
        # Mock emergency coordination results
        tool_results = [
            ToolResult("get_merchant_status", True, {
                "merchant_status": "emergency_closed",
                "incident_type": "fire",
                "estimated_closure": "24+ hours",
                "affected_orders": 15,
                "orders_in_preparation": 8,
                "orders_ready": 0,
                "emergency_services_on_site": True
            }, 1.2),
            ToolResult("get_nearby_merchants", True, {
                "emergency_alternatives": [
                    {"name": "McDonald's", "distance": "0.8 miles", "capacity": "high", "willing_to_help": True},
                    {"name": "Wendy's", "distance": "1.2 miles", "capacity": "medium", "willing_to_help": True},
                    {"name": "Subway", "distance": "0.5 miles", "capacity": "low", "willing_to_help": True}
                ],
                "total_capacity": "sufficient_for_emergency"
            }, 1.8),
            ToolResult("notify_customer", True, {
                "customers_notified": 15,
                "notification_method": "SMS_and_call",
                "customer_responses": {
                    "want_alternative": 8,
                    "want_refund": 5,
                    "will_wait": 2
                },
                "average_response_time": "3 minutes"
            }, 2.1),
            ToolResult("coordinate_replacement", True, {
                "drivers_reassigned": 8,
                "alternative_orders_placed": 8,
                "estimated_new_delivery_time": "45-60 minutes",
                "coordination_status": "in_progress"
            }, 2.5),
            ToolResult("escalate_to_support", True, {
                "escalation_successful": True,
                "support_team_assigned": "emergency_response_team",
                "case_priority": "critical",
                "ongoing_monitoring": True,
                "estimated_resolution_time": "4-6 hours"
            }, 0.8)
        ]
        mock_tool_manager.execute_tool.side_effect = tool_results
        
        # Process scenario
        result = agent.process_scenario(scenario)
        
        # Verify emergency handling
        assert result.success is True
        assert result.scenario.scenario_type == ScenarioType.MULTI_FACTOR
        assert result.scenario.urgency_level == UrgencyLevel.CRITICAL  # Emergency situation
        
        # Verify entity extraction
        entities = result.scenario.entities
        merchants = [e for e in entities if e.entity_type == EntityType.MERCHANT]
        delivery_ids = [e for e in entities if e.entity_type == EntityType.DELIVERY_ID]
        
        assert len(merchants) > 0 and "Burger King" in merchants[0].text
        assert len(delivery_ids) >= 3  # Should find DEL001, DEL002, DEL003
        
        # Verify reasoning was performed
        assert len(result.reasoning_trace.steps) >= 1
        
        # Verify multi-stakeholder coordination plan
        plan = result.resolution_plan
        assert len(plan.steps) >= 4
        assert len(plan.stakeholders) >= 4  # Customers, drivers, alternative merchants, support
        
        # Should include emergency response elements
        plan_text = " ".join([step.action.lower() for step in plan.steps])
        assert "emergency" in plan_text or "critical" in plan_text
        assert "alternative" in plan_text or "reassign" in plan_text
        assert "escalate" in plan_text or "support" in plan_text
        
        # Success probability should be moderate due to complexity
        assert 0.6 <= plan.success_probability <= 0.8
    
    def test_dispute_mediation_workflow(self, agent, mock_llm_provider, mock_tool_manager):
        """Test dispute mediation between customer and merchant."""
        scenario = "Customer Sarah claims her pizza from Tony's Pizza was cold and took 2 hours to arrive. Tony's Pizza says the order was delivered on time and hot. Both sides are demanding compensation. Delivery ID: DEL555."
        
        # Mock mediation workflow responses
        responses = [
            # Step 1: Collect evidence from customer
            LLMResponse(
                content="This requires careful mediation. I need to collect evidence from both parties starting with the customer.",
                messages=[],
                token_usage=TokenUsage(105, 52, 157),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.3,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "collect_evidence", "arguments": '{"delivery_id": "DEL555", "party": "customer", "customer_name": "Sarah", "evidence_types": ["delivery_time", "food_temperature", "photos"]}'}
                }]
            ),
            # Step 2: Collect evidence from merchant
            LLMResponse(
                content="Customer evidence collected. Now I need the merchant's side of the story and their records.",
                messages=[],
                token_usage=TokenUsage(95, 47, 142),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.2,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "collect_evidence", "arguments": '{"delivery_id": "DEL555", "party": "merchant", "merchant_name": "Tony\'s Pizza", "evidence_types": ["prep_time", "dispatch_time", "temperature_log"]}'}
                }]
            ),
            # Step 3: Get delivery tracking data
            LLMResponse(
                content="I have both sides' claims. Let me get objective delivery tracking data to mediate this dispute.",
                messages=[],
                token_usage=TokenUsage(90, 45, 135),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.1,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_3",
                    "type": "function",
                    "function": {"name": "collect_evidence", "arguments": '{"delivery_id": "DEL555", "evidence_type": "delivery_tracking", "include_gps_data": true, "include_timestamps": true}'}
                }]
            ),
            # Step 4: Mediated resolution
            LLMResponse(
                content="Based on evidence, there was a legitimate delay. I should provide a fair resolution that addresses both parties' concerns.",
                messages=[],
                token_usage=TokenUsage(110, 55, 165),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.4,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_4",
                    "type": "function",
                    "function": {"name": "issue_instant_refund", "arguments": '{"delivery_id": "DEL555", "customer_name": "Sarah", "refund_type": "partial", "reason": "delivery_delay_confirmed", "merchant_compensation": true}'}
                }]
            ),
            # Step 5: Final mediation
            LLMResponse(
                content="Fair resolution provided with evidence-based mediation. Both parties should be satisfied with the outcome.",
                messages=[],
                token_usage=TokenUsage(75, 37, 112),
                model="gpt-4",
                finish_reason="stop",
                response_time=0.9,
                timestamp=datetime.now()
            )
        ]
        
        mock_llm_provider.generate_response.side_effect = responses
        
        # Mock mediation evidence and resolution
        tool_results = [
            ToolResult("collect_evidence", True, {
                "customer_evidence": {
                    "claimed_delivery_time": "2 hours",
                    "food_condition": "cold",
                    "photos_provided": True,
                    "timestamp_photo": "2024-01-15 20:45:00"
                }
            }, 0.8),
            ToolResult("collect_evidence", True, {
                "merchant_evidence": {
                    "prep_completion": "18:30",
                    "dispatch_time": "18:35",
                    "claimed_delivery_time": "45 minutes",
                    "temperature_maintained": True
                }
            }, 0.9),
            ToolResult("collect_evidence", True, {
                "delivery_tracking": {
                    "actual_prep_time": "35 minutes",
                    "actual_delivery_time": "85 minutes",
                    "delay_cause": "traffic_and_multiple_stops",
                    "gps_confirmed_delay": True
                }
            }, 1.1),
            ToolResult("issue_instant_refund", True, {
                "mediated_resolution": True,
                "customer_refund": "$12.00",
                "merchant_compensation": "$8.00",
                "resolution_type": "shared_responsibility",
                "both_parties_notified": True
            }, 0.7)
        ]
        mock_tool_manager.execute_tool.side_effect = tool_results
        
        # Process scenario
        result = agent.process_scenario(scenario)
        
        # Verify mediation success
        assert result.success is True
        assert result.scenario.scenario_type == ScenarioType.CUSTOMER
        
        # Verify entity extraction
        entities = result.scenario.entities
        persons = [e for e in entities if e.entity_type == EntityType.PERSON]
        merchants = [e for e in entities if e.entity_type == EntityType.MERCHANT]
        delivery_ids = [e for e in entities if e.entity_type == EntityType.DELIVERY_ID]
        
        assert len(persons) > 0 and "Sarah" in persons[0].text
        assert len(merchants) > 0 and "Tony's Pizza" in merchants[0].text
        assert len(delivery_ids) > 0 and "DEL555" in delivery_ids[0].text
        
        # Verify reasoning was performed
        assert len(result.reasoning_trace.steps) >= 1
        
        # Verify fair resolution plan
        plan = result.resolution_plan
        assert len(plan.steps) >= 3
        
        # Should include evidence collection and fair resolution
        plan_text = " ".join([step.action.lower() for step in plan.steps])
        assert "evidence" in plan_text
        assert "mediate" in plan_text or "resolution" in plan_text
        
        # Should have high success probability due to evidence-based approach
        assert plan.success_probability > 0.8
    
    def test_instant_resolution_workflow(self, agent, mock_llm_provider, mock_tool_manager):
        """Test instant resolution for clear-cut customer issues."""
        scenario = "Customer received completely wrong order. Ordered vegetarian pizza, got meat lovers pizza. Customer is vegetarian for religious reasons and very upset. Delivery DEL777 from Green Garden Pizza."
        
        # Mock instant resolution responses
        responses = [
            # Step 1: Assess clear error
            LLMResponse(
                content="This is a clear order error with religious dietary implications. I need to provide immediate resolution.",
                messages=[],
                token_usage=TokenUsage(90, 45, 135),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.1,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "issue_instant_refund", "arguments": '{"delivery_id": "DEL777", "refund_type": "full_plus_credit", "reason": "wrong_order_dietary_restriction", "urgency": "immediate"}'}
                }]
            ),
            # Step 2: Coordinate replacement
            LLMResponse(
                content="Full refund processed. I should coordinate an immediate replacement order with special attention to dietary requirements.",
                messages=[],
                token_usage=TokenUsage(100, 50, 150),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.3,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "coordinate_replacement", "arguments": '{"original_delivery": "DEL777", "dietary_requirements": "vegetarian", "priority": "critical", "merchant": "Green Garden Pizza", "special_instructions": "verify_vegetarian"}'}
                }]
            ),
            # Step 3: Customer communication
            LLMResponse(
                content="Replacement coordinated. I need to personally communicate with the customer about the resolution and apologize for the error.",
                messages=[],
                token_usage=TokenUsage(85, 42, 127),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.0,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_3",
                    "type": "function",
                    "function": {"name": "notify_customer", "arguments": '{"delivery_id": "DEL777", "message_type": "personal_apology", "dietary_acknowledgment": true, "resolution_details": true, "priority_contact": true}'}
                }]
            ),
            # Step 4: Final resolution
            LLMResponse(
                content="Instant resolution provided with full refund, replacement order, and personal apology for dietary restriction violation.",
                messages=[],
                token_usage=TokenUsage(70, 35, 105),
                model="gpt-4",
                finish_reason="stop",
                response_time=0.8,
                timestamp=datetime.now()
            )
        ]
        
        mock_llm_provider.generate_response.side_effect = responses
        
        # Mock instant resolution results
        tool_results = [
            ToolResult("issue_instant_refund", True, {
                "refund_amount": "$18.50",
                "additional_credit": "$15.00",
                "refund_speed": "immediate",
                "reason_logged": "dietary_restriction_violation",
                "customer_priority_flagged": True
            }, 0.4),
            ToolResult("coordinate_replacement", True, {
                "replacement_order_id": "DEL777R",
                "estimated_delivery": "30 minutes",
                "dietary_verification": "triple_checked",
                "merchant_special_instructions": "vegetarian_only_prep_area",
                "priority_status": "critical"
            }, 1.2),
            ToolResult("notify_customer", True, {
                "personal_contact_made": True,
                "apology_delivered": True,
                "customer_response": "appreciative",
                "dietary_concerns_acknowledged": True,
                "satisfaction_level": "high"
            }, 0.6)
        ]
        mock_tool_manager.execute_tool.side_effect = tool_results
        
        # Process scenario
        result = agent.process_scenario(scenario)
        
        # Verify instant resolution
        assert result.success is True
        assert result.scenario.urgency_level == UrgencyLevel.CRITICAL  # Dietary restriction violation
        
        # Verify quick processing
        processing_time = (result.reasoning_trace.end_time - result.reasoning_trace.start_time).total_seconds()
        assert processing_time < 20  # Should be very quick for clear-cut cases
        
        # Verify comprehensive resolution
        plan = result.resolution_plan
        assert len(plan.steps) >= 3
        
        # Should include refund, replacement, and apology
        plan_text = " ".join([step.action.lower() for step in plan.steps])
        assert "refund" in plan_text
        assert "replacement" in plan_text or "coordinate" in plan_text
        assert "apology" in plan_text or "communicate" in plan_text
        
        # Should have very high success probability for clear-cut resolution
        assert plan.success_probability > 0.9
    
    @pytest.mark.parametrize("scenario_type,expected_tools", [
        ("Restaurant closed unexpectedly", ["get_merchant_status", "get_nearby_merchants", "notify_customer"]),
        ("Customer complaint about food quality", ["collect_evidence", "get_merchant_status", "issue_instant_refund"]),
        ("Multiple orders affected by kitchen fire", ["get_merchant_status", "get_nearby_merchants", "notify_customer", "escalate_to_support"]),
        ("Wrong delivery address dispute", ["collect_evidence", "notify_customer", "coordinate_replacement"])
    ])
    def test_merchant_customer_scenario_validation(self, agent, scenario_type, expected_tools):
        """Test validation of merchant/customer scenario handling."""
        scenario = f"{scenario_type} for delivery DEL123"
        
        result = agent.process_scenario(scenario)
        
        # Verify successful processing
        assert result.success is True
        
        # Verify appropriate scenario classification
        assert result.scenario.scenario_type in [ScenarioType.MERCHANT, ScenarioType.CUSTOMER, ScenarioType.MULTI_FACTOR]
        
        # Verify logical resolution plan
        assert len(result.resolution_plan.steps) >= 1
        assert result.resolution_plan.success_probability > 0.5
        
        # Verify stakeholder identification
        assert len(result.resolution_plan.stakeholders) >= 1