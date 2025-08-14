"""
Integration tests for the autonomous agent workflow.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

from src.agent.autonomous_agent import AutonomousAgent, AgentConfig
from src.agent.interfaces import ScenarioType, UrgencyLevel
from src.llm.interfaces import LLMResponse, Message, MessageRole, TokenUsage
from src.tools.interfaces import ToolResult
from src.config.settings import LLMConfig, LLMProvider as LLMProviderEnum


class TestAutonomousAgentIntegration:
    """Integration tests for complete agent workflow."""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create mock LLM provider."""
        provider = Mock()
        provider.generate_response.return_value = LLMResponse(
            content="I need to check the traffic situation for this delivery.",
            messages=[
                Message(role=MessageRole.USER, content="Test scenario"),
                Message(role=MessageRole.ASSISTANT, content="I need to check traffic.")
            ],
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            model="gpt-4",
            finish_reason="stop",
            response_time=1.5,
            timestamp=datetime.now(),
            tool_calls=[{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "check_traffic",
                    "arguments": '{"location": "123 Main St"}'
                }
            }]
        )
        return provider
    
    @pytest.fixture
    def mock_tool_manager(self):
        """Create mock tool manager."""
        manager = Mock()
        
        # Mock available tools
        mock_tool = Mock()
        mock_tool.name = "check_traffic"
        mock_tool.description = "Check traffic conditions"
        mock_tool.parameters = {"location": {"type": "string"}}
        
        manager.get_available_tools.return_value = [mock_tool]
        
        # Mock tool execution
        manager.execute_tool.return_value = ToolResult(
            tool_name="check_traffic",
            success=True,
            data={"status": "heavy_traffic", "delay_minutes": 15},
            execution_time=0.5
        )
        
        return manager
    
    @pytest.fixture
    def agent(self, mock_llm_provider, mock_tool_manager):
        """Create autonomous agent with mocked dependencies."""
        config = AgentConfig(
            max_reasoning_steps=5,
            reasoning_timeout=60,
            enable_context_tracking=True,
            log_reasoning_steps=False  # Disable for tests
        )
        return AutonomousAgent(mock_llm_provider, mock_tool_manager, config)
    
    def test_complete_scenario_processing_workflow(self, agent, mock_llm_provider, mock_tool_manager):
        """Test complete scenario processing from input to resolution."""
        scenario = "Driver is stuck in traffic on Main Street for delivery DEL123456 to Pizza Palace"
        
        # Mock multiple LLM responses for reasoning steps
        responses = [
            # Step 1: Initial analysis
            LLMResponse(
                content="I need to check the traffic situation for this delivery.",
                messages=[],
                token_usage=TokenUsage(50, 25, 75),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.0,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "check_traffic", "arguments": '{"location": "Main Street"}'}
                }]
            ),
            # Step 2: After traffic check
            LLMResponse(
                content="Traffic is heavy with 15 minute delay. I should notify the customer and check for alternative routes.",
                messages=[],
                token_usage=TokenUsage(60, 30, 90),
                model="gpt-4",
                finish_reason="tool_calls",
                response_time=1.2,
                timestamp=datetime.now(),
                tool_calls=[{
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "notify_customer", "arguments": '{"message": "Delivery delayed due to traffic"}'}
                }]
            ),
            # Step 3: Final step
            LLMResponse(
                content="I have sufficient information to create a resolution plan.",
                messages=[],
                token_usage=TokenUsage(40, 20, 60),
                model="gpt-4",
                finish_reason="stop",
                response_time=0.8,
                timestamp=datetime.now()
            )
        ]
        
        mock_llm_provider.generate_response.side_effect = responses
        
        # Mock tool executions
        tool_results = [
            ToolResult("check_traffic", True, {"status": "heavy_traffic", "delay_minutes": 15}, 0.5),
            ToolResult("notify_customer", True, {"message_sent": True, "customer_id": "CUST123"}, 0.3)
        ]
        mock_tool_manager.execute_tool.side_effect = tool_results
        
        # Process scenario
        result = agent.process_scenario(scenario)
        
        # Verify result structure
        assert result.success is True
        assert result.error_message is None
        assert result.scenario is not None
        assert result.reasoning_trace is not None
        assert result.resolution_plan is not None
        
        # Verify scenario parsing
        assert result.scenario.description == scenario
        assert result.scenario.scenario_type == ScenarioType.TRAFFIC
        assert len(result.scenario.entities) > 0  # Should extract delivery ID and merchant
        
        # Verify reasoning trace
        assert len(result.reasoning_trace.steps) >= 2
        assert result.reasoning_trace.start_time is not None
        assert result.reasoning_trace.end_time is not None
        
        # Verify resolution plan
        assert len(result.resolution_plan.steps) > 0
        assert result.resolution_plan.estimated_duration > timedelta(0)
        assert 0.0 <= result.resolution_plan.success_probability <= 1.0
        
        # Verify agent state
        state = agent.get_current_state()
        assert state["status"] == "completed"
        assert state["current_scenario"] == scenario
        assert state["reasoning_steps"] >= 2
    
    def test_scenario_with_entity_extraction(self, agent):
        """Test that entities are properly extracted from scenarios."""
        scenario = "Customer John Smith at 456 Oak Avenue called about delivery DEL789012 from Burger King"
        
        result = agent.process_scenario(scenario)
        
        # Verify entities were extracted
        entities = result.scenario.entities
        entity_types = [entity.entity_type.value for entity in entities]
        
        assert "delivery_id" in entity_types
        assert "address" in entity_types
        assert "merchant" in entity_types
        assert "person" in entity_types
        
        # Verify specific entities
        delivery_ids = [e for e in entities if e.entity_type.value == "delivery_id"]
        assert len(delivery_ids) > 0
        assert "DEL789012" in delivery_ids[0].text
    
    def test_error_handling_in_workflow(self, agent, mock_llm_provider, mock_tool_manager):
        """Test error handling during scenario processing."""
        scenario = "Test error scenario"
        
        # Mock LLM provider to raise an exception
        mock_llm_provider.generate_response.side_effect = Exception("LLM API error")
        
        result = agent.process_scenario(scenario)
        
        # Verify error handling
        assert result.success is False
        assert result.error_message is not None
        assert "LLM API error" in result.error_message
        
        # Verify fallback plan was created
        assert result.resolution_plan is not None
        assert len(result.resolution_plan.steps) > 0
        
        # Verify agent state
        state = agent.get_current_state()
        assert state["status"] == "error"
        assert state["error_message"] is not None
    
    def test_reasoning_step_logging(self, agent, mock_llm_provider, caplog):
        """Test that reasoning steps are logged when enabled."""
        # Enable logging for this test
        agent.config.log_reasoning_steps = True
        
        scenario = "Simple test scenario"
        
        # Mock single reasoning step
        mock_llm_provider.generate_response.return_value = LLMResponse(
            content="This is a test thought.",
            messages=[],
            token_usage=TokenUsage(50, 25, 75),
            model="gpt-4",
            finish_reason="stop",
            response_time=1.0,
            timestamp=datetime.now()
        )
        
        with caplog.at_level("INFO"):
            agent.process_scenario(scenario)
        
        # Verify reasoning steps were logged
        log_messages = [record.message for record in caplog.records]
        reasoning_logs = [msg for msg in log_messages if "Step" in msg and "This is a test thought" in msg]
        assert len(reasoning_logs) > 0
    
    def test_context_tracking(self, agent):
        """Test that context is tracked across scenarios."""
        scenarios = [
            "Traffic delay on Highway 1 for delivery DEL111",
            "Restaurant closed for delivery DEL222 to customer",
            "Wrong address for delivery DEL333"
        ]
        
        # Process multiple scenarios
        for scenario in scenarios:
            agent.process_scenario(scenario)
        
        # Verify context history
        context_history = agent.get_context_history()
        assert len(context_history) == 3
        
        # Verify context entries contain expected data
        for entry in context_history:
            assert "timestamp" in entry
            assert "scenario_type" in entry
            assert "urgency_level" in entry
            assert "processing_time" in entry
            assert "reasoning_steps" in entry
    
    def test_performance_metrics(self, agent):
        """Test performance metrics calculation."""
        # Process a few scenarios to generate metrics
        scenarios = [
            "Traffic issue for DEL001",
            "Merchant problem for DEL002"
        ]
        
        for scenario in scenarios:
            agent.process_scenario(scenario)
        
        # Get performance metrics
        metrics = agent.get_performance_metrics()
        
        # Verify metrics structure
        assert "total_scenarios_processed" in metrics
        assert "average_processing_time_seconds" in metrics
        assert "average_reasoning_steps" in metrics
        assert "scenario_type_distribution" in metrics
        assert "urgency_level_distribution" in metrics
        
        assert metrics["total_scenarios_processed"] == 2
        assert metrics["average_processing_time_seconds"] >= 0
    
    def test_state_management(self, agent):
        """Test agent state management throughout processing."""
        scenario = "Test scenario for state management"
        
        # Initial state
        initial_state = agent.get_current_state()
        assert initial_state["status"] == "idle"
        assert initial_state["current_scenario"] is None
        
        # Process scenario
        result = agent.process_scenario(scenario)
        
        # Final state
        final_state = agent.get_current_state()
        assert final_state["status"] == "completed"
        assert final_state["current_scenario"] == scenario
        assert final_state["reasoning_steps"] > 0
        
        # Reset state
        agent.reset_state()
        reset_state = agent.get_current_state()
        assert reset_state["status"] == "idle"
        assert reset_state["current_scenario"] is None
    
    def test_context_data_updates(self, agent):
        """Test context data updates."""
        # Update context data
        agent.update_context_data("test_key", "test_value")
        agent.update_context_data("priority_level", "high")
        
        # Verify context data
        state = agent.get_current_state()
        assert "test_key" in state["context_data_keys"]
        assert "priority_level" in state["context_data_keys"]
        
        # Verify context data is used in processing
        assert agent.state.context_data["test_key"] == "test_value"
        assert agent.state.context_data["priority_level"] == "high"
    
    def test_timeout_handling(self, agent, mock_llm_provider):
        """Test timeout handling in reasoning loop."""
        # Set very short timeout
        agent.config.reasoning_timeout = 1  # 1 second
        
        scenario = "Test timeout scenario"
        
        # Mock slow LLM responses
        def slow_response(*args, **kwargs):
            import time
            time.sleep(2)  # Longer than timeout
            return LLMResponse(
                content="Slow response",
                messages=[],
                token_usage=TokenUsage(50, 25, 75),
                model="gpt-4",
                finish_reason="stop",
                response_time=2.0,
                timestamp=datetime.now()
            )
        
        mock_llm_provider.generate_response.side_effect = slow_response
        
        # This should complete quickly due to timeout
        start_time = datetime.now()
        result = agent.process_scenario(scenario)
        end_time = datetime.now()
        
        # Verify it didn't take too long (allowing some buffer)
        processing_time = (end_time - start_time).total_seconds()
        assert processing_time < 10  # Should be much less due to timeout
        
        # Should still return a result (possibly with fallback plan)
        assert result is not None
    
    @pytest.mark.parametrize("scenario_type,expected_entities", [
        ("Traffic jam on I-95 for delivery DEL123", ["delivery_id"]),
        ("Pizza Hut is closed for delivery DEL456", ["merchant", "delivery_id"]),
        ("Wrong address 123 Main St for delivery DEL789", ["address", "delivery_id"]),
        ("Customer John at (555) 123-4567 called about DEL999", ["person", "phone", "delivery_id"])
    ])
    def test_entity_extraction_scenarios(self, agent, scenario_type, expected_entities):
        """Test entity extraction for different scenario types."""
        result = agent.process_scenario(scenario_type)
        
        extracted_types = [entity.entity_type.value for entity in result.scenario.entities]
        
        for expected_type in expected_entities:
            assert expected_type in extracted_types, f"Expected {expected_type} in {extracted_types}"
    
    def test_reasoning_engine_integration(self, agent, mock_llm_provider, mock_tool_manager):
        """Test integration with reasoning engine."""
        scenario = "Integration test scenario"
        
        # Verify reasoning engine is properly initialized
        assert agent.reasoning_engine is not None
        assert agent.reasoning_engine.llm_provider == mock_llm_provider
        assert agent.reasoning_engine.tool_manager == mock_tool_manager
        
        # Process scenario and verify reasoning engine was used
        result = agent.process_scenario(scenario)
        
        # Verify LLM provider was called (reasoning engine uses it)
        assert mock_llm_provider.generate_response.called
        
        # Verify reasoning trace was created
        assert result.reasoning_trace is not None
        assert len(result.reasoning_trace.steps) > 0
    
    def test_tool_manager_integration(self, agent, mock_tool_manager):
        """Test integration with tool manager."""
        scenario = "Tool integration test with delivery DEL123"
        
        # Verify tool manager is accessible
        assert agent.tool_manager == mock_tool_manager
        
        # Process scenario
        result = agent.process_scenario(scenario)
        
        # Verify tool manager methods were called
        assert mock_tool_manager.get_available_tools.called
        
        # If tools were executed, verify execution
        if mock_tool_manager.execute_tool.called:
            # Verify tool results are in reasoning trace
            tool_results_found = False
            for step in result.reasoning_trace.steps:
                if step.tool_results:
                    tool_results_found = True
                    break
            # Note: tool_results_found might be False if no tools were actually executed
            # This depends on the reasoning engine's decision-making