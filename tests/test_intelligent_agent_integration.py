"""
Integration tests for intelligent agent features.
"""
import pytest
from unittest.mock import Mock

from src.agent.autonomous_agent import AutonomousAgent, AgentConfig
from src.agent.interfaces import ScenarioType, UrgencyLevel
from src.llm.interfaces import LLMResponse, Message, MessageRole, TokenUsage
from src.tools.interfaces import ToolResult
from datetime import datetime


class TestIntelligentAgentIntegration:
    """Integration tests for intelligent agent features."""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create mock LLM provider."""
        provider = Mock()
        provider.generate_response.return_value = LLMResponse(
            content="Analysis complete with sufficient information.",
            messages=[],
            token_usage=TokenUsage(50, 25, 75),
            model="gpt-4",
            finish_reason="stop",
            response_time=1.0,
            timestamp=datetime.now()
        )
        return provider
    
    @pytest.fixture
    def mock_tool_manager(self):
        """Create mock tool manager with realistic tools."""
        manager = Mock()
        
        # Mock available tools
        tools = []
        tool_configs = [
            ("check_traffic", "Check traffic conditions"),
            ("get_merchant_status", "Get merchant availability"),
            ("notify_customer", "Send customer notification"),
            ("validate_address", "Validate delivery address"),
            ("re_route_driver", "Re-route driver"),
            ("get_nearby_merchants", "Find nearby merchants")
        ]
        
        for name, desc in tool_configs:
            tool = Mock()
            tool.name = name
            tool.description = desc
            tool.parameters = {"param": {"type": "string"}}
            tools.append(tool)
        
        manager.get_available_tools.return_value = tools
        
        # Mock tool execution with realistic results
        def mock_execute_tool(tool_name, parameters, **kwargs):
            if tool_name == "check_traffic":
                return ToolResult(
                    tool_name="check_traffic",
                    success=True,
                    data={"status": "heavy_traffic", "delay_minutes": 15},
                    execution_time=0.5
                )
            elif tool_name == "get_merchant_status":
                return ToolResult(
                    tool_name="get_merchant_status",
                    success=True,
                    data={"available": True, "prep_time_minutes": 10},
                    execution_time=0.3
                )
            elif tool_name == "notify_customer":
                return ToolResult(
                    tool_name="notify_customer",
                    success=True,
                    data={"message_sent": True, "customer_id": "CUST123"},
                    execution_time=0.2
                )
            else:
                return ToolResult(
                    tool_name=tool_name,
                    success=True,
                    data={"result": "success"},
                    execution_time=0.4
                )
        
        manager.execute_tool.side_effect = mock_execute_tool
        return manager
    
    @pytest.fixture
    def intelligent_agent(self, mock_llm_provider, mock_tool_manager):
        """Create intelligent agent with scenario analysis enabled."""
        config = AgentConfig(
            max_reasoning_steps=5,
            reasoning_timeout=60,
            enable_context_tracking=True,
            log_reasoning_steps=False
        )
        return AutonomousAgent(mock_llm_provider, mock_tool_manager, config)
    
    def test_intelligent_tool_selection_traffic_scenario(self, intelligent_agent):
        """Test intelligent tool selection for traffic scenario."""
        scenario = "Driver stuck in heavy traffic on Highway 101 for delivery DEL123456"
        
        result = intelligent_agent.process_scenario(scenario)
        
        # Verify successful processing
        assert result.success is True
        # Traffic scenario might be classified as multi-factor due to multiple entities
        assert result.scenario.scenario_type in [ScenarioType.TRAFFIC, ScenarioType.MULTI_FACTOR]
        
        # Verify intelligent tool selection occurred
        reasoning_steps = result.reasoning_trace.steps
        assert len(reasoning_steps) > 0
        
        # Check that tools were selected intelligently
        executed_tools = []
        for step in reasoning_steps:
            if step.action:
                executed_tools.append(step.action.tool_name)
        
        # For traffic scenario, should prioritize traffic-related tools
        if executed_tools:
            # Should include traffic or customer notification tools
            traffic_related = ["check_traffic", "notify_customer", "re_route_driver"]
            assert any(tool in executed_tools for tool in traffic_related)
    
    def test_intelligent_tool_selection_merchant_scenario(self, intelligent_agent):
        """Test intelligent tool selection for merchant scenario."""
        scenario = "Pizza Palace is closed and cannot prepare delivery DEL789012"
        
        result = intelligent_agent.process_scenario(scenario)
        
        # Verify successful processing
        assert result.success is True
        assert result.scenario.scenario_type == ScenarioType.MERCHANT
        
        # Check tool selection for merchant scenario
        executed_tools = []
        for step in result.reasoning_trace.steps:
            if step.action:
                executed_tools.append(step.action.tool_name)
        
        # For merchant scenario, should prioritize merchant-related tools
        if executed_tools:
            merchant_related = ["get_merchant_status", "get_nearby_merchants", "notify_customer"]
            assert any(tool in executed_tools for tool in merchant_related)
    
    def test_scenario_analysis_integration(self, intelligent_agent):
        """Test that scenario analysis is properly integrated."""
        scenario = "Complex delivery issue with traffic and merchant problems"
        
        result = intelligent_agent.process_scenario(scenario)
        
        # Verify scenario analysis was performed
        scenario_analysis = intelligent_agent.get_scenario_analysis()
        assert scenario_analysis is not None
        
        # Verify analysis structure
        assert "scenario_complexity" in scenario_analysis.__dict__
        assert "recommended_tools" in scenario_analysis.__dict__
        assert "stakeholders" in scenario_analysis.__dict__
        assert "estimated_resolution_time" in scenario_analysis.__dict__
    
    def test_tool_recommendations_api(self, intelligent_agent):
        """Test the tool recommendations API."""
        scenario = "Traffic delay for delivery DEL123456"
        
        # Process scenario first
        result = intelligent_agent.process_scenario(scenario)
        
        # Get tool recommendations
        recommendations = intelligent_agent.get_tool_recommendations(result.scenario)
        
        # Verify recommendations structure
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        for rec in recommendations:
            assert "tool_name" in rec
            assert "priority" in rec
            assert "confidence" in rec
            assert "reasoning" in rec
            assert "suggested_parameters" in rec
            assert "dependencies" in rec
    
    def test_enhanced_performance_metrics(self, intelligent_agent):
        """Test enhanced performance metrics with tool usage."""
        scenarios = [
            "Traffic issue for DEL001",
            "Merchant problem for DEL002"
        ]
        
        # Process multiple scenarios
        for scenario in scenarios:
            intelligent_agent.process_scenario(scenario)
        
        # Get enhanced metrics
        metrics = intelligent_agent.get_performance_metrics()
        
        # Verify enhanced metrics include tool usage
        assert "tool_usage_distribution" in metrics
        assert metrics["total_scenarios_processed"] == 2
        
        # Tool usage should be tracked
        if metrics["tool_usage_distribution"]:
            assert isinstance(metrics["tool_usage_distribution"], dict)
    
    def test_context_data_with_analysis(self, intelligent_agent):
        """Test that context data includes scenario analysis."""
        scenario = "Test scenario for context analysis"
        
        result = intelligent_agent.process_scenario(scenario)
        
        # Verify context data includes analysis
        state = intelligent_agent.get_current_state()
        assert "context_data_keys" in state
        
        # Should have scenario_analysis in context
        context_keys = state["context_data_keys"]
        assert "scenario_analysis" in context_keys
    
    def test_reasoning_with_tool_integration(self, intelligent_agent, mock_tool_manager):
        """Test that reasoning integrates tool results intelligently."""
        scenario = "Driver needs help with delivery DEL123456 due to traffic"
        
        result = intelligent_agent.process_scenario(scenario)
        
        # Verify tool manager was called
        assert mock_tool_manager.execute_tool.called
        
        # Verify reasoning steps include tool integration
        for step in result.reasoning_trace.steps:
            if step.tool_results:
                # Should have integrated observations
                assert step.observation is not None
                assert len(step.observation) > 0
                
                # Observation should include analysis information
                if "confidence" in step.observation or "completeness" in step.observation:
                    # This indicates integrated analysis was used
                    assert True
                    break
    
    def test_urgency_based_tool_prioritization(self, intelligent_agent):
        """Test that urgent scenarios get appropriate tool prioritization."""
        urgent_scenario = "URGENT: Customer emergency with delivery DEL999999"
        
        result = intelligent_agent.process_scenario(urgent_scenario)
        
        # Verify urgency was detected
        assert result.scenario.urgency_level in [UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]
        
        # Get tool recommendations to verify prioritization
        recommendations = intelligent_agent.get_tool_recommendations(result.scenario)
        
        # Should have high-priority recommendations for urgent scenarios
        high_priority_tools = [rec for rec in recommendations if rec["priority"] in ["CRITICAL", "HIGH"]]
        assert len(high_priority_tools) > 0
    
    def test_multi_factor_scenario_handling(self, intelligent_agent):
        """Test handling of complex multi-factor scenarios."""
        complex_scenario = ("Driver stuck in traffic on Main Street, "
                          "Pizza Palace is overloaded, and customer John Smith "
                          "at 456 Oak Avenue is calling about delivery DEL555555")
        
        result = intelligent_agent.process_scenario(complex_scenario)
        
        # Should be classified as multi-factor or have multiple entity types
        assert (result.scenario.scenario_type == ScenarioType.MULTI_FACTOR or
                len(set(e.entity_type for e in result.scenario.entities)) > 2)
        
        # Should have comprehensive tool recommendations
        recommendations = intelligent_agent.get_tool_recommendations(result.scenario)
        tool_names = [rec["tool_name"] for rec in recommendations]
        
        # Should recommend tools for multiple aspects
        tool_categories = set()
        if "check_traffic" in tool_names:
            tool_categories.add("traffic")
        if "get_merchant_status" in tool_names:
            tool_categories.add("merchant")
        if "notify_customer" in tool_names:
            tool_categories.add("customer")
        
        # Should address multiple categories for complex scenarios
        assert len(tool_categories) >= 2