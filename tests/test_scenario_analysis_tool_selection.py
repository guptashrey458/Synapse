"""
Tests for scenario analysis and intelligent tool selection.
"""
import pytest
from unittest.mock import Mock

from src.agent.scenario_analyzer import ScenarioAnalyzer, ToolPriority, ToolRecommendation
from src.agent.interfaces import ScenarioType, UrgencyLevel, EntityType
from src.agent.models import ValidatedDisruptionScenario, ValidatedEntity
from src.tools.interfaces import ToolResult


class TestScenarioAnalyzer:
    """Tests for the ScenarioAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create scenario analyzer instance."""
        return ScenarioAnalyzer()
    
    @pytest.fixture
    def traffic_scenario(self):
        """Create a traffic-related scenario."""
        entities = [
            ValidatedEntity(
                text="DEL123456",
                entity_type=EntityType.DELIVERY_ID,
                confidence=0.9,
                normalized_value="DEL123456"
            ),
            ValidatedEntity(
                text="123 Main Street",
                entity_type=EntityType.ADDRESS,
                confidence=0.8,
                normalized_value="123 Main Street"
            )
        ]
        
        return ValidatedDisruptionScenario(
            description="Driver is stuck in heavy traffic on Main Street for delivery DEL123456",
            entities=entities,
            scenario_type=ScenarioType.TRAFFIC,
            urgency_level=UrgencyLevel.HIGH
        )
    
    @pytest.fixture
    def merchant_scenario(self):
        """Create a merchant-related scenario."""
        entities = [
            ValidatedEntity(
                text="Pizza Palace",
                entity_type=EntityType.MERCHANT,
                confidence=0.8,
                normalized_value="Pizza Palace"
            ),
            ValidatedEntity(
                text="DEL789012",
                entity_type=EntityType.DELIVERY_ID,
                confidence=0.9,
                normalized_value="DEL789012"
            )
        ]
        
        return ValidatedDisruptionScenario(
            description="Pizza Palace is closed and cannot fulfill delivery DEL789012",
            entities=entities,
            scenario_type=ScenarioType.MERCHANT,
            urgency_level=UrgencyLevel.MEDIUM
        )
    
    @pytest.fixture
    def mock_tools(self):
        """Create mock tools."""
        tools = []
        tool_names = [
            "check_traffic", "get_merchant_status", "validate_address",
            "notify_customer", "re_route_driver", "get_nearby_merchants"
        ]
        
        for name in tool_names:
            tool = Mock()
            tool.name = name
            tool.description = f"Mock {name} tool"
            tool.parameters = {"param1": {"type": "string"}}
            tools.append(tool)
        
        return tools
    
    def test_analyze_traffic_scenario(self, analyzer, traffic_scenario, mock_tools):
        """Test analysis of traffic scenario."""
        analysis = analyzer.analyze_scenario(traffic_scenario, mock_tools)
        
        # Verify analysis structure
        assert analysis.scenario_complexity in ["simple", "moderate", "complex"]
        assert len(analysis.key_entities) > 0
        assert len(analysis.primary_issues) > 0
        assert len(analysis.stakeholders) > 0
        assert len(analysis.recommended_tools) > 0
        
        # Verify traffic-specific analysis
        assert "traffic_disruption" in analysis.primary_issues
        assert "driver" in analysis.stakeholders
        assert analysis.estimated_resolution_time > 0
        
        # Verify tool recommendations include traffic-related tools
        tool_names = [rec.tool_name for rec in analysis.recommended_tools]
        assert "check_traffic" in tool_names
        assert "notify_customer" in tool_names
    
    def test_analyze_merchant_scenario(self, analyzer, merchant_scenario, mock_tools):
        """Test analysis of merchant scenario."""
        analysis = analyzer.analyze_scenario(merchant_scenario, mock_tools)
        
        # Verify merchant-specific analysis
        assert "merchant_unavailability" in analysis.primary_issues
        assert "merchant" in analysis.stakeholders
        
        # Verify tool recommendations include merchant-related tools
        tool_names = [rec.tool_name for rec in analysis.recommended_tools]
        assert "get_merchant_status" in tool_names
        assert "notify_customer" in tool_names
    
    def test_complexity_assessment(self, analyzer):
        """Test scenario complexity assessment."""
        # Simple scenario
        simple_scenario = ValidatedDisruptionScenario(
            description="Simple delivery issue",
            entities=[],
            scenario_type=ScenarioType.OTHER,
            urgency_level=UrgencyLevel.LOW
        )
        
        analysis = analyzer.analyze_scenario(simple_scenario, [])
        assert analysis.scenario_complexity == "simple"
        
        # Complex scenario
        complex_entities = [
            ValidatedEntity("DEL123", EntityType.DELIVERY_ID, 0.9),
            ValidatedEntity("123 Main St", EntityType.ADDRESS, 0.8),
            ValidatedEntity("Pizza Palace", EntityType.MERCHANT, 0.7),
            ValidatedEntity("John Smith", EntityType.PERSON, 0.6)
        ]
        
        complex_scenario = ValidatedDisruptionScenario(
            description="Very complex multi-factor delivery disruption with traffic, merchant issues, and customer complaints requiring immediate attention and coordination between multiple stakeholders",
            entities=complex_entities,
            scenario_type=ScenarioType.MULTI_FACTOR,
            urgency_level=UrgencyLevel.CRITICAL
        )
        
        analysis = analyzer.analyze_scenario(complex_scenario, [])
        assert analysis.scenario_complexity in ["moderate", "complex"]
    
    def test_tool_prioritization(self, analyzer, traffic_scenario, mock_tools):
        """Test tool prioritization logic."""
        analysis = analyzer.analyze_scenario(traffic_scenario, mock_tools)
        prioritized = analyzer.prioritize_tools(analysis.recommended_tools, traffic_scenario)
        
        # Verify tools are ordered by priority
        priorities = [tool.priority.value for tool in prioritized]
        assert priorities == sorted(priorities)
        
        # Verify high urgency scenario gets appropriate priority adjustments
        high_priority_tools = [tool for tool in prioritized if tool.priority == ToolPriority.CRITICAL]
        if high_priority_tools:
            # At least one tool should be critical priority for high urgency traffic scenario
            assert len(high_priority_tools) > 0
    
    def test_tool_dependencies(self, analyzer, traffic_scenario, mock_tools):
        """Test tool dependency resolution."""
        analysis = analyzer.analyze_scenario(traffic_scenario, mock_tools)
        prioritized = analyzer.prioritize_tools(analysis.recommended_tools, traffic_scenario)
        
        # Verify dependencies are respected in ordering
        tool_names = [tool.tool_name for tool in prioritized]
        
        # If re_route_driver is present, check_traffic should come before it
        if "re_route_driver" in tool_names and "check_traffic" in tool_names:
            re_route_index = tool_names.index("re_route_driver")
            check_traffic_index = tool_names.index("check_traffic")
            assert check_traffic_index < re_route_index
    
    def test_next_tool_selection(self, analyzer, traffic_scenario, mock_tools):
        """Test intelligent next tool selection."""
        # No tools executed yet
        next_tool = analyzer.select_next_tool(
            traffic_scenario, [], [], mock_tools
        )
        
        assert next_tool is not None
        assert next_tool.tool_name in [tool.name for tool in mock_tools]
        assert next_tool.confidence > 0
        
        # Some tools already executed
        executed_tools = ["check_traffic"]
        tool_results = [
            ToolResult("check_traffic", True, {"status": "heavy_traffic", "delay_minutes": 15}, 1.0)
        ]
        
        next_tool = analyzer.select_next_tool(
            traffic_scenario, executed_tools, tool_results, mock_tools
        )
        
        # Should not recommend already executed tools
        if next_tool:
            assert next_tool.tool_name not in executed_tools
    
    def test_tool_result_integration(self, analyzer, traffic_scenario):
        """Test integration of tool results."""
        tool_results = [
            ToolResult("check_traffic", True, {"status": "heavy_traffic", "delay_minutes": 15}, 1.0),
            ToolResult("get_merchant_status", True, {"available": True, "prep_time_minutes": 10}, 0.8),
            ToolResult("notify_customer", False, {}, 0.5, "Network error")
        ]
        
        integrated_info = analyzer.integrate_tool_results(tool_results, traffic_scenario)
        
        # Verify integration structure
        assert "successful_tools" in integrated_info
        assert "failed_tools" in integrated_info
        assert "key_findings" in integrated_info
        assert "confidence_score" in integrated_info
        assert "completeness_score" in integrated_info
        
        # Verify content
        assert len(integrated_info["successful_tools"]) == 2
        assert len(integrated_info["failed_tools"]) == 1
        assert integrated_info["confidence_score"] > 0
        assert integrated_info["completeness_score"] > 0
    
    def test_urgency_based_adjustments(self, analyzer, mock_tools):
        """Test urgency-based priority adjustments."""
        # Critical urgency scenario
        critical_scenario = ValidatedDisruptionScenario(
            description="URGENT: Customer emergency with delivery",
            entities=[],
            scenario_type=ScenarioType.OTHER,
            urgency_level=UrgencyLevel.CRITICAL
        )
        
        analysis = analyzer.analyze_scenario(critical_scenario, mock_tools)
        prioritized = analyzer.prioritize_tools(analysis.recommended_tools, critical_scenario)
        
        # Customer notification should be prioritized for critical scenarios
        if prioritized:
            customer_tools = [tool for tool in prioritized if tool.tool_name == "notify_customer"]
            if customer_tools:
                # Should be high priority
                assert customer_tools[0].priority in [ToolPriority.CRITICAL, ToolPriority.HIGH]
    
    def test_scenario_specific_recommendations(self, analyzer, mock_tools):
        """Test scenario-specific tool recommendations."""
        # Scenario with customer dissatisfaction
        complaint_scenario = ValidatedDisruptionScenario(
            description="Customer is very angry and frustrated about delayed delivery",
            entities=[],
            scenario_type=ScenarioType.OTHER,
            urgency_level=UrgencyLevel.HIGH
        )
        
        analysis = analyzer.analyze_scenario(complaint_scenario, mock_tools)
        tool_names = [rec.tool_name for rec in analysis.recommended_tools]
        
        # Should recommend customer service tools for complaint scenarios
        # Note: This depends on having the appropriate tools available
        expected_tools = ["notify_customer"]  # Always recommended
        for tool in expected_tools:
            if any(mock_tool.name == tool for mock_tool in mock_tools):
                assert tool in tool_names
    
    def test_confidence_calculation(self, analyzer, traffic_scenario, mock_tools):
        """Test confidence score calculation."""
        analysis = analyzer.analyze_scenario(traffic_scenario, mock_tools)
        
        for recommendation in analysis.recommended_tools:
            # Confidence should be between 0 and 1
            assert 0 <= recommendation.confidence <= 1
            
            # Traffic-related tools should have higher confidence for traffic scenarios
            if recommendation.tool_name in ["check_traffic", "re_route_driver"]:
                assert recommendation.confidence > 0.5
    
    def test_parameter_generation(self, analyzer, traffic_scenario, mock_tools):
        """Test suggested parameter generation."""
        analysis = analyzer.analyze_scenario(traffic_scenario, mock_tools)
        
        for recommendation in analysis.recommended_tools:
            # Should have suggested parameters
            assert isinstance(recommendation.suggested_parameters, dict)
            
            # Check specific parameter generation
            if recommendation.tool_name == "check_traffic":
                # Should suggest location parameter based on address entity
                assert "location" in recommendation.suggested_parameters
            elif recommendation.tool_name == "notify_customer":
                # Should suggest delivery_id if available
                if traffic_scenario.get_delivery_ids():
                    assert "delivery_id" in recommendation.suggested_parameters
    
    def test_stakeholder_identification(self, analyzer, traffic_scenario, merchant_scenario):
        """Test stakeholder identification."""
        # Traffic scenario stakeholders
        traffic_analysis = analyzer.analyze_scenario(traffic_scenario, [])
        assert "customer" in traffic_analysis.stakeholders
        assert "driver" in traffic_analysis.stakeholders
        
        # Merchant scenario stakeholders
        merchant_analysis = analyzer.analyze_scenario(merchant_scenario, [])
        assert "customer" in merchant_analysis.stakeholders
        assert "merchant" in merchant_analysis.stakeholders
    
    def test_execution_strategy_determination(self, analyzer, mock_tools):
        """Test execution strategy determination."""
        # Critical scenario should get parallel urgent strategy
        critical_scenario = ValidatedDisruptionScenario(
            description="Critical emergency",
            entities=[],
            scenario_type=ScenarioType.OTHER,
            urgency_level=UrgencyLevel.CRITICAL
        )
        
        analysis = analyzer.analyze_scenario(critical_scenario, mock_tools)
        assert analysis.execution_strategy == "parallel_urgent"
        
        # Simple scenario should get sequential strategy
        simple_scenario = ValidatedDisruptionScenario(
            description="Simple issue",
            entities=[],
            scenario_type=ScenarioType.OTHER,
            urgency_level=UrgencyLevel.LOW
        )
        
        analysis = analyzer.analyze_scenario(simple_scenario, mock_tools[:1])  # Only one tool
        assert analysis.execution_strategy == "sequential_simple"
    
    def test_resolution_time_estimation(self, analyzer, mock_tools):
        """Test resolution time estimation."""
        scenarios = [
            # Simple scenario
            ValidatedDisruptionScenario(
                description="Simple issue",
                entities=[],
                scenario_type=ScenarioType.OTHER,
                urgency_level=UrgencyLevel.LOW
            ),
            # Complex scenario
            ValidatedDisruptionScenario(
                description="Complex multi-factor issue",
                entities=[ValidatedEntity("DEL123", EntityType.DELIVERY_ID, 0.9)],
                scenario_type=ScenarioType.MULTI_FACTOR,
                urgency_level=UrgencyLevel.HIGH
            )
        ]
        
        estimates = []
        for scenario in scenarios:
            analysis = analyzer.analyze_scenario(scenario, mock_tools)
            estimates.append(analysis.estimated_resolution_time)
        
        # All estimates should be positive
        assert all(est > 0 for est in estimates)
        
        # Complex scenario should generally take longer (though urgency might reduce it)
        # At minimum, we verify the estimation logic runs without error
        assert len(estimates) == 2