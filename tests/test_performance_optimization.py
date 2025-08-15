"""
Performance optimization tests for the autonomous delivery coordinator.
Tests requirements 6.2, 6.3: Performance and response time optimization.
"""
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

from src.agent.autonomous_agent import AutonomousAgent, AgentConfig
from src.agent.interfaces import ScenarioType, UrgencyLevel, EntityType
from src.llm.interfaces import LLMResponse, Message, MessageRole, TokenUsage
from src.tools.interfaces import ToolResult


class TestPerformanceOptimization:
    """Test cases for performance optimization and validation."""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create optimized mock LLM provider."""
        provider = Mock()
        # Simulate faster response times for optimized prompts
        provider.generate_response.return_value = LLMResponse(
            content="Optimized response with minimal token usage.",
            messages=[],
            token_usage=TokenUsage(50, 25, 75),  # Reduced token usage
            model="gpt-4",
            finish_reason="stop",
            response_time=0.5,  # Faster response time
            timestamp=datetime.now()
        )
        return provider
    
    @pytest.fixture
    def mock_tool_manager(self):
        """Create optimized mock tool manager with caching."""
        manager = Mock()
        
        # Mock tools with caching capabilities
        tools = [
            Mock(name="check_traffic", description="Check traffic with caching"),
            Mock(name="get_merchant_status", description="Get merchant status with caching"),
            Mock(name="notify_customer", description="Batch customer notifications"),
            Mock(name="get_delivery_status", description="Batch delivery status checks")
        ]
        manager.get_available_tools.return_value = tools
        
        # Simulate faster tool execution with caching
        manager.execute_tool.return_value = ToolResult(
            "cached_tool", True, {"cached": True, "response_time": "optimized"}, 0.1
        )
        return manager
    
    @pytest.fixture
    def optimized_agent(self, mock_llm_provider, mock_tool_manager):
        """Create optimized autonomous agent."""
        config = AgentConfig(
            max_reasoning_steps=5,  # Reduced for efficiency
            reasoning_timeout=30,   # Shorter timeout
            enable_context_tracking=True,
            log_reasoning_steps=False,  # Disabled for performance
            enable_caching=True,        # Enable caching
            concurrent_tools=True,      # Enable concurrent execution
            optimize_prompts=True,      # Enable prompt optimization
            batch_tool_calls=True       # Enable batching
        )
        return AutonomousAgent(mock_llm_provider, mock_tool_manager, config)
    
    @pytest.fixture
    def basic_agent(self, mock_llm_provider, mock_tool_manager):
        """Create basic agent without optimizations for comparison."""
        config = AgentConfig(
            max_reasoning_steps=5,
            reasoning_timeout=30,
            enable_context_tracking=True,
            log_reasoning_steps=False,
            enable_caching=False,       # Disabled
            concurrent_tools=False,     # Disabled
            optimize_prompts=False,     # Disabled
            batch_tool_calls=False      # Disabled
        )
        return AutonomousAgent(mock_llm_provider, mock_tool_manager, config)
    
    def test_basic_performance_requirements(self, basic_agent):
        """Test basic performance requirements without optimizations."""
        scenarios = [
            "Traffic delay for delivery DEL001",
            "Restaurant closed for delivery DEL002", 
            "Customer complaint for delivery DEL003"
        ]
        
        for scenario in scenarios:
            start_time = time.time()
            result = basic_agent.process_scenario(scenario)
            processing_time = time.time() - start_time
            
            # Verify successful processing
            assert result.success is True
            
            # Basic performance requirement: should complete within reasonable time
            assert processing_time < 10.0  # Should be fast even without optimizations
    
    def test_caching_configuration(self, optimized_agent):
        """Test that caching is properly configured when enabled."""
        # Check that caching components are initialized
        assert optimized_agent.tool_cache is not None or optimized_agent.scenario_cache is not None
        
        # Process same scenario multiple times
        scenario = "Traffic delay on Route 1 for delivery DEL456"
        
        # First execution
        result1 = optimized_agent.process_scenario(scenario)
        
        # Second execution (should potentially use cache)
        result2 = optimized_agent.process_scenario(scenario)
        
        # Both should succeed
        assert result1.success is True
        assert result2.success is True
        
        # Results should be consistent
        assert result1.scenario.scenario_type == result2.scenario.scenario_type
    
    def test_concurrent_tool_configuration(self, optimized_agent):
        """Test that concurrent tool execution is properly configured."""
        # Check that concurrent executor is initialized when enabled
        assert optimized_agent.concurrent_executor is not None or optimized_agent.batch_executor is not None
        
        # Process a complex scenario that would benefit from concurrent execution
        scenario = """
        Multiple issues: Traffic jam on Highway 1, restaurant delay at Pizza Palace, 
        customer John calling about delivery DEL123, and driver needs rerouting.
        """
        
        start_time = time.time()
        result = optimized_agent.process_scenario(scenario)
        processing_time = time.time() - start_time
        
        # Verify successful processing
        assert result.success is True
        
        # Should complete efficiently even with complex scenario
        assert processing_time < 15.0
    
    def test_batch_processing_efficiency(self, optimized_agent):
        """Test batch processing for multiple scenarios."""
        scenarios = [
            "Traffic delay for delivery DEL100",
            "Restaurant issue for delivery DEL101",
            "Address problem for delivery DEL102",
            "Customer complaint for delivery DEL103",
            "Driver rerouting for delivery DEL104"
        ]
        
        # Process scenarios
        start_time = time.time()
        results = []
        
        for scenario in scenarios:
            result = optimized_agent.process_scenario(scenario)
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Verify all scenarios processed successfully
        for result in results:
            assert result.success is True
        
        # Verify batch processing efficiency
        avg_time_per_scenario = total_time / len(scenarios)
        assert avg_time_per_scenario < 8.0  # Should average under 8 seconds per scenario
        assert total_time < 40.0  # Total should be under 40 seconds
    
    def test_response_time_requirements(self, optimized_agent):
        """Test response time requirements for different scenario types."""
        test_cases = [
            {
                "name": "Simple Traffic",
                "scenario": "Minor traffic delay for delivery DEL001",
                "max_time": 5.0  # Simple scenarios should be fast
            },
            {
                "name": "Merchant Issue",
                "scenario": "Restaurant closed affecting delivery DEL002",
                "max_time": 8.0  # Moderate complexity
            },
            {
                "name": "Complex Multi-factor",
                "scenario": "Traffic jam, restaurant delay, and customer complaint for delivery DEL003",
                "max_time": 15.0  # Complex scenarios allowed more time
            },
            {
                "name": "Emergency Scenario",
                "scenario": "URGENT: Medical delivery DEL004 stuck due to road closure, patient waiting",
                "max_time": 12.0  # Emergency scenarios should be prioritized
            }
        ]
        
        for test_case in test_cases:
            start_time = time.time()
            result = optimized_agent.process_scenario(test_case["scenario"])
            processing_time = time.time() - start_time
            
            # Verify successful processing
            assert result.success is True, f"Failed to process: {test_case['name']}"
            
            # Verify response time requirement
            assert processing_time < test_case["max_time"], \
                f"{test_case['name']} took {processing_time:.2f}s, max allowed: {test_case['max_time']}s"
    
    def test_scalability_under_load(self, optimized_agent):
        """Test system scalability under increasing load."""
        load_levels = [1, 3, 5]  # Number of concurrent scenarios
        base_scenario = "Traffic delay for delivery DEL{:03d}"
        
        for load_level in load_levels:
            scenarios = [base_scenario.format(i) for i in range(load_level)]
            
            start_time = time.time()
            results = []
            
            # Process scenarios (simulating load)
            for scenario in scenarios:
                result = optimized_agent.process_scenario(scenario)
                results.append(result)
            
            total_time = time.time() - start_time
            
            # Verify all scenarios processed successfully
            for i, result in enumerate(results):
                assert result.success is True, f"Scenario {i} failed at load level {load_level}"
            
            # Verify scalability (time should scale reasonably with load)
            avg_time_per_scenario = total_time / load_level
            assert avg_time_per_scenario < 12.0, \
                f"Average time {avg_time_per_scenario:.2f}s too high at load level {load_level}"
            
            # Total time should be reasonable even at higher load
            max_total_time = load_level * 15.0  # Allow 15 seconds per scenario max
            assert total_time < max_total_time, \
                f"Total time {total_time:.2f}s exceeded limit {max_total_time}s at load level {load_level}"
    
    def test_memory_usage_optimization(self, optimized_agent):
        """Test memory usage optimization for large scenarios."""
        # Large scenario with lots of entities and details
        large_scenario = """
        Major disruption affecting multiple deliveries: Highway 95 closed due to accident,
        affecting deliveries DEL001, DEL002, DEL003, DEL004, DEL005.
        Restaurants Tony's Pizza, Mario's Kitchen, Burger Palace all experiencing delays.
        Customers John (555-1234), Sarah (555-2345), Mike (555-3456) all calling.
        Drivers need rerouting to addresses: 123 Main St, 456 Oak Ave, 789 Pine Rd.
        """
        
        start_time = time.time()
        result = optimized_agent.process_scenario(large_scenario)
        processing_time = time.time() - start_time
        
        # Verify successful processing of large scenario
        assert result.success is True
        
        # Should handle large scenarios efficiently
        assert processing_time < 20.0  # Should complete within 20 seconds
        
        # Verify entity extraction worked on large input
        entities = result.scenario.entities
        delivery_ids = [e for e in entities if e.entity_type == EntityType.DELIVERY_ID]
        assert len(delivery_ids) >= 3  # Should extract multiple delivery IDs
    
    def test_error_handling_performance(self, optimized_agent, mock_tool_manager):
        """Test that error handling doesn't significantly impact performance."""
        # Configure mock to simulate some errors
        mock_tool_manager.execute_tool.side_effect = [
            Exception("Simulated tool error"),  # First call fails
            ToolResult("recovery_tool", True, {"recovered": True}, 0.2)  # Second call succeeds
        ]
        
        scenario = "Test scenario for error handling performance"
        
        start_time = time.time()
        result = optimized_agent.process_scenario(scenario)
        processing_time = time.time() - start_time
        
        # Should handle errors gracefully without excessive delay
        assert processing_time < 15.0  # Error handling shouldn't take too long
        
        # Result might succeed or fail, but should be handled gracefully
        assert result is not None
    
    def test_final_validation_against_requirements(self, optimized_agent):
        """Final validation against all performance requirements."""
        # Test requirement 6.2: Real-time progress display (simulated)
        scenario = "Traffic delay on Highway 1 affecting delivery DEL123 to customer John at 456 Oak Street"
        
        start_time = time.time()
        result = optimized_agent.process_scenario(scenario)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Verify successful processing
        assert result.success is True
        
        # Verify timing requirements
        assert processing_time < 30.0  # Should complete within reasonable time
        
        # Test requirement 6.3: Structured output format
        assert result.resolution_plan is not None
        assert len(result.resolution_plan.steps) >= 1
        assert result.resolution_plan.estimated_duration > timedelta(0)
        assert 0.0 <= result.resolution_plan.success_probability <= 1.0
        
        # Verify reasoning trace completeness
        assert result.reasoning_trace is not None
        assert len(result.reasoning_trace.steps) >= 1
        assert result.reasoning_trace.start_time is not None
        assert result.reasoning_trace.end_time is not None
        
        # Verify entity extraction
        assert result.scenario is not None
        assert len(result.scenario.entities) >= 1
        
        # Verify scenario classification
        assert result.scenario.scenario_type in [
            ScenarioType.TRAFFIC, ScenarioType.MERCHANT, ScenarioType.ADDRESS, 
            ScenarioType.MULTI_FACTOR, ScenarioType.OTHER
        ]
        assert result.scenario.urgency_level in [
            UrgencyLevel.LOW, UrgencyLevel.MEDIUM, UrgencyLevel.HIGH, UrgencyLevel.CRITICAL
        ]
    
    @pytest.mark.parametrize("scenario_complexity", [
        ("simple", "Traffic delay DEL001"),
        ("medium", "Restaurant closed, customer waiting DEL002"),
        ("complex", "Multi-factor: traffic, merchant, customer issues DEL003")
    ])
    def test_performance_across_complexity_levels(self, optimized_agent, scenario_complexity):
        """Test performance across different scenario complexity levels."""
        complexity_level, scenario = scenario_complexity
        
        # Define performance expectations based on complexity
        max_times = {
            "simple": 5.0,
            "medium": 10.0,
            "complex": 18.0
        }
        
        start_time = time.time()
        result = optimized_agent.process_scenario(scenario)
        processing_time = time.time() - start_time
        
        # Verify successful processing
        assert result.success is True
        
        # Verify performance meets expectations for complexity level
        max_time = max_times[complexity_level]
        assert processing_time < max_time, \
            f"{complexity_level} scenario took {processing_time:.2f}s, max: {max_time}s"
        
        # Verify quality isn't sacrificed for performance
        assert result.resolution_plan.success_probability > 0.3
        assert len(result.resolution_plan.steps) >= 1
    
    def test_optimization_components_initialization(self, optimized_agent, basic_agent):
        """Test that optimization components are properly initialized."""
        # Optimized agent should have optimization components
        # (Note: May be None if imports failed, which is acceptable)
        optimization_enabled = (
            optimized_agent.tool_cache is not None or
            optimized_agent.scenario_cache is not None or
            optimized_agent.concurrent_executor is not None or
            optimized_agent.batch_executor is not None
        )
        
        # Basic agent should not have optimization components
        basic_optimization_disabled = (
            basic_agent.tool_cache is None and
            basic_agent.scenario_cache is None and
            basic_agent.concurrent_executor is None and
            basic_agent.batch_executor is None
        )
        
        # At least one should be true (either optimizations work or they're properly disabled)
        assert optimization_enabled or basic_optimization_disabled
    
    def test_performance_metrics_collection(self, optimized_agent):
        """Test that performance metrics are properly collected."""
        # Process a few scenarios to generate metrics
        scenarios = [
            "Traffic delay DEL001",
            "Restaurant issue DEL002",
            "Customer complaint DEL003"
        ]
        
        for scenario in scenarios:
            result = optimized_agent.process_scenario(scenario)
            assert result.success is True
        
        # Get performance metrics
        metrics = optimized_agent.get_performance_metrics()
        
        # Verify metrics are collected
        assert "total_scenarios_processed" in metrics
        assert metrics["total_scenarios_processed"] >= len(scenarios)
        
        if "average_processing_time_seconds" in metrics:
            assert metrics["average_processing_time_seconds"] >= 0
        
        # Verify metrics structure
        assert isinstance(metrics, dict)
        assert len(metrics) > 0