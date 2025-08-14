"""
Tests for prompt effectiveness and output consistency.
"""
import pytest
from src.llm.templates import (
    ReActPromptTemplate, ChainOfThoughtTemplate, PromptTemplateManager,
    DEFAULT_DELIVERY_EXAMPLES
)
from src.llm.interfaces import Message, MessageRole


class TestReActPromptEffectiveness:
    """Test ReAct prompt effectiveness for delivery scenarios."""
    
    @pytest.fixture
    def react_template(self):
        return ReActPromptTemplate()
    
    @pytest.fixture
    def sample_tools(self):
        return [
            {
                "name": "check_traffic",
                "description": "Check current traffic conditions for a route",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "Starting location"},
                        "destination": {"type": "string", "description": "Destination"}
                    },
                    "required": ["location", "destination"]
                }
            },
            {
                "name": "notify_customer",
                "description": "Send notification to customer about delivery status",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer": {"type": "string", "description": "Customer name"},
                        "message": {"type": "string", "description": "Message to send"},
                        "estimated_arrival": {"type": "string", "description": "New ETA"}
                    },
                    "required": ["customer", "message"]
                }
            },
            {
                "name": "get_merchant_status",
                "description": "Check merchant availability and order status",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "merchant_id": {"type": "string", "description": "Merchant identifier"},
                        "order_id": {"type": "string", "description": "Order identifier"}
                    },
                    "required": ["merchant_id"]
                }
            }
        ]
    
    def test_traffic_scenario_prompt_structure(self, react_template, sample_tools):
        """Test that traffic scenario prompts have proper ReAct structure."""
        scenario = """
        Driver Mike is delivering pizza from Tony's Pizza to 456 Oak Street. 
        He's currently stuck in heavy traffic on Main Street due to a car accident. 
        The delivery was supposed to arrive at 8:00 PM but it's now 8:15 PM and 
        he's still 10 minutes away from the destination.
        """
        
        messages = react_template.format(
            scenario=scenario,
            available_tools=sample_tools,
            examples=DEFAULT_DELIVERY_EXAMPLES
        )
        
        assert len(messages) == 2
        system_msg, user_msg = messages
        
        # Check system message contains ReAct pattern instructions
        system_content = system_msg.content.lower()
        assert "thought:" in system_content
        assert "action:" in system_content
        assert "observation:" in system_content
        assert "react" in system_content
        
        # Check system message contains tool descriptions
        assert "check_traffic" in system_msg.content
        assert "notify_customer" in system_msg.content
        assert "get_merchant_status" in system_msg.content
        
        # Check user message contains scenario details
        user_content = user_msg.content
        assert "Mike" in user_content
        assert "Tony's Pizza" in user_content
        assert "456 Oak Street" in user_content
        assert "traffic" in user_content.lower()
    
    def test_merchant_scenario_prompt_structure(self, react_template, sample_tools):
        """Test merchant availability scenario prompt structure."""
        scenario = """
        Customer Sarah ordered sushi from Sakura Restaurant for delivery to 
        123 Pine Avenue. The restaurant just called to say they're running 
        30 minutes behind due to a large catering order. The original ETA 
        was 7:30 PM but now it will be 8:00 PM.
        """
        
        messages = react_template.format(
            scenario=scenario,
            available_tools=sample_tools
        )
        
        user_content = messages[1].content
        assert "Sarah" in user_content
        assert "Sakura Restaurant" in user_content
        assert "123 Pine Avenue" in user_content
        assert "30 minutes behind" in user_content
    
    def test_prompt_includes_examples(self, react_template, sample_tools):
        """Test that examples are properly included in prompts."""
        scenario = "Test scenario"
        
        messages = react_template.format(
            scenario=scenario,
            available_tools=sample_tools,
            examples=DEFAULT_DELIVERY_EXAMPLES
        )
        
        system_content = messages[0].content
        assert "Examples" in system_content
        assert "Traffic Disruption" in system_content
        # Should include reasoning steps from examples
        assert "Thought:" in system_content
        assert "Action:" in system_content
    
    def test_prompt_with_previous_steps(self, react_template, sample_tools):
        """Test prompt formatting with previous reasoning steps."""
        scenario = "Driver is delayed"
        previous_steps = [
            {
                "thought": "I need to check the traffic situation",
                "action": "check_traffic",
                "observation": "Heavy traffic due to construction"
            },
            {
                "thought": "I should notify the customer about the delay",
                "action": "notify_customer",
                "observation": "Customer notified successfully"
            }
        ]
        
        messages = react_template.format(
            scenario=scenario,
            available_tools=sample_tools,
            previous_steps=previous_steps
        )
        
        user_content = messages[1].content
        assert "Previous Reasoning Steps:" in user_content
        assert "I need to check the traffic situation" in user_content
        assert "Heavy traffic due to construction" in user_content
        assert "Customer notified successfully" in user_content
    
    def test_output_format_instructions(self, react_template, sample_tools):
        """Test that output format instructions are clear and complete."""
        scenario = "Test scenario"
        
        messages = react_template.format(
            scenario=scenario,
            available_tools=sample_tools
        )
        
        system_content = messages[0].content
        
        # Check for JSON output format instructions
        assert "```json" in system_content
        assert "reasoning_complete" in system_content
        assert "final_plan" in system_content
        assert "steps" in system_content
        assert "sequence" in system_content
        assert "responsible_party" in system_content
        assert "success_criteria" in system_content
    
    def test_tool_parameter_descriptions(self, react_template, sample_tools):
        """Test that tool parameters are clearly described."""
        scenario = "Test scenario"
        
        messages = react_template.format(
            scenario=scenario,
            available_tools=sample_tools
        )
        
        system_content = messages[0].content
        
        # Check that tool parameters are described
        assert "location" in system_content
        assert "destination" in system_content
        assert "customer" in system_content
        assert "message" in system_content
        assert "required" in system_content
        assert "optional" in system_content


class TestChainOfThoughtEffectiveness:
    """Test chain of thought prompt effectiveness."""
    
    @pytest.fixture
    def cot_template(self):
        return ChainOfThoughtTemplate()
    
    def test_problem_solving_structure(self, cot_template):
        """Test that CoT prompts encourage step-by-step thinking."""
        problem = "How should we handle a situation where multiple deliveries are delayed due to a restaurant fire?"
        
        messages = cot_template.format(problem=problem)
        
        assert len(messages) == 2
        system_msg, user_msg = messages
        
        # Check system message encourages step-by-step thinking
        system_content = system_msg.content.lower()
        assert "step by step" in system_content
        assert "understand the problem" in system_content
        assert "identify key information" in system_content
        assert "consider different approaches" in system_content
        assert "work through the solution" in system_content
        assert "verify your reasoning" in system_content
        
        # Check user message contains the problem
        assert problem in user_msg.content
    
    def test_context_integration(self, cot_template):
        """Test that context is properly integrated into prompts."""
        problem = "Optimize delivery routes for peak hours"
        context = "We have 15 drivers, 50 pending orders, and traffic is heavy downtown"
        
        messages = cot_template.format(problem=problem, context=context)
        
        user_content = messages[1].content
        assert problem in user_content
        assert context in user_content
        assert "Context:" in user_content
    
    def test_examples_formatting(self, cot_template):
        """Test that examples are properly formatted in CoT prompts."""
        problem = "Handle delivery disruptions"
        examples = [
            "Example 1: Traffic jam - reroute and notify customers",
            "Example 2: Restaurant delay - offer alternatives or compensation"
        ]
        
        messages = cot_template.format(problem=problem, examples=examples)
        
        user_content = messages[1].content
        assert "Examples:" in user_content
        assert "Example 1:" in user_content
        assert "Example 2:" in user_content
        assert "Traffic jam" in user_content
        assert "Restaurant delay" in user_content


class TestPromptConsistency:
    """Test prompt output consistency and reliability."""
    
    @pytest.fixture
    def template_manager(self):
        return PromptTemplateManager()
    
    def test_react_template_consistency(self, template_manager):
        """Test that ReAct template produces consistent output structure."""
        template = template_manager.get_template("react")
        
        scenario = "Driver delay scenario"
        tools = [{"name": "test_tool", "description": "Test tool"}]
        
        # Generate multiple prompts with same inputs
        messages1 = template.format(scenario=scenario, available_tools=tools)
        messages2 = template.format(scenario=scenario, available_tools=tools)
        
        # Should produce identical results
        assert len(messages1) == len(messages2)
        assert messages1[0].content == messages2[0].content
        assert messages1[1].content == messages2[1].content
    
    def test_template_variable_validation_consistency(self, template_manager):
        """Test that template validation is consistent."""
        template = template_manager.get_template("react")
        
        # Should consistently require the same variables
        required_vars = template.get_required_variables()
        
        # Test multiple times to ensure consistency
        for _ in range(5):
            assert template.get_required_variables() == required_vars
    
    def test_error_handling_consistency(self, template_manager):
        """Test that error handling is consistent across templates."""
        react_template = template_manager.get_template("react")
        cot_template = template_manager.get_template("chain_of_thought")
        
        # Both should handle missing required variables consistently
        with pytest.raises(Exception):  # Should be PromptTemplateError
            react_template.format()
        
        with pytest.raises(Exception):  # Should be PromptTemplateError
            cot_template.format()


class TestDeliveryScenarioExamples:
    """Test the quality and completeness of delivery scenario examples."""
    
    def test_example_completeness(self):
        """Test that examples contain all necessary components."""
        for example in DEFAULT_DELIVERY_EXAMPLES:
            # Each example should have required fields
            assert "title" in example
            assert "scenario" in example
            assert "reasoning_steps" in example
            assert "final_plan" in example
            
            # Scenario should be descriptive
            scenario = example["scenario"]
            assert len(scenario) > 50  # Should be reasonably detailed
            
            # Should contain delivery-related terms
            scenario_lower = scenario.lower()
            delivery_terms = ["delivery", "driver", "customer", "restaurant", "address", "order"]
            assert any(term in scenario_lower for term in delivery_terms)
    
    def test_reasoning_steps_quality(self):
        """Test that reasoning steps demonstrate good problem-solving."""
        for example in DEFAULT_DELIVERY_EXAMPLES:
            steps = example["reasoning_steps"]
            assert len(steps) >= 2  # Should have multiple reasoning steps
            
            for step in steps:
                assert "thought" in step
                # Thoughts should be substantive
                assert len(step["thought"]) > 20
                
                # If there's an action, it should be realistic
                if "action" in step and step["action"]:
                    action = step["action"].lower()
                    valid_actions = [
                        "check_traffic", "notify_customer", "get_merchant", "reroute",
                        "get_nearby_merchants", "cancel_order", "collect_evidence", 
                        "issue_instant_refund"
                    ]
                    assert any(valid_action in action for valid_action in valid_actions)
    
    def test_final_plan_structure(self):
        """Test that final plans have proper structure."""
        for example in DEFAULT_DELIVERY_EXAMPLES:
            plan = example["final_plan"]
            
            # Should have required plan components
            assert "summary" in plan
            assert "steps" in plan
            assert "estimated_duration" in plan
            assert "success_probability" in plan
            
            # Steps should be actionable
            steps = plan["steps"]
            assert len(steps) >= 1
            
            for step in steps:
                assert "sequence" in step
                assert "action" in step
                assert "responsible_party" in step
                assert "success_criteria" in step
                
                # Action should be specific and actionable
                assert len(step["action"]) > 10
                assert len(step["success_criteria"]) > 10
    
    def test_scenario_diversity(self):
        """Test that examples cover diverse delivery scenarios."""
        titles = [example["title"] for example in DEFAULT_DELIVERY_EXAMPLES]
        scenarios = [example["scenario"].lower() for example in DEFAULT_DELIVERY_EXAMPLES]
        
        # Should cover different types of disruptions
        disruption_types = ["traffic", "merchant", "restaurant", "address", "delay"]
        covered_types = []
        
        for scenario in scenarios:
            for disruption_type in disruption_types:
                if disruption_type in scenario:
                    covered_types.append(disruption_type)
        
        # Should cover at least 2 different types of disruptions
        assert len(set(covered_types)) >= 2