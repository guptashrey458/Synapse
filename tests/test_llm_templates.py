"""
Unit tests for LLM prompt templates.
"""
import pytest
from typing import List
from src.llm.interfaces import Message, MessageRole, PromptTemplateError
from src.llm.templates import (
    TemplateVariable, BasePromptTemplate, ReActPromptTemplate,
    ChainOfThoughtTemplate, PromptTemplateManager, DEFAULT_DELIVERY_EXAMPLES
)


class TestTemplateVariable:
    """Test cases for TemplateVariable."""
    
    def test_required_variable(self):
        """Test required template variable."""
        var = TemplateVariable("test_var", "str", required=True)
        
        assert var.name == "test_var"
        assert var.type == "str"
        assert var.required is True
        assert var.default is None
    
    def test_optional_variable_with_default(self):
        """Test optional template variable with default value."""
        var = TemplateVariable("test_var", "list", required=False, default=[])
        
        assert var.name == "test_var"
        assert var.required is False
        assert var.default == []


class TestPromptTemplate(BasePromptTemplate):
    """Concrete implementation for testing BasePromptTemplate."""
    
    def format(self, **kwargs) -> List[Message]:
        self.validate_variables(**kwargs)
        return [Message(role=MessageRole.USER, content="test")]


class TestBasePromptTemplate:
    """Test cases for BasePromptTemplate."""
    
    @pytest.fixture
    def template(self):
        variables = [
            TemplateVariable("required_var", "str", required=True),
            TemplateVariable("optional_var", "str", required=False, default="default")
        ]
        return TestPromptTemplate("test_template", variables)
    
    def test_initialization(self, template):
        """Test template initialization."""
        assert template.name == "test_template"
        assert len(template.variables) == 2
        assert "required_var" in template.variables
        assert "optional_var" in template.variables
    
    def test_get_required_variables(self, template):
        """Test getting required variables."""
        required = template.get_required_variables()
        
        assert required == ["required_var"]
    
    def test_validate_variables_success(self, template):
        """Test successful variable validation."""
        result = template.validate_variables(required_var="test")
        
        assert result is True
    
    def test_validate_variables_missing_required(self, template):
        """Test validation with missing required variables."""
        with pytest.raises(PromptTemplateError, match="Missing required variables"):
            template.validate_variables(optional_var="test")


class TestReActPromptTemplate:
    """Test cases for ReActPromptTemplate."""
    
    @pytest.fixture
    def template(self):
        return ReActPromptTemplate()
    
    @pytest.fixture
    def sample_tools(self):
        return [
            {
                "name": "check_traffic",
                "description": "Check traffic conditions for a route",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Starting location"
                        },
                        "destination": {
                            "type": "string", 
                            "description": "Destination location"
                        }
                    },
                    "required": ["location", "destination"]
                }
            }
        ]
    
    def test_initialization(self, template):
        """Test ReAct template initialization."""
        assert template.name == "react_reasoning"
        required = template.get_required_variables()
        assert "scenario" in required
        assert "available_tools" in required
    
    def test_format_basic_scenario(self, template, sample_tools):
        """Test formatting basic scenario without previous steps."""
        scenario = "Driver is stuck in traffic"
        
        messages = template.format(
            scenario=scenario,
            available_tools=sample_tools
        )
        
        assert len(messages) == 2
        assert messages[0].role == MessageRole.SYSTEM
        assert messages[1].role == MessageRole.USER
        
        # Check that system message contains ReAct instructions
        system_content = messages[0].content
        assert "ReAct" in system_content
        assert "Thought:" in system_content
        assert "Action:" in system_content
        assert "Observation:" in system_content
        
        # Check that user message contains scenario
        user_content = messages[1].content
        assert scenario in user_content
    
    def test_format_with_previous_steps(self, template, sample_tools):
        """Test formatting with previous reasoning steps."""
        scenario = "Driver is stuck in traffic"
        previous_steps = [
            {
                "thought": "I need to check traffic conditions",
                "action": "check_traffic",
                "observation": "Heavy traffic on Highway 101"
            }
        ]
        
        messages = template.format(
            scenario=scenario,
            available_tools=sample_tools,
            previous_steps=previous_steps
        )
        
        user_content = messages[1].content
        assert "Previous Reasoning Steps:" in user_content
        assert "I need to check traffic conditions" in user_content
        assert "Heavy traffic on Highway 101" in user_content
    
    def test_format_with_examples(self, template, sample_tools):
        """Test formatting with few-shot examples."""
        scenario = "Driver is stuck in traffic"
        examples = DEFAULT_DELIVERY_EXAMPLES
        
        messages = template.format(
            scenario=scenario,
            available_tools=sample_tools,
            examples=examples
        )
        
        system_content = messages[0].content
        assert "Examples" in system_content
        assert "Traffic Disruption" in system_content
    
    def test_format_tools_description(self, template, sample_tools):
        """Test tools description formatting."""
        tools_desc = template._format_tools_description(sample_tools)
        
        assert "check_traffic" in tools_desc
        assert "Check traffic conditions" in tools_desc
        assert "location" in tools_desc
        assert "required" in tools_desc
    
    def test_format_tools_description_empty(self, template):
        """Test tools description with no tools."""
        tools_desc = template._format_tools_description([])
        
        assert tools_desc == "No tools available."
    
    def test_validate_variables_missing_scenario(self, template, sample_tools):
        """Test validation with missing scenario."""
        with pytest.raises(PromptTemplateError):
            template.format(available_tools=sample_tools)
    
    def test_validate_variables_missing_tools(self, template):
        """Test validation with missing tools."""
        with pytest.raises(PromptTemplateError):
            template.format(scenario="Test scenario")


class TestChainOfThoughtTemplate:
    """Test cases for ChainOfThoughtTemplate."""
    
    @pytest.fixture
    def template(self):
        return ChainOfThoughtTemplate()
    
    def test_initialization(self, template):
        """Test chain of thought template initialization."""
        assert template.name == "chain_of_thought"
        required = template.get_required_variables()
        assert "problem" in required
    
    def test_format_basic_problem(self, template):
        """Test formatting basic problem."""
        problem = "How to optimize delivery routes?"
        
        messages = template.format(problem=problem)
        
        assert len(messages) == 2
        assert messages[0].role == MessageRole.SYSTEM
        assert messages[1].role == MessageRole.USER
        
        # Check system message contains step-by-step instructions
        system_content = messages[0].content
        assert "step by step" in system_content.lower()
        
        # Check user message contains problem
        user_content = messages[1].content
        assert problem in user_content
    
    def test_format_with_context(self, template):
        """Test formatting with additional context."""
        problem = "How to optimize delivery routes?"
        context = "We have 10 drivers and 50 deliveries per day"
        
        messages = template.format(problem=problem, context=context)
        
        user_content = messages[1].content
        assert problem in user_content
        assert context in user_content
    
    def test_format_with_examples(self, template):
        """Test formatting with examples."""
        problem = "How to optimize delivery routes?"
        examples = ["Example 1: Use shortest path algorithm", "Example 2: Consider traffic patterns"]
        
        messages = template.format(problem=problem, examples=examples)
        
        user_content = messages[1].content
        assert "Examples:" in user_content
        assert "Example 1" in user_content
        assert "Example 2" in user_content


class TestPromptTemplateManager:
    """Test cases for PromptTemplateManager."""
    
    @pytest.fixture
    def manager(self):
        return PromptTemplateManager()
    
    def test_initialization(self, manager):
        """Test manager initialization with default templates."""
        templates = manager.list_templates()
        
        assert "react" in templates
        assert "chain_of_thought" in templates
    
    def test_get_template(self, manager):
        """Test getting template by name."""
        template = manager.get_template("react")
        
        assert isinstance(template, ReActPromptTemplate)
    
    def test_get_nonexistent_template(self, manager):
        """Test getting non-existent template."""
        with pytest.raises(PromptTemplateError, match="Template 'nonexistent' not found"):
            manager.get_template("nonexistent")
    
    def test_register_custom_template(self, manager):
        """Test registering custom template."""
        custom_template = ChainOfThoughtTemplate()
        manager.register_template("custom", custom_template)
        
        retrieved = manager.get_template("custom")
        assert retrieved == custom_template
        
        templates = manager.list_templates()
        assert "custom" in templates
    
    def test_list_templates(self, manager):
        """Test listing all templates."""
        templates = manager.list_templates()
        
        assert isinstance(templates, list)
        assert len(templates) >= 2  # At least default templates


class TestDefaultDeliveryExamples:
    """Test cases for default delivery examples."""
    
    def test_examples_structure(self):
        """Test that default examples have correct structure."""
        examples = DEFAULT_DELIVERY_EXAMPLES
        
        assert len(examples) > 0
        
        for example in examples:
            assert "title" in example
            assert "scenario" in example
            assert "reasoning_steps" in example
            assert "final_plan" in example
            
            # Check reasoning steps structure
            for step in example["reasoning_steps"]:
                assert "thought" in step
                # Action and observation are optional
            
            # Check final plan structure
            plan = example["final_plan"]
            assert "summary" in plan
            assert "steps" in plan
            assert "estimated_duration" in plan
            assert "success_probability" in plan
    
    def test_traffic_disruption_example(self):
        """Test specific traffic disruption example."""
        traffic_example = None
        for example in DEFAULT_DELIVERY_EXAMPLES:
            if "Traffic Disruption" in example["title"]:
                traffic_example = example
                break
        
        assert traffic_example is not None
        assert "Highway 101" in traffic_example["scenario"]
        assert "pizza delivery" in traffic_example["scenario"]
        
        # Check that it has realistic reasoning steps
        steps = traffic_example["reasoning_steps"]
        assert len(steps) >= 2
        
        # Should include traffic check and customer notification
        actions = [step.get("action", "") for step in steps]
        assert "check_traffic" in actions
        assert "notify_customer" in actions