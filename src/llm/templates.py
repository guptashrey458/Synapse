"""
Prompt template management and ReAct pattern templates.
"""
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .interfaces import PromptTemplate, Message, MessageRole, PromptTemplateError


@dataclass
class TemplateVariable:
    """Represents a template variable with validation."""
    name: str
    type: str
    required: bool = True
    default: Optional[Any] = None
    description: Optional[str] = None


class BasePromptTemplate(PromptTemplate):
    """Base implementation of prompt template."""
    
    def __init__(self, name: str, variables: List[TemplateVariable]):
        self.name = name
        self.variables = {var.name: var for var in variables}
    
    def get_required_variables(self) -> List[str]:
        """Get list of required template variables."""
        return [name for name, var in self.variables.items() if var.required]
    
    def validate_variables(self, **kwargs) -> bool:
        """Validate that all required variables are provided."""
        required = self.get_required_variables()
        missing = [var for var in required if var not in kwargs]
        
        if missing:
            raise PromptTemplateError(f"Missing required variables: {missing}")
        
        return True


class ReActPromptTemplate(BasePromptTemplate):
    """ReAct pattern prompt template for reasoning and tool usage."""
    
    def __init__(self):
        variables = [
            TemplateVariable("scenario", "str", description="The disruption scenario to analyze"),
            TemplateVariable("available_tools", "list", description="List of available tools"),
            TemplateVariable("previous_steps", "list", required=False, default=[], 
                           description="Previous reasoning steps"),
            TemplateVariable("examples", "list", required=False, default=[],
                           description="Few-shot examples")
        ]
        super().__init__("react_reasoning", variables)
    
    def format(self, **kwargs) -> List[Message]:
        """Format ReAct prompt with provided variables."""
        self.validate_variables(**kwargs)
        
        scenario = kwargs["scenario"]
        available_tools = kwargs["available_tools"]
        previous_steps = kwargs.get("previous_steps", [])
        examples = kwargs.get("examples", [])
        
        # System message with ReAct instructions
        system_content = self._build_system_message(available_tools, examples)
        
        # User message with scenario and context
        user_content = self._build_user_message(scenario, previous_steps)
        
        messages = [
            Message(role=MessageRole.SYSTEM, content=system_content),
            Message(role=MessageRole.USER, content=user_content)
        ]
        
        return messages
    
    def _build_system_message(self, available_tools: List[Dict], examples: List[Dict]) -> str:
        """Build the system message with ReAct instructions."""
        tools_description = self._format_tools_description(available_tools)
        examples_text = self._format_examples(examples)
        
        return f"""You are an autonomous delivery coordinator agent that resolves last-mile delivery disruptions using a ReAct (Reasoning + Acting) approach.

Your task is to analyze delivery disruption scenarios and create actionable resolution plans by:
1. **Reasoning** about the problem step by step
2. **Acting** by using available tools to gather information
3. **Observing** the results and incorporating them into your reasoning
4. **Planning** a comprehensive resolution strategy

## Available Tools

{tools_description}

## ReAct Pattern Instructions

Follow this pattern for each reasoning step:

**Thought:** [Your reasoning about the current situation and what you need to do next]
**Action:** [The tool you want to use and why]
**Action Input:** [The specific parameters for the tool call]
**Observation:** [The result from the tool - this will be provided after tool execution]

Continue this pattern until you have enough information to create a comprehensive resolution plan.

## Output Format

When you have sufficient information, provide your final response in this format:

```json
{{
  "reasoning_complete": true,
  "final_plan": {{
    "summary": "Brief summary of the resolution strategy",
    "steps": [
      {{
        "sequence": 1,
        "action": "Specific action to take",
        "responsible_party": "Who should perform this action",
        "estimated_time": "How long this should take",
        "success_criteria": "How to know this step succeeded"
      }}
    ],
    "estimated_duration": "Total estimated resolution time",
    "success_probability": 0.85,
    "alternatives": ["Alternative approach if primary plan fails"],
    "stakeholders": ["List of people/systems that need to be involved"]
  }}
}}
```

{examples_text}

Remember:
- Be thorough in your reasoning but efficient in tool usage
- Consider all stakeholders (customer, driver, merchant, support team)
- Prioritize actions that have the highest impact on successful delivery
- Always provide clear, actionable steps with specific responsibilities
- Consider alternative solutions for complex scenarios"""
    
    def _build_user_message(self, scenario: str, previous_steps: List[Dict]) -> str:
        """Build the user message with scenario and context."""
        message = f"**Disruption Scenario:**\n{scenario}\n\n"
        
        if previous_steps:
            message += "**Previous Reasoning Steps:**\n"
            for i, step in enumerate(previous_steps, 1):
                message += f"{i}. **Thought:** {step.get('thought', '')}\n"
                if step.get('action'):
                    message += f"   **Action:** {step['action']}\n"
                if step.get('observation'):
                    message += f"   **Observation:** {step['observation']}\n"
                message += "\n"
        
        message += "Please analyze this scenario using the ReAct pattern and provide your reasoning step by step."
        
        return message
    
    def _format_tools_description(self, tools: List[Dict]) -> str:
        """Format available tools for the prompt."""
        if not tools:
            return "No tools available."
        
        tools_text = ""
        for tool in tools:
            name = tool.get("name", "Unknown")
            description = tool.get("description", "No description")
            parameters = tool.get("parameters", {})
            
            tools_text += f"**{name}:** {description}\n"
            
            if parameters and "properties" in parameters:
                tools_text += "  Parameters:\n"
                for param_name, param_info in parameters["properties"].items():
                    param_desc = param_info.get("description", "")
                    param_type = param_info.get("type", "")
                    required = param_name in parameters.get("required", [])
                    req_text = " (required)" if required else " (optional)"
                    tools_text += f"  - {param_name} ({param_type}){req_text}: {param_desc}\n"
            
            tools_text += "\n"
        
        return tools_text
    
    def _format_examples(self, examples: List[Dict]) -> str:
        """Format few-shot examples for the prompt."""
        if not examples:
            return ""
        
        examples_text = "\n## Examples\n\n"
        
        for i, example in enumerate(examples, 1):
            examples_text += f"### Example {i}: {example.get('title', 'Delivery Disruption')}\n\n"
            examples_text += f"**Scenario:** {example.get('scenario', '')}\n\n"
            
            if 'reasoning_steps' in example:
                examples_text += "**Reasoning Process:**\n"
                for step in example['reasoning_steps']:
                    examples_text += f"**Thought:** {step.get('thought', '')}\n"
                    if step.get('action'):
                        examples_text += f"**Action:** {step['action']}\n"
                        examples_text += f"**Action Input:** {step.get('action_input', '')}\n"
                    if step.get('observation'):
                        examples_text += f"**Observation:** {step['observation']}\n"
                    examples_text += "\n"
            
            if 'final_plan' in example:
                examples_text += f"**Final Plan:** {json.dumps(example['final_plan'], indent=2)}\n\n"
        
        return examples_text


class ChainOfThoughtTemplate(BasePromptTemplate):
    """Chain of thought prompt template for step-by-step reasoning."""
    
    def __init__(self):
        variables = [
            TemplateVariable("problem", "str", description="The problem to solve"),
            TemplateVariable("context", "str", required=False, default="",
                           description="Additional context"),
            TemplateVariable("examples", "list", required=False, default=[],
                           description="Few-shot examples")
        ]
        super().__init__("chain_of_thought", variables)
    
    def format(self, **kwargs) -> List[Message]:
        """Format chain of thought prompt."""
        self.validate_variables(**kwargs)
        
        problem = kwargs["problem"]
        context = kwargs.get("context", "")
        examples = kwargs.get("examples", [])
        
        system_content = """You are an expert problem solver. Break down complex problems into clear, logical steps.

Think through problems step by step:
1. Understand the problem clearly
2. Identify key information and constraints
3. Consider different approaches
4. Work through the solution systematically
5. Verify your reasoning

Always show your thinking process clearly."""
        
        user_content = f"Problem: {problem}\n"
        if context:
            user_content += f"\nContext: {context}\n"
        
        if examples:
            user_content += "\nExamples:\n"
            for i, example in enumerate(examples, 1):
                user_content += f"{i}. {example}\n"
        
        user_content += "\nPlease solve this step by step, showing your reasoning clearly."
        
        return [
            Message(role=MessageRole.SYSTEM, content=system_content),
            Message(role=MessageRole.USER, content=user_content)
        ]


class PromptTemplateManager:
    """Manages prompt templates and provides template registry."""
    
    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}
        self._register_default_templates()
    
    def register_template(self, name: str, template: PromptTemplate) -> None:
        """Register a new template."""
        self._templates[name] = template
    
    def get_template(self, name: str) -> PromptTemplate:
        """Get a template by name."""
        if name not in self._templates:
            raise PromptTemplateError(f"Template '{name}' not found")
        return self._templates[name]
    
    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self._templates.keys())
    
    def _register_default_templates(self) -> None:
        """Register default templates."""
        self.register_template("react", ReActPromptTemplate())
        self.register_template("chain_of_thought", ChainOfThoughtTemplate())


# Default few-shot examples for delivery scenarios
DEFAULT_DELIVERY_EXAMPLES = [
    {
        "title": "Traffic Disruption",
        "scenario": "Driver John is stuck in traffic on Highway 101 due to an accident. He has a pizza delivery for customer Sarah at 123 Main St that was supposed to arrive at 7:30 PM. It's now 7:45 PM and he's still 15 minutes away.",
        "reasoning_steps": [
            {
                "thought": "The driver is delayed due to traffic and the customer is waiting. I need to check the current traffic situation and find alternative routes.",
                "action": "check_traffic",
                "action_input": '{"location": "Highway 101", "destination": "123 Main St"}',
                "observation": "Heavy traffic due to accident, estimated delay 20 minutes. Alternative route via Oak Street available, adds 5 minutes but avoids traffic."
            },
            {
                "thought": "There's an alternative route that's faster. I should also notify the customer about the delay and get the driver rerouted.",
                "action": "notify_customer",
                "action_input": '{"customer": "Sarah", "message": "Your pizza delivery is running about 10 minutes late due to traffic. Driver is taking alternate route.", "estimated_arrival": "8:00 PM"}',
                "observation": "Customer notified successfully. Customer responded: 'Thanks for letting me know!'"
            }
        ],
        "final_plan": {
            "summary": "Reroute driver via Oak Street and keep customer informed",
            "steps": [
                {
                    "sequence": 1,
                    "action": "Reroute driver John via Oak Street to avoid Highway 101 traffic",
                    "responsible_party": "Dispatch system",
                    "estimated_time": "2 minutes",
                    "success_criteria": "Driver receives new route and confirms"
                },
                {
                    "sequence": 2,
                    "action": "Monitor driver progress and update customer if needed",
                    "responsible_party": "Customer service",
                    "estimated_time": "15 minutes",
                    "success_criteria": "Delivery completed successfully"
                }
            ],
            "estimated_duration": "15 minutes",
            "success_probability": 0.9,
            "alternatives": ["If Oak Street also has issues, consider refund and reorder from closer restaurant"],
            "stakeholders": ["Driver John", "Customer Sarah", "Dispatch team"]
        }
    },
    {
        "title": "Restaurant Overloaded",
        "scenario": "Bella's Italian Restaurant is overwhelmed with orders during dinner rush. Customer Mike's pasta order was supposed to be ready at 6:45 PM for pickup by driver Lisa, but the restaurant says it will be another 25 minutes. Mike lives 15 minutes away and was expecting delivery at 7:15 PM.",
        "reasoning_steps": [
            {
                "thought": "The restaurant is significantly delayed, which will impact the customer's expected delivery time. I need to check if there are alternative restaurants nearby that could fulfill a similar order faster.",
                "action": "get_nearby_merchants",
                "action_input": '{"location": "Mike\'s address", "cuisine_type": "Italian", "max_distance": "3 miles"}',
                "observation": "Found 2 Italian restaurants within 3 miles: Mario's Pizza (10 min prep time) and Giuseppe's (15 min prep time). Both have similar pasta dishes available."
            },
            {
                "thought": "There are alternatives available that could deliver faster. I should notify the customer about the delay and offer alternatives, while also checking if they prefer to wait for their original order.",
                "action": "notify_customer",
                "action_input": '{"customer": "Mike", "message": "Your order from Bella\'s is running 25 minutes late due to high demand. We found similar pasta dishes at Mario\'s Pizza that could be delivered 20 minutes sooner. Would you like to switch or wait for your original order?", "alternatives": ["Switch to Mario\'s Pizza", "Wait for original order"]}',
                "observation": "Customer responded: 'I'll switch to Mario's Pizza if they have penne arrabbiata. Thanks for the quick solution!'"
            },
            {
                "thought": "Customer wants to switch to Mario's Pizza. I need to cancel the original order and place a new one at Mario's.",
                "action": "cancel_order",
                "action_input": '{"restaurant": "Bella\'s Italian", "order_id": "B12345", "reason": "Customer switched due to delay"}',
                "observation": "Order cancelled successfully. Refund processed."
            }
        ],
        "final_plan": {
            "summary": "Switch customer to faster alternative restaurant with similar cuisine",
            "steps": [
                {
                    "sequence": 1,
                    "action": "Place new order at Mario's Pizza for penne arrabbiata",
                    "responsible_party": "Order management system",
                    "estimated_time": "3 minutes",
                    "success_criteria": "New order confirmed and payment processed"
                },
                {
                    "sequence": 2,
                    "action": "Assign driver Lisa to pickup from Mario's Pizza instead",
                    "responsible_party": "Dispatch system",
                    "estimated_time": "2 minutes",
                    "success_criteria": "Driver receives new pickup location and confirms"
                },
                {
                    "sequence": 3,
                    "action": "Update customer with new ETA and order details",
                    "responsible_party": "Customer service",
                    "estimated_time": "1 minute",
                    "success_criteria": "Customer confirms satisfaction with new arrangement"
                }
            ],
            "estimated_duration": "25 minutes total (vs 40 minutes with original)",
            "success_probability": 0.95,
            "alternatives": ["If Mario's also gets busy, try Giuseppe's as backup", "Offer partial refund if customer prefers to cancel entirely"],
            "stakeholders": ["Customer Mike", "Driver Lisa", "Mario's Pizza", "Bella's Italian", "Order management"]
        }
    },
    {
        "title": "Damaged Package Dispute",
        "scenario": "Driver Carlos delivered a Thai food order to customer Jennifer at 789 Pine Street. Jennifer claims the food containers were damaged and sauce leaked all over the bag, making the food inedible. She's requesting a full refund and wants to file a complaint. Carlos says the food was fine when he picked it up.",
        "reasoning_steps": [
            {
                "thought": "This is a dispute situation where I need to gather evidence from both parties to determine what happened and provide a fair resolution. I should collect photos and statements from both the customer and driver.",
                "action": "collect_evidence",
                "action_input": '{"incident_type": "damaged_package", "customer": "Jennifer", "driver": "Carlos", "order_id": "T67890"}',
                "observation": "Customer provided photos showing damaged containers with sauce leakage. Driver provided photos from pickup showing intact packaging. Timestamp shows 15-minute gap between pickup and delivery."
            },
            {
                "thought": "The evidence shows the package was intact at pickup but damaged at delivery. This could be due to rough handling during transport or poor packaging. Given the customer has clear evidence of damage, I should offer immediate resolution while also addressing the root cause.",
                "action": "issue_instant_refund",
                "action_input": '{"customer": "Jennifer", "order_id": "T67890", "amount": "full", "reason": "Damaged food containers with sauce leakage"}',
                "observation": "Full refund of $24.50 processed successfully. Customer notified of refund."
            },
            {
                "thought": "I should also offer to reorder the meal from the same restaurant if the customer wants, and provide feedback to the driver about careful handling of liquid containers.",
                "action": "notify_customer",
                "action_input": '{"customer": "Jennifer", "message": "We\'ve processed your full refund for the damaged order. Would you like us to reorder the same meal with extra careful packaging? We apologize for the inconvenience.", "offer_reorder": true}',
                "observation": "Customer accepted reorder offer and appreciated the quick resolution."
            }
        ],
        "final_plan": {
            "summary": "Provide immediate refund and reorder with improved packaging protocols",
            "steps": [
                {
                    "sequence": 1,
                    "action": "Place new order with special handling instructions for liquid containers",
                    "responsible_party": "Order management",
                    "estimated_time": "5 minutes",
                    "success_criteria": "Restaurant confirms extra secure packaging"
                },
                {
                    "sequence": 2,
                    "action": "Assign different driver with training on fragile item handling",
                    "responsible_party": "Dispatch system",
                    "estimated_time": "3 minutes",
                    "success_criteria": "Experienced driver assigned and briefed"
                },
                {
                    "sequence": 3,
                    "action": "Provide additional training to Carlos on careful handling of liquid containers",
                    "responsible_party": "Driver management",
                    "estimated_time": "15 minutes",
                    "success_criteria": "Carlos completes handling training module"
                },
                {
                    "sequence": 4,
                    "action": "Follow up with customer after successful redelivery",
                    "responsible_party": "Customer service",
                    "estimated_time": "2 minutes",
                    "success_criteria": "Customer confirms satisfaction with resolution"
                }
            ],
            "estimated_duration": "45 minutes for complete resolution",
            "success_probability": 0.92,
            "alternatives": ["If restaurant can't improve packaging, switch to different Thai restaurant", "Offer store credit instead of reorder if customer prefers"],
            "stakeholders": ["Customer Jennifer", "Driver Carlos", "Thai restaurant", "Driver management", "Customer service"]
        }
    }
]