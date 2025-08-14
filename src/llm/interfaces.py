"""
Core interfaces for LLM providers and prompt management.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import json


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    role: MessageRole
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: Optional[float] = None


@dataclass
class LLMResponse:
    content: str
    messages: List[Message]
    token_usage: TokenUsage
    model: str
    finish_reason: str
    response_time: float
    timestamp: datetime
    tool_calls: Optional[List[Dict[str, Any]]] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_response(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of conversation messages
            tools: Optional list of available tools for function calling
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse containing the generated response and metadata
        """
        pass
    
    @abstractmethod
    def parse_structured_output(
        self,
        response: LLMResponse,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse structured output from LLM response.
        
        Args:
            response: LLM response to parse
            schema: JSON schema for expected structure
            
        Returns:
            Parsed structured data
        """
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for given text.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        pass


class PromptTemplate(ABC):
    """Abstract base class for prompt templates."""
    
    @abstractmethod
    def format(self, **kwargs) -> List[Message]:
        """
        Format the template with provided variables.
        
        Args:
            **kwargs: Template variables
            
        Returns:
            List of formatted messages
        """
        pass
    
    @abstractmethod
    def get_required_variables(self) -> List[str]:
        """
        Get list of required template variables.
        
        Returns:
            List of required variable names
        """
        pass
    
    @abstractmethod
    def validate_variables(self, **kwargs) -> bool:
        """
        Validate that all required variables are provided.
        
        Args:
            **kwargs: Variables to validate
            
        Returns:
            True if all required variables are present
        """
        pass


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """Represents the result of a tool execution."""
    tool_call_id: str
    name: str
    result: Any
    success: bool
    error_message: Optional[str] = None
    execution_time: Optional[float] = None


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMProviderError(LLMError):
    """Exception for LLM provider-specific errors."""
    pass


class PromptTemplateError(LLMError):
    """Exception for prompt template errors."""
    pass


class TokenLimitError(LLMError):
    """Exception for token limit exceeded errors."""
    pass


class RateLimitError(LLMError):
    """Exception for rate limit errors."""
    pass