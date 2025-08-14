"""
LLM provider implementations for OpenAI and Anthropic.
"""
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

from ..config.settings import LLMConfig, LLMProvider as LLMProviderEnum
from .interfaces import (
    LLMProvider, LLMResponse, Message, MessageRole, TokenUsage,
    LLMProviderError, TokenLimitError, RateLimitError
)

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
        self._token_costs = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002}
        }
    
    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout
                )
            except ImportError:
                raise LLMProviderError("OpenAI package not installed. Install with: pip install openai")
        return self._client
    
    def generate_response(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        start_time = time.time()
        
        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            openai_msg = {
                "role": msg.role.value,
                "content": msg.content
            }
            if msg.tool_calls:
                openai_msg["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id
            openai_messages.append(openai_msg)
        
        # Prepare request parameters
        request_params = {
            "model": self.config.model,
            "messages": openai_messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            **kwargs
        }
        
        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = "auto"
        
        try:
            response = self.client.chat.completions.create(**request_params)
            response_time = time.time() - start_time
            
            # Extract response data
            choice = response.choices[0]
            content = choice.message.content or ""
            tool_calls = None
            
            if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in choice.message.tool_calls
                ]
            
            # Calculate token usage and cost
            usage = response.usage
            token_usage = TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                cost_usd=self._calculate_cost(usage.prompt_tokens, usage.completion_tokens)
            )
            
            # Create response messages
            response_messages = messages + [
                Message(
                    role=MessageRole.ASSISTANT,
                    content=content,
                    tool_calls=tool_calls
                )
            ]
            
            return LLMResponse(
                content=content,
                messages=response_messages,
                token_usage=token_usage,
                model=self.config.model,
                finish_reason=choice.finish_reason,
                response_time=response_time,
                timestamp=datetime.now(),
                tool_calls=tool_calls
            )
            
        except Exception as e:
            if "rate_limit" in str(e).lower():
                raise RateLimitError(f"OpenAI rate limit exceeded: {e}")
            elif "token" in str(e).lower() and "limit" in str(e).lower():
                raise TokenLimitError(f"OpenAI token limit exceeded: {e}")
            else:
                raise LLMProviderError(f"OpenAI API error: {e}")
    
    def parse_structured_output(
        self,
        response: LLMResponse,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse structured output from OpenAI response."""
        try:
            # Try to parse JSON from content
            content = response.content.strip()
            
            # Handle code blocks
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            parsed = json.loads(content)
            
            # Basic schema validation (simplified)
            if "type" in schema and schema["type"] == "object":
                if not isinstance(parsed, dict):
                    raise ValueError("Expected object type")
                
                if "required" in schema:
                    for field in schema["required"]:
                        if field not in parsed:
                            raise ValueError(f"Required field '{field}' missing")
            
            return parsed
            
        except json.JSONDecodeError as e:
            raise LLMProviderError(f"Failed to parse JSON from response: {e}")
        except ValueError as e:
            raise LLMProviderError(f"Schema validation failed: {e}")
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using simple heuristic."""
        # Rough estimation: ~4 characters per token for English text
        return len(text) // 4
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information."""
        return {
            "provider": "openai",
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "supports_tools": True,
            "supports_structured_output": True
        }
    
    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
        """Calculate cost based on token usage."""
        if self.config.model not in self._token_costs:
            return None
        
        costs = self._token_costs[self.config.model]
        prompt_cost = (prompt_tokens / 1000) * costs["input"]
        completion_cost = (completion_tokens / 1000) * costs["output"]
        
        return prompt_cost + completion_cost


class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider implementation."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
        self._token_costs = {
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125}
        }
    
    @property
    def client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(
                    api_key=self.config.api_key,
                    timeout=self.config.timeout
                )
            except ImportError:
                raise LLMProviderError("Anthropic package not installed. Install with: pip install anthropic")
        return self._client
    
    def generate_response(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Anthropic API."""
        start_time = time.time()
        
        # Convert messages to Anthropic format
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
        
        # Prepare request parameters
        request_params = {
            "model": self.config.model,
            "messages": anthropic_messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            **kwargs
        }
        
        if system_message:
            request_params["system"] = system_message
        
        if tools:
            request_params["tools"] = tools
        
        try:
            response = self.client.messages.create(**request_params)
            response_time = time.time() - start_time
            
            # Extract response data
            content = ""
            tool_calls = None
            
            for content_block in response.content:
                if content_block.type == "text":
                    content += content_block.text
                elif content_block.type == "tool_use":
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append({
                        "id": content_block.id,
                        "type": "function",
                        "function": {
                            "name": content_block.name,
                            "arguments": json.dumps(content_block.input)
                        }
                    })
            
            # Calculate token usage and cost
            token_usage = TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                cost_usd=self._calculate_cost(response.usage.input_tokens, response.usage.output_tokens)
            )
            
            # Create response messages
            response_messages = messages + [
                Message(
                    role=MessageRole.ASSISTANT,
                    content=content,
                    tool_calls=tool_calls
                )
            ]
            
            return LLMResponse(
                content=content,
                messages=response_messages,
                token_usage=token_usage,
                model=self.config.model,
                finish_reason=response.stop_reason,
                response_time=response_time,
                timestamp=datetime.now(),
                tool_calls=tool_calls
            )
            
        except Exception as e:
            if "rate_limit" in str(e).lower():
                raise RateLimitError(f"Anthropic rate limit exceeded: {e}")
            elif "token" in str(e).lower() and "limit" in str(e).lower():
                raise TokenLimitError(f"Anthropic token limit exceeded: {e}")
            else:
                raise LLMProviderError(f"Anthropic API error: {e}")
    
    def parse_structured_output(
        self,
        response: LLMResponse,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse structured output from Anthropic response."""
        # Same implementation as OpenAI since both return JSON
        try:
            content = response.content.strip()
            
            # Handle code blocks
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            parsed = json.loads(content)
            
            # Basic schema validation
            if "type" in schema and schema["type"] == "object":
                if not isinstance(parsed, dict):
                    raise ValueError("Expected object type")
                
                if "required" in schema:
                    for field in schema["required"]:
                        if field not in parsed:
                            raise ValueError(f"Required field '{field}' missing")
            
            return parsed
            
        except json.JSONDecodeError as e:
            raise LLMProviderError(f"Failed to parse JSON from response: {e}")
        except ValueError as e:
            raise LLMProviderError(f"Schema validation failed: {e}")
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using simple heuristic."""
        # Rough estimation: ~4 characters per token for English text
        return len(text) // 4
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Anthropic model information."""
        return {
            "provider": "anthropic",
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "supports_tools": True,
            "supports_structured_output": True
        }
    
    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
        """Calculate cost based on token usage."""
        if self.config.model not in self._token_costs:
            return None
        
        costs = self._token_costs[self.config.model]
        prompt_cost = (prompt_tokens / 1000) * costs["input"]
        completion_cost = (completion_tokens / 1000) * costs["output"]
        
        return prompt_cost + completion_cost


def get_llm_provider(config: LLMConfig) -> LLMProvider:
    """
    Factory function to create LLM provider based on configuration.
    
    Args:
        config: LLM configuration
        
    Returns:
        LLM provider instance
    """
    if config.provider == LLMProviderEnum.OPENAI:
        return OpenAIProvider(config)
    elif config.provider == LLMProviderEnum.ANTHROPIC:
        return AnthropicProvider(config)
    else:
        raise LLMProviderError(f"Unsupported LLM provider: {config.provider}")