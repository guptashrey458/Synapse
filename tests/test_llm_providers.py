"""
Unit tests for LLM provider implementations.
"""
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.config.settings import LLMConfig, LLMProvider as LLMProviderEnum
from src.llm.interfaces import (
    Message, MessageRole, TokenUsage, LLMResponse,
    LLMProviderError, TokenLimitError, RateLimitError
)
from src.llm.providers import OpenAIProvider, AnthropicProvider, get_llm_provider


class TestOpenAIProvider:
    """Test cases for OpenAI provider."""
    
    @pytest.fixture
    def config(self):
        return LLMConfig(
            provider=LLMProviderEnum.OPENAI,
            model="gpt-4",
            api_key="test-key",
            max_tokens=1000,
            temperature=0.1
        )
    
    @pytest.fixture
    def provider(self, config):
        return OpenAIProvider(config)
    
    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI API response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        return mock_response
    
    def test_initialization(self, provider, config):
        """Test provider initialization."""
        assert provider.config == config
        assert provider._client is None
    
    @patch('builtins.__import__')
    def test_client_lazy_initialization(self, mock_import, provider):
        """Test lazy initialization of OpenAI client."""
        mock_openai = Mock()
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        
        def import_side_effect(name, *args, **kwargs):
            if name == 'openai':
                return mock_openai
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_side_effect
        
        client = provider.client
        
        assert client == mock_client
        mock_openai.OpenAI.assert_called_once_with(
            api_key="test-key",
            base_url=None,
            timeout=30
        )
    
    @patch('builtins.__import__')
    def test_client_import_error(self, mock_import, provider):
        """Test handling of missing OpenAI package."""
        def import_side_effect(name, *args, **kwargs):
            if name == 'openai':
                raise ImportError("No module named 'openai'")
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_side_effect
        
        with pytest.raises(LLMProviderError, match="OpenAI package not installed"):
            _ = provider.client
    
    @patch('builtins.__import__')
    def test_generate_response_success(self, mock_import, provider, mock_openai_response):
        """Test successful response generation."""
        mock_openai = Mock()
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai.OpenAI.return_value = mock_client
        
        def import_side_effect(name, *args, **kwargs):
            if name == 'openai':
                return mock_openai
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_side_effect
        
        messages = [
            Message(role=MessageRole.USER, content="Test message")
        ]
        
        response = provider.generate_response(messages)
        
        assert isinstance(response, LLMResponse)
        assert response.content == "Test response"
        assert response.model == "gpt-4"
        assert response.finish_reason == "stop"
        assert response.token_usage.prompt_tokens == 100
        assert response.token_usage.completion_tokens == 50
        assert response.token_usage.total_tokens == 150
        assert len(response.messages) == 2  # Original + response
    
    @patch('builtins.__import__')
    def test_generate_response_with_tools(self, mock_import, provider, mock_openai_response):
        """Test response generation with tool calls."""
        # Mock tool call response
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"param": "value"}'
        
        mock_openai_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_openai_response.choices[0].message.content = None
        
        mock_openai = Mock()
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai.OpenAI.return_value = mock_client
        
        def import_side_effect(name, *args, **kwargs):
            if name == 'openai':
                return mock_openai
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_side_effect
        
        messages = [Message(role=MessageRole.USER, content="Use a tool")]
        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        
        response = provider.generate_response(messages, tools=tools)
        
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["id"] == "call_123"
        assert response.tool_calls[0]["function"]["name"] == "test_tool"
    
    @patch('builtins.__import__')
    def test_generate_response_rate_limit_error(self, mock_import, provider):
        """Test handling of rate limit errors."""
        mock_openai = Mock()
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("rate_limit_exceeded")
        mock_openai.OpenAI.return_value = mock_client
        
        def import_side_effect(name, *args, **kwargs):
            if name == 'openai':
                return mock_openai
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_side_effect
        
        messages = [Message(role=MessageRole.USER, content="Test")]
        
        with pytest.raises(RateLimitError):
            provider.generate_response(messages)
    
    @patch('builtins.__import__')
    def test_generate_response_token_limit_error(self, mock_import, provider):
        """Test handling of token limit errors."""
        mock_openai = Mock()
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("token limit exceeded")
        mock_openai.OpenAI.return_value = mock_client
        
        def import_side_effect(name, *args, **kwargs):
            if name == 'openai':
                return mock_openai
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_side_effect
        
        messages = [Message(role=MessageRole.USER, content="Test")]
        
        with pytest.raises(TokenLimitError):
            provider.generate_response(messages)
    
    def test_parse_structured_output_success(self, provider):
        """Test successful structured output parsing."""
        response = LLMResponse(
            content='{"key": "value", "number": 42}',
            messages=[],
            token_usage=TokenUsage(0, 0, 0),
            model="gpt-4",
            finish_reason="stop",
            response_time=1.0,
            timestamp=datetime.now()
        )
        
        schema = {
            "type": "object",
            "required": ["key", "number"]
        }
        
        result = provider.parse_structured_output(response, schema)
        
        assert result == {"key": "value", "number": 42}
    
    def test_parse_structured_output_with_code_blocks(self, provider):
        """Test parsing structured output with JSON code blocks."""
        response = LLMResponse(
            content='```json\n{"key": "value"}\n```',
            messages=[],
            token_usage=TokenUsage(0, 0, 0),
            model="gpt-4",
            finish_reason="stop",
            response_time=1.0,
            timestamp=datetime.now()
        )
        
        schema = {"type": "object"}
        
        result = provider.parse_structured_output(response, schema)
        
        assert result == {"key": "value"}
    
    def test_parse_structured_output_invalid_json(self, provider):
        """Test handling of invalid JSON in structured output."""
        response = LLMResponse(
            content='invalid json',
            messages=[],
            token_usage=TokenUsage(0, 0, 0),
            model="gpt-4",
            finish_reason="stop",
            response_time=1.0,
            timestamp=datetime.now()
        )
        
        schema = {"type": "object"}
        
        with pytest.raises(LLMProviderError, match="Failed to parse JSON"):
            provider.parse_structured_output(response, schema)
    
    def test_parse_structured_output_schema_validation_error(self, provider):
        """Test schema validation errors."""
        response = LLMResponse(
            content='{"key": "value"}',
            messages=[],
            token_usage=TokenUsage(0, 0, 0),
            model="gpt-4",
            finish_reason="stop",
            response_time=1.0,
            timestamp=datetime.now()
        )
        
        schema = {
            "type": "object",
            "required": ["missing_field"]
        }
        
        with pytest.raises(LLMProviderError, match="Schema validation failed"):
            provider.parse_structured_output(response, schema)
    
    def test_estimate_tokens(self, provider):
        """Test token estimation."""
        text = "This is a test message with some words"
        estimated = provider.estimate_tokens(text)
        
        # Should be roughly len(text) // 4
        expected = len(text) // 4
        assert estimated == expected
    
    def test_get_model_info(self, provider):
        """Test model information retrieval."""
        info = provider.get_model_info()
        
        assert info["provider"] == "openai"
        assert info["model"] == "gpt-4"
        assert info["max_tokens"] == 1000
        assert info["temperature"] == 0.1
        assert info["supports_tools"] is True
        assert info["supports_structured_output"] is True
    
    def test_calculate_cost(self, provider):
        """Test cost calculation."""
        cost = provider._calculate_cost(1000, 500)
        
        # gpt-4: $0.03/1k input, $0.06/1k output
        expected = (1000/1000 * 0.03) + (500/1000 * 0.06)
        assert cost == expected
    
    def test_calculate_cost_unknown_model(self, config):
        """Test cost calculation for unknown model."""
        config.model = "unknown-model"
        provider = OpenAIProvider(config)
        
        cost = provider._calculate_cost(1000, 500)
        assert cost is None


class TestAnthropicProvider:
    """Test cases for Anthropic provider."""
    
    @pytest.fixture
    def config(self):
        return LLMConfig(
            provider=LLMProviderEnum.ANTHROPIC,
            model="claude-3-sonnet-20240229",
            api_key="test-key",
            max_tokens=1000,
            temperature=0.1
        )
    
    @pytest.fixture
    def provider(self, config):
        return AnthropicProvider(config)
    
    @pytest.fixture
    def mock_anthropic_response(self):
        """Mock Anthropic API response."""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].type = "text"
        mock_response.content[0].text = "Test response"
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        return mock_response
    
    @patch('builtins.__import__')
    def test_client_lazy_initialization(self, mock_import, provider):
        """Test lazy initialization of Anthropic client."""
        mock_anthropic = Mock()
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        
        def import_side_effect(name, *args, **kwargs):
            if name == 'anthropic':
                return mock_anthropic
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_side_effect
        
        client = provider.client
        
        assert client == mock_client
        mock_anthropic.Anthropic.assert_called_once_with(
            api_key="test-key",
            timeout=30
        )
    
    @patch('builtins.__import__')
    def test_generate_response_success(self, mock_import, provider, mock_anthropic_response):
        """Test successful response generation."""
        mock_anthropic = Mock()
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.Anthropic.return_value = mock_client
        
        def import_side_effect(name, *args, **kwargs):
            if name == 'anthropic':
                return mock_anthropic
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_side_effect
        
        messages = [
            Message(role=MessageRole.SYSTEM, content="System message"),
            Message(role=MessageRole.USER, content="User message")
        ]
        
        response = provider.generate_response(messages)
        
        assert isinstance(response, LLMResponse)
        assert response.content == "Test response"
        assert response.model == "claude-3-sonnet-20240229"
        assert response.finish_reason == "end_turn"
        assert response.token_usage.prompt_tokens == 100
        assert response.token_usage.completion_tokens == 50
    
    @patch('builtins.__import__')
    def test_generate_response_with_tool_use(self, mock_import, provider):
        """Test response generation with tool use."""
        mock_response = Mock()
        
        # Mock text content
        text_content = Mock()
        text_content.type = "text"
        text_content.text = "I'll use a tool"
        
        # Mock tool use content
        tool_content = Mock()
        tool_content.type = "tool_use"
        tool_content.id = "tool_123"
        tool_content.name = "test_tool"
        tool_content.input = {"param": "value"}
        
        mock_response.content = [text_content, tool_content]
        mock_response.stop_reason = "tool_use"
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        
        mock_anthropic = Mock()
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        
        def import_side_effect(name, *args, **kwargs):
            if name == 'anthropic':
                return mock_anthropic
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_side_effect
        
        messages = [Message(role=MessageRole.USER, content="Use a tool")]
        tools = [{"name": "test_tool", "description": "Test tool"}]
        
        response = provider.generate_response(messages, tools=tools)
        
        assert "I'll use a tool" in response.content
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["id"] == "tool_123"
        assert response.tool_calls[0]["function"]["name"] == "test_tool"
    
    def test_calculate_cost(self, provider):
        """Test cost calculation for Anthropic."""
        cost = provider._calculate_cost(1000, 500)
        
        # claude-3-sonnet: $0.003/1k input, $0.015/1k output
        expected = (1000/1000 * 0.003) + (500/1000 * 0.015)
        assert cost == expected


class TestLLMProviderFactory:
    """Test cases for LLM provider factory function."""
    
    def test_get_openai_provider(self):
        """Test creating OpenAI provider."""
        config = LLMConfig(
            provider=LLMProviderEnum.OPENAI,
            model="gpt-4",
            api_key="test-key"
        )
        
        provider = get_llm_provider(config)
        
        assert isinstance(provider, OpenAIProvider)
        assert provider.config == config
    
    def test_get_anthropic_provider(self):
        """Test creating Anthropic provider."""
        config = LLMConfig(
            provider=LLMProviderEnum.ANTHROPIC,
            model="claude-3-sonnet-20240229",
            api_key="test-key"
        )
        
        provider = get_llm_provider(config)
        
        assert isinstance(provider, AnthropicProvider)
        assert provider.config == config
    
    def test_unsupported_provider(self):
        """Test handling of unsupported provider."""
        config = LLMConfig(
            provider=LLMProviderEnum.LOCAL,  # Not implemented
            model="local-model",
            api_key="test-key"
        )
        
        with pytest.raises(LLMProviderError, match="Unsupported LLM provider"):
            get_llm_provider(config)