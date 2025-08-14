"""
Configuration management for LLM providers and tool settings.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Any
import os
import json
from pathlib import Path


class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


@dataclass
class LLMConfig:
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.1
    timeout: int = 30
    max_retries: int = 3


@dataclass
class ToolConfig:
    enabled: bool = True
    timeout: int = 10
    max_retries: int = 2
    cache_results: bool = True
    cache_ttl: int = 300  # seconds


@dataclass
class ReasoningConfig:
    max_steps: int = 20
    confidence_threshold: float = 0.8
    enable_chain_of_thought: bool = True
    enable_debugging: bool = False


@dataclass
class CLIConfig:
    verbose: bool = False
    output_format: str = "structured"  # structured, json, plain
    show_reasoning: bool = True
    show_timing: bool = False


@dataclass
class Config:
    llm: LLMConfig
    tools: Dict[str, ToolConfig] = field(default_factory=dict)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    cli: CLIConfig = field(default_factory=CLIConfig)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Config instance loaded from file
        """
        if not os.path.exists(config_path):
            return cls.default()
        
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """
        Create Config from dictionary.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            Config instance
        """
        llm_data = data.get('llm', {})
        llm_config = LLMConfig(
            provider=LLMProvider(llm_data.get('provider', 'openai')),
            model=llm_data.get('model', 'gpt-4'),
            api_key=llm_data.get('api_key') or os.getenv('OPENAI_API_KEY'),
            base_url=llm_data.get('base_url'),
            max_tokens=llm_data.get('max_tokens', 4000),
            temperature=llm_data.get('temperature', 0.1),
            timeout=llm_data.get('timeout', 30),
            max_retries=llm_data.get('max_retries', 3)
        )
        
        tools_data = data.get('tools', {})
        tools_config = {}
        for tool_name, tool_data in tools_data.items():
            tools_config[tool_name] = ToolConfig(
                enabled=tool_data.get('enabled', True),
                timeout=tool_data.get('timeout', 10),
                max_retries=tool_data.get('max_retries', 2),
                cache_results=tool_data.get('cache_results', True),
                cache_ttl=tool_data.get('cache_ttl', 300)
            )
        
        reasoning_data = data.get('reasoning', {})
        reasoning_config = ReasoningConfig(
            max_steps=reasoning_data.get('max_steps', 20),
            confidence_threshold=reasoning_data.get('confidence_threshold', 0.8),
            enable_chain_of_thought=reasoning_data.get('enable_chain_of_thought', True),
            enable_debugging=reasoning_data.get('enable_debugging', False)
        )
        
        cli_data = data.get('cli', {})
        cli_config = CLIConfig(
            verbose=cli_data.get('verbose', False),
            output_format=cli_data.get('output_format', 'structured'),
            show_reasoning=cli_data.get('show_reasoning', True),
            show_timing=cli_data.get('show_timing', False)
        )
        
        return cls(
            llm=llm_config,
            tools=tools_config,
            reasoning=reasoning_config,
            cli=cli_config
        )
    
    @classmethod
    def default(cls) -> 'Config':
        """
        Create default configuration.
        
        Returns:
            Config instance with default values
        """
        return cls(
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4",
                api_key=os.getenv('OPENAI_API_KEY')
            )
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            'llm': {
                'provider': self.llm.provider.value,
                'model': self.llm.model,
                'api_key': self.llm.api_key,
                'base_url': self.llm.base_url,
                'max_tokens': self.llm.max_tokens,
                'temperature': self.llm.temperature,
                'timeout': self.llm.timeout,
                'max_retries': self.llm.max_retries
            },
            'tools': {
                name: {
                    'enabled': config.enabled,
                    'timeout': config.timeout,
                    'max_retries': config.max_retries,
                    'cache_results': config.cache_results,
                    'cache_ttl': config.cache_ttl
                }
                for name, config in self.tools.items()
            },
            'reasoning': {
                'max_steps': self.reasoning.max_steps,
                'confidence_threshold': self.reasoning.confidence_threshold,
                'enable_chain_of_thought': self.reasoning.enable_chain_of_thought,
                'enable_debugging': self.reasoning.enable_debugging
            },
            'cli': {
                'verbose': self.cli.verbose,
                'output_format': self.cli.output_format,
                'show_reasoning': self.cli.show_reasoning,
                'show_timing': self.cli.show_timing
            }
        }
    
    def save_to_file(self, config_path: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            config_path: Path where to save configuration
        """
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or environment.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Config instance
    """
    if config_path is None:
        # Try default locations
        default_paths = [
            'config.json',
            'config/config.json',
            os.path.expanduser('~/.autonomous-delivery-coordinator/config.json')
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path and os.path.exists(config_path):
        return Config.from_file(config_path)
    else:
        return Config.default()