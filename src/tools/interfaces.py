"""
Tool interfaces for logistics API interactions.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    data: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class Tool(ABC):
    """Base interface for all logistics tools."""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with provided parameters.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResult containing execution outcome and data
        """
        pass
    
    @abstractmethod
    def validate_parameters(self, **kwargs) -> bool:
        """
        Validate input parameters before execution.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        pass


class ToolManager(ABC):
    """Interface for managing and executing tools."""
    
    @abstractmethod
    def get_available_tools(self) -> List[Tool]:
        """
        Get list of all available tools.
        
        Returns:
            List of available Tool instances
        """
        pass
    
    @abstractmethod
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute a specific tool with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters to pass to the tool
            
        Returns:
            ToolResult containing execution outcome
        """
        pass
    
    @abstractmethod
    def register_tool(self, tool: Tool) -> None:
        """
        Register a new tool with the manager.
        
        Args:
            tool: Tool instance to register
        """
        pass
    
    @abstractmethod
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """
        Get a specific tool by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            Tool instance if found, None otherwise
        """
        pass