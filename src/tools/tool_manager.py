"""
Tool management system for autonomous delivery coordination.
"""
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import threading

from .interfaces import Tool, ToolResult, ToolManager
from .error_handling import (
    ErrorHandlerRegistry, ErrorContext, ErrorCategory, ErrorSeverity,
    CircuitBreaker, CircuitBreakerConfig, categorize_error, determine_severity
)


class ToolExecutionStatus(Enum):
    """Status of tool execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CACHED = "cached"


@dataclass
class ToolExecution:
    """Represents a tool execution instance."""
    execution_id: str
    tool_name: str
    parameters: Dict[str, Any]
    status: ToolExecutionStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Optional[ToolResult] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    cached: bool = False


@dataclass
class ToolMetadata:
    """Metadata about a registered tool."""
    tool: Tool
    registration_time: datetime
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    average_execution_time: float = 0.0
    last_execution_time: Optional[datetime] = None
    enabled: bool = True


@dataclass
class CacheEntry:
    """Cache entry for tool results."""
    result: ToolResult
    timestamp: datetime
    parameters_hash: str
    ttl_seconds: int = 300  # 5 minutes default
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl_seconds)


class ConcreteToolManager(ToolManager):
    """Concrete implementation of ToolManager with advanced features."""
    
    def __init__(self, max_workers: int = 4, default_timeout: int = 30, 
                 enable_caching: bool = True, cache_ttl: int = 300,
                 enable_circuit_breaker: bool = True):
        """
        Initialize the tool manager.
        
        Args:
            max_workers: Maximum number of concurrent tool executions
            default_timeout: Default timeout for tool execution in seconds
            enable_caching: Whether to enable result caching
            cache_ttl: Default cache time-to-live in seconds
            enable_circuit_breaker: Whether to enable circuit breaker pattern
        """
        self._tools: Dict[str, ToolMetadata] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._default_timeout = default_timeout
        self._enable_caching = enable_caching
        self._cache_ttl = cache_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._execution_history: List[ToolExecution] = []
        self._lock = threading.RLock()
        
        # Configuration for retry logic
        self._max_retries = 3
        self._retry_delay_base = 1.0  # Base delay in seconds
        self._retry_backoff_multiplier = 2.0
        
        # Advanced error handling
        self._error_handler_registry = ErrorHandlerRegistry()
        self._enable_circuit_breaker = enable_circuit_breaker
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Setup logging
        self._logger = logging.getLogger(__name__)
        
    def get_available_tools(self) -> List[Tool]:
        """Get list of all available and enabled tools."""
        with self._lock:
            return [metadata.tool for metadata in self._tools.values() 
                   if metadata.enabled]
    
    def register_tool(self, tool: Tool) -> None:
        """
        Register a new tool with the manager.
        
        Args:
            tool: Tool instance to register
            
        Raises:
            ValueError: If tool with same name already exists
        """
        with self._lock:
            if tool.name in self._tools:
                raise ValueError(f"Tool '{tool.name}' is already registered")
            
            self._tools[tool.name] = ToolMetadata(
                tool=tool,
                registration_time=datetime.now()
            )
            
            # Initialize circuit breaker if enabled
            if self._enable_circuit_breaker:
                self._circuit_breakers[tool.name] = CircuitBreaker(
                    tool.name, CircuitBreakerConfig()
                )
            
            self._logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """
        Get a specific tool by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            Tool instance if found and enabled, None otherwise
        """
        with self._lock:
            metadata = self._tools.get(tool_name)
            return metadata.tool if metadata and metadata.enabled else None
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any], 
                    timeout: Optional[int] = None, enable_retry: bool = True,
                    cache_result: bool = True) -> ToolResult:
        """
        Execute a specific tool with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters to pass to the tool
            timeout: Execution timeout in seconds (uses default if None)
            enable_retry: Whether to enable retry logic on failure
            cache_result: Whether to cache the result
            
        Returns:
            ToolResult containing execution outcome
        """
        execution_id = f"exec_{int(time.time() * 1000)}_{tool_name}"
        execution_timeout = timeout or self._default_timeout
        
        # Create execution record
        execution = ToolExecution(
            execution_id=execution_id,
            tool_name=tool_name,
            parameters=parameters.copy(),
            status=ToolExecutionStatus.PENDING,
            start_time=datetime.now()
        )
        
        with self._lock:
            self._execution_history.append(execution)
        
        try:
            # Check if tool exists and is enabled
            tool_metadata = self._tools.get(tool_name)
            if not tool_metadata:
                return self._create_error_result(
                    tool_name, f"Tool '{tool_name}' not found", execution
                )
            
            if not tool_metadata.enabled:
                return self._create_error_result(
                    tool_name, f"Tool '{tool_name}' is disabled", execution
                )
            
            # Check circuit breaker
            if self._enable_circuit_breaker:
                circuit_breaker = self._circuit_breakers.get(tool_name)
                if circuit_breaker and not circuit_breaker.can_execute():
                    return self._create_error_result(
                        tool_name, "Circuit breaker is OPEN - service unavailable", execution
                    )
            
            # Check cache first
            if self._enable_caching and cache_result:
                cached_result = self._get_cached_result(tool_name, parameters)
                if cached_result:
                    execution.status = ToolExecutionStatus.CACHED
                    execution.result = cached_result
                    execution.end_time = datetime.now()
                    execution.cached = True
                    return cached_result
            
            # Execute tool with advanced error handling
            if enable_retry:
                result = self._execute_with_advanced_retry(
                    tool_metadata.tool, parameters, execution_timeout, execution
                )
            else:
                result = self._execute_single(
                    tool_metadata.tool, parameters, execution_timeout, execution
                )
            
            # Update circuit breaker
            if self._enable_circuit_breaker:
                circuit_breaker = self._circuit_breakers.get(tool_name)
                if circuit_breaker:
                    if result.success:
                        circuit_breaker.record_success()
                    else:
                        circuit_breaker.record_failure()
            
            # Update metadata
            self._update_tool_metadata(tool_metadata, result)
            
            # Cache result if successful and caching is enabled
            if (self._enable_caching and cache_result and result.success):
                self._cache_result(tool_name, parameters, result)
            
            execution.result = result
            execution.end_time = datetime.now()
            
            return result
            
        except Exception as e:
            self._logger.error(f"Unexpected error executing tool {tool_name}: {str(e)}")
            return self._create_error_result(
                tool_name, f"Unexpected error: {str(e)}", execution
            )
    
    def _execute_with_retry(self, tool: Tool, parameters: Dict[str, Any], 
                           timeout: int, execution: ToolExecution) -> ToolResult:
        """Execute tool with retry logic."""
        last_result = None
        
        for attempt in range(self._max_retries + 1):
            execution.retry_count = attempt
            
            try:
                result = self._execute_single(tool, parameters, timeout, execution)
                
                # If successful, return immediately
                if result.success:
                    return result
                
                last_result = result
                
                # If this was the last attempt, return the failed result
                if attempt == self._max_retries:
                    break
                
                # Calculate retry delay with exponential backoff
                delay = self._retry_delay_base * (self._retry_backoff_multiplier ** attempt)
                self._logger.warning(
                    f"Tool {tool.name} failed (attempt {attempt + 1}), "
                    f"retrying in {delay:.1f}s: {result.error_message}"
                )
                time.sleep(delay)
                
            except Exception as e:
                last_result = self._create_error_result(
                    tool.name, f"Retry attempt {attempt + 1} failed: {str(e)}", execution
                )
                
                if attempt == self._max_retries:
                    break
                
                delay = self._retry_delay_base * (self._retry_backoff_multiplier ** attempt)
                time.sleep(delay)
        
        return last_result or self._create_error_result(
            tool.name, "All retry attempts failed", execution
        )
    
    def _execute_with_advanced_retry(self, tool: Tool, parameters: Dict[str, Any], 
                                   timeout: int, execution: ToolExecution) -> ToolResult:
        """Execute tool with advanced error handling and retry logic."""
        last_result = None
        
        for attempt in range(self._max_retries + 1):
            execution.retry_count = attempt
            
            try:
                result = self._execute_single(tool, parameters, timeout, execution)
                
                # If successful, return immediately
                if result.success:
                    return result
                
                last_result = result
                
                # If this was the last attempt, try error handling
                if attempt == self._max_retries:
                    return self._handle_final_error(tool, parameters, result, execution)
                
                # Create error context for advanced handling
                error_context = ErrorContext(
                    tool_name=tool.name,
                    parameters=parameters,
                    attempt_number=attempt + 1,
                    total_attempts=self._max_retries + 1,
                    error_message=result.error_message or "Unknown error",
                    error_category=categorize_error(
                        result.error_message or "", result.execution_time, timeout
                    ),
                    severity=determine_severity(
                        categorize_error(result.error_message or "", result.execution_time, timeout),
                        attempt + 1
                    ),
                    timestamp=datetime.now(),
                    execution_time=result.execution_time
                )
                
                # Try error handler
                handled_result = self._error_handler_registry.handle_error(error_context)
                if handled_result and handled_result.success:
                    return handled_result
                elif handled_result:
                    # Error handler returned a result but it's not successful
                    # This might be a degraded result we should return
                    if attempt == self._max_retries:
                        return handled_result
                
                # Calculate retry delay based on error type
                delay = self._calculate_retry_delay(error_context, attempt)
                if delay > 0:
                    self._logger.warning(
                        f"Tool {tool.name} failed (attempt {attempt + 1}), "
                        f"retrying in {delay:.1f}s: {result.error_message}"
                    )
                    time.sleep(delay)
                
            except Exception as e:
                last_result = self._create_error_result(
                    tool.name, f"Retry attempt {attempt + 1} failed: {str(e)}", execution
                )
                
                if attempt == self._max_retries:
                    break
                
                delay = self._retry_delay_base * (self._retry_backoff_multiplier ** attempt)
                time.sleep(delay)
        
        return last_result or self._create_error_result(
            tool.name, "All retry attempts failed", execution
        )
    
    def _calculate_retry_delay(self, error_context: ErrorContext, attempt: int) -> float:
        """Calculate retry delay based on error type and attempt number."""
        base_delay = self._retry_delay_base * (self._retry_backoff_multiplier ** attempt)
        
        # Adjust delay based on error category
        if error_context.error_category == ErrorCategory.RATE_LIMIT:
            return base_delay * 2  # Longer delay for rate limits
        elif error_context.error_category == ErrorCategory.NETWORK:
            return base_delay * 1.5  # Slightly longer for network issues
        elif error_context.error_category == ErrorCategory.SERVICE_UNAVAILABLE:
            return 0  # Don't retry service unavailable
        else:
            return base_delay
    
    def _handle_final_error(self, tool: Tool, parameters: Dict[str, Any], 
                          result: ToolResult, execution: ToolExecution) -> ToolResult:
        """Handle final error after all retries exhausted."""
        error_context = ErrorContext(
            tool_name=tool.name,
            parameters=parameters,
            attempt_number=self._max_retries + 1,
            total_attempts=self._max_retries + 1,
            error_message=result.error_message or "Unknown error",
            error_category=categorize_error(
                result.error_message or "", result.execution_time, self._default_timeout
            ),
            severity=ErrorSeverity.HIGH,  # Final error is always high severity
            timestamp=datetime.now(),
            execution_time=result.execution_time
        )
        
        # Try to get a degraded result from error handler
        handled_result = self._error_handler_registry.handle_error(error_context)
        return handled_result or result
    
    def _execute_single(self, tool: Tool, parameters: Dict[str, Any], 
                       timeout: int, execution: ToolExecution) -> ToolResult:
        """Execute tool once with timeout handling."""
        execution.status = ToolExecutionStatus.RUNNING
        
        try:
            # Submit tool execution to thread pool
            future = self._executor.submit(self._safe_tool_execution, tool, parameters)
            
            # Wait for completion with timeout
            result = future.result(timeout=timeout)
            execution.status = ToolExecutionStatus.COMPLETED
            return result
            
        except FutureTimeoutError:
            execution.status = ToolExecutionStatus.TIMEOUT
            return self._create_error_result(
                tool.name, f"Tool execution timed out after {timeout} seconds", execution
            )
        except Exception as e:
            execution.status = ToolExecutionStatus.FAILED
            return self._create_error_result(
                tool.name, f"Tool execution failed: {str(e)}", execution
            )
    
    def _safe_tool_execution(self, tool: Tool, parameters: Dict[str, Any]) -> ToolResult:
        """Safely execute tool with parameter validation."""
        try:
            # Validate parameters
            if not tool.validate_parameters(**parameters):
                return ToolResult(
                    tool_name=tool.name,
                    success=False,
                    data={},
                    execution_time=0.0,
                    error_message="Parameter validation failed"
                )
            
            # Execute tool
            return tool.execute(**parameters)
            
        except Exception as e:
            return ToolResult(
                tool_name=tool.name,
                success=False,
                data={},
                execution_time=0.0,
                error_message=f"Tool execution error: {str(e)}"
            )
    
    def _create_error_result(self, tool_name: str, error_message: str, 
                           execution: ToolExecution) -> ToolResult:
        """Create a standardized error result."""
        execution.status = ToolExecutionStatus.FAILED
        execution.error_message = error_message
        execution.end_time = datetime.now()
        
        return ToolResult(
            tool_name=tool_name,
            success=False,
            data={},
            execution_time=0.0,
            error_message=error_message
        )
    
    def _update_tool_metadata(self, metadata: ToolMetadata, result: ToolResult) -> None:
        """Update tool metadata with execution results."""
        with self._lock:
            metadata.execution_count += 1
            metadata.last_execution_time = datetime.now()
            
            if result.success:
                metadata.success_count += 1
            else:
                metadata.failure_count += 1
            
            # Update average execution time
            if metadata.execution_count == 1:
                metadata.average_execution_time = result.execution_time
            else:
                # Running average calculation
                metadata.average_execution_time = (
                    (metadata.average_execution_time * (metadata.execution_count - 1) + 
                     result.execution_time) / metadata.execution_count
                )
    
    def _get_cache_key(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Generate cache key for tool execution."""
        # Create a deterministic hash of parameters
        param_str = str(sorted(parameters.items()))
        return f"{tool_name}:{hash(param_str)}"
    
    def _get_cached_result(self, tool_name: str, parameters: Dict[str, Any]) -> Optional[ToolResult]:
        """Get cached result if available and not expired."""
        cache_key = self._get_cache_key(tool_name, parameters)
        
        with self._lock:
            entry = self._cache.get(cache_key)
            if entry and not entry.is_expired():
                self._logger.debug(f"Cache hit for {tool_name}")
                return entry.result
            elif entry:
                # Remove expired entry
                del self._cache[cache_key]
                self._logger.debug(f"Cache expired for {tool_name}")
        
        return None
    
    def _cache_result(self, tool_name: str, parameters: Dict[str, Any], result: ToolResult) -> None:
        """Cache tool result."""
        cache_key = self._get_cache_key(tool_name, parameters)
        
        with self._lock:
            self._cache[cache_key] = CacheEntry(
                result=result,
                timestamp=datetime.now(),
                parameters_hash=cache_key,
                ttl_seconds=self._cache_ttl
            )
            self._logger.debug(f"Cached result for {tool_name}")
    
    def get_tool_metadata(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific tool."""
        with self._lock:
            metadata = self._tools.get(tool_name)
            if not metadata:
                return None
            
            return {
                "name": tool_name,
                "description": metadata.tool.description,
                "parameters": metadata.tool.parameters,
                "registration_time": metadata.registration_time.isoformat(),
                "execution_count": metadata.execution_count,
                "success_count": metadata.success_count,
                "failure_count": metadata.failure_count,
                "success_rate": (metadata.success_count / metadata.execution_count 
                               if metadata.execution_count > 0 else 0.0),
                "average_execution_time": metadata.average_execution_time,
                "last_execution_time": (metadata.last_execution_time.isoformat() 
                                      if metadata.last_execution_time else None),
                "enabled": metadata.enabled
            }
    
    def get_all_tool_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all registered tools."""
        return {name: self.get_tool_metadata(name) 
                for name in self._tools.keys()}
    
    def enable_tool(self, tool_name: str) -> bool:
        """Enable a tool."""
        with self._lock:
            metadata = self._tools.get(tool_name)
            if metadata:
                metadata.enabled = True
                self._logger.info(f"Enabled tool: {tool_name}")
                return True
            return False
    
    def disable_tool(self, tool_name: str) -> bool:
        """Disable a tool."""
        with self._lock:
            metadata = self._tools.get(tool_name)
            if metadata:
                metadata.enabled = False
                self._logger.info(f"Disabled tool: {tool_name}")
                return True
            return False
    
    def clear_cache(self, tool_name: Optional[str] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            tool_name: If specified, only clear cache for this tool
            
        Returns:
            Number of cache entries cleared
        """
        with self._lock:
            if tool_name:
                # Clear cache for specific tool
                keys_to_remove = [key for key in self._cache.keys() 
                                if key.startswith(f"{tool_name}:")]
                for key in keys_to_remove:
                    del self._cache[key]
                return len(keys_to_remove)
            else:
                # Clear all cache
                count = len(self._cache)
                self._cache.clear()
                return count
    
    def cleanup_expired_cache(self) -> int:
        """Remove expired cache entries."""
        with self._lock:
            expired_keys = [key for key, entry in self._cache.items() 
                          if entry.is_expired()]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)
    
    def get_execution_history(self, tool_name: Optional[str] = None, 
                            limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get execution history.
        
        Args:
            tool_name: If specified, only return history for this tool
            limit: Maximum number of entries to return
            
        Returns:
            List of execution history entries
        """
        with self._lock:
            history = self._execution_history
            
            if tool_name:
                history = [exec for exec in history if exec.tool_name == tool_name]
            
            if limit:
                history = history[-limit:]
            
            return [
                {
                    "execution_id": exec.execution_id,
                    "tool_name": exec.tool_name,
                    "parameters": exec.parameters,
                    "status": exec.status.value,
                    "start_time": exec.start_time.isoformat(),
                    "end_time": exec.end_time.isoformat() if exec.end_time else None,
                    "execution_time": (exec.result.execution_time 
                                     if exec.result else None),
                    "success": exec.result.success if exec.result else False,
                    "error_message": exec.error_message,
                    "retry_count": exec.retry_count,
                    "cached": exec.cached
                }
                for exec in history
            ]
    
    def get_circuit_breaker_status(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get circuit breaker status for tools.
        
        Args:
            tool_name: If specified, get status for specific tool
            
        Returns:
            Circuit breaker status information
        """
        if not self._enable_circuit_breaker:
            return {"circuit_breaker_enabled": False}
        
        with self._lock:
            if tool_name:
                breaker = self._circuit_breakers.get(tool_name)
                return breaker.get_state_info() if breaker else {}
            else:
                return {
                    "circuit_breaker_enabled": True,
                    "tools": {name: breaker.get_state_info() 
                             for name, breaker in self._circuit_breakers.items()}
                }
    
    def reset_circuit_breaker(self, tool_name: str) -> bool:
        """
        Reset circuit breaker for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if reset successful, False otherwise
        """
        if not self._enable_circuit_breaker:
            return False
        
        with self._lock:
            breaker = self._circuit_breakers.get(tool_name)
            if breaker:
                breaker.state = breaker.state.CLOSED
                breaker.failure_count = 0
                breaker.success_count = 0
                breaker.last_failure_time = None
                self._logger.info(f"Reset circuit breaker for {tool_name}")
                return True
            return False
    
    def get_error_statistics(self, tool_name: Optional[str] = None, 
                           hours: int = 24) -> Dict[str, Any]:
        """
        Get error statistics for tools.
        
        Args:
            tool_name: If specified, get stats for specific tool
            hours: Number of hours to look back
            
        Returns:
            Error statistics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            relevant_executions = [
                exec for exec in self._execution_history
                if exec.start_time >= cutoff_time and
                (not tool_name or exec.tool_name == tool_name)
            ]
            
            if not relevant_executions:
                return {"no_data": True, "period_hours": hours}
            
            total_executions = len(relevant_executions)
            failed_executions = [exec for exec in relevant_executions 
                               if exec.result and not exec.result.success]
            
            error_categories = {}
            for exec in failed_executions:
                if exec.result and exec.result.error_message:
                    category = categorize_error(
                        exec.result.error_message, 
                        exec.result.execution_time, 
                        self._default_timeout
                    )
                    error_categories[category.value] = error_categories.get(category.value, 0) + 1
            
            return {
                "period_hours": hours,
                "total_executions": total_executions,
                "failed_executions": len(failed_executions),
                "success_rate": (total_executions - len(failed_executions)) / total_executions,
                "error_categories": error_categories,
                "average_execution_time": sum(
                    exec.result.execution_time for exec in relevant_executions 
                    if exec.result
                ) / total_executions if total_executions > 0 else 0
            }
    
    def configure_error_handling(self, max_retries: Optional[int] = None,
                               retry_delay_base: Optional[float] = None,
                               retry_backoff_multiplier: Optional[float] = None) -> None:
        """
        Configure error handling parameters.
        
        Args:
            max_retries: Maximum number of retries
            retry_delay_base: Base delay for retries in seconds
            retry_backoff_multiplier: Multiplier for exponential backoff
        """
        if max_retries is not None:
            self._max_retries = max_retries
        if retry_delay_base is not None:
            self._retry_delay_base = retry_delay_base
        if retry_backoff_multiplier is not None:
            self._retry_backoff_multiplier = retry_backoff_multiplier
        
        self._logger.info(f"Updated error handling config: retries={self._max_retries}, "
                         f"base_delay={self._retry_delay_base}, "
                         f"backoff={self._retry_backoff_multiplier}")
    
    def get_performance_metrics(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics for tools.
        
        Args:
            tool_name: If specified, get metrics for specific tool
            
        Returns:
            Performance metrics
        """
        with self._lock:
            if tool_name:
                metadata = self._tools.get(tool_name)
                if not metadata:
                    return {"error": "Tool not found"}
                
                return {
                    "tool_name": tool_name,
                    "execution_count": metadata.execution_count,
                    "success_count": metadata.success_count,
                    "failure_count": metadata.failure_count,
                    "success_rate": (metadata.success_count / metadata.execution_count 
                                   if metadata.execution_count > 0 else 0),
                    "average_execution_time": metadata.average_execution_time,
                    "last_execution": (metadata.last_execution_time.isoformat() 
                                     if metadata.last_execution_time else None)
                }
            else:
                return {
                    "tools": {name: self.get_performance_metrics(name) 
                             for name in self._tools.keys()},
                    "cache_stats": {
                        "total_entries": len(self._cache),
                        "cache_enabled": self._enable_caching
                    },
                    "circuit_breaker_enabled": self._enable_circuit_breaker
                }
    
    def shutdown(self) -> None:
        """Shutdown the tool manager and cleanup resources."""
        self._logger.info("Shutting down tool manager")
        self._executor.shutdown(wait=True)
        with self._lock:
            self._cache.clear()
            self._execution_history.clear()
            self._circuit_breakers.clear()