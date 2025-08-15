"""
Concurrent tool execution for performance optimization.
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

from ..tools.interfaces import ToolManager
from ..tools.interfaces import ToolResult


logger = logging.getLogger(__name__)


@dataclass
class ToolExecution:
    """Represents a tool execution request."""
    tool_name: str
    parameters: Dict[str, Any]
    priority: int = 0  # Higher numbers = higher priority
    timeout: Optional[float] = None


@dataclass
class ExecutionResult:
    """Result of concurrent tool execution."""
    tool_name: str
    parameters: Dict[str, Any]
    result: Optional[ToolResult]
    error: Optional[Exception]
    execution_time: float
    success: bool


class ConcurrentToolExecutor:
    """Manages concurrent execution of tools for improved performance."""
    
    def __init__(self, tool_manager: ToolManager, max_workers: int = 3):
        """
        Initialize concurrent tool executor.
        
        Args:
            tool_manager: Tool manager instance
            max_workers: Maximum number of concurrent tool executions
        """
        self.tool_manager = tool_manager
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Performance tracking
        self.total_executions = 0
        self.total_time_saved = 0.0
        self.concurrent_executions = 0
    
    def execute_single(self, tool_execution: ToolExecution) -> ExecutionResult:
        """Execute a single tool."""
        start_time = time.time()
        
        try:
            result = self.tool_manager.execute_tool(
                tool_execution.tool_name, 
                tool_execution.parameters
            )
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                tool_name=tool_execution.tool_name,
                parameters=tool_execution.parameters,
                result=result,
                error=None,
                execution_time=execution_time,
                success=True
            )
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Tool execution failed: {tool_execution.tool_name} - {e}")
            
            return ExecutionResult(
                tool_name=tool_execution.tool_name,
                parameters=tool_execution.parameters,
                result=None,
                error=e,
                execution_time=execution_time,
                success=False
            )
    
    def execute_concurrent(self, tool_executions: List[ToolExecution]) -> List[ExecutionResult]:
        """
        Execute multiple tools concurrently.
        
        Args:
            tool_executions: List of tool executions to perform
            
        Returns:
            List of execution results in the same order as input
        """
        if not tool_executions:
            return []
        
        if len(tool_executions) == 1:
            return [self.execute_single(tool_executions[0])]
        
        start_time = time.time()
        
        # Sort by priority (higher priority first)
        sorted_executions = sorted(tool_executions, key=lambda x: x.priority, reverse=True)
        
        # Submit all executions to thread pool
        future_to_execution = {}
        for execution in sorted_executions:
            future = self.executor.submit(self.execute_single, execution)
            future_to_execution[future] = execution
        
        # Collect results as they complete
        results = {}
        for future in as_completed(future_to_execution.keys()):
            execution = future_to_execution[future]
            try:
                result = future.result()
                results[id(execution)] = result
            except Exception as e:
                logger.error(f"Concurrent execution failed: {execution.tool_name} - {e}")
                results[id(execution)] = ExecutionResult(
                    tool_name=execution.tool_name,
                    parameters=execution.parameters,
                    result=None,
                    error=e,
                    execution_time=0.0,
                    success=False
                )
        
        # Return results in original order
        ordered_results = []
        for execution in tool_executions:
            ordered_results.append(results[id(execution)])
        
        # Update performance metrics
        total_time = time.time() - start_time
        sequential_time = sum(r.execution_time for r in ordered_results)
        time_saved = max(0, sequential_time - total_time)
        
        self.total_executions += len(tool_executions)
        self.total_time_saved += time_saved
        self.concurrent_executions += 1
        
        logger.info(f"Concurrent execution completed: {len(tool_executions)} tools in {total_time:.2f}s "
                   f"(saved {time_saved:.2f}s)")
        
        return ordered_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for concurrent execution."""
        avg_time_saved = (self.total_time_saved / self.concurrent_executions 
                         if self.concurrent_executions > 0 else 0.0)
        
        return {
            "total_executions": self.total_executions,
            "concurrent_batches": self.concurrent_executions,
            "total_time_saved": self.total_time_saved,
            "average_time_saved_per_batch": avg_time_saved,
            "max_workers": self.max_workers
        }
    
    def shutdown(self):
        """Shutdown the executor and clean up resources."""
        self.executor.shutdown(wait=True)


class BatchToolExecutor:
    """Specialized executor for batching similar tool calls."""
    
    def __init__(self, tool_manager: ToolManager):
        """Initialize batch tool executor."""
        self.tool_manager = tool_manager
    
    def batch_delivery_status_checks(self, delivery_ids: List[str]) -> Dict[str, ToolResult]:
        """Batch multiple delivery status checks into a single call."""
        if not delivery_ids:
            return {}
        
        try:
            # Use batch-capable tool call if available
            batch_result = self.tool_manager.execute_tool(
                "get_delivery_status",
                {"delivery_ids": delivery_ids, "batch": True}
            )
            
            if batch_result.success and isinstance(batch_result.data, dict):
                # Convert batch result to individual results
                individual_results = {}
                for delivery_id in delivery_ids:
                    if delivery_id in batch_result.data:
                        individual_results[delivery_id] = ToolResult(
                            tool_name="get_delivery_status",
                            success=True,
                            data=batch_result.data[delivery_id],
                            execution_time=batch_result.execution_time / len(delivery_ids)
                        )
                return individual_results
        except Exception as e:
            logger.warning(f"Batch delivery status check failed, falling back to individual calls: {e}")
        
        # Fallback to individual calls
        results = {}
        for delivery_id in delivery_ids:
            try:
                result = self.tool_manager.execute_tool(
                    "get_delivery_status",
                    {"delivery_id": delivery_id}
                )
                results[delivery_id] = result
            except Exception as e:
                logger.error(f"Individual delivery status check failed for {delivery_id}: {e}")
                results[delivery_id] = ToolResult(
                    tool_name="get_delivery_status",
                    success=False,
                    data={},
                    execution_time=0.0,
                    error_message=str(e)
                )
        
        return results