"""
Comprehensive error handling and resilience system for the autonomous agent.
"""
import time
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
from contextlib import contextmanager

from ..tools.error_handling import (
    ErrorCategory, ErrorSeverity, ErrorContext, CircuitBreaker, 
    CircuitBreakerConfig, CircuitBreakerState
)
from ..llm.interfaces import LLMProviderError
from ..tools.interfaces import ToolResult
from ..reasoning.interfaces import ReasoningTrace, ReasoningStep


class AgentErrorType(Enum):
    """Types of errors that can occur in the agent system."""
    LLM_ERROR = "llm_error"
    TOOL_ERROR = "tool_error"
    REASONING_ERROR = "reasoning_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    CIRCUIT_BREAKER_ERROR = "circuit_breaker_error"
    CONFIGURATION_ERROR = "configuration_error"
    SYSTEM_ERROR = "system_error"


@dataclass
class AgentErrorContext:
    """Extended error context for agent operations."""
    error_type: AgentErrorType
    component: str  # Which component failed (llm, tool, reasoning, etc.)
    operation: str  # What operation was being performed
    error_message: str
    original_exception: Optional[Exception] = None
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 3
    context_data: Dict[str, Any] = field(default_factory=dict)
    severity: ErrorSeverity = ErrorSeverity.MEDIUM


@dataclass
class RecoveryAction:
    """Represents a recovery action that can be taken."""
    action_type: str
    description: str
    execute: Callable[[], Any]
    success_probability: float
    estimated_time: timedelta


class AgentErrorHandler(ABC):
    """Abstract base class for agent error handlers."""
    
    @abstractmethod
    def can_handle(self, error_context: AgentErrorContext) -> bool:
        """Check if this handler can handle the given error."""
        pass
    
    @abstractmethod
    def handle(self, error_context: AgentErrorContext) -> Optional[Any]:
        """Handle the error and optionally return a recovery result."""
        pass
    
    @abstractmethod
    def get_recovery_actions(self, error_context: AgentErrorContext) -> List[RecoveryAction]:
        """Get available recovery actions for this error."""
        pass


class ReasoningLoopDetector:
    """Detects and prevents infinite reasoning loops."""
    
    def __init__(self, max_similar_thoughts: int = 3, similarity_threshold: float = 0.8):
        self.max_similar_thoughts = max_similar_thoughts
        self.similarity_threshold = similarity_threshold
        self.recent_thoughts: List[str] = []
        self.loop_detected = False
        self.loop_count = 0
        
    def add_thought(self, thought: str) -> bool:
        """
        Add a new thought and check for loops.
        
        Returns:
            True if a loop is detected
        """
        self.recent_thoughts.append(thought.lower().strip())
        
        # Keep only recent thoughts
        if len(self.recent_thoughts) > self.max_similar_thoughts * 2:
            self.recent_thoughts = self.recent_thoughts[-self.max_similar_thoughts * 2:]
        
        # Check for loops
        if len(self.recent_thoughts) >= self.max_similar_thoughts:
            recent = self.recent_thoughts[-self.max_similar_thoughts:]
            
            # Simple similarity check - count identical thoughts
            unique_thoughts = set(recent)
            if len(unique_thoughts) <= 1:
                self.loop_detected = True
                self.loop_count += 1
                return True
            
            # Check for semantic similarity (simplified)
            similarity_count = 0
            for i in range(len(recent) - 1):
                for j in range(i + 1, len(recent)):
                    if self._calculate_similarity(recent[i], recent[j]) > self.similarity_threshold:
                        similarity_count += 1
            
            # If most pairs are similar, consider it a loop
            total_pairs = len(recent) * (len(recent) - 1) // 2
            if total_pairs > 0 and similarity_count / total_pairs > 0.5:
                self.loop_detected = True
                self.loop_count += 1
                return True
        
        return False
    
    def _calculate_similarity(self, thought1: str, thought2: str) -> float:
        """Calculate similarity between two thoughts (simplified)."""
        words1 = set(thought1.split())
        words2 = set(thought2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def reset(self):
        """Reset the loop detector."""
        self.recent_thoughts.clear()
        self.loop_detected = False
        self.loop_count = 0


class ReasoningCircuitBreaker(CircuitBreaker):
    """Circuit breaker specifically for reasoning operations."""
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        reasoning_config = config or CircuitBreakerConfig(
            failure_threshold=3,  # Lower threshold for reasoning
            recovery_timeout=30,  # Shorter recovery time
            success_threshold=2,
            timeout_threshold=60
        )
        super().__init__("reasoning_engine", reasoning_config)
        self.loop_detector = ReasoningLoopDetector()
        
    def record_reasoning_step(self, step: ReasoningStep) -> bool:
        """
        Record a reasoning step and check for issues.
        
        Returns:
            True if reasoning should continue, False if circuit should open
        """
        # Check for loops
        if self.loop_detector.add_thought(step.thought):
            self.record_failure()
            return False
        
        # Check for other failure indicators
        if hasattr(step, 'tool_results') and step.tool_results:
            failed_tools = [r for r in step.tool_results if not r.success]
            if len(failed_tools) > len(step.tool_results) / 2:  # More than half failed
                self.record_failure()
                return False
        
        # Record success if step seems healthy
        self.record_success()
        return True
    
    def get_loop_info(self) -> Dict[str, Any]:
        """Get information about detected loops."""
        return {
            "loop_detected": self.loop_detector.loop_detected,
            "loop_count": self.loop_detector.loop_count,
            "recent_thoughts_count": len(self.loop_detector.recent_thoughts)
        }


class LLMErrorHandler(AgentErrorHandler):
    """Handler for LLM provider errors."""
    
    def can_handle(self, error_context: AgentErrorContext) -> bool:
        return error_context.error_type == AgentErrorType.LLM_ERROR
    
    def handle(self, error_context: AgentErrorContext) -> Optional[Any]:
        """Handle LLM errors with appropriate recovery strategies."""
        if error_context.retry_count < error_context.max_retries:
            # Implement exponential backoff
            delay = min(2 ** error_context.retry_count, 30)
            time.sleep(delay)
            return None  # Signal retry
        
        # After max retries, try fallback strategies
        return self._get_fallback_response(error_context)
    
    def get_recovery_actions(self, error_context: AgentErrorContext) -> List[RecoveryAction]:
        actions = []
        
        if "rate limit" in error_context.error_message.lower():
            actions.append(RecoveryAction(
                action_type="wait_and_retry",
                description="Wait for rate limit to reset and retry",
                execute=lambda: time.sleep(60),
                success_probability=0.9,
                estimated_time=timedelta(minutes=1)
            ))
        
        if "authentication" in error_context.error_message.lower():
            actions.append(RecoveryAction(
                action_type="check_credentials",
                description="Verify API credentials and configuration",
                execute=lambda: self._check_credentials(),
                success_probability=0.7,
                estimated_time=timedelta(minutes=2)
            ))
        
        actions.append(RecoveryAction(
            action_type="use_fallback",
            description="Use rule-based fallback for simple scenarios",
            execute=lambda: self._get_fallback_response(error_context),
            success_probability=0.5,
            estimated_time=timedelta(seconds=5)
        ))
        
        return actions
    
    def _get_fallback_response(self, error_context: AgentErrorContext) -> Dict[str, Any]:
        """Generate a fallback response when LLM is unavailable."""
        return {
            "fallback": True,
            "message": "LLM unavailable - using rule-based response",
            "suggested_actions": [
                "Check network connectivity",
                "Verify API credentials",
                "Try again in a few minutes"
            ]
        }
    
    def _check_credentials(self) -> bool:
        """Check if API credentials are properly configured."""
        import os
        return bool(os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY'))


class ToolErrorHandler(AgentErrorHandler):
    """Handler for tool execution errors."""
    
    def can_handle(self, error_context: AgentErrorContext) -> bool:
        return error_context.error_type == AgentErrorType.TOOL_ERROR
    
    def handle(self, error_context: AgentErrorContext) -> Optional[ToolResult]:
        """Handle tool errors with graceful degradation."""
        tool_name = error_context.context_data.get('tool_name', 'unknown')
        
        # Try alternative tools first
        alternative_result = self._try_alternative_tools(error_context)
        if alternative_result:
            return alternative_result
        
        # Return degraded result with fallback data
        return ToolResult(
            tool_name=tool_name,
            success=False,
            data=self._get_fallback_data(tool_name),
            execution_time=0.0,
            error_message=f"Tool failed, using fallback data: {error_context.error_message}"
        )
    
    def get_recovery_actions(self, error_context: AgentErrorContext) -> List[RecoveryAction]:
        tool_name = error_context.context_data.get('tool_name', 'unknown')
        
        actions = [
            RecoveryAction(
                action_type="retry_tool",
                description=f"Retry {tool_name} with same parameters",
                execute=lambda: None,  # Handled by retry logic
                success_probability=0.6,
                estimated_time=timedelta(seconds=10)
            ),
            RecoveryAction(
                action_type="use_alternative",
                description=f"Try alternative tools for {tool_name}",
                execute=lambda: self._try_alternative_tools(error_context),
                success_probability=0.4,
                estimated_time=timedelta(seconds=15)
            ),
            RecoveryAction(
                action_type="use_fallback_data",
                description="Use cached or default data",
                execute=lambda: self._get_fallback_data(tool_name),
                success_probability=0.8,
                estimated_time=timedelta(seconds=1)
            )
        ]
        
        return actions
    
    def _try_alternative_tools(self, error_context: AgentErrorContext) -> Optional[ToolResult]:
        """Try alternative tools that might provide similar functionality."""
        tool_name = error_context.context_data.get('tool_name', '')
        
        # Define tool alternatives
        alternatives = {
            'check_traffic': ['get_route_info', 'estimate_travel_time'],
            'get_merchant_status': ['get_nearby_merchants', 'check_restaurant_hours'],
            'validate_address': ['geocode_address', 'check_delivery_zone']
        }
        
        # This would need to be implemented with actual tool manager
        # For now, return None to indicate no alternative found
        return None
    
    def _get_fallback_data(self, tool_name: str) -> Dict[str, Any]:
        """Get fallback data for specific tools."""
        fallback_data = {
            'check_traffic': {
                'status': 'unknown',
                'estimated_delay': 15,
                'recommendation': 'Assume moderate traffic delays'
            },
            'get_merchant_status': {
                'status': 'unknown',
                'estimated_prep_time': 20,
                'recommendation': 'Contact merchant directly'
            },
            'validate_address': {
                'valid': True,
                'confidence': 0.5,
                'recommendation': 'Verify address with customer'
            }
        }
        
        return fallback_data.get(tool_name, {'fallback': True})


class ReasoningErrorHandler(AgentErrorHandler):
    """Handler for reasoning loop and logic errors."""
    
    def __init__(self):
        self.circuit_breaker = ReasoningCircuitBreaker()
    
    def can_handle(self, error_context: AgentErrorContext) -> bool:
        return error_context.error_type == AgentErrorType.REASONING_ERROR
    
    def handle(self, error_context: AgentErrorContext) -> Optional[Any]:
        """Handle reasoning errors by breaking loops and providing fallback logic."""
        if not self.circuit_breaker.can_execute():
            return self._create_emergency_plan(error_context)
        
        # Try to recover from reasoning issues
        if "loop" in error_context.error_message.lower():
            return self._break_reasoning_loop(error_context)
        elif "timeout" in error_context.error_message.lower():
            return self._handle_reasoning_timeout(error_context)
        else:
            return self._provide_fallback_reasoning(error_context)
    
    def get_recovery_actions(self, error_context: AgentErrorContext) -> List[RecoveryAction]:
        actions = [
            RecoveryAction(
                action_type="break_loop",
                description="Break reasoning loop and continue with available data",
                execute=lambda: self._break_reasoning_loop(error_context),
                success_probability=0.8,
                estimated_time=timedelta(seconds=5)
            ),
            RecoveryAction(
                action_type="simplify_reasoning",
                description="Use simplified reasoning approach",
                execute=lambda: self._provide_fallback_reasoning(error_context),
                success_probability=0.7,
                estimated_time=timedelta(seconds=10)
            ),
            RecoveryAction(
                action_type="emergency_plan",
                description="Create basic emergency response plan",
                execute=lambda: self._create_emergency_plan(error_context),
                success_probability=0.9,
                estimated_time=timedelta(seconds=3)
            )
        ]
        
        return actions
    
    def record_reasoning_step(self, step: ReasoningStep) -> bool:
        """Record a reasoning step and check for issues."""
        return self.circuit_breaker.record_reasoning_step(step)
    
    def _break_reasoning_loop(self, error_context: AgentErrorContext) -> Dict[str, Any]:
        """Break out of reasoning loop with current best information."""
        return {
            "loop_broken": True,
            "message": "Reasoning loop detected and broken",
            "action": "proceed_with_available_data",
            "confidence": 0.6
        }
    
    def _handle_reasoning_timeout(self, error_context: AgentErrorContext) -> Dict[str, Any]:
        """Handle reasoning timeout by providing quick resolution."""
        return {
            "timeout_handled": True,
            "message": "Reasoning timeout - providing quick resolution",
            "action": "use_standard_procedure",
            "confidence": 0.5
        }
    
    def _provide_fallback_reasoning(self, error_context: AgentErrorContext) -> Dict[str, Any]:
        """Provide simple fallback reasoning."""
        return {
            "fallback_reasoning": True,
            "message": "Using simplified reasoning approach",
            "steps": [
                "Identify the main issue",
                "Apply standard resolution procedure",
                "Notify relevant parties",
                "Monitor for resolution"
            ],
            "confidence": 0.4
        }
    
    def _create_emergency_plan(self, error_context: AgentErrorContext) -> Dict[str, Any]:
        """Create an emergency response plan when reasoning fails completely."""
        return {
            "emergency_plan": True,
            "message": "Reasoning system failure - using emergency procedures",
            "immediate_actions": [
                "Notify customer of delay",
                "Escalate to human operator",
                "Log incident for review"
            ],
            "confidence": 0.3
        }


class ComprehensiveErrorHandler:
    """Main error handler that coordinates all error handling strategies."""
    
    def __init__(self):
        self.handlers: List[AgentErrorHandler] = [
            LLMErrorHandler(),
            ToolErrorHandler(),
            ReasoningErrorHandler()
        ]
        self.error_history: List[AgentErrorContext] = []
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Global circuit breakers
        self.global_circuit_breaker = CircuitBreaker(
            "global_system", 
            CircuitBreakerConfig(failure_threshold=10, recovery_timeout=120)
        )
        
    def handle_error(self, error_context: AgentErrorContext) -> Optional[Any]:
        """Handle error using appropriate handler."""
        with self.lock:
            self.error_history.append(error_context)
            
            # Check global circuit breaker
            if not self.global_circuit_breaker.can_execute():
                self.logger.error("Global circuit breaker is OPEN - system in failure mode")
                return self._create_system_failure_response(error_context)
            
            # Find appropriate handler
            for handler in self.handlers:
                if handler.can_handle(error_context):
                    try:
                        result = handler.handle(error_context)
                        if result is not None:
                            self.global_circuit_breaker.record_success()
                            return result
                    except Exception as e:
                        self.logger.error(f"Error handler {handler.__class__.__name__} failed: {e}")
                        self.global_circuit_breaker.record_failure()
            
            # No handler could resolve the error
            self.global_circuit_breaker.record_failure()
            return self._create_unhandled_error_response(error_context)
    
    def get_recovery_suggestions(self, error_context: AgentErrorContext) -> List[RecoveryAction]:
        """Get all available recovery actions for an error."""
        suggestions = []
        
        for handler in self.handlers:
            if handler.can_handle(error_context):
                suggestions.extend(handler.get_recovery_actions(error_context))
        
        # Sort by success probability
        suggestions.sort(key=lambda x: x.success_probability, reverse=True)
        return suggestions
    
    def create_error_context(self, error_type: AgentErrorType, component: str, 
                           operation: str, error: Exception, 
                           context_data: Optional[Dict[str, Any]] = None) -> AgentErrorContext:
        """Create standardized error context."""
        return AgentErrorContext(
            error_type=error_type,
            component=component,
            operation=operation,
            error_message=str(error),
            original_exception=error,
            context_data=context_data or {},
            severity=self._determine_severity(error_type, error)
        )
    
    def _determine_severity(self, error_type: AgentErrorType, error: Exception) -> ErrorSeverity:
        """Determine error severity based on type and content."""
        if error_type == AgentErrorType.SYSTEM_ERROR:
            return ErrorSeverity.CRITICAL
        elif error_type in [AgentErrorType.LLM_ERROR, AgentErrorType.REASONING_ERROR]:
            return ErrorSeverity.HIGH
        elif error_type == AgentErrorType.TOOL_ERROR:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _create_system_failure_response(self, error_context: AgentErrorContext) -> Dict[str, Any]:
        """Create response when system is in failure mode."""
        return {
            "system_failure": True,
            "message": "System is experiencing critical failures",
            "action": "emergency_mode",
            "recommendations": [
                "Contact system administrator",
                "Check system logs",
                "Restart system components"
            ]
        }
    
    def _create_unhandled_error_response(self, error_context: AgentErrorContext) -> Dict[str, Any]:
        """Create response for unhandled errors."""
        return {
            "unhandled_error": True,
            "error_type": error_context.error_type.value,
            "message": f"Unhandled error in {error_context.component}: {error_context.error_message}",
            "recommendations": [
                "Review error logs",
                "Check system configuration",
                "Contact support if issue persists"
            ]
        }
    
    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            recent_errors = [
                error for error in self.error_history 
                if error.timestamp >= cutoff_time
            ]
            
            if not recent_errors:
                return {"no_errors": True, "period_hours": hours}
            
            # Count by type
            error_counts = {}
            for error in recent_errors:
                error_type = error.error_type.value
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            # Count by severity
            severity_counts = {}
            for error in recent_errors:
                severity = error.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            return {
                "period_hours": hours,
                "total_errors": len(recent_errors),
                "error_types": error_counts,
                "severity_distribution": severity_counts,
                "most_common_error": max(error_counts.items(), key=lambda x: x[1])[0] if error_counts else None,
                "global_circuit_breaker": self.global_circuit_breaker.get_state_info()
            }
    
    def reset_circuit_breakers(self):
        """Reset all circuit breakers."""
        self.global_circuit_breaker.state = CircuitBreakerState.CLOSED
        self.global_circuit_breaker.failure_count = 0
        
        for handler in self.handlers:
            if hasattr(handler, 'circuit_breaker'):
                handler.circuit_breaker.state = CircuitBreakerState.CLOSED
                handler.circuit_breaker.failure_count = 0
        
        self.logger.info("All circuit breakers reset")
    
    @contextmanager
    def error_handling_context(self, error_type: AgentErrorType, component: str, operation: str):
        """Context manager for automatic error handling."""
        try:
            yield
        except Exception as e:
            error_context = self.create_error_context(error_type, component, operation, e)
            result = self.handle_error(error_context)
            if result is None:
                raise  # Re-raise if no recovery possible


# Convenience functions for creating error contexts
def create_llm_error_context(operation: str, error: Exception, 
                           context_data: Optional[Dict[str, Any]] = None) -> AgentErrorContext:
    """Create error context for LLM errors."""
    return AgentErrorContext(
        error_type=AgentErrorType.LLM_ERROR,
        component="llm_provider",
        operation=operation,
        error_message=str(error),
        original_exception=error,
        context_data=context_data or {}
    )


def create_tool_error_context(tool_name: str, operation: str, error: Exception,
                            context_data: Optional[Dict[str, Any]] = None) -> AgentErrorContext:
    """Create error context for tool errors."""
    context_data = context_data or {}
    context_data['tool_name'] = tool_name
    
    return AgentErrorContext(
        error_type=AgentErrorType.TOOL_ERROR,
        component="tool_manager",
        operation=operation,
        error_message=str(error),
        original_exception=error,
        context_data=context_data
    )


def create_reasoning_error_context(operation: str, error: Exception,
                                 context_data: Optional[Dict[str, Any]] = None) -> AgentErrorContext:
    """Create error context for reasoning errors."""
    return AgentErrorContext(
        error_type=AgentErrorType.REASONING_ERROR,
        component="reasoning_engine",
        operation=operation,
        error_message=str(error),
        original_exception=error,
        context_data=context_data or {}
    )