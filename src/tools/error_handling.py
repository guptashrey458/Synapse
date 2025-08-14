"""
Advanced error handling and resilience features for tool execution.
"""
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from .interfaces import ToolResult


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors."""
    NETWORK = "network"
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    SERVICE_UNAVAILABLE = "service_unavailable"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    tool_name: str
    parameters: Dict[str, Any]
    attempt_number: int
    total_attempts: int
    error_message: str
    error_category: ErrorCategory
    severity: ErrorSeverity
    timestamp: datetime
    execution_time: float


class ErrorHandler(ABC):
    """Abstract base class for error handlers."""
    
    @abstractmethod
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this handler can handle the given error."""
        pass
    
    @abstractmethod
    def handle(self, error_context: ErrorContext) -> Optional[ToolResult]:
        """Handle the error and optionally return a fallback result."""
        pass


class CircuitBreakerState(Enum):
    """States of a circuit breaker."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: int = 60  # Seconds before trying half-open
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout_threshold: int = 30  # Seconds to consider a call timeout


class CircuitBreaker:
    """Circuit breaker implementation for tool resilience."""
    
    def __init__(self, tool_name: str, config: CircuitBreakerConfig):
        self.tool_name = tool_name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.logger = logging.getLogger(__name__)
    
    def can_execute(self) -> bool:
        """Check if execution is allowed based on circuit breaker state."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if (self.last_failure_time and 
                datetime.now() - self.last_failure_time > 
                timedelta(seconds=self.config.recovery_timeout)):
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                self.logger.info(f"Circuit breaker for {self.tool_name} moved to HALF_OPEN")
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self) -> None:
        """Record a successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.logger.info(f"Circuit breaker for {self.tool_name} CLOSED")
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0  # Reset failure count on success
    
    def record_failure(self) -> None:
        """Record a failed execution."""
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitBreakerState.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.logger.warning(f"Circuit breaker for {self.tool_name} OPENED")
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.failure_count += 1
            self.logger.warning(f"Circuit breaker for {self.tool_name} returned to OPEN")
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information."""
        return {
            "tool_name": self.tool_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": (self.last_failure_time.isoformat() 
                                if self.last_failure_time else None),
            "can_execute": self.can_execute()
        }


class NetworkErrorHandler(ErrorHandler):
    """Handler for network-related errors."""
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this is a network error."""
        return error_context.error_category == ErrorCategory.NETWORK
    
    def handle(self, error_context: ErrorContext) -> Optional[ToolResult]:
        """Handle network errors with exponential backoff."""
        if error_context.attempt_number < error_context.total_attempts:
            # Calculate backoff delay
            delay = min(2 ** error_context.attempt_number, 30)  # Max 30 seconds
            time.sleep(delay)
            return None  # Retry
        
        # Return degraded result after all retries
        return ToolResult(
            tool_name=error_context.tool_name,
            success=False,
            data={"degraded": True, "reason": "network_unavailable"},
            execution_time=error_context.execution_time,
            error_message=f"Network error after {error_context.total_attempts} attempts"
        )


class TimeoutErrorHandler(ErrorHandler):
    """Handler for timeout errors."""
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this is a timeout error."""
        return error_context.error_category == ErrorCategory.TIMEOUT
    
    def handle(self, error_context: ErrorContext) -> Optional[ToolResult]:
        """Handle timeout errors."""
        if error_context.attempt_number < error_context.total_attempts:
            # Increase timeout for retry
            return None  # Retry with longer timeout
        
        # Return partial result if possible
        return ToolResult(
            tool_name=error_context.tool_name,
            success=False,
            data={"timeout": True, "partial_data": {}},
            execution_time=error_context.execution_time,
            error_message=f"Operation timed out after {error_context.total_attempts} attempts"
        )


class RateLimitErrorHandler(ErrorHandler):
    """Handler for rate limiting errors."""
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this is a rate limit error."""
        return error_context.error_category == ErrorCategory.RATE_LIMIT
    
    def handle(self, error_context: ErrorContext) -> Optional[ToolResult]:
        """Handle rate limit errors with progressive delays."""
        if error_context.attempt_number < error_context.total_attempts:
            # Progressive delay for rate limits
            delay = 5 * error_context.attempt_number  # 5, 10, 15 seconds
            time.sleep(delay)
            return None  # Retry
        
        return ToolResult(
            tool_name=error_context.tool_name,
            success=False,
            data={"rate_limited": True},
            execution_time=error_context.execution_time,
            error_message="Rate limit exceeded - please try again later"
        )


class ServiceUnavailableHandler(ErrorHandler):
    """Handler for service unavailable errors."""
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this is a service unavailable error."""
        return error_context.error_category == ErrorCategory.SERVICE_UNAVAILABLE
    
    def handle(self, error_context: ErrorContext) -> Optional[ToolResult]:
        """Handle service unavailable errors."""
        # Don't retry service unavailable - return degraded result immediately
        return ToolResult(
            tool_name=error_context.tool_name,
            success=False,
            data={
                "service_unavailable": True,
                "fallback_data": self._get_fallback_data(error_context.tool_name)
            },
            execution_time=error_context.execution_time,
            error_message="Service temporarily unavailable"
        )
    
    def _get_fallback_data(self, tool_name: str) -> Dict[str, Any]:
        """Get fallback data based on tool type."""
        fallback_data = {
            "check_traffic": {
                "route": {"estimated_delay_minutes": 15},
                "recommendations": ["Assume moderate traffic delays"]
            },
            "get_merchant_status": {
                "status": "unknown",
                "recommendations": ["Contact merchant directly"]
            },
            "notify_customer": {
                "notification_sent": False,
                "recommendations": ["Use alternative communication method"]
            }
        }
        return fallback_data.get(tool_name, {})


class ErrorHandlerRegistry:
    """Registry for error handlers."""
    
    def __init__(self):
        self.handlers: List[ErrorHandler] = []
        self.logger = logging.getLogger(__name__)
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default error handlers."""
        self.handlers.extend([
            NetworkErrorHandler(),
            TimeoutErrorHandler(),
            RateLimitErrorHandler(),
            ServiceUnavailableHandler()
        ])
    
    def register_handler(self, handler: ErrorHandler):
        """Register a custom error handler."""
        self.handlers.append(handler)
    
    def handle_error(self, error_context: ErrorContext) -> Optional[ToolResult]:
        """Handle error using appropriate handler."""
        for handler in self.handlers:
            if handler.can_handle(error_context):
                self.logger.debug(f"Handling error with {handler.__class__.__name__}")
                return handler.handle(error_context)
        
        # No specific handler found - return generic error
        return ToolResult(
            tool_name=error_context.tool_name,
            success=False,
            data={},
            execution_time=error_context.execution_time,
            error_message=f"Unhandled error: {error_context.error_message}"
        )


def categorize_error(error_message: str, execution_time: float, timeout: int) -> ErrorCategory:
    """Categorize error based on error message and context."""
    error_lower = error_message.lower()
    
    if "timeout" in error_lower or execution_time >= timeout:
        return ErrorCategory.TIMEOUT
    elif any(keyword in error_lower for keyword in ["network", "connection", "dns", "unreachable"]):
        return ErrorCategory.NETWORK
    elif any(keyword in error_lower for keyword in ["rate limit", "too many requests", "quota"]):
        return ErrorCategory.RATE_LIMIT
    elif any(keyword in error_lower for keyword in ["service unavailable", "503", "502", "500"]):
        return ErrorCategory.SERVICE_UNAVAILABLE
    elif any(keyword in error_lower for keyword in ["auth", "unauthorized", "forbidden", "401", "403"]):
        return ErrorCategory.AUTHENTICATION
    elif any(keyword in error_lower for keyword in ["validation", "invalid", "parameter"]):
        return ErrorCategory.VALIDATION
    else:
        return ErrorCategory.UNKNOWN


def determine_severity(error_category: ErrorCategory, attempt_number: int) -> ErrorSeverity:
    """Determine error severity based on category and attempt number."""
    if error_category == ErrorCategory.CRITICAL:
        return ErrorSeverity.CRITICAL
    elif error_category in [ErrorCategory.SERVICE_UNAVAILABLE, ErrorCategory.AUTHENTICATION]:
        return ErrorSeverity.HIGH
    elif error_category in [ErrorCategory.NETWORK, ErrorCategory.TIMEOUT] and attempt_number > 2:
        return ErrorSeverity.HIGH
    elif error_category == ErrorCategory.RATE_LIMIT:
        return ErrorSeverity.MEDIUM
    else:
        return ErrorSeverity.LOW