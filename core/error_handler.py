"""Advanced error handling and monitoring for NEXUS AI"""
import asyncio
import functools
import time
import logging
from typing import Any, Callable, Optional, Dict, List, Type, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import traceback
import json
from pathlib import Path

# Configure structured logging
import structlog

logger = structlog.get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification"""
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error"
    DATA_ERROR = "data_error"
    VALIDATION_ERROR = "validation_error"
    AUTH_ERROR = "auth_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    SCRAPING_ERROR = "scraping_error"
    AI_ERROR = "ai_error"
    DATABASE_ERROR = "database_error"
    SYSTEM_ERROR = "system_error"


@dataclass
class ErrorContext:
    """Context information for error tracking"""
    timestamp: datetime = field(default_factory=datetime.now)
    function: str = ""
    module: str = ""
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    error_type: str = ""
    error_message: str = ""
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.SYSTEM_ERROR
    stack_trace: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "function": self.function,
            "module": self.module,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "severity": self.severity.value,
            "category": self.category.value,
            "metadata": self.metadata
        }


class ErrorTracker:
    """Track and manage errors across the application"""
    
    def __init__(self, max_errors: int = 1000, log_file: Optional[str] = None):
        self.max_errors = max_errors
        self.errors: List[ErrorContext] = []
        self.error_counts: Dict[str, int] = {}
        self.last_errors: Dict[str, datetime] = {}
        self.log_file = Path(log_file) if log_file else Path("logs/errors.json")
        self.log_file.parent.mkdir(exist_ok=True)
        
        # Rate limiting tracking
        self.rate_limit_hits: Dict[str, List[datetime]] = {}
        
        # Circuit breaker states
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
    
    def add_error(self, error_context: ErrorContext):
        """Add error to tracker"""
        self.errors.append(error_context)
        
        # Keep only recent errors
        if len(self.errors) > self.max_errors:
            self.errors = self.errors[-self.max_errors:]
        
        # Update error counts
        key = f"{error_context.category.value}:{error_context.error_type}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        self.last_errors[key] = error_context.timestamp
        
        # Log error
        self._log_error(error_context)
        
        # Check for patterns
        self._check_error_patterns(error_context)
    
    def _log_error(self, error_context: ErrorContext):
        """Log error to file"""
        try:
            log_entry = error_context.to_dict()
            
            # Append to log file
            if self.log_file.exists():
                with open(self.log_file, 'r') as f:
                    try:
                        logs = json.load(f)
                    except json.JSONDecodeError:
                        logs = []
            else:
                logs = []
            
            logs.append(log_entry)
            
            # Keep only recent logs
            if len(logs) > self.max_errors:
                logs = logs[-self.max_errors:]
            
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to log error: {e}")
    
    def _check_error_patterns(self, error_context: ErrorContext):
        """Check for error patterns and take action"""
        # Check rate limiting
        if error_context.category == ErrorCategory.RATE_LIMIT_ERROR:
            service = error_context.metadata.get("service", "unknown")
            if service not in self.rate_limit_hits:
                self.rate_limit_hits[service] = []
            
            self.rate_limit_hits[service].append(error_context.timestamp)
            
            # Clean old entries
            cutoff = datetime.now() - timedelta(minutes=60)
            self.rate_limit_hits[service] = [
                t for t in self.rate_limit_hits[service] if t > cutoff
            ]
            
            # Check if rate limited too often
            if len(self.rate_limit_hits[service]) > 5:
                logger.warning(f"Service {service} is being rate limited frequently")
        
        # Check circuit breaker
        key = f"{error_context.category.value}:{error_context.error_type}"
        if self.error_counts.get(key, 0) > 10:
            self._trigger_circuit_breaker(key, error_context)
    
    def _trigger_circuit_breaker(self, key: str, error_context: ErrorContext):
        """Trigger circuit breaker for failing service"""
        if key not in self.circuit_breakers:
            self.circuit_breakers[key] = {
                "state": "open",
                "opened_at": error_context.timestamp,
                "error_count": 0,
                "last_error": error_context
            }
            logger.critical(f"Circuit breaker opened for {key}")
    
    def is_circuit_breaker_open(self, key: str) -> bool:
        """Check if circuit breaker is open for given key"""
        if key not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[key]
        if breaker["state"] == "open":
            # Check if enough time has passed to try again
            opened_at = breaker["opened_at"]
            if datetime.now() - opened_at > timedelta(minutes=15):
                breaker["state"] = "half_open"
                return False
            return True
        
        return False
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of current errors"""
        return {
            "total_errors": len(self.errors),
            "error_counts": self.error_counts,
            "circuit_breakers": self.circuit_breakers,
            "rate_limit_hits": {
                service: len(hits) 
                for service, hits in self.rate_limit_hits.items()
            }
        }


# Global error tracker
error_tracker = ErrorTracker()


def error_handler(
    category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    circuit_breaker_key: Optional[str] = None
):
    """
    Decorator for comprehensive error handling
    
    Args:
        category: Error category for classification
        severity: Error severity level
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds
        circuit_breaker_key: Key for circuit breaker functionality
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await _handle_error(
                func, args, kwargs, category, severity, 
                max_retries, retry_delay, circuit_breaker_key, is_async=True
            )
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return _handle_error(
                func, args, kwargs, category, severity,
                max_retries, retry_delay, circuit_breaker_key, is_async=False
            )
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


async def _handle_error(
    func: Callable,
    args: tuple,
    kwargs: Dict[str, Any],
    category: ErrorCategory,
    severity: ErrorSeverity,
    max_retries: int,
    retry_delay: float,
    circuit_breaker_key: Optional[str],
    is_async: bool
):
    """Internal error handling logic"""
    last_exception = None
    
    # Check circuit breaker
    if circuit_breaker_key and error_tracker.is_circuit_breaker_open(circuit_breaker_key):
        logger.warning(f"Circuit breaker open for {circuit_breaker_key}, skipping execution")
        return None
    
    for attempt in range(max_retries + 1):
        try:
            if is_async:
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Reset error count on success
            if circuit_breaker_key and attempt > 0:
                error_tracker.circuit_breakers.pop(circuit_breaker_key, None)
            
            return result
            
        except Exception as e:
            last_exception = e
            
            # Create error context
            error_context = ErrorContext(
                function=func.__name__,
                module=func.__module__,
                args=args,
                kwargs=kwargs,
                error_type=type(e).__name__,
                error_message=str(e),
                severity=severity,
                category=category,
                stack_trace=traceback.format_exc(),
                metadata={
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "circuit_breaker_key": circuit_breaker_key
                }
            )
            
            # Add to error tracker
            error_tracker.add_error(error_context)
            
            # Log error
            if attempt < max_retries:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}"
                )
                await asyncio.sleep(retry_delay * (attempt + 1))
            else:
                logger.error(
                    f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                )
    
    # All retries failed
    return None


class ErrorRecoveryStrategy:
    """Strategies for error recovery"""
    
    @staticmethod
    def fallback_to_cache(service_name: str):
        """Fallback to cached data"""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Falling back to cache for {service_name}: {e}")
                    # Implementation would check cache here
                    return None
            return wrapper
        return decorator
    
    @staticmethod
    def degrade_gracefully(default_value: Any = None):
        """Graceful degradation"""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Graceful degradation for {func.__name__}: {e}")
                    return default_value
            return wrapper
        return decorator


# Utility functions for common error scenarios
def handle_rate_limit_error(service: str, retry_after: Optional[int] = None):
    """Handle rate limit errors specifically"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    logger.warning(f"Rate limit hit for {service}")
                    # Add rate limit tracking
                    error_context = ErrorContext(
                        function=func.__name__,
                        module=func.__module__,
                        error_type="RateLimitError",
                        error_message=str(e),
                        severity=ErrorSeverity.MEDIUM,
                        category=ErrorCategory.RATE_LIMIT_ERROR,
                        metadata={"service": service, "retry_after": retry_after}
                    )
                    error_tracker.add_error(error_context)
                    
                    if retry_after:
                        logger.info(f"Waiting {retry_after} seconds before retry")
                        await asyncio.sleep(retry_after)
                
                raise
        return wrapper
    return decorator


def handle_auth_error(service: str):
    """Handle authentication errors"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if any(auth_error in str(e).lower() for auth_error in ["unauthorized", "forbidden", "401", "403"]):
                    logger.error(f"Authentication failed for {service}: {e}")
                    error_context = ErrorContext(
                        function=func.__name__,
                        module=func.__module__,
                        error_type="AuthError",
                        error_message=str(e),
                        severity=ErrorSeverity.HIGH,
                        category=ErrorCategory.AUTH_ERROR,
                        metadata={"service": service}
                    )
                    error_tracker.add_error(error_context)
                
                raise
        return wrapper
    return decorator
