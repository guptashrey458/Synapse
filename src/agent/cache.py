"""
Caching mechanisms for performance optimization.
"""
import hashlib
import time
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class CacheEntry:
    """Represents a cached entry with metadata."""
    value: Any
    timestamp: datetime
    hit_count: int = 0
    ttl_seconds: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl_seconds)
    
    def increment_hit_count(self):
        """Increment hit count for cache statistics."""
        self.hit_count += 1


class ToolResultCache:
    """Cache for tool execution results to improve performance."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """
        Initialize tool result cache.
        
        Args:
            max_size: Maximum number of entries to cache
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # For LRU eviction
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _generate_key(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Generate cache key from tool name and parameters."""
        # Create a deterministic hash of tool name and parameters
        param_str = str(sorted(parameters.items()))
        key_data = f"{tool_name}:{param_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, tool_name: str, parameters: Dict[str, Any]) -> Optional[Any]:
        """
        Get cached result for tool execution.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            
        Returns:
            Cached result if available and not expired, None otherwise
        """
        key = self._generate_key(tool_name, parameters)
        
        if key not in self._cache:
            self.misses += 1
            return None
        
        entry = self._cache[key]
        
        # Check if expired
        if entry.is_expired():
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            self.misses += 1
            return None
        
        # Update access order for LRU
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        entry.increment_hit_count()
        self.hits += 1
        return entry.value
    
    def put(self, tool_name: str, parameters: Dict[str, Any], result: Any, ttl: Optional[int] = None) -> None:
        """
        Cache tool execution result.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            result: Tool execution result
            ttl: Time-to-live in seconds (uses default if None)
        """
        key = self._generate_key(tool_name, parameters)
        
        # Use default TTL if not specified
        if ttl is None:
            ttl = self.default_ttl
        
        # Evict oldest entry if cache is full
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_lru()
        
        # Create cache entry
        entry = CacheEntry(
            value=result,
            timestamp=datetime.now(),
            ttl_seconds=ttl
        )
        
        self._cache[key] = entry
        
        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._access_order:
            return
        
        lru_key = self._access_order.pop(0)
        if lru_key in self._cache:
            del self._cache[lru_key]
            self.evictions += 1
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
            "max_size": self.max_size
        }


class ScenarioCache:
    """Cache for complete scenario processing results."""
    
    def __init__(self, max_size: int = 100, ttl: int = 1800):  # 30 minutes TTL
        """
        Initialize scenario cache.
        
        Args:
            max_size: Maximum number of scenarios to cache
            ttl: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, scenario: str) -> str:
        """Generate cache key from scenario description."""
        # Normalize scenario text and create hash
        normalized = scenario.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get(self, scenario: str) -> Optional[Any]:
        """Get cached result for scenario."""
        key = self._generate_key(scenario)
        
        if key not in self._cache:
            self.misses += 1
            return None
        
        entry = self._cache[key]
        
        if entry.is_expired():
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            self.misses += 1
            return None
        
        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        entry.increment_hit_count()
        self.hits += 1
        return entry.value
    
    def put(self, scenario: str, result: Any) -> None:
        """Cache scenario processing result."""
        key = self._generate_key(scenario)
        
        # Evict if necessary
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_lru()
        
        entry = CacheEntry(
            value=result,
            timestamp=datetime.now(),
            ttl_seconds=self.ttl
        )
        
        self._cache[key] = entry
        
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._access_order:
            return
        
        lru_key = self._access_order.pop(0)
        if lru_key in self._cache:
            del self._cache[lru_key]
    
    def should_cache(self, scenario: str) -> bool:
        """Determine if scenario result should be cached."""
        # Don't cache scenarios with time-sensitive or unique elements
        time_sensitive_keywords = [
            "urgent", "emergency", "now", "immediate", "asap",
            "critical", "breaking", "live", "real-time"
        ]
        
        scenario_lower = scenario.lower()
        return not any(keyword in scenario_lower for keyword in time_sensitive_keywords)


class PromptCache:
    """Cache for optimized LLM prompts."""
    
    def __init__(self):
        """Initialize prompt cache with optimized templates."""
        self._optimized_prompts = {
            "traffic_analysis": """Analyze traffic disruption: {scenario}
Key info needed: route, delay estimate, alternatives.
Response format: JSON with route_issue, estimated_delay, suggested_action.""",
            
            "merchant_analysis": """Analyze merchant issue: {scenario}
Key info needed: merchant status, alternatives, customer impact.
Response format: JSON with merchant_status, alternatives, recommended_action.""",
            
            "customer_analysis": """Analyze customer issue: {scenario}
Key info needed: complaint type, resolution options, urgency.
Response format: JSON with issue_type, resolution_options, priority_level.""",
            
            "general_analysis": """Analyze delivery disruption: {scenario}
Identify: problem type, affected parties, resolution steps.
Be concise and actionable."""
        }
    
    def get_optimized_prompt(self, scenario_type: str, scenario: str) -> str:
        """Get optimized prompt for scenario type."""
        template = self._optimized_prompts.get(f"{scenario_type}_analysis", 
                                             self._optimized_prompts["general_analysis"])
        return template.format(scenario=scenario)
    
    def get_token_optimized_prompt(self, base_prompt: str) -> str:
        """Optimize prompt for minimal token usage."""
        # Remove unnecessary words and phrases
        optimizations = [
            ("please", ""),
            ("could you", ""),
            ("I would like you to", ""),
            ("can you help me", ""),
            ("it would be great if", ""),
            ("  ", " "),  # Remove double spaces
        ]
        
        optimized = base_prompt
        for old, new in optimizations:
            optimized = optimized.replace(old, new)
        
        return optimized.strip()