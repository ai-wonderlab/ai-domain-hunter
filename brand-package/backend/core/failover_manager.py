"""
Failover Manager for Image Generation APIs
"""
import asyncio
from typing import Optional, List, Dict, Any
from enum import Enum
import logging
from datetime import datetime, timedelta

from core.exceptions import ImageGenerationError

logger = logging.getLogger(__name__)


class FailoverStrategy(Enum):
    """Failover strategies"""
    SEQUENTIAL = "sequential"  # Try in order
    RANDOM = "random"  # Random selection
    LEAST_FAILED = "least_failed"  # Prefer least failures
    FASTEST = "fastest"  # Prefer fastest response
    CHEAPEST = "cheapest"  # Prefer cheapest option


class FailoverManager:
    """
    Manages failover between multiple image generation providers
    Tracks performance and automatically selects best provider
    """
    
    def __init__(self, strategy: FailoverStrategy = FailoverStrategy.SEQUENTIAL):
        """
        Initialize failover manager
        
        Args:
            strategy: Failover strategy to use
        """
        self.strategy = strategy
        self.provider_stats: Dict[str, Dict] = {}
        self.blacklist: Dict[str, datetime] = {}
        self.blacklist_duration = timedelta(minutes=30)
        
        logger.info(f"✅ Failover manager initialized with {strategy.value} strategy")
    
    def register_provider(self, provider: str, priority: int = 0, cost: float = 0.01):
        """
        Register a provider with the failover manager
        
        Args:
            provider: Provider name
            priority: Priority (lower is better)
            cost: Cost per image
        """
        if provider not in self.provider_stats:
            self.provider_stats[provider] = {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "last_success": None,
                "last_failure": None,
                "priority": priority,
                "cost": cost,
                "error_streak": 0
            }
            logger.info(f"Registered provider: {provider} (priority={priority}, cost=${cost})")
    
    def record_success(self, provider: str, response_time: float):
        """
        Record successful generation
        
        Args:
            provider: Provider name
            response_time: Time taken in seconds
        """
        if provider not in self.provider_stats:
            self.register_provider(provider)
        
        stats = self.provider_stats[provider]
        stats["attempts"] += 1
        stats["successes"] += 1
        stats["total_time"] += response_time
        stats["avg_time"] = stats["total_time"] / stats["successes"]
        stats["last_success"] = datetime.now()
        stats["error_streak"] = 0  # Reset error streak
        
        # Remove from blacklist if present
        if provider in self.blacklist:
            del self.blacklist[provider]
            logger.info(f"✅ {provider} removed from blacklist after success")
        
        logger.debug(f"✅ {provider}: Success in {response_time:.2f}s (avg: {stats['avg_time']:.2f}s)")
    
    def record_failure(self, provider: str, error: str):
        """
        Record failed generation
        
        Args:
            provider: Provider name
            error: Error message
        """
        if provider not in self.provider_stats:
            self.register_provider(provider)
        
        stats = self.provider_stats[provider]
        stats["attempts"] += 1
        stats["failures"] += 1
        stats["last_failure"] = datetime.now()
        stats["error_streak"] += 1
        
        # Blacklist if too many consecutive errors
        if stats["error_streak"] >= 3:
            self.blacklist[provider] = datetime.now()
            logger.warning(f"⚠️ {provider} blacklisted after {stats['error_streak']} consecutive errors")
        
        logger.debug(f"❌ {provider}: Failure ({error})")
    
    def get_next_provider(self, exclude: Optional[List[str]] = None) -> Optional[str]:
        """
        Get next provider to try based on strategy
        
        Args:
            exclude: Providers to exclude
        
        Returns:
            Provider name or None if all are blacklisted
        """
        exclude = exclude or []
        
        # Filter available providers
        available = []
        for provider, stats in self.provider_stats.items():
            # Skip excluded
            if provider in exclude:
                continue
            
            # Check blacklist
            if provider in self.blacklist:
                blacklist_time = self.blacklist[provider]
                if datetime.now() - blacklist_time < self.blacklist_duration:
                    continue
                else:
                    # Remove from blacklist
                    del self.blacklist[provider]
                    logger.info(f"✅ {provider} blacklist expired")
            
            available.append(provider)
        
        if not available:
            return None
        
        # Apply strategy
        if self.strategy == FailoverStrategy.SEQUENTIAL:
            # Sort by priority
            available.sort(key=lambda p: self.provider_stats[p]["priority"])
            return available[0]
        
        elif self.strategy == FailoverStrategy.LEAST_FAILED:
            # Sort by success rate
            def success_rate(p):
                stats = self.provider_stats[p]
                if stats["attempts"] == 0:
                    return 0.5  # No data, neutral rating
                return stats["successes"] / stats["attempts"]
            
            available.sort(key=success_rate, reverse=True)
            return available[0]
        
        elif self.strategy == FailoverStrategy.FASTEST:
            # Sort by average response time
            available.sort(key=lambda p: self.provider_stats[p]["avg_time"] or float('inf'))
            return available[0]
        
        elif self.strategy == FailoverStrategy.CHEAPEST:
            # Sort by cost
            available.sort(key=lambda p: self.provider_stats[p]["cost"])
            return available[0]
        
        elif self.strategy == FailoverStrategy.RANDOM:
            import random
            return random.choice(available)
        
        return available[0] if available else None
    
    def get_provider_order(self) -> List[str]:
        """
        Get complete provider order based on strategy
        
        Returns:
            Ordered list of providers
        """
        providers = []
        exclude = []
        
        while True:
            provider = self.get_next_provider(exclude)
            if not provider:
                break
            providers.append(provider)
            exclude.append(provider)
        
        return providers
    
    def is_provider_healthy(self, provider: str) -> bool:
        """
        Check if provider is healthy
        
        Args:
            provider: Provider name
        
        Returns:
            True if healthy
        """
        if provider in self.blacklist:
            blacklist_time = self.blacklist[provider]
            if datetime.now() - blacklist_time < self.blacklist_duration:
                return False
        
        if provider in self.provider_stats:
            stats = self.provider_stats[provider]
            # Unhealthy if more than 50% failures in last 10 attempts
            if stats["attempts"] >= 10:
                recent_success_rate = stats["successes"] / stats["attempts"]
                if recent_success_rate < 0.5:
                    return False
        
        return True
    
    def reset_provider(self, provider: str):
        """
        Reset provider statistics
        
        Args:
            provider: Provider name
        """
        if provider in self.provider_stats:
            priority = self.provider_stats[provider]["priority"]
            cost = self.provider_stats[provider]["cost"]
            self.register_provider(provider, priority, cost)
        
        if provider in self.blacklist:
            del self.blacklist[provider]
        
        logger.info(f"♻️ {provider} statistics reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get failover statistics
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "strategy": self.strategy.value,
            "providers": {},
            "blacklisted": list(self.blacklist.keys()),
            "recommended_order": self.get_provider_order()
        }
        
        for provider, provider_stats in self.provider_stats.items():
            success_rate = 0
            if provider_stats["attempts"] > 0:
                success_rate = provider_stats["successes"] / provider_stats["attempts"]
            
            stats["providers"][provider] = {
                "healthy": self.is_provider_healthy(provider),
                "attempts": provider_stats["attempts"],
                "successes": provider_stats["successes"],
                "failures": provider_stats["failures"],
                "success_rate": round(success_rate * 100, 2),
                "avg_response_time": round(provider_stats["avg_time"], 2),
                "cost": provider_stats["cost"],
                "error_streak": provider_stats["error_streak"]
            }
        
        return stats
    
    def set_strategy(self, strategy: FailoverStrategy):
        """
        Change failover strategy
        
        Args:
            strategy: New strategy
        """
        self.strategy = strategy
        logger.info(f"Failover strategy changed to: {strategy.value}")