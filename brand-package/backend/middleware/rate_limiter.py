"""
Rate Limiting Middleware
"""
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Tuple, Optional
import time
import hashlib
import logging
from collections import defaultdict
import asyncio

from config.settings import settings
from core.exceptions import RateLimitExceeded

logger = logging.getLogger(__name__)


class RateLimiter:
    """In-memory rate limiter with sliding window"""
    
    def __init__(self):
        # Store request timestamps for each client
        self.requests: Dict[str, list] = defaultdict(list)
        # Lock for thread safety
        self.lock = asyncio.Lock()
        # Cleanup task
        self.cleanup_interval = 300  # 5 minutes
        self.cleanup_task = None
    
    async def start_cleanup(self):
        """Start periodic cleanup of old entries"""
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop_cleanup(self):
        """Stop cleanup task"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
    
    async def _cleanup_loop(self):
        """Cleanup old entries periodically"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_old_entries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    async def _cleanup_old_entries(self):
        """Remove old request entries"""
        async with self.lock:
            current_time = time.time()
            clients_to_remove = []
            
            for client_id, timestamps in self.requests.items():
                # Keep only recent timestamps
                self.requests[client_id] = [
                    ts for ts in timestamps
                    if current_time - ts < 3600  # Keep last hour
                ]
                
                # Remove client if no recent requests
                if not self.requests[client_id]:
                    clients_to_remove.append(client_id)
            
            for client_id in clients_to_remove:
                del self.requests[client_id]
            
            if clients_to_remove:
                logger.info(f"Cleaned up {len(clients_to_remove)} inactive clients")
    
    async def is_allowed(
        self,
        client_id: str,
        limit: int = 60,
        window: int = 60
    ) -> Tuple[bool, Dict]:
        """Check if request is allowed
        
        Args:
            client_id: Client identifier
            limit: Maximum requests allowed
            window: Time window in seconds
            
        Returns:
            Tuple of (is_allowed, metadata)
        """
        async with self.lock:
            current_time = time.time()
            window_start = current_time - window
            
            # Get client's request timestamps
            timestamps = self.requests[client_id]
            
            # Remove old timestamps outside window
            timestamps = [ts for ts in timestamps if ts > window_start]
            self.requests[client_id] = timestamps
            
            # Check if limit exceeded
            if len(timestamps) >= limit:
                # Calculate retry after
                oldest_request = min(timestamps)
                retry_after = int(oldest_request + window - current_time)
                
                return False, {
                    "limit": limit,
                    "remaining": 0,
                    "reset": int(oldest_request + window),
                    "retry_after": max(1, retry_after)
                }
            
            # Add current request
            timestamps.append(current_time)
            
            # Calculate remaining
            remaining = limit - len(timestamps)
            
            # Calculate reset time
            reset_time = int(current_time + window)
            if timestamps:
                reset_time = int(min(timestamps) + window)
            
            return True, {
                "limit": limit,
                "remaining": remaining,
                "reset": reset_time
            }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(self, app):
        super().__init__(app)
        self.limiter = RateLimiter()
        # Start cleanup task
        asyncio.create_task(self.limiter.start_cleanup())
        
        # Configure limits
        self.limits = {
            "default": {
                "requests": settings.rate_limit_requests_per_minute,
                "window": 60
            },
            "auth": {
                "requests": 5,
                "window": 60
            },
            "generation": {
                "requests": 10,
                "window": 60
            }
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting"""
        
        # Skip rate limiting for health endpoints
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Determine rate limit to apply
        limit_type = self._get_limit_type(request.url.path)
        limit_config = self.limits.get(limit_type, self.limits["default"])
        
        # Check rate limit
        is_allowed, metadata = await self.limiter.is_allowed(
            client_id,
            limit_config["requests"],
            limit_config["window"]
        )
        
        if not is_allowed:
            # Return 429 Too Many Requests
            return Response(
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests. Please retry after some time.",
                    "retry_after": metadata["retry_after"]
                },
                status_code=429,
                headers={
                    "X-RateLimit-Limit": str(metadata["limit"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(metadata["reset"]),
                    "Retry-After": str(metadata["retry_after"])
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(metadata["limit"])
        response.headers["X-RateLimit-Remaining"] = str(metadata["remaining"])
        response.headers["X-RateLimit-Reset"] = str(metadata["reset"])
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting
        
        Args:
            request: Request object
            
        Returns:
            Client identifier string
        """
        # Try to get user ID from request state (set by auth middleware)
        if hasattr(request.state, "user_id"):
            return f"user_{request.state.user_id}"
        
        # Try to get from authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header:
            # Hash the token to use as identifier
            token = auth_header.replace("Bearer ", "")
            return f"token_{hashlib.md5(token.encode()).hexdigest()}"
        
        # Fall back to IP address
        if request.client:
            return f"ip_{request.client.host}"
        
        # Last resort
        return "anonymous"
    
    def _get_limit_type(self, path: str) -> str:
        """Determine rate limit type based on path
        
        Args:
            path: Request path
            
        Returns:
            Limit type identifier
        """
        if path.startswith("/api/v1/auth"):
            return "auth"
        elif path.startswith("/api/v1/generation"):
            return "generation"
        else:
            return "default"