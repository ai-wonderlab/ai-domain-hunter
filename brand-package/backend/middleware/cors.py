"""
CORS Middleware Configuration
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


def setup_cors(app: FastAPI) -> None:
    """Configure CORS middleware
    
    Args:
        app: FastAPI application instance
    """
    # Get allowed origins from settings
    origins = get_allowed_origins()
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Request-ID",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset"
        ],
        max_age=3600  # Cache preflight requests for 1 hour
    )
    
    logger.info(f"✅ CORS configured for {len(origins)} origins")
    logger.debug(f"Allowed origins: {origins}")


def get_allowed_origins() -> List[str]:
    """Get list of allowed origins based on environment
    
    Returns:
        List of allowed origin URLs
    """
    # Start with configured origins
    origins = settings.allowed_origins.copy()
    
    # Add environment-specific origins
    if settings.app_env == "development":
        # Add common development URLs
        dev_origins = [
            "http://localhost:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3000",
            "http://localhost:5173",  # Vite
            "http://localhost:4200",   # Angular
            "http://localhost:8080",   # Vue
        ]
        origins.extend(dev_origins)
        
    elif settings.app_env == "staging":
        staging_origins = [
            "https://staging.yourdomain.com",
            "https://preview.yourdomain.com",
            "https://test.yourdomain.com"
        ]
        origins.extend(staging_origins)
        
    elif settings.app_env == "production":
        prod_origins = [
            "https://yourdomain.com",
            "https://www.yourdomain.com",
            "https://app.yourdomain.com"
        ]
        origins.extend(prod_origins)
    
    # Remove duplicates and return
    return list(set(origins))


def is_origin_allowed(origin: str) -> bool:
    """Check if an origin is allowed
    
    Args:
        origin: Origin URL to check
        
    Returns:
        True if origin is allowed
    """
    allowed_origins = get_allowed_origins()
    
    # Check exact match
    if origin in allowed_origins:
        return True
    
    # Check wildcard patterns
    for allowed in allowed_origins:
        if allowed == "*":
            return True
        if allowed.endswith("*"):
            # Prefix match
            prefix = allowed[:-1]
            if origin.startswith(prefix):
                return True
    
    return False