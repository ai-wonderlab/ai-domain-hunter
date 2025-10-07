"""
Middleware Package
"""
from fastapi import FastAPI
import logging
from middleware.cors import setup_cors
from middleware.rate_limiter import RateLimitMiddleware
from middleware.logging_middleware import LoggingMiddleware, AccessLogMiddleware
from middleware.error_handler import register_exception_handlers
from config.settings import settings

logger = logging.getLogger(__name__)

def setup_middleware(app: FastAPI) -> None:
    """
    Configure all middleware for the application
    
    Args:
        app: FastAPI application instance
    """
    
    logger.info("Setting up middleware...")
    
    # 1. CORS (must be first for preflight requests)
    setup_cors(app)
    
    # 2. Access logging (early in chain)
    if settings.app_env == "production":
        app.add_middleware(AccessLogMiddleware)
    
    # 3. Rate limiting
    if settings.enable_rate_limiting:
        app.add_middleware(RateLimitMiddleware)
        logger.info("✅ Rate limiting middleware enabled")
    
    # 4. Request/Response logging
    if settings.enable_request_logging:
        app.add_middleware(LoggingMiddleware)
        logger.info("✅ Request logging middleware enabled")
    
    # 5. Exception handlers (not middleware, but related)
    register_exception_handlers(app)
    
    logger.info("✅ All middleware configured")

__all__ = [
    "setup_middleware",
    "setup_cors",
    "RateLimitMiddleware",
    "LoggingMiddleware",
    "AccessLogMiddleware",
    "register_exception_handlers"
]
