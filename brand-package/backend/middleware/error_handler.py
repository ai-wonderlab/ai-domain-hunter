"""
Global Error Handler Middleware
"""
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
import logging
import traceback
from typing import Union, Dict, Any
import json

from config.settings import settings
from core.exceptions import (
    BaseAppException,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    RateLimitExceeded,
    GenerationError,
    AIGenerationError,
    DatabaseError,
    ExternalServiceError
)

logger = logging.getLogger(__name__)


async def http_exception_handler(
    request: Request,
    exc: Union[HTTPException, StarletteHTTPException]
) -> JSONResponse:
    """Handle HTTP exceptions"""
    
    # Get request ID if available
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Build error response
    error_response = {
        "error": {
            "type": "http_error",
            "message": exc.detail if hasattr(exc, "detail") else str(exc),
            "status_code": exc.status_code,
            "request_id": request_id
        }
    }
    
    # Add headers if present
    headers = getattr(exc, "headers", None)
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response,
        headers=headers
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """Handle validation errors"""
    
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Format validation errors
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"] if loc != "body")
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"]
        })
    
    error_response = {
        "error": {
            "type": "validation_error",
            "message": "Request validation failed",
            "details": errors,
            "request_id": request_id
        }
    }
    
    return JSONResponse(
        status_code=422,
        content=error_response
    )


async def app_exception_handler(
    request: Request,
    exc: BaseAppException
) -> JSONResponse:
    """Handle custom application exceptions"""
    
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Map exception types to status codes
    status_code_map = {
        ValidationError: 400,
        AuthenticationError: 401,
        AuthorizationError: 403,
        NotFoundError: 404,
        RateLimitExceeded: 429,
        GenerationError: 500,
        AIGenerationError: 500,
        DatabaseError: 503,
        ExternalServiceError: 502
    }
    
    status_code = status_code_map.get(type(exc), 500)
    
    # Build error response
    error_response = {
        "error": {
            "type": exc.error_code,
            "message": str(exc),
            "request_id": request_id
        }
    }
    
    # Add extra details if available
    if hasattr(exc, "details") and exc.details:
        error_response["error"]["details"] = exc.details
    
    # Log error
    if status_code >= 500:
        logger.error(
            f"Application error [{request_id}]: {type(exc).__name__}: {exc}",
            exc_info=True
        )
    else:
        logger.warning(
            f"Client error [{request_id}]: {type(exc).__name__}: {exc}"
        )
    
    return JSONResponse(
        status_code=status_code,
        content=error_response
    )


async def general_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """Handle unexpected exceptions"""
    
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Log full traceback
    logger.error(
        f"Unexpected error [{request_id}]: {type(exc).__name__}: {exc}",
        exc_info=True
    )
    
    # Build error response
    error_response = {
        "error": {
            "type": "internal_error",
            "message": "An unexpected error occurred",
            "request_id": request_id
        }
    }
    
    # In debug mode, include traceback
    if settings.debug:
        error_response["error"]["debug"] = {
            "exception": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc().split("\n")
        }
    
    return JSONResponse(
        status_code=500,
        content=error_response
    )


def register_exception_handlers(app):
    """Register all exception handlers with the app"""
    
    # HTTP exceptions
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    
    # Validation exceptions
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    
    # Custom application exceptions
    app.add_exception_handler(BaseAppException, app_exception_handler)
    
    # Specific custom exceptions
    app.add_exception_handler(ValidationError, app_exception_handler)
    app.add_exception_handler(AuthenticationError, app_exception_handler)
    app.add_exception_handler(AuthorizationError, app_exception_handler)
    app.add_exception_handler(NotFoundError, app_exception_handler)
    app.add_exception_handler(RateLimitExceeded, app_exception_handler)
    app.add_exception_handler(GenerationError, app_exception_handler)
    app.add_exception_handler(AIGenerationError, app_exception_handler)
    app.add_exception_handler(DatabaseError, app_exception_handler)
    app.add_exception_handler(ExternalServiceError, app_exception_handler)
    
    # General exceptions (must be last)
    app.add_exception_handler(Exception, general_exception_handler)
    
    logger.info("✅ Exception handlers registered")