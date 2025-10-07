"""
Request/Response Logging Middleware
"""
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
import json
import logging
import uuid
from typing import Optional, Dict, Any
import traceback

from config.settings import settings

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive logging middleware for requests and responses
    """
    
    def __init__(self, app, **kwargs):
        super().__init__(app)
        
        # Paths to skip detailed logging
        self.skip_paths = [
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/favicon.ico"
        ]
        
        # Sensitive headers to redact
        self.sensitive_headers = [
            "authorization",
            "x-api-key",
            "cookie",
            "set-cookie"
        ]
    
    async def dispatch(self, request: Request, call_next):
        """Log request and response details"""
        
        # Skip logging for certain paths
        if any(request.url.path.startswith(path) for path in self.skip_paths):
            return await call_next(request)
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log request
        await self._log_request(request, request_id)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = (time.time() - start_time) * 1000  # Convert to ms
            
            # Log response
            await self._log_response(
                request, response, duration, request_id
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Log error
            duration = (time.time() - start_time) * 1000
            await self._log_error(request, e, duration, request_id)
            raise
    
    async def _log_request(self, request: Request, request_id: str):
        """Log incoming request details"""
        
        # Get client info
        client_host = request.client.host if request.client else "unknown"
        
        # Get headers (with sensitive data redacted)
        headers = self._get_safe_headers(request.headers)
        
        # Build log entry
        log_data = {
            "type": "request",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query": dict(request.query_params),
            "client_ip": client_host,
            "user_agent": headers.get("user-agent", "unknown")
        }
        
        # SKIP BODY LOGGING - it causes issues with FastAPI body parsing
        # Body will be available to the handler anyway
        
        # Log at appropriate level
        if settings.debug:
            logger.info(f"→ Request: {json.dumps(log_data)}")
        else:
            logger.info(
                f"→ {request.method} {request.url.path} "
                f"[{request_id}] from {client_host}"
            )
    
    async def _log_response(
        self,
        request: Request,
        response: Response,
        duration: float,
        request_id: str
    ):
        """Log response details"""
        
        log_data = {
            "type": "response",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration, 2)
        }
        
        # Determine log level based on status code
        if response.status_code >= 500:
            log_level = logging.ERROR
            log_data["level"] = "error"
        elif response.status_code >= 400:
            log_level = logging.WARNING
            log_data["level"] = "warning"
        else:
            log_level = logging.INFO
            log_data["level"] = "info"
        
        # Log
        if settings.debug:
            logger.log(log_level, f"← Response: {json.dumps(log_data)}")
        else:
            logger.log(
                log_level,
                f"← {response.status_code} {request.method} {request.url.path} "
                f"({duration:.2f}ms) [{request_id}]"
            )
    
    async def _log_error(
        self,
        request: Request,
        error: Exception,
        duration: float,
        request_id: str
    ):
        """Log error details"""
        
        log_data = {
            "type": "error",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "duration_ms": round(duration, 2)
        }
        
        # Add traceback in debug mode
        if settings.debug:
            log_data["traceback"] = traceback.format_exc()
        
        logger.error(f"✗ Error: {json.dumps(log_data)}")
    
    def _get_safe_headers(self, headers: Dict) -> Dict:
        """Get headers with sensitive data redacted"""
        safe_headers = {}
        
        for key, value in headers.items():
            key_lower = key.lower()
            if key_lower in self.sensitive_headers:
                safe_headers[key] = "***REDACTED***"
            else:
                safe_headers[key] = value
        
        return safe_headers
    
    def _redact_sensitive_data(self, data: Any) -> Any:
        """Redact sensitive data from request/response bodies"""
        
        if isinstance(data, dict):
            redacted = {}
            sensitive_fields = [
                "password", "token", "api_key", "secret",
                "credit_card", "ssn"
            ]
            
            for key, value in data.items():
                key_lower = key.lower()
                if any(field in key_lower for field in sensitive_fields):
                    redacted[key] = "***REDACTED***"
                elif isinstance(value, (dict, list)):
                    redacted[key] = self._redact_sensitive_data(value)
                else:
                    redacted[key] = value
            
            return redacted
            
        elif isinstance(data, list):
            return [self._redact_sensitive_data(item) for item in data]
        
        return data


class AccessLogMiddleware(BaseHTTPMiddleware):
    """
    Simplified access log middleware (Apache/Nginx style)
    """
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Get client IP
        client_ip = request.client.host if request.client else "-"
        
        # Format log line (Common Log Format with extras)
        log_line = (
            f'{client_ip} - - '
            f'[{time.strftime("%d/%b/%Y:%H:%M:%S %z")}] '
            f'"{request.method} {request.url.path} HTTP/1.1" '
            f'{response.status_code} - '
            f'"{request.headers.get("referer", "-")}" '
            f'"{request.headers.get("user-agent", "-")}" '
            f'{process_time:.3f}s'
        )
        
        # Log to access logger
        access_logger = logging.getLogger("access")
        access_logger.info(log_line)
        
        return response
