"""
Error Response Schemas
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ValidationErrorDetail(BaseModel):
    """Validation error detail"""
    field: str
    message: str
    type: str


class ValidationErrorResponse(BaseModel):
    """Validation error response"""
    error: str = "validation_error"
    message: str = "Validation failed"
    details: List[ValidationErrorDetail]


class RateLimitErrorResponse(BaseModel):
    """Rate limit error response"""
    error: str = "rate_limit_exceeded"
    message: str
    current_usage: int
    limit: int
    resets_at: Optional[str] = None
    upgrade_url: Optional[str] = "/pricing"


class AuthenticationErrorResponse(BaseModel):
    """Authentication error response"""
    error: str = "authentication_failed"
    message: str
    details: Optional[Dict[str, Any]] = None


class NotFoundErrorResponse(BaseModel):
    """Not found error response"""
    error: str = "not_found"
    message: str
    resource: Optional[str] = None
    id: Optional[str] = None


class GenerationErrorResponse(BaseModel):
    """Generation error response"""
    error: str = "generation_failed"
    message: str
    service: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    partial_results: Optional[Dict[str, Any]] = None