"""
Custom Exception Classes
"""
from typing import Optional, Dict, Any


class BrandGeneratorException(Exception):
    """Base exception for all application exceptions"""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "internal_error"
        self.status_code = status_code
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API response"""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }


# Authentication Exceptions
class AuthenticationError(BrandGeneratorException):
    """Raised when authentication fails"""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message=message,
            error_code="authentication_failed",
            status_code=401,
            **kwargs
        )


class TokenExpiredError(AuthenticationError):
    """Raised when JWT token is expired"""
    
    def __init__(self, message: str = "Token has expired", **kwargs):
        super().__init__(
            message=message,
            error_code="token_expired",
            **kwargs
        )


class InvalidTokenError(AuthenticationError):
    """Raised when JWT token is invalid"""
    
    def __init__(self, message: str = "Invalid token", **kwargs):
        super().__init__(
            message=message,
            error_code="invalid_token",
            **kwargs
        )


# Authorization Exceptions
class AuthorizationError(BrandGeneratorException):
    """Raised when user is not authorized"""
    
    def __init__(self, message: str = "Not authorized", **kwargs):
        super().__init__(
            message=message,
            error_code="not_authorized",
            status_code=403,
            **kwargs
        )


# Rate Limiting Exceptions
class RateLimitExceeded(BrandGeneratorException):
    """Raised when rate limit is exceeded"""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if retry_after:
            details["retry_after"] = retry_after
        
        super().__init__(
            message=message,
            error_code="rate_limit_exceeded",
            status_code=429,
            details=details,
            **kwargs
        )


# Validation Exceptions
class ValidationError(BrandGeneratorException):
    """Raised when input validation fails"""
    
    def __init__(
        self,
        message: str = "Validation failed",
        errors: Optional[list] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if errors:
            details["errors"] = errors
        
        super().__init__(
            message=message,
            error_code="validation_error",
            status_code=400,
            details=details,
            **kwargs
        )

# Generation Exceptions
class GenerationError(BrandGeneratorException):
    """Base exception for generation errors"""
    
    def __init__(self, message: str = "Generation failed", **kwargs):
        kwargs.pop('error_code', None)  # Avoid conflicts
        super().__init__(
            message=message,
            error_code="generation_failed",
            status_code=500,
            **kwargs
        )


class AIGenerationError(GenerationError):
    """Raised when AI generation fails"""
    
    def __init__(
        self,
        message: str = "AI generation failed",
        model: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if model:
            details["model"] = model
        kwargs['details'] = details
        kwargs.pop('error_code', None)  # Avoid conflicts
        
        super().__init__(message=message, **kwargs)


class ImageGenerationError(GenerationError):
    """Raised when image generation fails"""
    
    def __init__(
        self,
        message: str = "Image generation failed",
        provider: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if provider:
            details["provider"] = provider
        kwargs['details'] = details
        kwargs.pop('error_code', None)  # Avoid conflicts
        
        super().__init__(message=message, **kwargs)

class ConfigurationError(BrandGeneratorException):
    """Raised when configuration is invalid or missing"""
    
    def __init__(self, message: str = "Configuration error", **kwargs):
        super().__init__(
            message=message,
            error_code="configuration_error",
            status_code=500,
            **kwargs
        )

class MissingAPIKeyError(ConfigurationError):
    """Raised when required API key is missing"""
    
    def __init__(
        self,
        api_name: str,
        message: Optional[str] = None,
        **kwargs
    ):
        if not message:
            message = f"API key for {api_name} is missing"
        
        super().__init__(
            message=message,
            error_code="missing_api_key",
            details={"api": api_name},
            **kwargs
        )

# Database Exceptions
class DatabaseError(BrandGeneratorException):
    """Raised when database operation fails"""
    
    def __init__(self, message: str = "Database operation failed", **kwargs):
        super().__init__(
            message=message,
            error_code="database_error",
            status_code=500,
            **kwargs
        )


class RecordNotFoundError(DatabaseError):
    """Raised when requested record is not found"""
    
    def __init__(
        self,
        resource: str,
        identifier: Any,
        message: Optional[str] = None,
        **kwargs
    ):
        if not message:
            message = f"{resource} with id '{identifier}' not found"
        
        super().__init__(
            message=message,
            error_code="record_not_found",
            status_code=404,
            details={"resource": resource, "id": identifier},
            **kwargs
        )


class DuplicateRecordError(DatabaseError):
    """Raised when trying to create duplicate record"""
    
    def __init__(
        self,
        resource: str,
        message: Optional[str] = None,
        **kwargs
    ):
        if not message:
            message = f"{resource} already exists"
        
        super().__init__(
            message=message,
            error_code="duplicate_record",
            status_code=409,
            details={"resource": resource},
            **kwargs
        )


# Storage Exceptions
class StorageError(BrandGeneratorException):
    """Raised when storage operation fails"""
    
    def __init__(self, message: str = "Storage operation failed", **kwargs):
        super().__init__(
            message=message,
            error_code="storage_error",
            status_code=500,
            **kwargs
        )


class FileUploadError(StorageError):
    """Raised when file upload fails"""
    
    def __init__(
        self,
        filename: str,
        message: Optional[str] = None,
        **kwargs
    ):
        if not message:
            message = f"Failed to upload file: {filename}"
        
        super().__init__(
            message=message,
            error_code="file_upload_failed",
            details={"filename": filename},
            **kwargs
        )


# Service Exceptions
class ServiceUnavailableError(BrandGeneratorException):
    """Raised when external service is unavailable"""
    
    def __init__(
        self,
        service: str,
        message: Optional[str] = None,
        **kwargs
    ):
        if not message:
            message = f"Service '{service}' is currently unavailable"
        
        super().__init__(
            message=message,
            error_code="service_unavailable",
            status_code=503,
            details={"service": service},
            **kwargs
        )

BaseAppException = BrandGeneratorException

# Resource Exceptions
class NotFoundError(BrandGeneratorException):
    """Raised when a resource is not found"""
    
    def __init__(self, resource: str, resource_id: str = None, **kwargs):
        message = f"{resource} not found"
        if resource_id:
            message += f": {resource_id}"
        super().__init__(
            message=message,
            error_code="not_found",
            status_code=404,
            **kwargs
        )


# External Service Exceptions
class ExternalServiceError(BrandGeneratorException):
    """Raised when an external service fails"""
    
    def __init__(self, service: str, message: str = None, **kwargs):
        msg = f"External service error: {service}"
        if message:
            msg += f" - {message}"
        super().__init__(
            message=msg,
            error_code="external_service_error",
            status_code=502,
            **kwargs
        )

# Exports
__all__ = [
    "BrandGeneratorException",
    "BaseAppException",
    "AuthenticationError",
    "TokenExpiredError",
    "InvalidTokenError",
    "AuthorizationError",
    "RateLimitExceeded",
    "ValidationError",
    "GenerationError",
    "AIGenerationError",
    "ImageGenerationError",
    "ConfigurationError",
    "MissingAPIKeyError",
    "DatabaseError",
    "RecordNotFoundError",
    "DuplicateRecordError",
    "StorageError",
    "FileUploadError",
    "ServiceUnavailableError",
    "NotFoundError",
    "ExternalServiceError"  
]