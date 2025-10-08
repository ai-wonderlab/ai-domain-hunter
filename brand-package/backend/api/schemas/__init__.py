"""
API Schemas Package
"""

# Base schemas
from api.schemas.base_schema import (
    BaseResponse,
    ErrorResponse,
    PaginationParams,
    PaginatedResponse,
    GenerationBase,
    UserBase,
    ProjectBase
)

# Request schemas
from api.schemas.request_schemas import (
    GenerateNamesRequest,
    CheckDomainsRequest,
    GenerateDomainsRequest,
    GenerateLogosRequest,
    GenerateColorsRequest,
    GenerateTaglinesRequest,
    GeneratePackageRequest,
    RegenerateRequest,
    LoginRequest,
    RegisterRequest,
    CreateProjectRequest,
    UpdateProjectRequest
)

# Response schemas
from api.schemas.response_schemas import (
    NameOption,
    GenerateNamesResponse,
    DomainResult,
    CheckDomainsResponse,
    LogoOption,
    GenerateLogosResponse,
    ColorInfo,
    ColorPalette,
    GenerateColorsResponse,
    TaglineOption,
    GenerateTaglinesResponse,
    PackageSummary,
    GeneratePackageResponse,
    UserInfo,
    UserHistoryItem,
    UserHistoryResponse,
    UserUsageResponse,
    LoginResponse,
    ProjectInfo,
    ProjectListResponse,
    ProjectDetailResponse,
    HealthStatus,
    HealthCheckResponse
)

# Error schemas
from api.schemas.error_schemas import (
    ValidationErrorDetail,
    ValidationErrorResponse,
    RateLimitErrorResponse,
    AuthenticationErrorResponse,
    NotFoundErrorResponse,
    GenerationErrorResponse
)

__all__ = [
    # Base
    "BaseResponse",
    "ErrorResponse",
    "PaginationParams",
    "PaginatedResponse",
    "GenerationBase",
    "UserBase",
    "ProjectBase",
    
    # Requests
    "GenerateNamesRequest",
    "CheckDomainsRequest",
    "GenerateDomainsRequest",
    "GenerateLogosRequest",
    "GenerateColorsRequest",
    "GenerateTaglinesRequest",
    "GeneratePackageRequest",
    "RegenerateRequest",
    "LoginRequest",
    "RegisterRequest",
    "CreateProjectRequest",
    "UpdateProjectRequest",
    
    # Responses
    "NameOption",
    "GenerateNamesResponse",
    "DomainResult",
    "CheckDomainsResponse",
    "LogoOption",
    "GenerateLogosResponse",
    "ColorInfo",
    "ColorPalette",
    "GenerateColorsResponse",
    "TaglineOption",
    "GenerateTaglinesResponse",
    "PackageSummary",
    "GeneratePackageResponse",
    "UserInfo",
    "UserHistoryItem",
    "UserHistoryResponse",
    "UserUsageResponse",
    "LoginResponse",
    "ProjectInfo",
    "ProjectListResponse",
    "ProjectDetailResponse",
    "HealthStatus",
    "HealthCheckResponse",
    
    # Errors
    "ValidationErrorDetail",
    "ValidationErrorResponse",
    "RateLimitErrorResponse",
    "AuthenticationErrorResponse",
    "NotFoundErrorResponse",
    "GenerationErrorResponse"
]