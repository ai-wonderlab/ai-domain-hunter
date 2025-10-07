"""
Response Schemas
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from api.schemas.base_schema import BaseResponse


# Name Generation Response
class NameOption(BaseModel):
    """Single name option"""
    name: str
    reasoning: str
    score: float = Field(ge=0, le=10)
    style: Optional[str] = None
    memorability: Optional[int] = None
    pronounceability: Optional[int] = None
    uniqueness: Optional[int] = None


class GenerateNamesResponse(BaseResponse):
    """Response for name generation"""
    names: List[NameOption]
    generation_id: str
    count: int


# Domain Checking Response
class DomainResult(BaseModel):
    """Single domain result"""
    domain: str
    available: bool
    status: str
    price: Optional[str] = None
    registrar: Optional[str] = None
    registrar_link: Optional[str] = None
    checked_at: datetime
    method: str


class CheckDomainsResponse(BaseResponse):
    """Response for domain checking"""
    results: List[DomainResult]
    generation_id: str
    total_checked: int
    available_count: int


# Logo Generation Response
class LogoOption(BaseModel):
    """Single logo option"""
    id: str
    concept_name: str
    description: str
    style: str
    colors: List[str]
    rationale: str
    urls: Dict[str, str]  # Format -> URL mapping


class GenerateLogosResponse(BaseResponse):
    """Response for logo generation"""
    logos: List[LogoOption]
    generation_id: str
    business_name: str
    count: int


# Color Palette Response
class ColorInfo(BaseModel):
    """Single color information"""
    hex: str
    rgb: str
    hsl: Optional[str] = None
    name: str
    role: str


class ColorPalette(BaseModel):
    """Single color palette"""
    id: str
    name: str
    description: str
    theme: Optional[str] = None
    colors: List[ColorInfo]
    primary_color: Optional[ColorInfo] = None
    secondary_color: Optional[ColorInfo] = None
    accent_color: Optional[ColorInfo] = None
    psychology: Optional[str] = None
    use_cases: List[str] = Field(default_factory=list)
    contrasts: Dict[str, float] = Field(default_factory=dict)
    accessibility: Dict[str, Any] = Field(default_factory=dict)


class GenerateColorsResponse(BaseResponse):
    """Response for color generation"""
    palettes: List[ColorPalette]
    generation_id: str
    count: int


# Tagline Generation Response
class TaglineOption(BaseModel):
    """Single tagline option"""
    id: str
    text: str
    tone: str
    style: Optional[str] = None
    reasoning: str
    target_emotion: Optional[str] = None
    call_to_action: bool = False
    word_count: int
    character_count: int
    readability: str
    use_cases: List[str] = Field(default_factory=list)


class GenerateTaglinesResponse(BaseResponse):
    """Response for tagline generation"""
    taglines: List[TaglineOption]
    generation_id: str
    business_name: str
    count: int


# Package Generation Response
class PackageSummary(BaseModel):
    """Package summary"""
    business_name: str
    components_generated: List[str]
    recommendations: List[str] = Field(default_factory=list)
    featured_tagline: Optional[str] = None


class GeneratePackageResponse(BaseResponse):
    """Response for package generation"""
    project_id: str
    generation_id: str
    business_name: str
    names: Optional[List[NameOption]] = None
    domains: Optional[List[DomainResult]] = None
    logos: Optional[List[LogoOption]] = None
    color_palettes: Optional[List[ColorPalette]] = None
    taglines: Optional[List[TaglineOption]] = None
    summary: PackageSummary
    errors: List[Dict[str, str]] = Field(default_factory=list)
    total_cost: float
    created_at: datetime


# User Response
class UserInfo(BaseModel):
    """User information"""
    id: str
    email: str
    created_at: datetime
    generation_count: int = 0
    generation_limit: int = 2


class UserHistoryItem(BaseModel):
    """Single history item"""
    id: str
    type: str
    created_at: datetime
    input_data: Dict[str, Any]
    preview: Optional[Dict[str, Any]] = None


class UserHistoryResponse(BaseResponse):
    """User history response"""
    projects: List[Dict[str, Any]]
    recent_generations: List[UserHistoryItem]
    total_count: int


class UserUsageResponse(BaseResponse):
    """User usage response"""
    generation_count: int
    limit: int
    remaining: int
    resets_at: Optional[datetime] = None
    is_paid: bool = False


# Authentication Response
class LoginResponse(BaseResponse):
    """Login response"""
    user: UserInfo
    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 86400  # 24 hours


# Project Response
class ProjectInfo(BaseModel):
    """Project information"""
    id: str
    user_id: str
    name: str
    description: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime
    generation_count: int = 0


class ProjectListResponse(BaseResponse):
    """Project list response"""
    projects: List[ProjectInfo]
    total: int


class ProjectDetailResponse(BaseResponse):
    """Project detail response"""
    project: ProjectInfo
    generations: List[Dict[str, Any]]
    assets: List[Dict[str, Any]] = Field(default_factory=list)


# Health Check Response
class HealthStatus(BaseModel):
    """Service health status"""
    service: str
    status: str  # healthy, degraded, unhealthy
    latency_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class HealthCheckResponse(BaseResponse):
    """Health check response"""
    status: str  # healthy, degraded, unhealthy
    version: str
    environment: str
    services: List[HealthStatus]
    uptime_seconds: float