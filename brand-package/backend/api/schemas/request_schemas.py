"""
Request Schemas
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from api.schemas.base_schema import GenerationBase


# Name Generation
class GenerateNamesRequest(GenerationBase):
    """Request for generating business names"""
    style: Optional[str] = Field(None, description="Naming style: modern, playful, professional")
    keywords: Optional[List[str]] = Field(default_factory=list, max_items=10)
    count: int = Field(default=10, ge=1, le=10)
    
    @validator('keywords')
    def validate_keywords(cls, v):
        if v:
            return [k.strip()[:50] for k in v if k.strip()]
        return []


# Domain Checking
class CheckDomainsRequest(BaseModel):
    """Request for checking domain availability"""
    domains: List[str] = Field(..., min_items=1, max_items=20)
    
    @validator('domains')
    def validate_domains(cls, v):
        cleaned = []
        for domain in v:
            domain = domain.lower().strip()
            if len(domain) > 253:  # Max domain length
                continue
            cleaned.append(domain)
        
        if not cleaned:
            raise ValueError("No valid domains provided")
        return cleaned


# Logo Generation
class GenerateLogosRequest(GenerationBase):
    """Request for generating logos"""
    business_name: str = Field(..., min_length=1, max_length=50)
    style: Optional[str] = Field(
        None, 
        description="Logo style: minimalist, modern, playful, elegant, bold"
    )
    colors: Optional[List[str]] = Field(default_factory=list, max_items=5)
    count: int = Field(default=3, ge=1, le=3)
    
    @validator('colors')
    def validate_colors(cls, v):
        import re
        if v:
            validated = []
            for color in v:
                if re.match(r'^#[A-Fa-f0-9]{6}$', color):
                    validated.append(color.upper())
            return validated
        return []


# Color Palette Generation
class GenerateColorsRequest(GenerationBase):
    """Request for generating color palettes"""
    business_name: Optional[str] = Field(None, max_length=50)
    theme: Optional[str] = Field(
        None,
        description="Color theme: vibrant, pastel, monochrome, earth, ocean"
    )
    logo_colors: Optional[List[str]] = Field(default_factory=list, max_items=5)
    count: int = Field(default=3, ge=1, le=5)


# Tagline Generation
class GenerateTaglinesRequest(GenerationBase):
    """Request for generating taglines"""
    business_name: str = Field(..., min_length=1, max_length=50)
    tone: Optional[str] = Field(
        None,
        description="Tone: professional, friendly, bold, inspirational, witty"
    )
    target_audience: Optional[str] = Field(None, max_length=200)
    keywords: Optional[List[str]] = Field(default_factory=list, max_items=10)
    count: int = Field(default=5, ge=1, le=5)


# Package Generation
class GeneratePackageRequest(GenerationBase):
    """Request for generating complete brand package"""
    business_name: Optional[str] = Field(None, max_length=50)
    style_preferences: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Style preferences for each component"
    )
    include_services: Optional[List[str]] = Field(
        default_factory=lambda: ['name', 'domain', 'logo', 'color', 'tagline'],
        description="Services to include in package"
    )
    
    @validator('include_services')
    def validate_services(cls, v):
        valid_services = {'name', 'domain', 'logo', 'color', 'tagline'}
        if v:
            return [s for s in v if s in valid_services]
        return list(valid_services)


# Regeneration
class RegenerateRequest(BaseModel):
    """Request for regenerating with feedback"""
    generation_id: str = Field(..., description="Original generation ID")
    feedback: str = Field(..., min_length=10, max_length=500)
    service_to_regenerate: Optional[str] = Field(
        None,
        description="Specific service to regenerate (for packages)"
    )


# User Authentication
class LoginRequest(BaseModel):
    """Login request"""
    email: str
    google_token: Optional[str] = Field(None, description="Google OAuth token")


class RegisterRequest(BaseModel):
    """Registration request"""
    email: str
    google_id: Optional[str] = None
    full_name: Optional[str] = Field(None, max_length=100)


# Project Management
class CreateProjectRequest(BaseModel):
    """Create project request"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)


class UpdateProjectRequest(BaseModel):
    """Update project request"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    status: Optional[str] = Field(None, pattern='^(draft|in_progress|completed|archived)$')