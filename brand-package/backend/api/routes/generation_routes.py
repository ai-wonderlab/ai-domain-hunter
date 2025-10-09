"""
Generation API Routes
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel

from api.schemas.request_schemas import (
    GenerateNamesRequest,
    CheckDomainsRequest,
    GenerateDomainsRequest,
    GenerateLogosRequest,
    GenerateColorsRequest,
    GenerateTaglinesRequest,
    GeneratePackageRequest,
    RegenerateRequest
)
from api.schemas.response_schemas import (
    GenerateNamesResponse,
    CheckDomainsResponse,
    GenerateLogosResponse,
    GenerateColorsResponse,
    GenerateTaglinesResponse,
    GeneratePackageResponse
)

# Preferences analysis schemas
class AnalyzePreferencesRequest(BaseModel):
    """Request for preferences analysis"""
    business_name: str
    description: str
    for_type: Literal["logo", "tagline"]
    industry: Optional[str] = None

class AnalyzePreferencesResponse(BaseModel):
    """Response for preferences analysis"""
    success: bool
    style: str
    colors: List[str]
    reasoning: str
    suggestions: Dict[str, Any]

from api.dependencies import get_current_user, check_rate_limit
from services.name_service import NameService
from services.domain_service import DomainService
from services.logo_service import LogoService
from services.color_service import ColorService
from services.tagline_service import TaglineService
from services.package_service import PackageService
from core.exceptions import (
    GenerationError,
    RateLimitExceeded,
    ValidationError
)
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/names", response_model=GenerateNamesResponse)
async def generate_names(
    request: GenerateNamesRequest,
    user_id: str = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
) -> GenerateNamesResponse:
    """Generate business names"""
    try:
        service = NameService()
        result = await service.generate(
            description=request.description,
            user_id=user_id,
            industry=request.industry,
            style=request.style,
            keywords=request.keywords,
            count=request.count
        )
        
        return GenerateNamesResponse(
            success=True,
            names=result['names'],
            generation_id=result['generation_id'],
            count=result['count']
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except GenerationError as e:
        logger.error(f"Name generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate names"
        )


@router.post("/domains", response_model=CheckDomainsResponse)
async def check_domains(
    request: CheckDomainsRequest,
    user_id: str = Depends(get_current_user)
) -> CheckDomainsResponse:
    """Check domain availability"""
    try:
        async with DomainService() as service:
            result = await service.check_domains(
                domains=request.domains,
                user_id=user_id
            )
            
            return CheckDomainsResponse(
                success=True,
                results=result['results'],
                generation_id=result['generation_id'],
                total_checked=result['total_checked'],
                available_count=result['available_count']
            )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Domain check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check domains"
        )


@router.post("/logos", response_model=GenerateLogosResponse)
async def generate_logos(
    request: GenerateLogosRequest,
    user_id: str = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
) -> GenerateLogosResponse:
    """Generate logo concepts"""
    try:
        service = LogoService()
        result = await service.generate(
            description=request.description,
            user_id=user_id,
            business_name=request.business_name,
            style=request.style,
            colors=request.colors,
            industry=request.industry,
            count=request.count
        )
        
        return GenerateLogosResponse(
            success=True,
            logos=result['logos'],
            generation_id=result['generation_id'],
            business_name=result['business_name'],
            count=result['count']
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except GenerationError as e:
        logger.error(f"Logo generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate logos"
        )


@router.post("/colors", response_model=GenerateColorsResponse)
async def generate_colors(
    request: GenerateColorsRequest,
    user_id: str = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
) -> GenerateColorsResponse:
    """Generate color palettes"""
    try:
        service = ColorService()
        result = await service.generate(
            description=request.description,
            user_id=user_id,
            business_name=request.business_name,
            industry=request.industry,
            theme=request.theme,
            logo_colors=request.logo_colors,
            count=request.count
        )
        
        return GenerateColorsResponse(
            success=True,
            palettes=result['palettes'],
            generation_id=result['generation_id'],
            count=result['count']
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except GenerationError as e:
        logger.error(f"Color generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate color palettes"
        )


@router.post("/taglines", response_model=GenerateTaglinesResponse)
async def generate_taglines(
    request: GenerateTaglinesRequest,
    user_id: str = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
) -> GenerateTaglinesResponse:
    """Generate taglines"""
    try:
        service = TaglineService()
        result = await service.generate(
            description=request.description,
            user_id=user_id,
            business_name=request.business_name,
            tone=request.tone,
            industry=request.industry,
            target_audience=request.target_audience,
            keywords=request.keywords,
            count=request.count
        )
        
        return GenerateTaglinesResponse(
            success=True,
            taglines=result['taglines'],
            generation_id=result['generation_id'],
            business_name=result['business_name'],
            count=result['count']
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except GenerationError as e:
        logger.error(f"Tagline generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate taglines"
        )


@router.post("/package", response_model=GeneratePackageResponse)
async def generate_package(
    request: GeneratePackageRequest,
    user_id: str = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
) -> GeneratePackageResponse:
    """Generate complete brand package"""
    try:
        service = PackageService()
        result = await service.generate(
            description=request.description,
            user_id=user_id,
            business_name=request.business_name,
            industry=request.industry,
            style_preferences=request.style_preferences,
            include_services=request.include_services
        )
        
        return GeneratePackageResponse(
            success=True,
            project_id=result['project_id'],
            generation_id=result['generation_id'],
            business_name=result['business_name'],
            names=result.get('names'),
            domains=result.get('domains'),
            logos=result.get('logos'),
            color_palettes=result.get('color_palettes'),
            taglines=result.get('taglines'),
            summary=result['summary'],
            errors=result.get('errors', []),
            total_cost=result['total_cost'],
            created_at=result['created_at']
        )
        
    except RateLimitExceeded as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "rate_limit_exceeded",
                "message": str(e)
            }
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except GenerationError as e:
        logger.error(f"Package generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate package"
        )

@router.post("/domains/generate", response_model=CheckDomainsResponse)
async def generate_domains(
    request: GenerateDomainsRequest,
    user_id: str = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
) -> CheckDomainsResponse:
    """Generate domain variations with AI and check availability"""
    try:
        async with DomainService() as service:
            logger.info(f"🚀 Generating domains for: {request.business_name}")
            
            result = await service.generate(
                description=request.description,
                user_id=user_id,
                business_name=request.business_name
            )
            
            return CheckDomainsResponse(
                success=True,
                results=result['results'],
                generation_id=result['generation_id'],
                total_checked=result['total_checked'],
                available_count=result['available_count'],
                rounds=result.get('rounds', 1)  # ✅ ADD THIS
            )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Domain generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate domains: {str(e)}"
        )

@router.post("/domains/find-available", response_model=CheckDomainsResponse)
async def find_available_domains(
    request: CheckDomainsRequest,
    user_id: str = Depends(get_current_user)
) -> CheckDomainsResponse:
    """Check domain availability"""
    try:
        async with DomainService() as service:  # ✅ ADD async with
            logger.info(f"Domain APIs configured: {service.domain_apis}")
            result = await service.check_domains(
                domains=request.domains,
                user_id=user_id
            )
            
            return CheckDomainsResponse(
                success=True,
                results=result['results'],
                generation_id=result['generation_id'],
                total_checked=result['total_checked'],
                available_count=result['available_count']
            )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Domain check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check domains"
        )

@router.post("/domains/regenerate", response_model=CheckDomainsResponse)
async def regenerate_domains(
    request: Dict[str, Any],  # Use Dict for flexibility
    user_id: str = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
) -> CheckDomainsResponse:
    """Regenerate domains with feedback"""
    try:
        async with DomainService() as service:
            logger.info(f"🔄 Regenerating domains for: {request.get('business_name')}")
            
            result = await service.regenerate(
                business_name=request.get('business_name'),
                description=request.get('description', ''),
                feedback=request.get('feedback', ''),
                exclude_domains=request.get('exclude_domains', []),
                user_id=user_id
            )
            
            return CheckDomainsResponse(
                success=True,
                results=result['results'],
                generation_id=result['generation_id'],
                total_checked=result['total_checked'],
                available_count=result['available_count'],
                rounds=result.get('rounds', 1)
            )
        
    except Exception as e:
        logger.error(f"Domain regeneration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to regenerate domains: {str(e)}"
        )

@router.post("/regenerate/{component}")
async def regenerate_component(
    component: str,
    request: RegenerateRequest,
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """Regenerate a specific component with feedback"""
    
    valid_components = ['name', 'logo', 'color', 'tagline', 'package']
    if component not in valid_components:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid component. Must be one of: {valid_components}"
        )
    
    try:
        # Get appropriate service
        services = {
            'name': NameService(),
            'logo': LogoService(),
            'color': ColorService(),
            'tagline': TaglineService(),
            'package': PackageService()
        }
        
        service = services[component]
        
        # Regenerate
        if component == 'package':
            result = await service.regenerate(
                generation_id=request.generation_id,
                feedback=request.feedback,
                user_id=user_id,
                service_to_regenerate=request.service_to_regenerate
            )
        else:
            result = await service.regenerate(
                generation_id=request.generation_id,
                feedback=request.feedback,
                user_id=user_id
            )
        
        return {
            "success": True,
            "component": component,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Regeneration failed for {component}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to regenerate {component}"
        )

@router.post("/analyze/preferences", response_model=AnalyzePreferencesResponse)
async def analyze_preferences(
    request: AnalyzePreferencesRequest,
    user_id: str = Depends(get_current_user)
) -> AnalyzePreferencesResponse:
    """AI analyzes business and suggests preferences"""
    try:
        from core.ai_manager import AIManager
        ai_manager = AIManager()
        
        if request.for_type == "logo":
            prompt = f"""Analyze this business and suggest logo design preferences.
Business Name: {request.business_name}
Description: {request.description}

Return JSON with: style (modern/classic/playful/minimalist/bold/elegant), 
colors (2-3 hex codes), icon_type (abstract/literal/lettermark/wordmark), 
reasoning (2-3 sentences)"""
        
        result = await ai_manager.generate_text(
            prompt=prompt,
            model="claude-3-5-sonnet",
            temperature=0.7
        )
        
        import json
        try:
            suggestions = json.loads(result)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse AI response: {result[:100]}")
            suggestions = {
                "style": "modern",
                "colors": ["#2E86AB", "#FFFFFF"],
                "icon_type": "abstract",
                "reasoning": "Using default preferences due to parsing error"
            }
        
        return AnalyzePreferencesResponse(
            success=True,
            style=suggestions.get("style", "modern"),
            colors=suggestions.get("colors", ["#000000", "#FFFFFF"]),
            reasoning=suggestions.get("reasoning", ""),
            suggestions=suggestions
        )
    except Exception as e:
        logger.error(f"Preferences analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze preferences"
        )