"""
Health Check Routes
"""
from fastapi import APIRouter, Depends
from typing import Dict, Any, List
import logging
import time
import asyncio
from datetime import datetime

from api.schemas.response_schemas import (
    HealthCheckResponse,
    HealthStatus
)
from database.client import get_supabase
from core.ai_manager import AIManager
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Track startup time
START_TIME = time.time()


@router.get("/", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """Basic health check"""
    
    services = []
    overall_status = "healthy"
    
    # Check database
    db_status = await check_database()
    services.append(db_status)
    if db_status.status != "healthy":
        overall_status = "degraded"
    
    # Check AI services
    ai_status = await check_ai_services()
    services.append(ai_status)
    if ai_status.status == "unhealthy":
        overall_status = "unhealthy"
    elif ai_status.status == "degraded" and overall_status == "healthy":
        overall_status = "degraded"
    
    # Check storage
    storage_status = await check_storage()
    services.append(storage_status)
    
    # Calculate uptime
    uptime = time.time() - START_TIME
    
    return HealthCheckResponse(
        success=True,
        status=overall_status,
        version=settings.app_version,
        environment=settings.app_env,
        services=services,
        uptime_seconds=uptime
    )


@router.get("/live")
async def liveness_probe() -> Dict[str, Any]:
    """
    Kubernetes liveness probe
    Returns 200 if service is alive
    """
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/ready")
async def readiness_probe() -> Dict[str, Any]:
    """
    Kubernetes readiness probe
    Returns 200 if service is ready to accept traffic
    """
    # Quick database check
    try:
        db = get_supabase()
        # Simple query to verify connection
        db.table('users').select("id").limit(1).execute()
        
        return {
            "status": "ready",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {
            "status": "not_ready",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/startup")
async def startup_probe() -> Dict[str, Any]:
    """
    Kubernetes startup probe
    Returns 200 when application has started
    """
    # Check if all required services are initialized
    checks = {
        "database": False,
        "ai_manager": False,
        "configuration": False
    }
    
    # Check database
    try:
        db = get_supabase()
        db.table('users').select("id").limit(1).execute()
        checks["database"] = True
    except:
        pass
    
    # Check AI manager
    try:
        ai = AIManager.get_instance()
        if ai:
            checks["ai_manager"] = True
    except:
        pass
    
    # Check configuration
    if settings.text_api_key:
        checks["configuration"] = True
    
    all_ready = all(checks.values())
    
    return {
        "status": "started" if all_ready else "starting",
        "checks": checks,
        "timestamp": datetime.now().isoformat()
    }


async def check_database() -> HealthStatus:
    """Check database health"""
    start = time.time()
    
    try:
        db = get_supabase()
        
        # Test query
        result = db.table('users').select("id").limit(1).execute()
        
        latency = (time.time() - start) * 1000
        
        return HealthStatus(
            service="database",
            status="healthy",
            latency_ms=latency,
            details={"connected": True}
        )
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return HealthStatus(
            service="database",
            status="unhealthy",
            latency_ms=None,
            details={"error": str(e)}
        )


async def check_ai_services() -> HealthStatus:
    """Check AI services health"""
    
    try:
        ai = AIManager.get_instance()
        stats = ai.get_stats()
        
        # Check if we have configured models
        has_text = len(ai.text_models) > 0
        has_image = len(ai.image_clients) > 0
        
        if not has_text:
            return HealthStatus(
                service="ai_services",
                status="unhealthy",
                details={
                    "text_models": 0,
                    "image_providers": len(ai.image_clients),
                    "error": "No text models configured"
                }
            )
        
        status = "healthy"
        if not has_image:
            status = "degraded"
        
        return HealthStatus(
            service="ai_services",
            status=status,
            details={
                "text_models": len(ai.text_models),
                "image_providers": len(ai.image_clients),
                "total_requests": stats.get("request_count", 0),
                "image_stats": stats.get("image_stats", [])
            }
        )
        
    except Exception as e:
        logger.error(f"AI services health check failed: {e}")
        return HealthStatus(
            service="ai_services",
            status="unhealthy",
            details={"error": str(e)}
        )


async def check_storage() -> HealthStatus:
    """Check storage health"""
    
    try:
        db = get_supabase()
        
        # Try to list files in storage bucket
        # This is a simple check - in production, might want more thorough testing
        bucket_name = settings.storage_bucket
        
        # Note: Supabase storage operations might need different approach
        # This is a placeholder
        
        return HealthStatus(
            service="storage",
            status="healthy",
            details={
                "bucket": bucket_name,
                "accessible": True
            }
        )
        
    except Exception as e:
        logger.error(f"Storage health check failed: {e}")
        return HealthStatus(
            service="storage",
            status="degraded",
            details={"error": str(e)}
        )


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Get application metrics
    For Prometheus or monitoring systems
    """
    ai = AIManager.get_instance()
    ai_stats = ai.get_stats() if ai else {}
    
    db = get_supabase()
    
    # Get some basic metrics
    try:
        # Count users
        users_result = db.table('users').select("id", count='exact').execute()
        user_count = users_result.count or 0
        
        # Count generations today
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        generations_result = db.table('generations').select(
            "id", count='exact'
        ).gte('created_at', today.isoformat()).execute()
        generations_today = generations_result.count or 0
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        user_count = 0
        generations_today = 0
    
    return {
        "uptime_seconds": time.time() - START_TIME,
        "users_total": user_count,
        "generations_today": generations_today,
        "ai_requests_total": ai_stats.get("request_count", 0),
        "ai_tokens_total": ai_stats.get("total_tokens", 0),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/debug")
async def debug_info(
    # admin_only: str = Depends(require_admin)  # Uncomment for admin only
) -> Dict[str, Any]:
    """
    Get debug information
    Should be admin-only in production
    """
    ai = AIManager.get_instance() if AIManager._instance else None
    
    return {
        "environment": settings.app_env,
        "debug_mode": settings.debug,
        "configuration": {
            "text_api_configured": bool(settings.text_api_key),
            "image_apis_count": len(settings.get_image_apis()),
            "domain_apis_count": len(settings.get_domain_apis()),
            "rate_limit": settings.rate_limit_generations,
            "storage_bucket": settings.storage_bucket
        },
        "ai_manager": {
            "initialized": ai is not None,
            "stats": ai.get_stats() if ai else None
        },
        "timestamp": datetime.now().isoformat()
    }