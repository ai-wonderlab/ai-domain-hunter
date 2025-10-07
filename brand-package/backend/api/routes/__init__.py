"""
API Routes Package
"""
from fastapi import APIRouter

# Import all routers
from api.routes.auth_routes import router as auth_router
from api.routes.generation_routes import router as generation_router
from api.routes.user_routes import router as user_router
from api.routes.project_routes import router as project_router
from api.routes.health_routes import router as health_router

# Create main API router
api_router = APIRouter()

# Include all sub-routers
api_router.include_router(
    auth_router,
    prefix="/auth",
    tags=["authentication"]
)

api_router.include_router(
    generation_router,
    prefix="/generation",
    tags=["generation"]
)

api_router.include_router(
    user_router,
    prefix="/user",
    tags=["user"]
)

api_router.include_router(
    project_router,
    prefix="/projects",
    tags=["projects"]
)

api_router.include_router(
    health_router,
    prefix="/health",
    tags=["health"]
)

__all__ = [
    "api_router",
    "auth_router",
    "generation_router",
    "user_router",
    "project_router",
    "health_router"
]