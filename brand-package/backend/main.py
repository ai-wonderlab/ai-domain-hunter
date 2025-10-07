"""
FastAPI Main Application Entry Point
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import logging
from typing import Dict

from config.settings import settings
from config.logging_config import setup_logging
from core.ai_manager import AIManager
from core.research_loader import ResearchLoader
from database.client import init_supabase, close_supabase
from middleware import setup_middleware
from api.routes import (
    health_routes,
    auth_routes,
    generation_routes,
    user_routes,
    project_routes
)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle - startup and shutdown
    """
    # Startup
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.app_env}")
    
    try:
        # Initialize database
        await init_supabase()
        logger.info("✅ Database connected")
        
        # Initialize AI Manager
        AIManager.initialize()
        logger.info("✅ AI Manager initialized")
        
        # Load research patterns
        ResearchLoader.load_all_patterns()
        logger.info("✅ Research patterns loaded")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        raise
    finally:
        # Shutdown
        logger.info("🛑 Shutting down...")
        await close_supabase()
        await AIManager.cleanup()
        logger.info("✅ Cleanup complete")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered brand package generation API",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# Setup middleware (includes CORS, error handlers, rate limiting, etc.)
setup_middleware(app)

# Include routers
app.include_router(health_routes.router, prefix="/api/health", tags=["Health"])
app.include_router(auth_routes.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(generation_routes.router, prefix="/api/generate", tags=["Generation"])
app.include_router(user_routes.router, prefix="/api/users", tags=["Users"])
app.include_router(project_routes.router, prefix="/api/projects", tags=["Projects"])

# Root endpoint
@app.get("/")
async def root() -> Dict:
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs" if settings.debug else "disabled"
    }

# Internal error handler (500 errors)
@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 errors"""
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )
