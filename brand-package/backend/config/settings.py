"""
Application settings using Pydantic for validation - Agnostic version
"""
from typing import List, Optional, Dict
from pydantic_settings import BaseSettings
from functools import lru_cache
import os
from pathlib import Path

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings with environment variable validation"""
    
    # Application
    app_env: str = "development"
    app_name: str = "Brand Package Generator"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # Supabase
    supabase_url: str
    supabase_key: str
    supabase_service_key: str
    
    # Text Generation API (Primary)
    text_api_provider: str = "openrouter"
    text_api_key: str
    text_api_url: str = "https://openrouter.ai/api/v1"
    
    # Text Generation Models (up to 5)
    text_model_1: Optional[str] = None
    text_model_2: Optional[str] = None
    text_model_3: Optional[str] = None
    text_model_4: Optional[str] = None
    text_model_5: Optional[str] = None
    
    # Image Generation APIs (up to 3)
    # API 1 (Primary)
    image_api_1_provider: Optional[str] = None
    image_api_1_key: Optional[str] = None
    image_api_1_url: Optional[str] = None
    image_api_1_model: Optional[str] = None
    
    # API 2 (Backup)
    image_api_2_provider: Optional[str] = None
    image_api_2_key: Optional[str] = None
    image_api_2_url: Optional[str] = None
    image_api_2_model: Optional[str] = None
    
    # API 3 (Tertiary)
    image_api_3_provider: Optional[str] = None
    image_api_3_key: Optional[str] = None
    image_api_3_url: Optional[str] = None
    image_api_3_model: Optional[str] = None
    
    # Domain Checking APIs
    # API 1
    domain_api_1_provider: Optional[str] = None
    domain_api_1_key: Optional[str] = None
    domain_api_1_url: Optional[str] = None
    
    # API 2
    domain_api_2_provider: Optional[str] = None
    domain_api_2_key: Optional[str] = None
    domain_api_2_url: Optional[str] = None
    
    # Storage
    storage_bucket: str = "brand-assets"
    cdn_url: Optional[str] = None
    
    # Rate Limiting
    rate_limit_generations: int = 2
    rate_limit_window_minutes: int = 1440
    
    # Security
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440
    
    # Redis
    redis_url: Optional[str] = None
    
    # Celery
    celery_broker_url: Optional[str] = None
    celery_result_backend: Optional[str] = None
    
    # Frontend
    frontend_url: str = "http://localhost:3000"
    allowed_origins: List[str] = ["http://localhost:3000", "http://localhost:8001"]
    
    # Analytics
    analytics_provider: Optional[str] = None
    analytics_api_key: Optional[str] = None
    monitoring_provider: Optional[str] = None
    monitoring_dsn: Optional[str] = None
    
    # Feature Flags
    enable_logo_generation: bool = True
    enable_color_generation: bool = True
    enable_tagline_generation: bool = True
    enable_domain_checking: bool = True
    enable_name_generation: bool = True
    # Middleware feature flags
    enable_rate_limiting: bool = False  # Disabled by default in dev
    enable_request_logging: bool = True
    
    
    # Paths
    @property
    def research_dir(self) -> Path:
        return BASE_DIR / "research" / "documents"
    
    @property
    def cache_dir(self) -> Path:
        return BASE_DIR / "research" / "cache"
    
    @property
    def temp_dir(self) -> Path:
        temp = BASE_DIR / "temp"
        temp.mkdir(exist_ok=True)
        return temp
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def is_development(self) -> bool:
        return self.app_env == "development"
    
    def is_production(self) -> bool:
        return self.app_env == "production"
    
    def get_text_models(self) -> Dict[str, str]:
        """Get all configured text models"""
        models = {}
        for i in range(1, 6):
            model = getattr(self, f"text_model_{i}", None)
            if model:
                models[f"model_{i}"] = model
        return models
    
    def get_image_apis(self) -> List[Dict]:
        """Get all configured image APIs"""
        apis = []
        for i in range(1, 4):
            provider = getattr(self, f"image_api_{i}_provider", None)
            if provider:
                apis.append({
                    "provider": provider,
                    "key": getattr(self, f"image_api_{i}_key"),
                    "url": getattr(self, f"image_api_{i}_url"),
                    "model": getattr(self, f"image_api_{i}_model")
                })
        return apis
    
    def get_domain_apis(self) -> List[Dict]:
        """Get all configured domain checking APIs"""
        apis = []
        for i in range(1, 3):
            provider = getattr(self, f"domain_api_{i}_provider", None)
            if provider:
                apis.append({
                    "provider": provider,
                    "key": getattr(self, f"domain_api_{i}_key"),
                    "url": getattr(self, f"domain_api_{i}_url")
                })
        return apis
    
    def has_image_api(self) -> bool:
        """Check if at least one image API is configured"""
        return bool(self.image_api_1_provider)
    
    def has_domain_api(self) -> bool:
        """Check if domain checking APIs are configured"""
        return bool(self.domain_api_1_provider)
    
    def has_text_models(self) -> bool:
        """Check if text models are configured"""
        return bool(self.text_model_1)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Create a global settings instance
settings = get_settings()

