"""
FastAPI Dependencies
"""
from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import jwt
import logging

from database.client import get_supabase
from config.settings import settings
from core.exceptions import (
    AuthenticationError,
    TokenExpiredError,
    InvalidTokenError,
    RateLimitExceeded
)

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> str:
    """
    Get current user from JWT token
    
    Returns:
        User ID
    """
    token = credentials.credentials
    
    try:
        # Decode JWT token
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        
        user_id = payload.get("sub")
        if not user_id:
            raise InvalidTokenError("Invalid token payload")
        
        return user_id  # ✅ ΠΡΟΣΘΕΣΕ ΑΥΤΟ!
                
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_optional_user(
    authorization: Optional[str] = Header(None)
) -> Optional[str]:
    """
    Get optional user from JWT token
    
    Returns:
        User ID or None
    """
    if not authorization:
        return None
    
    try:
        # Extract token from "Bearer <token>"
        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
        
        token = parts[1]
        
        # Decode JWT token
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        
        return payload.get("sub")
        
    except Exception:
        return None


async def check_rate_limit(
    user_id: str = Depends(get_current_user)
) -> None:
    """
    Check if user has exceeded rate limit
    
    Raises:
        HTTPException if rate limit exceeded
    """
    db = get_supabase()
    
    try:
        # Get user's usage
        result = db.table('usage_tracking').select("*").eq(
            'user_id', user_id
        ).execute()
        
        if not result.data:
            # No usage record yet, allow
            return
        
        usage = result.data[0]
        generation_count = usage.get('generation_count', 0)
        
        # Check limit
        if generation_count >= settings.rate_limit_generations:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": f"You have used all {settings.rate_limit_generations} generations",
                    "current_usage": generation_count,
                    "limit": settings.rate_limit_generations
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rate limit check failed: {e}")
        # Allow in case of error
        return


async def get_project_access(
    project_id: str,
    user_id: str = Depends(get_current_user)
) -> bool:
    """
    Check if user has access to project
    
    Returns:
        True if user has access
    """
    db = get_supabase()
    
    try:
        result = db.table('projects').select("id").eq(
            'id', project_id
        ).eq(
            'user_id', user_id
        ).execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found or access denied"
            )
        
        return True
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Project access check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check project access"
        )


# Admin check
async def require_admin(
    user_id: str = Depends(get_current_user)
) -> str:
    """
    Require admin user
    
    Returns:
        User ID if admin
    """
    # For now, simple email check
    # In production, add proper role management
    db = get_supabase()
    
    try:
        result = db.table('users').select("email").eq(
            'id', user_id
        ).execute()
        
        if result.data and result.data[0]['email'] in settings.admin_emails:
            return user_id
        
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify admin access"
        )


# Database dependency
async def get_db():
    """Get database client"""
    return get_supabase()