"""
API Dependencies with Supabase Auth
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
from typing import Optional

from database.client import get_supabase
from config.settings import settings

logger = logging.getLogger(__name__)
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """Verify Supabase JWT token and extract user ID"""
    token = credentials.credentials
    
    try:
        db = get_supabase()
        user_response = db.auth.get_user(token)
        
        if not user_response or not user_response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token"
            )
        
        user_id = user_response.user.id
        await sync_user_profile(user_id, user_response.user.email)
        
        return user_id
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


async def sync_user_profile(user_id: str, email: Optional[str]) -> None:
    """Sync auth.users → users table"""
    db = get_supabase()
    
    try:
        result = db.table('users').select("id").eq('id', user_id).execute()
        
        if not result.data:
            db.table('users').insert({
                'id': user_id,
                'email': email or 'unknown@example.com',
                'created_at': 'now()'
            }).execute()
            
            db.table('usage_tracking').insert({
                'user_id': user_id,
                'generation_count': 0
            }).execute()
            
            logger.info(f"✅ Synced new user: {email}")
    except Exception as e:
        logger.error(f"Failed to sync user: {e}")


async def check_rate_limit(user_id: str) -> None:
    """Check if user exceeded rate limit"""
    db = get_supabase()
    
    try:
        result = db.table('usage_tracking').select("*").eq(
            'user_id', user_id
        ).execute()
        
        if not result.data:
            db.table('usage_tracking').insert({
                'user_id': user_id,
                'generation_count': 0
            }).execute()
            return
        
        usage = result.data[0]
        generation_count = usage.get('generation_count', 0)
        
        if generation_count >= settings.rate_limit_generations:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. {generation_count}/{settings.rate_limit_generations} used."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rate limit check failed: {e}")


async def get_project_access(
    project_id: str,
    user_id: str = Depends(get_current_user)
) -> str:
    """Verify user has access to project"""
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
                detail="Project not found"
            )
        
        return project_id
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Project access check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify project access"
        )