"""
Authentication Routes
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from datetime import datetime, timedelta
import jwt
import logging
from typing import Dict, Any
import httpx

from api.schemas.auth_schemas import SimpleLoginRequest as LoginRequest, SimpleRegisterRequest as RegisterRequest
from api.schemas.auth_schemas import SimpleLoginResponse as LoginResponse
from api.schemas.auth_schemas import SimpleUserInfo as UserInfo
from api.dependencies import security, get_current_user
from database.client import get_supabase
from config.settings import settings
from core.exceptions import (
    AuthenticationError,
    DatabaseError,
    DuplicateRecordError
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Dependency functions
def get_current_user_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """Get current token"""
    return credentials.credentials

def create_access_token(user_id: str, email: str) -> str:
    """Create JWT access token"""
    expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    payload = {
        "sub": user_id,
        "email": email,
        "exp": expire,
        "iat": datetime.utcnow()
    }
    
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)

    
    payload = {
        "sub": user_id,
        "email": email,
        "exp": expire,
        "iat": datetime.utcnow()
    }
    
    token = jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )
    
    return token


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest) -> LoginResponse:
    """
    Login with email or Google OAuth
    """
    db = get_supabase()
    
    try:
        # If Google token provided, verify with Google
        if request.google_token:
            # Verify Google token
            user_info = await verify_google_token(request.google_token)
            
            if not user_info:
                raise AuthenticationError("Invalid Google token")
            
            email = user_info.get('email')
            google_id = user_info.get('sub')
            
            # Check if user exists
            result = db.table('users').select("*").eq(
                'email', email
            ).execute()
            
            if result.data:
                # User exists, update Google ID if needed
                user = result.data[0]
                if not user.get('google_id'):
                    db.table('users').update({
                        'google_id': google_id
                    }).eq('id', user['id']).execute()
            else:
                # Create new user
                result = db.table('users').insert({
                    'email': email,
                    'google_id': google_id,
                    'created_at': datetime.now().isoformat()
                }).execute()
                
                user = result.data[0]
                
                # Initialize usage tracking
                db.table('usage_tracking').insert({
                    'user_id': user['id'],
                    'generation_count': 0
                }).execute()
        
        else:
            # Email-only login (simplified for MVP)
            result = db.table('users').select("*").eq(
                'email', request.email
            ).execute()
            
            if not result.data:
                # Auto-create user for MVP
                result = db.table('users').insert({
                    'email': request.email,
                    'created_at': datetime.now().isoformat()
                }).execute()
                
                user = result.data[0]
                
                # Initialize usage tracking
                db.table('usage_tracking').insert({
                    'user_id': user['id'],
                    'generation_count': 0
                }).execute()
            else:
                user = result.data[0]
        
        # Get usage info
        usage_result = db.table('usage_tracking').select("*").eq(
            'user_id', user['id']
        ).execute()
        
        generation_count = 0
        if usage_result.data:
            generation_count = usage_result.data[0].get('generation_count', 0)
        
        # Create access token
        access_token = create_access_token(user['id'], user['email'])
        
        # Create user info
        user_info = UserInfo(
            id=user['id'],
            email=user['email'],
            created_at=user['created_at'],
            generation_count=generation_count,
            generation_limit=settings.rate_limit_generations
        )
        
        return LoginResponse(
            success=True,
            user=user_info,
            access_token=access_token,
            token_type="Bearer",
            expires_in=settings.access_token_expire_minutes * 60
        )
        
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/register")
async def register(request: RegisterRequest):
    """
    Register new user
    """
    db = get_supabase()
    try:
        # Check if user exists
        existing = db.table('users').select("id").eq(
            'email', request.email
        ).execute()
        
        if existing.data:
            raise DuplicateRecordError(
                resource="User",
                message="Email already registered"
            )
        
        # Create user
        result = db.table('users').insert({
            'email': request.email,
            'google_id': request.google_id,
            'created_at': datetime.now().isoformat()
        }).execute()
        
        user = result.data[0]
        
        # Initialize usage tracking
        db.table('usage_tracking').insert({
            'user_id': user['id'],
            'generation_count': 0
        }).execute()
        
        # Create access token
        access_token = create_access_token(user['id'], user['email'])
        
        # Create user info
        user_info = UserInfo(
            id=user['id'],
            email=user['email'],
            created_at=user['created_at'],
            generation_count=0,
            generation_limit=settings.rate_limit_generations
        )
        
        return LoginResponse(
            success=True,
            user=user_info,
            access_token=access_token,
            token_type="Bearer",
            expires_in=settings.access_token_expire_minutes * 60
        )
        
    except DuplicateRecordError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/refresh")
async def refresh_token(
) -> Dict[str, Any]:
    """
    Refresh access token
    """
    try:
        # Decode current token (even if expired, for user info)
        payload = jwt.decode(
            current_token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            options={"verify_exp": False}  # Don't verify expiration
        )
        
        user_id = payload.get("sub")
        email = payload.get("email")
        
        if not user_id or not email:
            raise AuthenticationError("Invalid token")
        
        # Create new token
        new_token = create_access_token(user_id, email)
        
        return {
            "access_token": new_token,
            "token_type": "Bearer",
            "expires_in": settings.access_token_expire_minutes * 60
        }
        
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to refresh token"
        )


@router.post("/logout")
async def logout(
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Logout user (client should discard token)
    """
    # For JWT, logout is handled client-side
    # Here we could blacklist the token if needed
    
    return {
        "success": True,
        "message": "Logged out successfully"
    }


@router.post("/verify")
async def verify_token(
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Verify if token is valid
    """
    db = get_supabase()
    
    try:
        # Get user info
        result = db.table('users').select("*").eq(
            'id', user_id
        ).execute()
        
        if not result.data:
            raise AuthenticationError("User not found")
        
        user = result.data[0]
        
        return {
            "valid": True,
            "user_id": user['id'],
            "email": user['email']
        }
        
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


async def verify_google_token(token: str) -> Dict[str, Any]:
    """
    Verify Google OAuth token
    """
    try:
        # Verify with Google's tokeninfo endpoint
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://oauth2.googleapis.com/tokeninfo?id_token={token}"
            )
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            # Verify audience (client ID) if configured
            # if settings.google_client_id and data.get('aud') != settings.google_client_id:
            #     return None
            
            return data
            
    except Exception as e:
        logger.error(f"Google token verification failed: {e}")
        return None