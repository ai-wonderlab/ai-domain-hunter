"""
Auth Routes - Supabase Auth Integration
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr
import logging

from database.client import get_supabase

logger = logging.getLogger(__name__)
router = APIRouter()


class SignUpRequest(BaseModel):
    email: EmailStr
    password: str


class SignInRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    success: bool = True
    user: dict
    access_token: str
    refresh_token: str


@router.post("/signup", response_model=AuthResponse)
async def sign_up(request: SignUpRequest):
    """Sign up with email and password"""
    db = get_supabase()
    
    try:
        # ✅ Use Supabase Auth
        response = db.auth.sign_up({
            "email": request.email,
            "password": request.password
        })
        
        if not response.user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Registration failed"
            )
        
        return AuthResponse(
            user={
                "id": response.user.id,
                "email": response.user.email
            },
            access_token=response.session.access_token,
            refresh_token=response.session.refresh_token
        )
        
    except Exception as e:
        logger.error(f"Signup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/login", response_model=AuthResponse)
async def login(request: SignInRequest):
    """Login with email and password"""
    db = get_supabase()
    
    try:
        # ✅ Use Supabase Auth
        response = db.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password
        })
        
        if not response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        return AuthResponse(
            user={
                "id": response.user.id,
                "email": response.user.email
            },
            access_token=response.session.access_token,
            refresh_token=response.session.refresh_token
        )
        
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )


@router.post("/logout")
async def logout():
    """Logout (client-side token removal)"""
    db = get_supabase()
    
    try:
        db.auth.sign_out()
        return {"success": True, "message": "Logged out"}
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return {"success": True, "message": "Logged out"}


@router.post("/refresh")
async def refresh_token(refresh_token: str):
    """Refresh access token"""
    db = get_supabase()
    
    try:
        response = db.auth.refresh_session(refresh_token)
        
        return {
            "access_token": response.session.access_token,
            "refresh_token": response.session.refresh_token
        }
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )