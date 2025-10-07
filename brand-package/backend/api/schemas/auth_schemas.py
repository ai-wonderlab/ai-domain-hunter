"""
Isolated Auth Schemas - No dependencies
"""
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class SimpleRegisterRequest(BaseModel):
    email: str
    full_name: Optional[str] = None
    google_id: Optional[str] = None

class SimpleLoginRequest(BaseModel):
    email: str
    google_token: Optional[str] = None

class SimpleUserInfo(BaseModel):
    id: str
    email: str
    created_at: datetime
    generation_count: int = 0
    generation_limit: int = 2

class SimpleLoginResponse(BaseModel):
    success: bool = True
    user: SimpleUserInfo
    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 86400
