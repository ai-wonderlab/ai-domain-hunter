"""
Session Management Routes
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from api.dependencies import get_current_user
from database.client import get_supabase
from core.exceptions import NotFoundError, DatabaseError

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/save")
async def save_session(
    session_data: Dict[str, Any],
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Save or update user session
    
    Body:
        session_data: Complete session object (from frontend)
    
    Returns:
        Success message
    """
    db = get_supabase()
    
    try:
        session_id = session_data.get('sessionId')
        if not session_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="session_id is required"
            )
        
        # Upsert session
        result = db.table('sessions').upsert({
            'user_id': user_id,
            'session_id': session_id,
            'session_data': session_data,
            'updated_at': datetime.now().isoformat()
        }).execute()
        
        if not result.data:
            raise DatabaseError("Failed to save session")
        
        return {
            "success": True,
            "message": "Session saved",
            "session_id": session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save session"
        )


@router.get("/{session_id}")
async def load_session(
    session_id: str,
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Load a specific session
    
    Returns:
        Session data
    """
    db = get_supabase()
    
    try:
        result = db.table('sessions').select("*").eq(
            'session_id', session_id
        ).eq(
            'user_id', user_id
        ).single().execute()
        
        if not result.data:
            raise NotFoundError("Session", session_id)
        
        return {
            "success": True,
            "session": result.data['session_data']
        }
        
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    except Exception as e:
        logger.error(f"Failed to load session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load session"
        )


@router.get("/list")
async def list_sessions(
    user_id: str = Depends(get_current_user),
    limit: int = 10
) -> Dict[str, Any]:
    """
    List user's sessions
    
    Returns:
        List of sessions (metadata only)
    """
    db = get_supabase()
    
    try:
        result = db.table('sessions').select(
            "session_id, created_at, updated_at"
        ).eq(
            'user_id', user_id
        ).order(
            'updated_at', desc=True
        ).limit(limit).execute()
        
        return {
            "success": True,
            "sessions": result.data or [],
            "count": len(result.data) if result.data else 0
        }
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list sessions"
        )


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Delete a session
    
    Returns:
        Success message
    """
    db = get_supabase()
    
    try:
        result = db.table('sessions').delete().eq(
            'session_id', session_id
        ).eq(
            'user_id', user_id
        ).execute()
        
        if not result.data:
            raise NotFoundError("Session", session_id)
        
        return {
            "success": True,
            "message": "Session deleted"
        }
        
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete session"
        )