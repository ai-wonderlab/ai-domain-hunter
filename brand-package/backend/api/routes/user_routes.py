"""
User Management Routes
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from api.schemas.response_schemas import (
    UserInfo,
    UserHistoryResponse,
    UserUsageResponse,
    UserHistoryItem
)
from api.schemas.base_schema import PaginationParams
from api.dependencies import get_current_user
from database.client import get_supabase
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/profile", response_model=UserInfo)
async def get_profile(
    user_id: str = Depends(get_current_user)
) -> UserInfo:
    """Get user profile"""
    db = get_supabase()
    
    try:
        # Get user
        user_result = db.table('users').select("*").eq(
            'id', user_id
        ).execute()
        
        if not user_result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        user = user_result.data[0]
        
        # Get usage
        usage_result = db.table('usage_tracking').select("*").eq(
            'user_id', user_id
        ).execute()
        
        generation_count = 0
        if usage_result.data:
            generation_count = usage_result.data[0].get('generation_count', 0)
        
        return UserInfo(
            id=user['id'],
            email=user['email'],
            created_at=user['created_at'],
            generation_count=generation_count,
            generation_limit=settings.rate_limit_generations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get profile"
        )


@router.put("/profile")
async def update_profile(
    update_data: Dict[str, Any],
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """Update user profile"""
    db = get_supabase()
    
    # Only allow certain fields to be updated
    allowed_fields = ['full_name', 'company_name', 'industry']
    filtered_data = {
        k: v for k, v in update_data.items() 
        if k in allowed_fields
    }
    
    if not filtered_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid fields to update"
        )
    
    try:
        result = db.table('users').update(
            filtered_data
        ).eq('id', user_id).execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "success": True,
            "message": "Profile updated",
            "updated_fields": list(filtered_data.keys())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )


@router.get("/history", response_model=UserHistoryResponse)
async def get_history(
    user_id: str = Depends(get_current_user),
    pagination: PaginationParams = Depends(),
    project_id: Optional[str] = None,
    generation_type: Optional[str] = None
) -> UserHistoryResponse:
    """Get user generation history"""
    db = get_supabase()
    
    try:
        # Get projects
        projects_query = db.table('projects').select("*").eq(
            'user_id', user_id
        ).order('created_at', desc=True)
        
        if project_id:
            projects_query = projects_query.eq('id', project_id)
        
        projects_result = projects_query.execute()
        
        # Get generations
        generations_query = db.table('generations').select("*").eq(
            'user_id', user_id
        ).order('created_at', desc=True)
        
        if project_id:
            generations_query = generations_query.eq('project_id', project_id)
        
        if generation_type:
            generations_query = generations_query.eq('type', generation_type)
        
        # Apply pagination
        generations_query = generations_query.range(
            pagination.offset,
            pagination.offset + pagination.per_page - 1
        )
        
        generations_result = generations_query.execute()
        
        # Format history items
        history_items = []
        for gen in generations_result.data:
            # Create preview based on type
            preview = {}
            output_data = gen.get('output_data', {})
            
            if gen['type'] == 'name' and 'names' in output_data:
                preview['names'] = [n['name'] for n in output_data['names'][:3]]
            elif gen['type'] == 'logo' and 'logos' in output_data:
                preview['logo_count'] = len(output_data['logos'])
            elif gen['type'] == 'tagline' and 'taglines' in output_data:
                preview['taglines'] = [t['text'] for t in output_data['taglines'][:2]]
            
            history_items.append(UserHistoryItem(
                id=gen['id'],
                type=gen['type'],
                created_at=gen['created_at'],
                input_data=gen.get('input_data', {}),
                preview=preview
            ))
        
        return UserHistoryResponse(
            success=True,
            projects=projects_result.data,
            recent_generations=history_items,
            total_count=len(history_items)
        )
        
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get history"
        )


@router.get("/usage", response_model=UserUsageResponse)
async def get_usage(
    user_id: str = Depends(get_current_user)
) -> UserUsageResponse:
    """Get user usage statistics"""
    db = get_supabase()
    
    try:
        # Get usage tracking
        result = db.table('usage_tracking').select("*").eq(
            'user_id', user_id
        ).execute()
        
        if not result.data:
            # Create usage record
            db.table('usage_tracking').insert({
                'user_id': user_id,
                'generation_count': 0,
                'period_start': datetime.now().isoformat()
            }).execute()
            
            generation_count = 0
            resets_at = None
        else:
            usage = result.data[0]
            generation_count = usage.get('generation_count', 0)
            
            # Calculate reset time (if implementing daily/monthly resets)
            period_start = usage.get('period_start')
            if period_start:
                # For MVP, no reset - just show None
                resets_at = None
            else:
                resets_at = None
        
        remaining = max(0, settings.rate_limit_generations - generation_count)
        
        return UserUsageResponse(
            success=True,
            generation_count=generation_count,
            limit=settings.rate_limit_generations,
            remaining=remaining,
            resets_at=resets_at,
            is_paid=False  # For future paid plans
        )
        
    except Exception as e:
        logger.error(f"Failed to get usage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get usage"
        )


@router.delete("/account")
async def delete_account(
    user_id: str = Depends(get_current_user),
    confirm: bool = Query(False, description="Confirm deletion")
) -> Dict[str, Any]:
    """
    Delete user account and all data
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please confirm account deletion"
        )
    
    db = get_supabase()
    
    try:
        # Delete in order due to foreign keys
        
        # 1. Delete assets
        db.table('assets').delete().eq('user_id', user_id).execute()
        
        # 2. Delete generations
        db.table('generations').delete().eq('user_id', user_id).execute()
        
        # 3. Delete projects
        db.table('projects').delete().eq('user_id', user_id).execute()
        
        # 4. Delete usage tracking
        db.table('usage_tracking').delete().eq('user_id', user_id).execute()
        
        # 5. Finally, delete user
        result = db.table('users').delete().eq('id', user_id).execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "success": True,
            "message": "Account deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete account: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete account"
        )


@router.post("/reset-usage")
async def reset_usage(
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Reset usage counter (admin only in production)
    For MVP, allow self-reset for testing
    """
    db = get_supabase()
    
    try:
        # In production, check if admin
        # For MVP, allow anyone to reset their own
        
        result = db.table('usage_tracking').update({
            'generation_count': 0,
            'last_reset_at': datetime.now().isoformat()
        }).eq('user_id', user_id).execute()
        
        if not result.data:
            # Create new record
            db.table('usage_tracking').insert({
                'user_id': user_id,
                'generation_count': 0,
                'last_reset_at': datetime.now().isoformat()
            }).execute()
        
        return {
            "success": True,
            "message": "Usage reset successfully",
            "generation_count": 0,
            "limit": settings.rate_limit_generations
        }
        
    except Exception as e:
        logger.error(f"Failed to reset usage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset usage"
        )