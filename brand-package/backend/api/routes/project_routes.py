"""
Project Management Routes
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import uuid

from api.schemas.request_schemas import (
    CreateProjectRequest,
    UpdateProjectRequest
)
from api.schemas.response_schemas import (
    ProjectInfo,
    ProjectListResponse,
    ProjectDetailResponse
)
from api.schemas.base_schema import PaginationParams
from api.dependencies import get_current_user, get_project_access
from database.client import get_supabase

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=ProjectListResponse)
async def list_projects(
    user_id: str = Depends(get_current_user),
    pagination: PaginationParams = Depends(),
    status: Optional[str] = None
) -> ProjectListResponse:
    """List user projects"""
    db = get_supabase()
    
    try:
        # Build query
        query = db.table('projects').select("*").eq(
            'user_id', user_id
        ).order('created_at', desc=True)
        
        if status:
            query = query.eq('status', status)
        
        # Get total count
        count_result = query.execute()
        total = len(count_result.data)
        
        # Apply pagination
        query = query.range(
            pagination.offset,
            pagination.offset + pagination.per_page - 1
        )
        
        result = query.execute()
        
        # Format projects
        projects = []
        for proj in result.data:
            # Get generation count
            gen_count_result = db.table('generations').select(
                "id", count='exact'
            ).eq('project_id', proj['id']).execute()
            
            projects.append(ProjectInfo(
                id=proj['id'],
                user_id=proj['user_id'],
                name=proj['name'],
                description=proj.get('description'),
                status=proj['status'],
                created_at=proj['created_at'],
                updated_at=proj.get('updated_at', proj['created_at']),
                generation_count=gen_count_result.count or 0
            ))
        
        return ProjectListResponse(
            success=True,
            projects=projects,
            total=total
        )
        
    except Exception as e:
        logger.error(f"Failed to list projects: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list projects"
        )


@router.post("/", response_model=ProjectInfo)
async def create_project(
    request: CreateProjectRequest,
    user_id: str = Depends(get_current_user)
) -> ProjectInfo:
    """Create new project"""
    db = get_supabase()
    
    try:
        project_id = str(uuid.uuid4())
        
        result = db.table('projects').insert({
            'id': project_id,
            'user_id': user_id,
            'name': request.name,
            'description': request.description,
            'status': 'draft',
            'created_at': datetime.now().isoformat()
        }).execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create project"
            )
        
        project = result.data[0]
        
        return ProjectInfo(
            id=project['id'],
            user_id=project['user_id'],
            name=project['name'],
            description=project.get('description'),
            status=project['status'],
            created_at=project['created_at'],
            updated_at=project['created_at'],
            generation_count=0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create project: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create project"
        )


@router.get("/{project_id}", response_model=ProjectDetailResponse)
async def get_project(
    project_id: str,
    user_id: str = Depends(get_current_user)
) -> ProjectDetailResponse:
    """Get project details"""
    db = get_supabase()
    
    # Check access
    await get_project_access(project_id, user_id)
    
    try:
        # Get project
        project_result = db.table('projects').select("*").eq(
            'id', project_id
        ).execute()
        
        if not project_result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        
        project = project_result.data[0]
        
        # Get generations
        generations_result = db.table('generations').select("*").eq(
            'project_id', project_id
        ).order('created_at', desc=True).execute()
        
        # Get assets
        assets_result = db.table('assets').select("*").eq(
            'user_id', user_id
        ).execute()
        
        # Filter assets for this project's generations
        generation_ids = [g['id'] for g in generations_result.data]
        project_assets = [
            a for a in assets_result.data 
            if a.get('generation_id') in generation_ids
        ]
        
        # Format project info
        project_info = ProjectInfo(
            id=project['id'],
            user_id=project['user_id'],
            name=project['name'],
            description=project.get('description'),
            status=project['status'],
            created_at=project['created_at'],
            updated_at=project.get('updated_at', project['created_at']),
            generation_count=len(generations_result.data)
        )
        
        return ProjectDetailResponse(
            success=True,
            project=project_info,
            generations=generations_result.data,
            assets=project_assets
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get project: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get project"
        )


@router.put("/{project_id}")
async def update_project(
    project_id: str,
    request: UpdateProjectRequest,
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """Update project"""
    db = get_supabase()
    
    # Check access
    await get_project_access(project_id, user_id)
    
    # Build update data
    update_data = {}
    if request.name is not None:
        update_data['name'] = request.name
    if request.description is not None:
        update_data['description'] = request.description
    if request.status is not None:
        update_data['status'] = request.status
    
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update"
        )
    
    update_data['updated_at'] = datetime.now().isoformat()
    
    try:
        result = db.table('projects').update(
            update_data
        ).eq('id', project_id).execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        
        return {
            "success": True,
            "message": "Project updated",
            "updated_fields": list(update_data.keys())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update project: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update project"
        )


@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """Delete project and all associated data"""
    db = get_supabase()
    
    # Check access
    await get_project_access(project_id, user_id)
    
    try:
        # Get all generations for this project
        generations_result = db.table('generations').select("id").eq(
            'project_id', project_id
        ).execute()
        
        generation_ids = [g['id'] for g in generations_result.data]
        
        # Delete assets
        if generation_ids:
            db.table('assets').delete().in_(
                'generation_id', generation_ids
            ).execute()
        
        # Delete generations
        db.table('generations').delete().eq(
            'project_id', project_id
        ).execute()
        
        # Delete project
        result = db.table('projects').delete().eq(
            'id', project_id
        ).execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        
        return {
            "success": True,
            "message": "Project deleted",
            "deleted_generations": len(generation_ids)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete project: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete project"
        )


@router.post("/{project_id}/duplicate")
async def duplicate_project(
    project_id: str,
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """Duplicate a project"""
    db = get_supabase()
    
    # Check access
    await get_project_access(project_id, user_id)
    
    try:
        # Get original project
        original_result = db.table('projects').select("*").eq(
            'id', project_id
        ).execute()
        
        if not original_result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        
        original = original_result.data[0]
        
        # Create new project
        new_project_id = str(uuid.uuid4())
        
        new_project_result = db.table('projects').insert({
            'id': new_project_id,
            'user_id': user_id,
            'name': f"{original['name']} (Copy)",
            'description': original.get('description'),
            'status': 'draft',
            'created_at': datetime.now().isoformat()
        }).execute()
        
        # Duplicate generations (optional)
        # For MVP, we'll just copy the project shell
        
        return {
            "success": True,
            "message": "Project duplicated",
            "new_project_id": new_project_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to duplicate project: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to duplicate project"
        )