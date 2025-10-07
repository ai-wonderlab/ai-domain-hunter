"""
Project Repository
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import uuid

from database.repositories.base_repository import BaseRepository
from core.exceptions import NotFoundError, DuplicateRecordError

logger = logging.getLogger(__name__)


class ProjectRepository(BaseRepository):
    """Repository for project operations"""
    
    def __init__(self):
        super().__init__("projects")
    
    async def create_project(
        self,
        user_id: str,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new project
        
        Args:
            user_id: User ID
            name: Project name
            description: Project description
            metadata: Additional metadata
            
        Returns:
            Created project
        """
        # Check for duplicate name for user
        existing = await self.get_by_name(user_id, name)
        if existing:
            raise DuplicateRecordError(
                "Project",
                f"Project '{name}' already exists"
            )
        
        project_data = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "name": name,
            "description": description,
            "status": "draft",
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        return await self.create(project_data)
    
    async def get_user_projects(
        self,
        user_id: str,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get projects for a user
        
        Args:
            user_id: User ID
            status: Filter by status
            limit: Maximum records
            offset: Skip records
            
        Returns:
            List of projects
        """
        filters = {"user_id": user_id}
        
        if status:
            filters["status"] = status
        
        return await self.get_all(
            filters=filters,
            limit=limit,
            offset=offset,
            order_by="-updated_at"
        )
    
    async def get_by_name(
        self,
        user_id: str,
        name: str
    ) -> Optional[Dict[str, Any]]:
        """Get project by name for user
        
        Args:
            user_id: User ID
            name: Project name
            
        Returns:
            Project or None
        """
        try:
            result = self.db.table(self.table_name).select("*").eq(
                "user_id", user_id
            ).eq(
                "name", name
            ).execute()
            
            if result.data:
                return result.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Failed to get project by name: {e}")
            return None
    
    async def update_project(
        self,
        project_id: str,
        user_id: str,
        updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update project
        
        Args:
            project_id: Project ID
            user_id: User ID (for verification)
            updates: Update data
            
        Returns:
            Updated project or None
        """
        # Verify ownership
        existing = await self.get_by_id(project_id)
        if not existing or existing.get("user_id") != user_id:
            raise NotFoundError("Project", project_id)
        
        # Don't allow changing user_id or id
        updates.pop("user_id", None)
        updates.pop("id", None)
        
        return await self.update(project_id, updates)
    
    async def delete_project(
        self,
        project_id: str,
        user_id: str,
        cascade: bool = True
    ) -> bool:
        """Delete project
        
        Args:
            project_id: Project ID
            user_id: User ID (for verification)
            cascade: Delete related generations
            
        Returns:
            Success status
        """
        # Verify ownership
        existing = await self.get_by_id(project_id)
        if not existing or existing.get("user_id") != user_id:
            raise NotFoundError("Project", project_id)
        
        if cascade:
            # Delete related generations
            self.db.table("generations").delete().eq(
                "project_id", project_id
            ).execute()
            
            # Delete related assets
            self.db.table("assets").delete().eq(
                "project_id", project_id
            ).execute()
        
        return await self.delete(project_id)
    
    async def get_project_statistics(
        self,
        project_id: str
    ) -> Dict[str, Any]:
        """Get project statistics
        
        Args:
            project_id: Project ID
            
        Returns:
            Statistics dictionary
        """
        try:
            # Get generation count by type
            generations = self.db.table("generations").select("*").eq(
                "project_id", project_id
            ).execute()
            
            stats = {
                "total_generations": len(generations.data),
                "generation_types": {},
                "total_cost": 0.0,
                "last_activity": None
            }
            
            for gen in generations.data:
                gen_type = gen.get("type", "unknown")
                if gen_type not in stats["generation_types"]:
                    stats["generation_types"][gen_type] = 0
                stats["generation_types"][gen_type] += 1
                stats["total_cost"] += gen.get("cost", 0.0)
                
                # Track last activity
                created_at = gen.get("created_at")
                if created_at and (not stats["last_activity"] or created_at > stats["last_activity"]):
                    stats["last_activity"] = created_at
            
            # Get asset count
            assets = self.db.table("assets").select(
                "id", count="exact"
            ).eq(
                "project_id", project_id
            ).execute()
            
            stats["total_assets"] = assets.count or 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get project statistics: {e}")
            return {
                "total_generations": 0,
                "generation_types": {},
                "total_cost": 0.0,
                "total_assets": 0,
                "last_activity": None
            }
    
    async def duplicate_project(
        self,
        project_id: str,
        user_id: str,
        new_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Duplicate a project
        
        Args:
            project_id: Source project ID
            user_id: User ID
            new_name: New project name
            
        Returns:
            New project
        """
        # Get original project
        original = await self.get_by_id(project_id)
        if not original or original.get("user_id") != user_id:
            raise NotFoundError("Project", project_id)
        
        # Create new project
        new_name = new_name or f"{original['name']} (Copy)"
        
        return await self.create_project(
            user_id=user_id,
            name=new_name,
            description=original.get("description"),
            metadata=original.get("metadata", {})
        )