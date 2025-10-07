"""
Generation Repository
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging
import uuid

from database.repositories.base_repository import BaseRepository
from core.exceptions import NotFoundError, DatabaseError

logger = logging.getLogger(__name__)


class GenerationRepository(BaseRepository):
    """Repository for generation operations"""
    
    def __init__(self):
        super().__init__("generations")
    
    async def create_generation(
        self,
        user_id: str,
        generation_type: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        project_id: Optional[str] = None,
        cost: float = 0.0,
        ai_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new generation record
        
        Args:
            user_id: User ID
            generation_type: Type of generation
            input_data: Input parameters
            output_data: Generated results
            project_id: Optional project ID
            cost: Generation cost
            ai_model: AI model used
            
        Returns:
            Created generation record
        """
        generation_data = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "project_id": project_id,
            "type": generation_type,
            "input_data": input_data,
            "output_data": output_data,
            "cost": cost,
            "ai_model_used": ai_model,
            "created_at": datetime.now().isoformat()
        }
        
        return await self.create(generation_data)
    
    async def get_user_generations(
        self,
        user_id: str,
        generation_type: Optional[str] = None,
        project_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get generations for a user
        
        Args:
            user_id: User ID
            generation_type: Filter by type
            project_id: Filter by project
            limit: Maximum records
            offset: Skip records
            
        Returns:
            List of generations
        """
        filters = {"user_id": user_id}
        
        if generation_type:
            filters["type"] = generation_type
        
        if project_id:
            filters["project_id"] = project_id
        
        return await self.get_all(
            filters=filters,
            limit=limit,
            offset=offset,
            order_by="-created_at"
        )
    
    async def get_generation_stats(
        self,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get generation statistics
        
        Args:
            user_id: Filter by user
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            Statistics dictionary
        """
        try:
            query = self.db.table(self.table_name).select("*")
            
            if user_id:
                query = query.eq("user_id", user_id)
            
            if start_date:
                query = query.gte("created_at", start_date.isoformat())
            
            if end_date:
                query = query.lt("created_at", end_date.isoformat())
            
            result = query.execute()
            
            # Calculate statistics
            stats = {
                "total_count": len(result.data),
                "by_type": {},
                "total_cost": 0.0,
                "average_cost": 0.0
            }
            
            for gen in result.data:
                gen_type = gen.get("type", "unknown")
                if gen_type not in stats["by_type"]:
                    stats["by_type"][gen_type] = {
                        "count": 0,
                        "cost": 0.0
                    }
                
                stats["by_type"][gen_type]["count"] += 1
                stats["by_type"][gen_type]["cost"] += gen.get("cost", 0.0)
                stats["total_cost"] += gen.get("cost", 0.0)
            
            if stats["total_count"] > 0:
                stats["average_cost"] = stats["total_cost"] / stats["total_count"]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get generation stats: {e}")
            return {
                "total_count": 0,
                "by_type": {},
                "total_cost": 0.0,
                "average_cost": 0.0
            }
    
    async def get_recent_generations(
        self,
        limit: int = 10,
        generation_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent generations across all users
        
        Args:
            limit: Maximum records
            generation_type: Filter by type
            
        Returns:
            List of recent generations
        """
        filters = {}
        if generation_type:
            filters["type"] = generation_type
        
        return await self.get_all(
            filters=filters,
            limit=limit,
            order_by="-created_at"
        )
    
    async def cleanup_old_generations(
        self,
        days: int = 30,
        keep_favorites: bool = True
    ) -> int:
        """Delete old generations
        
        Args:
            days: Age in days
            keep_favorites: Keep favorited items
            
        Returns:
            Number of deleted records
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            query = self.db.table(self.table_name).delete().lt(
                "created_at", cutoff_date
            )
            
            if keep_favorites:
                query = query.neq("is_favorite", True)
            
            result = query.execute()
            
            deleted_count = len(result.data) if result.data else 0
            logger.info(f"Deleted {deleted_count} old generations")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup generations: {e}")
            return 0
    
    async def mark_as_favorite(
        self,
        generation_id: str,
        user_id: str,
        is_favorite: bool = True
    ) -> bool:
        """Mark generation as favorite
        
        Args:
            generation_id: Generation ID
            user_id: User ID (for verification)
            is_favorite: Favorite status
            
        Returns:
            Success status
        """
        try:
            # Verify ownership
            existing = await self.get_by_id(generation_id)
            if not existing or existing.get("user_id") != user_id:
                raise NotFoundError("Generation", generation_id)
            
            # Update favorite status
            result = await self.update(
                generation_id,
                {"is_favorite": is_favorite}
            )
            
            return result is not None
            
        except Exception as e:
            logger.error(f"Failed to mark favorite: {e}")
            return False