"""
Asset Repository
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import uuid

from database.repositories.base_repository import BaseRepository
from utils.storage import StorageManager
from core.exceptions import NotFoundError, DatabaseError

logger = logging.getLogger(__name__)


class AssetRepository(BaseRepository):
    """Repository for asset operations"""
    
    def __init__(self):
        super().__init__("assets")
        self.storage = StorageManager()
    
    async def create_asset(
        self,
        user_id: str,
        file_path: str,
        file_url: str,
        file_type: str,
        file_size: int,
        generation_id: Optional[str] = None,
        project_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new asset record
        
        Args:
            user_id: User ID
            file_path: Path in storage
            file_url: Public URL
            file_type: MIME type
            file_size: Size in bytes
            generation_id: Related generation ID
            project_id: Related project ID
            metadata: Additional metadata
            
        Returns:
            Created asset record
        """
        asset_data = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "generation_id": generation_id,
            "project_id": project_id,
            "file_path": file_path,
            "file_url": file_url,
            "file_type": file_type,
            "file_size": file_size,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat()
        }
        
        return await self.create(asset_data)
    
    async def get_user_assets(
        self,
        user_id: str,
        file_type: Optional[str] = None,
        project_id: Optional[str] = None,
        generation_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get assets for a user
        
        Args:
            user_id: User ID
            file_type: Filter by file type
            project_id: Filter by project
            generation_id: Filter by generation
            limit: Maximum records
            offset: Skip records
            
        Returns:
            List of assets
        """
        filters = {"user_id": user_id}
        
        if file_type:
            filters["file_type"] = file_type
        
        if project_id:
            filters["project_id"] = project_id
        
        if generation_id:
            filters["generation_id"] = generation_id
        
        return await self.get_all(
            filters=filters,
            limit=limit,
            offset=offset,
            order_by="-created_at"
        )
    
    async def delete_asset(
        self,
        asset_id: str,
        user_id: str,
        delete_file: bool = True
    ) -> bool:
        """Delete asset and optionally its file
        
        Args:
            asset_id: Asset ID
            user_id: User ID (for verification)
            delete_file: Delete from storage
            
        Returns:
            Success status
        """
        try:
            # Get asset
            asset = await self.get_by_id(asset_id)
            if not asset or asset.get("user_id") != user_id:
                raise NotFoundError("Asset", asset_id)
            
            # Delete from storage if requested
            if delete_file and asset.get("file_path"):
                try:
                    await self.storage.delete_file(asset["file_path"])
                except Exception as e:
                    logger.warning(f"Failed to delete file: {e}")
            
            # Delete from database
            return await self.delete(asset_id)
            
        except Exception as e:
            logger.error(f"Failed to delete asset: {e}")
            return False
    
    async def get_storage_usage(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """Get storage usage for user
        
        Args:
            user_id: User ID
            
        Returns:
            Usage statistics
        """
        try:
            result = self.db.table(self.table_name).select("*").eq(
                "user_id", user_id
            ).execute()
            
            total_size = 0
            by_type = {}
            
            for asset in result.data:
                file_size = asset.get("file_size", 0)
                file_type = asset.get("file_type", "unknown")
                
                total_size += file_size
                
                if file_type not in by_type:
                    by_type[file_type] = {
                        "count": 0,
                        "size": 0
                    }
                
                by_type[file_type]["count"] += 1
                by_type[file_type]["size"] += file_size
            
            return {
                "total_count": len(result.data),
                "total_size": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "by_type": by_type
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage usage: {e}")
            return {
                "total_count": 0,
                "total_size": 0,
                "total_size_mb": 0,
                "by_type": {}
            }
    
    async def cleanup_orphaned_assets(self) -> int:
        """Delete assets without parent records
        
        Returns:
            Number of deleted assets
        """
        try:
            # Find orphaned assets
            assets = self.db.table(self.table_name).select("*").execute()
            
            deleted_count = 0
            
            for asset in assets.data:
                # Check if generation exists
                if asset.get("generation_id"):
                    gen = self.db.table("generations").select("id").eq(
                        "id", asset["generation_id"]
                    ).execute()
                    
                    if not gen.data:
                        # Orphaned - delete
                        await self.delete_asset(
                            asset["id"],
                            asset["user_id"],
                            delete_file=True
                        )
                        deleted_count += 1
            
            logger.info(f"Deleted {deleted_count} orphaned assets")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup orphaned assets: {e}")
            return 0
    
    async def copy_asset(
        self,
        asset_id: str,
        user_id: str,
        new_project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Copy an asset
        
        Args:
            asset_id: Source asset ID
            user_id: User ID
            new_project_id: Target project ID
            
        Returns:
            New asset record
        """
        try:
            # Get original asset
            original = await self.get_by_id(asset_id)
            if not original or original.get("user_id") != user_id:
                raise NotFoundError("Asset", asset_id)
            
            # Copy file in storage
            new_path = f"{user_id}/copies/{uuid.uuid4()}"
            success = await self.storage.copy_file(
                original["file_path"],
                new_path
            )
            
            if not success:
                raise DatabaseError("Failed to copy file in storage")
            
            # Get new URL
            new_url = await self.storage.get_file_url(new_path)
            
            # Create new asset record
            return await self.create_asset(
                user_id=user_id,
                file_path=new_path,
                file_url=new_url,
                file_type=original["file_type"],
                file_size=original["file_size"],
                generation_id=None,  # New copy not linked to original generation
                project_id=new_project_id,
                metadata={
                    **original.get("metadata", {}),
                    "copied_from": asset_id,
                    "copied_at": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to copy asset: {e}")
            raise DatabaseError(f"Failed to copy asset: {str(e)}")