"""
Storage Utilities for Supabase Storage
"""
from typing import Optional, Dict, Any, List
from pathlib import Path
import mimetypes
import logging
from datetime import datetime
import uuid

from database.client import get_supabase
from config.settings import settings
from utils.security import sanitize_filename

logger = logging.getLogger(__name__)


class StorageManager:
    """Manage file storage in Supabase Storage"""
    
    def __init__(self, bucket: Optional[str] = None):
        """Initialize storage manager
        
        Args:
            bucket: Storage bucket name
        """
        self.bucket = bucket or settings.storage_bucket
        self.client = get_supabase()
        
    async def upload_file(
        self,
        file_bytes: bytes,
        filename: str,
        folder: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Upload file to storage
        
        Args:
            file_bytes: File content
            filename: File name
            folder: Optional folder path
            content_type: MIME type
            
        Returns:
            Upload result with URL
        """
        try:
            # Sanitize filename
            safe_filename = sanitize_filename(filename)
            
            # Generate unique filename
            unique_id = str(uuid.uuid4())[:8]
            name, ext = safe_filename.rsplit('.', 1) if '.' in safe_filename else (safe_filename, '')
            unique_filename = f"{name}_{unique_id}.{ext}" if ext else f"{name}_{unique_id}"
            
            # Build path
            if folder:
                path = f"{folder}/{unique_filename}"
            else:
                path = unique_filename
            
            # Detect content type if not provided
            if not content_type:
                content_type, _ = mimetypes.guess_type(filename)
                content_type = content_type or 'application/octet-stream'
            
            # Upload to Supabase Storage
            response = self.client.storage.from_(self.bucket).upload(
                path,
                file_bytes,
                {
                    "content-type": content_type,
                    "x-upsert": "false"  # Don't overwrite existing
                }
            )
            
            # Get public URL
            url = self.client.storage.from_(self.bucket).get_public_url(path)
            
            return {
                "success": True,
                "path": path,
                "url": url,
                "size": len(file_bytes),
                "content_type": content_type,
                "filename": unique_filename
            }
            
        except Exception as e:
            logger.error(f"File upload failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def download_file(self, path: str) -> Optional[bytes]:
        """Download file from storage
        
        Args:
            path: File path in bucket
            
        Returns:
            File bytes or None
        """
        try:
            response = self.client.storage.from_(self.bucket).download(path)
            return response
        except Exception as e:
            logger.error(f"File download failed: {e}")
            return None
    
    async def delete_file(self, path: str) -> bool:
        """Delete file from storage
        
        Args:
            path: File path in bucket
            
        Returns:
            True if deleted
        """
        try:
            self.client.storage.from_(self.bucket).remove([path])
            return True
        except Exception as e:
            logger.error(f"File deletion failed: {e}")
            return False
    
    async def list_files(
        self,
        folder: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List files in bucket/folder
        
        Args:
            folder: Folder path
            limit: Maximum files to return
            
        Returns:
            List of file info
        """
        try:
            response = self.client.storage.from_(self.bucket).list(
                path=folder,
                options={
                    "limit": limit,
                    "sortBy": {"column": "created_at", "order": "desc"}
                }
            )
            
            files = []
            for item in response:
                files.append({
                    "name": item.get("name"),
                    "path": f"{folder}/{item.get('name')}" if folder else item.get("name"),
                    "size": item.get("metadata", {}).get("size"),
                    "content_type": item.get("metadata", {}).get("mimetype"),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at")
                })
            
            return files
            
        except Exception as e:
            logger.error(f"File listing failed: {e}")
            return []
    
    async def get_file_url(
        self,
        path: str,
        expires_in: Optional[int] = None
    ) -> Optional[str]:
        """Get public or signed URL for file
        
        Args:
            path: File path in bucket
            expires_in: Expiration time in seconds for signed URL
            
        Returns:
            File URL or None
        """
        try:
            if expires_in:
                # Get signed URL with expiration
                response = self.client.storage.from_(self.bucket).create_signed_url(
                    path,
                    expires_in
                )
                return response["signedURL"]
            else:
                # Get public URL
                return self.client.storage.from_(self.bucket).get_public_url(path)
                
        except Exception as e:
            logger.error(f"Failed to get file URL: {e}")
            return None
    
    async def move_file(
        self,
        source_path: str,
        dest_path: str
    ) -> bool:
        """Move/rename file in storage
        
        Args:
            source_path: Current file path
            dest_path: New file path
            
        Returns:
            True if moved
        """
        try:
            self.client.storage.from_(self.bucket).move(source_path, dest_path)
            return True
        except Exception as e:
            logger.error(f"File move failed: {e}")
            return False
    
    async def copy_file(
        self,
        source_path: str,
        dest_path: str
    ) -> bool:
        """Copy file in storage
        
        Args:
            source_path: Source file path
            dest_path: Destination file path
            
        Returns:
            True if copied
        """
        try:
            # Download and re-upload
            file_bytes = await self.download_file(source_path)
            if file_bytes:
                result = await self.upload_file(
                    file_bytes,
                    Path(dest_path).name,
                    folder=str(Path(dest_path).parent)
                )
                return result.get("success", False)
            return False
            
        except Exception as e:
            logger.error(f"File copy failed: {e}")
            return False


# Convenience functions
async def upload_image(
    image_bytes: bytes,
    user_id: str,
    image_type: str = "logo"
) -> Optional[str]:
    """Upload image and return URL
    
    Args:
        image_bytes: Image data
        user_id: User ID for organization
        image_type: Type of image (logo, asset, etc.)
        
    Returns:
        Public URL or None
    """
    storage = StorageManager()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{image_type}_{timestamp}.png"
    folder = f"{user_id}/{image_type}s"
    
    result = await storage.upload_file(
        image_bytes,
        filename,
        folder=folder,
        content_type="image/png"
    )
    
    if result.get("success"):
        return result.get("url")
    
    return None


async def cleanup_user_storage(user_id: str) -> int:
    """Clean up all storage for a user
    
    Args:
        user_id: User ID
        
    Returns:
        Number of files deleted
    """
    storage = StorageManager()
    
    # List all user files
    files = await storage.list_files(folder=user_id, limit=1000)
    
    deleted_count = 0
    for file in files:
        if await storage.delete_file(file["path"]):
            deleted_count += 1
    
    logger.info(f"Deleted {deleted_count} files for user {user_id}")
    return deleted_count