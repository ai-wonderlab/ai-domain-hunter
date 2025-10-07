"""
Supabase Database Client
"""
import os
from typing import Optional
from supabase import create_client, Client
from supabase.client import ClientOptions
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

# Global client instance
_supabase_client: Optional[Client] = None


class SupabaseClient:
    """Singleton Supabase client wrapper"""
    
    _instance: Optional[Client] = None
    
    @classmethod
    def get_client(cls) -> Client:
        """Get or create Supabase client instance"""
        global _supabase_client
        
        if _supabase_client is None:
            _supabase_client = cls._create_client()
        
        return _supabase_client
    
    @classmethod
    def _create_client(cls) -> Client:
        """Create new Supabase client"""
        try:
            # Use service key for server-side operations
            client = create_client(
                supabase_url=settings.supabase_url,
                supabase_key=settings.supabase_service_key,
                options=ClientOptions(
                    auto_refresh_token=True,
                    persist_session=False
                )
            )
            
            logger.info("✅ Supabase client created")
            return client
            
        except Exception as e:
            logger.error(f"❌ Failed to create Supabase client: {e}")
            raise
    
    @classmethod
    async def close(cls):
        """Close Supabase client connection"""
        global _supabase_client
        
        if _supabase_client:
            # Supabase client doesn't have explicit close, but we can cleanup
            _supabase_client = None
            logger.info("✅ Supabase client closed")


async def init_supabase() -> Client:
    """Initialize Supabase connection"""
    client = SupabaseClient.get_client()
    
    # Test connection
    try:
        # Try a simple query to verify connection
        result = client.table('projects').select("id").limit(1).execute()
        logger.info(f"✅ Supabase connection verified")
        return client
    except Exception as e:
        logger.error(f"❌ Supabase connection test failed: {e}")
        raise


async def close_supabase():
    """Close Supabase connection"""
    await SupabaseClient.close()


def get_supabase() -> Client:
    """
    Dependency to get Supabase client
    Use this in FastAPI dependency injection
    """
    return SupabaseClient.get_client()


# Storage utilities
class SupabaseStorage:
    """Utilities for Supabase Storage operations"""
    
    @staticmethod
    def get_public_url(bucket: str, path: str) -> str:
        """Get public URL for a storage object"""
        client = get_supabase()
        return client.storage.from_(bucket).get_public_url(path)
    
    @staticmethod
    async def upload_file(
        bucket: str,
        path: str,
        file_data: bytes,
        content_type: str = "application/octet-stream"
    ) -> str:
        """Upload file to Supabase Storage"""
        client = get_supabase()
        
        try:
            # Upload file
            response = client.storage.from_(bucket).upload(
                path,
                file_data,
                {"content-type": content_type}
            )
            
            # Return public URL
            return SupabaseStorage.get_public_url(bucket, path)
            
        except Exception as e:
            logger.error(f"Failed to upload file to {bucket}/{path}: {e}")
            raise
    
    @staticmethod
    async def delete_file(bucket: str, path: str) -> bool:
        """Delete file from Supabase Storage"""
        client = get_supabase()
        
        try:
            response = client.storage.from_(bucket).remove([path])
            return True
        except Exception as e:
            logger.error(f"Failed to delete file {bucket}/{path}: {e}")
            return False
    
    @staticmethod
    async def list_files(bucket: str, prefix: str = "") -> list:
        """List files in a bucket with optional prefix"""
        client = get_supabase()
        
        try:
            response = client.storage.from_(bucket).list(prefix)
            return response
        except Exception as e:
            logger.error(f"Failed to list files in {bucket}/{prefix}: {e}")
            return []


# Database transaction utilities
class DatabaseTransaction:
    """Context manager for database transactions"""
    
    def __init__(self):
        self.client = get_supabase()
    
    async def __aenter__(self):
        """Start transaction"""
        # Supabase doesn't have explicit transactions in the client
        # but we can implement optimistic locking or use RLS
        return self.client
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """End transaction"""
        if exc_type:
            logger.error(f"Transaction failed: {exc_val}")
            # Handle rollback logic if needed
        return False


# Exports
__all__ = [
    "init_supabase",
    "close_supabase",
    "get_supabase",
    "SupabaseClient",
    "SupabaseStorage",
    "DatabaseTransaction"
]