"""
Base Repository Pattern
"""
from typing import Generic, TypeVar, Optional, List, Dict, Any
from datetime import datetime
import logging
from supabase import Client

from database.client import get_supabase
from core.exceptions import DatabaseError, NotFoundError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseRepository(Generic[T]):
    """Base repository with common database operations"""
    
    def __init__(self, table_name: str):
        """Initialize repository
        
        Args:
            table_name: Name of the database table
        """
        self.table_name = table_name
        self.db: Client = get_supabase()
    
    async def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new record
        
        Args:
            data: Record data
            
        Returns:
            Created record
        """
        try:
            # Add timestamp if not present
            if 'created_at' not in data:
                data['created_at'] = datetime.now().isoformat()
            
            result = self.db.table(self.table_name).insert(data).execute()
            
            if not result.data:
                raise DatabaseError(f"Failed to create {self.table_name} record")
            
            return result.data[0]
            
        except Exception as e:
            logger.error(f"Create failed in {self.table_name}: {e}")
            raise DatabaseError(f"Database operation failed: {str(e)}")
    
    async def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """Get record by ID
        
        Args:
            id: Record ID
            
        Returns:
            Record or None
        """
        try:
            result = self.db.table(self.table_name).select("*").eq('id', id).execute()
            
            if result.data:
                return result.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Get by ID failed in {self.table_name}: {e}")
            raise DatabaseError(f"Database operation failed: {str(e)}")
    
    async def get_all(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all records with optional filtering
        
        Args:
            filters: Filter conditions
            limit: Maximum records to return
            offset: Number of records to skip
            order_by: Order by column
            
        Returns:
            List of records
        """
        try:
            query = self.db.table(self.table_name).select("*")
            
            # Apply filters
            if filters:
                for key, value in filters.items():
                    query = query.eq(key, value)
            
            # Apply ordering
            if order_by:
                desc = order_by.startswith('-')
                column = order_by[1:] if desc else order_by
                query = query.order(column, desc=desc)
            
            # Apply pagination
            if limit and offset is not None:
                query = query.range(offset, offset + limit - 1)
            elif limit:
                query = query.limit(limit)
            
            result = query.execute()
            return result.data or []
            
        except Exception as e:
            logger.error(f"Get all failed in {self.table_name}: {e}")
            raise DatabaseError(f"Database operation failed: {str(e)}")
    
    async def update(
        self,
        id: str,
        data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update a record
        
        Args:
            id: Record ID
            data: Update data
            
        Returns:
            Updated record or None
        """
        try:
            # Add updated timestamp
            data['updated_at'] = datetime.now().isoformat()
            
            result = self.db.table(self.table_name).update(data).eq('id', id).execute()
            
            if result.data:
                return result.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Update failed in {self.table_name}: {e}")
            raise DatabaseError(f"Database operation failed: {str(e)}")
    
    async def delete(self, id: str) -> bool:
        """Delete a record
        
        Args:
            id: Record ID
            
        Returns:
            True if deleted
        """
        try:
            result = self.db.table(self.table_name).delete().eq('id', id).execute()
            return len(result.data) > 0
            
        except Exception as e:
            logger.error(f"Delete failed in {self.table_name}: {e}")
            raise DatabaseError(f"Database operation failed: {str(e)}")
    
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count records
        
        Args:
            filters: Filter conditions
            
        Returns:
            Record count
        """
        try:
            query = self.db.table(self.table_name).select("id", count='exact')
            
            if filters:
                for key, value in filters.items():
                    query = query.eq(key, value)
            
            result = query.execute()
            return result.count or 0
            
        except Exception as e:
            logger.error(f"Count failed in {self.table_name}: {e}")
            raise DatabaseError(f"Database operation failed: {str(e)}")