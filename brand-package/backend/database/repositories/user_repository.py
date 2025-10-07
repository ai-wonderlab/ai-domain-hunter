"""
User Repository
"""
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from database.repositories.base_repository import BaseRepository
from core.exceptions import DuplicateRecordError

logger = logging.getLogger(__name__)


class UserRepository(BaseRepository):
    """Repository for user operations"""
    
    def __init__(self):
        super().__init__("users")
    
    async def get_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email
        
        Args:
            email: User email
            
        Returns:
            User record or None
        """
        try:
            result = self.db.table(self.table_name).select("*").eq('email', email).execute()
            
            if result.data:
                return result.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Get by email failed: {e}")
            return None
    
    async def create_user(
        self,
        email: str,
        google_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a new user
        
        Args:
            email: User email
            google_id: Google OAuth ID
            **kwargs: Additional user data
            
        Returns:
            Created user
        """
        # Check if user exists
        existing = await self.get_by_email(email)
        if existing:
            raise DuplicateRecordError("User", f"User with email {email} already exists")
        
        user_data = {
            "email": email,
            "google_id": google_id,
            "created_at": datetime.now().isoformat(),
            **kwargs
        }
        
        return await self.create(user_data)
    
    async def get_usage(self, user_id: str) -> Dict[str, Any]:
        """Get user usage statistics
        
        Args:
            user_id: User ID
            
        Returns:
            Usage statistics
        """
        try:
            result = self.db.table('usage_tracking').select("*").eq('user_id', user_id).execute()
            
            if result.data:
                return result.data[0]
            
            # Create default usage record
            usage = {
                'user_id': user_id,
                'generation_count': 0,
                'created_at': datetime.now().isoformat()
            }
            
            self.db.table('usage_tracking').insert(usage).execute()
            return usage
            
        except Exception as e:
            logger.error(f"Get usage failed: {e}")
            return {'generation_count': 0}
    
    async def increment_usage(self, user_id: str) -> Dict[str, Any]:
        """Increment user generation count
        
        Args:
            user_id: User ID
            
        Returns:
            Updated usage
        """
        usage = await self.get_usage(user_id)
        new_count = usage.get('generation_count', 0) + 1
        
        try:
            result = self.db.table('usage_tracking').update({
                'generation_count': new_count,
                'last_used_at': datetime.now().isoformat()
            }).eq('user_id', user_id).execute()
            
            if result.data:
                return result.data[0]
            return {'generation_count': new_count}
            
        except Exception as e:
            logger.error(f"Increment usage failed: {e}")
            return {'generation_count': new_count}