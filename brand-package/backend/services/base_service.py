"""
Base Service Class
"""
from typing import Optional, Dict, Any
import logging
from abc import ABC, abstractmethod
from datetime import datetime
import uuid

from core.ai_manager import AIManager
from core.research_loader import ResearchLoader
from database.client import get_supabase
from core.exceptions import GenerationError

logger = logging.getLogger(__name__)


class BaseService(ABC):
    """
    Abstract base class for all generation services
    """
    
    def __init__(self, ai_manager=None, research_loader=None, db=None, db_client=None):
        """Initialize base service with optional dependency injection for testing"""
        self.ai = ai_manager if ai_manager is not None else AIManager.get_instance()
        self.research_loader = research_loader if research_loader is not None else ResearchLoader.get_instance()
        self.db = db if db is not None else (db_client if db_client is not None else get_supabase())
        self.service_name = self.__class__.__name__
        self.service_name = self.__class__.__name__
        
    @abstractmethod
    async def generate(
        self,
        description: str,
        user_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Abstract generate method that all services must implement
        
        Args:
            description: Business description
            user_id: User identifier
            **kwargs: Service-specific parameters
            
        Returns:
            Generation results
        """
        pass
    
    @abstractmethod
    async def regenerate(
        self,
        generation_id: str,
        feedback: str,
        user_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Regenerate with user feedback
        
        Args:
            generation_id: Original generation ID
            feedback: User feedback for improvement
            user_id: User identifier
            **kwargs: Additional parameters
            
        Returns:
            New generation results
        """
        pass
    
    async def save_generation(
        self,
        user_id: str,
        generation_type: str,
        input_data: Dict,
        output_data: Dict,
        project_id: Optional[str] = None,
        cost: float = 0.0
    ) -> str:
        """
        Save generation to database
        
        Args:
            user_id: User ID
            generation_type: Type of generation
            input_data: Input parameters
            output_data: Generated results
            project_id: Optional project ID
            cost: Generation cost
            
        Returns:
            Generation ID
        """
        try:
            generation_id = str(uuid.uuid4())
            
            result = self.db.table('generations').insert({
                'id': generation_id,
                'user_id': user_id,
                'project_id': project_id,
                'type': generation_type,
                'input_data': input_data,
                'output_data': output_data,
                'ai_model_used': self._get_models_used(),
                'cost': cost,
                'created_at': datetime.now().isoformat()
            }).execute()
            
            logger.info(f"✅ Saved generation {generation_id} for user {user_id}")
            return generation_id
            
        except Exception as e:
            logger.error(f"Failed to save generation: {e}")
            raise GenerationError(f"Failed to save generation: {str(e)}")
    
    async def get_generation(self, generation_id: str, user_id: str) -> Dict:
        """
        Get generation from database
        
        Args:
            generation_id: Generation ID
            user_id: User ID for authorization
            
        Returns:
            Generation data
        """
        try:
            result = self.db.table('generations').select("*").eq(
                'id', generation_id
            ).eq(
                'user_id', user_id
            ).execute()
            
            if not result.data:
                raise GenerationError(f"Generation {generation_id} not found")
            
            return result.data[0]
            
        except Exception as e:
            logger.error(f"Failed to get generation: {e}")
            raise GenerationError(f"Failed to get generation: {str(e)}")
    
    def get_patterns(self, pattern_type: str) -> str:
        """
        Get research patterns for generation
        
        Args:
            pattern_type: Type of pattern (name, logo, etc)
            
        Returns:
            Pattern content
        """
        return self.research_loader.get_pattern(pattern_type)
    
    def _get_models_used(self) -> str:
        """Get string of models used"""
        models = []
        
        # Get text models
        text_models = self.ai.text_models
        if text_models:
            models.append(f"text:{list(text_models.keys())[0]}")
        
        # Get image models
        if self.ai.image_clients:
            providers = [c["provider"] for c in self.ai.image_clients[:2]]
            models.extend([f"image:{p}" for p in providers])
        
        return ", ".join(models)
    
    def _extract_json_from_response(self, response: str) -> Any:
        """
        Extract JSON from AI response
        
        Args:
            response: AI response text
            
        Returns:
            Parsed JSON object
        """
        import json
        import re
        
        # Try direct parsing first
        try:
            return json.loads(response)
        except:
            pass
        
        # Try to find JSON in response
        json_patterns = [
            r'\[.*?\]',  # Array
            r'\{.*?\}',  # Object
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue
        
        # Fallback - try to clean and parse
        cleaned = response.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        
        try:
            return json.loads(cleaned.strip())
        except:
            logger.warning(f"Could not parse JSON from response: {response[:100]}...")
            return None
    
    async def check_usage_limit(self, user_id: str) -> bool:
        """
        Check if user has reached usage limit
        
        Args:
            user_id: User ID
            
        Returns:
            True if within limit
        """
        try:
            # Get usage tracking
            result = self.db.table('usage_tracking').select("*").eq(
                'user_id', user_id
            ).execute()
            
            if not result.data:
                # Create usage record
                self.db.table('usage_tracking').insert({
                    'user_id': user_id,
                    'generation_count': 0
                }).execute()
                return True
            
            usage = result.data[0]
            from config.constants import FREE_TIER_GENERATIONS
            
            return usage['generation_count'] < FREE_TIER_GENERATIONS
            
        except Exception as e:
            logger.error(f"Failed to check usage limit: {e}")
            return True  # Allow in case of error
    
    async def increment_usage(self, user_id: str):
        """
        Increment usage count
        
        Args:
            user_id: User ID
        """
        try:
            # Get current count
            result = self.db.table('usage_tracking').select("*").eq(
                'user_id', user_id
            ).execute()
            
            if result.data:
                current_count = result.data[0]['generation_count']
                
                # Update count
                self.db.table('usage_tracking').update({
                    'generation_count': current_count + 1
                }).eq('user_id', user_id).execute()
            else:
                # Create with count 1
                self.db.table('usage_tracking').insert({
                    'user_id': user_id,
                    'generation_count': 1
                }).execute()
                
        except Exception as e:
            logger.error(f"Failed to increment usage: {e}")