"""
Asynchronous Generation Tasks
"""
from typing import Dict, Any, Optional
import asyncio
import logging
from celery import current_task

from workers.celery_app import app
from services.name_service import NameService
from services.logo_service import LogoService
from services.color_service import ColorService
from services.tagline_service import TaglineService
from services.package_service import PackageService
from core.ai_manager import AIManager
from database.client import get_supabase

logger = logging.getLogger(__name__)


@app.task(bind=True, max_retries=3)
def generate_logo_async(
    self,
    user_id: str,
    business_name: str,
    description: str,
    **kwargs
) -> Dict[str, Any]:
    """Generate logo asynchronously
    
    Args:
        user_id: User ID
        business_name: Business name
        description: Business description
        **kwargs: Additional parameters
        
    Returns:
        Generation result
    """
    try:
        # Update task state
        current_task.update_state(
            state='PROGRESS',
            meta={'status': 'Initializing logo generation...'}
        )
        
        # Initialize service
        service = LogoService()
        
        # Update progress
        current_task.update_state(
            state='PROGRESS',
            meta={'status': 'Generating logo concepts...'}
        )
        
        # Run generation
        result = asyncio.run(
            service.generate(
                description=description,
                user_id=user_id,
                business_name=business_name,
                **kwargs
            )
        )
        
        # Update task state
        current_task.update_state(
            state='SUCCESS',
            meta={'status': 'Logo generation complete', 'result': result}
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Logo generation task failed: {e}")
        
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))


@app.task(bind=True, max_retries=3)
def generate_package_async(
    self,
    user_id: str,
    description: str,
    **kwargs
) -> Dict[str, Any]:
    """Generate complete brand package asynchronously
    
    Args:
        user_id: User ID
        description: Business description
        **kwargs: Additional parameters
        
    Returns:
        Package result
    """
    try:
        # Update task state with progress
        stages = [
            'Initializing package generation...',
            'Generating business names...',
            'Checking domain availability...',
            'Creating logo concepts...',
            'Designing color palettes...',
            'Writing taglines...',
            'Finalizing package...'
        ]
        
        current_stage = 0
        
        def update_progress(message: str):
            nonlocal current_stage
            current_task.update_state(
                state='PROGRESS',
                meta={
                    'status': message,
                    'progress': (current_stage / len(stages)) * 100
                }
            )
            current_stage += 1
        
        # Initialize
        update_progress(stages[0])
        service = PackageService()
        
        # Run generation with progress updates
        # Note: This is simplified - in production you'd want to
        # integrate progress updates within the service
        update_progress(stages[1])
        
        result = asyncio.run(
            service.generate(
                description=description,
                user_id=user_id,
                **kwargs
            )
        )
        
        # Final update
        current_task.update_state(
            state='SUCCESS',
            meta={
                'status': 'Package generation complete',
                'progress': 100,
                'result': result
            }
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Package generation task failed: {e}")
        raise self.retry(exc=e, countdown=120 * (2 ** self.request.retries))


@app.task
def regenerate_component_async(
    component_type: str,
    generation_id: str,
    feedback: str,
    user_id: str
) -> Dict[str, Any]:
    """Regenerate a specific component with feedback
    
    Args:
        component_type: Type of component
        generation_id: Original generation ID
        feedback: User feedback
        user_id: User ID
        
    Returns:
        Regeneration result
    """
    try:
        # Map component types to services
        services = {
            'name': NameService(),
            'logo': LogoService(),
            'color': ColorService(),
            'tagline': TaglineService()
        }
        
        service = services.get(component_type)
        if not service:
            raise ValueError(f"Unknown component type: {component_type}")
        
        # Run regeneration
        result = asyncio.run(
            service.regenerate(
                generation_id=generation_id,
                feedback=feedback,
                user_id=user_id
            )
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Regeneration task failed: {e}")
        raise


@app.task
def batch_generate_names(
    descriptions: list,
    user_id: str,
    **kwargs
) -> list:
    """Generate names for multiple businesses
    
    Args:
        descriptions: List of business descriptions
        user_id: User ID
        **kwargs: Additional parameters
        
    Returns:
        List of generation results
    """
    results = []
    service = NameService()
    
    for i, description in enumerate(descriptions):
        try:
            # Update progress
            current_task.update_state(
                state='PROGRESS',
                meta={
                    'status': f'Processing {i+1}/{len(descriptions)}',
                    'progress': (i / len(descriptions)) * 100
                }
            )
            
            # Generate names
            result = asyncio.run(
                service.generate(
                    description=description,
                    user_id=user_id,
                    **kwargs
                )
            )
            
            results.append({
                'success': True,
                'result': result
            })
            
        except Exception as e:
            logger.error(f"Batch generation failed for item {i}: {e}")
            results.append({
                'success': False,
                'error': str(e)
            })
    
    return results


@app.task
def warmup_ai_models():
    """Warm up AI models for faster response times"""
    try:
        ai = AIManager.get_instance()
        
        # Send a simple request to warm up text model
        asyncio.run(
            ai.generate_text(
                "Hello",
                temperature=0.1,
                max_tokens=10
            )
        )
        
        logger.info("✅ AI models warmed up")
        
    except Exception as e:
        logger.error(f"Model warmup failed: {e}")