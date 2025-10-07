"""
Celery Application Configuration
"""
from celery import Celery
from celery.schedules import crontab
import logging
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)

# Create Celery app
app = Celery(
    'brand_generator',
    broker=settings.redis_url or 'redis://localhost:6379/0',
    backend=settings.redis_url or 'redis://localhost:6379/0',
    include=['workers.tasks.generation_tasks', 'workers.tasks.cleanup_tasks']
)

# Celery configuration
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task settings
    task_soft_time_limit=300,  # 5 minutes
    task_time_limit=600,  # 10 minutes
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Worker settings
    worker_prefetch_multiplier=2,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=False,
    
    # Result backend settings
    result_expires=3600,  # 1 hour
    result_compression='gzip',
    
    # Beat schedule for periodic tasks
    beat_schedule={
        'cleanup-old-files': {
            'task': 'workers.tasks.cleanup_tasks.cleanup_old_files',
            'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
        },
        'cleanup-expired-sessions': {
            'task': 'workers.tasks.cleanup_tasks.cleanup_expired_sessions',
            'schedule': crontab(minute='*/30'),  # Every 30 minutes
        },
        'generate-usage-reports': {
            'task': 'workers.tasks.cleanup_tasks.generate_usage_reports',
            'schedule': crontab(hour=0, minute=0, day_of_week=1),  # Weekly on Monday
        },
        'check-api-health': {
            'task': 'workers.tasks.cleanup_tasks.check_external_apis',
            'schedule': crontab(minute='*/15'),  # Every 15 minutes
        }
    },
    
    # Queue routing
    task_routes={
        'workers.tasks.generation_tasks.*': {'queue': 'generation'},
        'workers.tasks.cleanup_tasks.*': {'queue': 'maintenance'},
    },
    
    # Rate limits
    task_annotations={
        'workers.tasks.generation_tasks.generate_logo_async': {'rate_limit': '10/m'},
        'workers.tasks.generation_tasks.generate_package_async': {'rate_limit': '5/m'},
    }
)

# Initialize app on startup
@app.task
def test_task() -> str:
    """Test task to verify Celery is working"""
    return "Celery is working!"


def init_celery():
    """Initialize Celery app with Flask/FastAPI context if needed"""
    logger.info("✅ Celery app initialized")
    return app


# Error handling
@app.task(bind=True, max_retries=3)
def retry_task(self, func, *args, **kwargs):
    """Generic retry wrapper for tasks"""
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        logger.error(f"Task failed: {exc}, retrying...")
        raise self.retry(exc=exc, countdown=60)


if __name__ == '__main__':
    app.start()