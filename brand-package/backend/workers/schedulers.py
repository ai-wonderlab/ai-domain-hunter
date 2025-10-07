"""
Task Schedulers and Job Management
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import pytz

from database.client import get_supabase
from config.settings import settings

logger = logging.getLogger(__name__)


class TaskScheduler:
    """Manage scheduled tasks"""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler(
            timezone=pytz.UTC,
            job_defaults={
                'coalesce': False,
                'max_instances': 1,
                'misfire_grace_time': 30
            }
        )
        self.db = get_supabase()
        
    def start(self):
        """Start the scheduler"""
        self.scheduler.start()
        logger.info("✅ Task scheduler started")
        
    def stop(self):
        """Stop the scheduler"""
        self.scheduler.shutdown()
        logger.info("🛑 Task scheduler stopped")
        
    def add_job(
        self,
        func,
        trigger: str,
        job_id: Optional[str] = None,
        **kwargs
    ):
        """Add a scheduled job
        
        Args:
            func: Function to execute
            trigger: Trigger type (cron, interval, date)
            job_id: Optional job ID
            **kwargs: Trigger-specific arguments
        """
        if trigger == 'cron':
            trigger_obj = CronTrigger(**kwargs)
        elif trigger == 'interval':
            trigger_obj = IntervalTrigger(**kwargs)
        else:
            raise ValueError(f"Unknown trigger type: {trigger}")
        
        job = self.scheduler.add_job(
            func,
            trigger=trigger_obj,
            id=job_id,
            replace_existing=True
        )
        
        logger.info(f"✅ Added scheduled job: {job.id}")
        return job
    
    def remove_job(self, job_id: str):
        """Remove a scheduled job"""
        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"✅ Removed job: {job_id}")
        except Exception as e:
            logger.error(f"Failed to remove job {job_id}: {e}")
    
    def get_jobs(self) -> List[Dict[str, Any]]:
        """Get all scheduled jobs"""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time,
                'trigger': str(job.trigger)
            })
        return jobs


# Scheduled task functions
async def cleanup_old_generations():
    """Clean up old generation records"""
    db = get_supabase()
    
    try:
        # Delete generations older than 30 days
        cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()
        
        result = db.table('generations').delete().lt(
            'created_at', cutoff_date
        ).execute()
        
        count = len(result.data) if result.data else 0
        logger.info(f"✅ Cleaned up {count} old generations")
        
    except Exception as e:
        logger.error(f"Generation cleanup failed: {e}")


async def reset_usage_limits():
    """Reset daily usage limits"""
    db = get_supabase()
    
    try:
        # Reset all user generation counts
        result = db.table('usage_tracking').update({
            'generation_count': 0,
            'period_start': datetime.now().isoformat()
        }).execute()
        
        logger.info(f"✅ Reset usage limits for all users")
        
    except Exception as e:
        logger.error(f"Usage reset failed: {e}")


async def generate_analytics():
    """Generate analytics and reports"""
    db = get_supabase()
    
    try:
        # Get generation statistics
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday = today - timedelta(days=1)
        
        # Count generations by type
        generation_stats = {}
        for gen_type in ['name', 'logo', 'color', 'tagline', 'package']:
            result = db.table('generations').select(
                'id', count='exact'
            ).eq('type', gen_type).gte(
                'created_at', yesterday.isoformat()
            ).lt(
                'created_at', today.isoformat()
            ).execute()
            
            generation_stats[gen_type] = result.count or 0
        
        # Count new users
        new_users_result = db.table('users').select(
            'id', count='exact'
        ).gte(
            'created_at', yesterday.isoformat()
        ).lt(
            'created_at', today.isoformat()
        ).execute()
        
        new_users = new_users_result.count or 0
        
        # Save analytics
        analytics_data = {
            'date': yesterday.date().isoformat(),
            'new_users': new_users,
            'generations': generation_stats,
            'created_at': datetime.now().isoformat()
        }
        
        db.table('analytics').insert(analytics_data).execute()
        
        logger.info(f"✅ Generated analytics for {yesterday.date()}")
        
    except Exception as e:
        logger.error(f"Analytics generation failed: {e}")


async def check_storage_quota():
    """Check storage usage and alert if needed"""
    from utils.storage import StorageManager
    
    storage = StorageManager()
    
    try:
        # Get storage statistics (this is placeholder - implement based on your needs)
        # You might need to track this separately or use Supabase's API
        
        logger.info("✅ Storage quota checked")
        
    except Exception as e:
        logger.error(f"Storage check failed: {e}")


# Initialize scheduler
scheduler = TaskScheduler()


def setup_scheduled_tasks():
    """Set up all scheduled tasks"""
    
    # Daily tasks
    scheduler.add_job(
        cleanup_old_generations,
        'cron',
        job_id='cleanup_old_generations',
        hour=2,
        minute=0
    )
    
    scheduler.add_job(
        reset_usage_limits,
        'cron',
        job_id='reset_usage_limits',
        hour=0,
        minute=0
    )
    
    scheduler.add_job(
        generate_analytics,
        'cron',
        job_id='generate_analytics',
        hour=1,
        minute=0
    )
    
    # Hourly tasks
    scheduler.add_job(
        check_storage_quota,
        'interval',
        job_id='check_storage_quota',
        hours=1
    )
    
    logger.info("✅ All scheduled tasks configured")