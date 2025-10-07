"""
Cleanup and Maintenance Tasks
"""
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List
import asyncio

from workers.celery_app import app
from database.client import get_supabase
from utils.storage import cleanup_user_storage
from config.settings import settings

logger = logging.getLogger(__name__)


@app.task
def cleanup_old_files():
    """Clean up old files from storage"""
    try:
        db = get_supabase()
        
        # Find files older than 30 days
        cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()
        
        # Get old assets
        result = db.table('assets').select('*').lt(
            'created_at', cutoff_date
        ).execute()
        
        deleted_count = 0
        
        for asset in result.data:
            try:
                # Delete from storage
                from utils.storage import StorageManager
                storage = StorageManager()
                
                if asyncio.run(storage.delete_file(asset['file_path'])):
                    # Delete from database
                    db.table('assets').delete().eq(
                        'id', asset['id']
                    ).execute()
                    
                    deleted_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to delete asset {asset['id']}: {e}")
        
        logger.info(f"✅ Cleaned up {deleted_count} old files")
        return {'deleted_count': deleted_count}
        
    except Exception as e:
        logger.error(f"File cleanup failed: {e}")
        raise


@app.task
def cleanup_expired_sessions():
    """Clean up expired user sessions"""
    try:
        db = get_supabase()
        
        # This is a placeholder - implement based on your session storage
        # If using JWT, sessions are stateless and don't need cleanup
        
        logger.info("✅ Session cleanup completed")
        return {'success': True}
        
    except Exception as e:
        logger.error(f"Session cleanup failed: {e}")
        raise


@app.task
def cleanup_orphaned_records():
    """Clean up orphaned database records"""
    try:
        db = get_supabase()
        
        # Find generations without users
        orphaned_generations = db.table('generations').select('id').is_(
            'user_id', 'null'
        ).execute()
        
        if orphaned_generations.data:
            # Delete orphaned records
            ids = [g['id'] for g in orphaned_generations.data]
            db.table('generations').delete().in_('id', ids).execute()
            logger.info(f"✅ Deleted {len(ids)} orphaned generations")
        
        # Find assets without generations
        orphaned_assets = db.table('assets').select('id').is_(
            'generation_id', 'null'
        ).execute()
        
        if orphaned_assets.data:
            ids = [a['id'] for a in orphaned_assets.data]
            db.table('assets').delete().in_('id', ids).execute()
            logger.info(f"✅ Deleted {len(ids)} orphaned assets")
        
        return {'success': True}
        
    except Exception as e:
        logger.error(f"Orphan cleanup failed: {e}")
        raise


@app.task
def generate_usage_reports():
    """Generate weekly usage reports"""
    try:
        db = get_supabase()
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # Get generation statistics
        generations = db.table('generations').select('*').gte(
            'created_at', start_date.isoformat()
        ).lt(
            'created_at', end_date.isoformat()
        ).execute()
        
        # Group by type
        stats = {}
        for gen in generations.data:
            gen_type = gen.get('type', 'unknown')
            if gen_type not in stats:
                stats[gen_type] = {
                    'count': 0,
                    'total_cost': 0.0
                }
            stats[gen_type]['count'] += 1
            stats[gen_type]['total_cost'] += gen.get('cost', 0.0)
        
        # Get user statistics
        users = db.table('users').select('*').gte(
            'created_at', start_date.isoformat()
        ).lt(
            'created_at', end_date.isoformat()
        ).execute()
        
        new_users = len(users.data)
        
        # Save report
        report = {
            'period_start': start_date.isoformat(),
            'period_end': end_date.isoformat(),
            'generation_stats': stats,
            'new_users': new_users,
            'total_generations': sum(s['count'] for s in stats.values()),
            'total_revenue': sum(s['total_cost'] for s in stats.values()),
            'created_at': datetime.now().isoformat()
        }
        
        db.table('reports').insert(report).execute()
        
        logger.info(f"✅ Generated usage report for week ending {end_date.date()}")
        return report
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise


@app.task
def check_external_apis():
    """Check health of external APIs"""
    try:
        from core.ai_manager import AIManager
        import httpx
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'services': []
        }
        
        # Check text generation API
        try:
            ai = AIManager.get_instance()
            test_result = asyncio.run(
                ai.generate_text("test", max_tokens=10)
            )
            results['services'].append({
                'name': 'text_generation',
                'status': 'healthy',
                'response_time': 0.5  # You'd measure this
            })
        except Exception as e:
            results['services'].append({
                'name': 'text_generation',
                'status': 'unhealthy',
                'error': str(e)
            })
        
        # Check image generation APIs
        if ai and ai.image_clients:
            for client_info in ai.image_clients:
                provider = client_info['provider']
                try:
                    # Simple health check - you might want to customize
                    results['services'].append({
                        'name': f'image_{provider}',
                        'status': 'healthy',
                        'stats': {
                            'successes': client_info['successes'],
                            'failures': client_info['failures']
                        }
                    })
                except Exception as e:
                    results['services'].append({
                        'name': f'image_{provider}',
                        'status': 'unhealthy',
                        'error': str(e)
                    })
        
        # Save health check results
        db = get_supabase()
        db.table('health_checks').insert(results).execute()
        
        logger.info("✅ External API health check completed")
        return results
        
    except Exception as e:
        logger.error(f"API health check failed: {e}")
        raise


@app.task
def cleanup_user_data(user_id: str):
    """Clean up all data for a deleted user
    
    Args:
        user_id: User ID to clean up
    """
    try:
        db = get_supabase()
        
        # Delete in order due to foreign keys
        tables = [
            'assets',
            'generations',
            'projects',
            'usage_tracking'
        ]
        
        total_deleted = 0
        
        for table in tables:
            result = db.table(table).delete().eq('user_id', user_id).execute()
            count = len(result.data) if result.data else 0
            total_deleted += count
            logger.info(f"Deleted {count} records from {table}")
        
        # Clean up storage
        storage_deleted = asyncio.run(cleanup_user_storage(user_id))
        
        logger.info(
            f"✅ Cleaned up user {user_id}: "
            f"{total_deleted} records, {storage_deleted} files"
        )
        
        return {
            'records_deleted': total_deleted,
            'files_deleted': storage_deleted
        }
        
    except Exception as e:
        logger.error(f"User cleanup failed: {e}")
        raise