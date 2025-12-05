"""
Celery Configuration for PromptCritic
Handles async processing of prompt evaluations to avoid timeouts
"""

from celery import Celery
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Get Redis URL from environment
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# Initialize Celery app
celery_app = Celery(
    'promptcritic',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=['celery_tasks']
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task execution
    task_track_started=True,
    task_time_limit=300,  # 5 minutes hard limit
    task_soft_time_limit=240,  # 4 minutes soft limit
    
    # Result backend
    result_expires=3600,  # Results expire after 1 hour
    result_backend_transport_options={
        'master_name': 'mymaster'
    },
    
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    
    # Queue settings
    task_default_queue='promptcritic_default',
    task_queues={
        'promptcritic_default': {
            'exchange': 'promptcritic',
            'routing_key': 'default',
        },
        'promptcritic_high_priority': {
            'exchange': 'promptcritic',
            'routing_key': 'high',
        },
        'promptcritic_low_priority': {
            'exchange': 'promptcritic',
            'routing_key': 'low',
        }
    },
    
    # Monitoring
    task_send_sent_event=True,
    worker_send_task_events=True,
)

# Optional: Configure Celery Beat for periodic tasks (future use)
celery_app.conf.beat_schedule = {
    # Example: Clean up old evaluations every day
    'cleanup-old-evaluations': {
        'task': 'celery_tasks.cleanup_old_evaluations',
        'schedule': 86400.0,  # 24 hours
    },
}

if __name__ == '__main__':
    celery_app.start()
