"""
Celery Tasks for PromptCritic
Background tasks for async evaluation processing
"""

from celery_config import celery_app
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from pathlib import Path
import json

# Load environment
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB connection info
MONGO_URL = os.environ['MONGO_URL']
DB_NAME = os.environ['DB_NAME']


def get_db_client():
    """Get MongoDB client"""
    client = AsyncIOMotorClient(MONGO_URL)
    return client[DB_NAME]


@celery_app.task(bind=True, name='celery_tasks.evaluate_prompt_async')
def evaluate_prompt_async(self, evaluation_id: str, prompt_text: str, 
                          llm_provider: str, api_key: str, 
                          model_name: str = None, evaluation_mode: str = "standard"):
    """
    Async task for prompt evaluation
    
    Args:
        self: Celery task instance (for progress updates)
        evaluation_id: Unique evaluation ID
        prompt_text: Prompt to evaluate
        llm_provider: LLM provider (openai, claude, gemini)
        api_key: API key for LLM
        model_name: Optional model name
        evaluation_mode: Evaluation mode
        
    Returns:
        Evaluation result dictionary
    """
    try:
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'current': 10, 'total': 100, 'status': 'Starting evaluation...'}
        )
        
        # Import here to avoid circular imports
        from server import get_llm_evaluation
        from cache import get_cache_manager
        
        # Check cache first
        cache = get_cache_manager()
        cached_result = cache.get_evaluation(
            prompt_text=prompt_text,
            evaluation_mode=evaluation_mode,
            provider=llm_provider,
            model=model_name or "default"
        )
        
        if cached_result:
            logger.info(f"✅ Cache hit for evaluation {evaluation_id}")
            self.update_state(
                state='SUCCESS',
                meta={'current': 100, 'total': 100, 'status': 'Completed (cached)'}
            )
            return cached_result
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={'current': 30, 'total': 100, 'status': 'Calling LLM API...'}
        )
        
        # Run async evaluation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            evaluation_data = loop.run_until_complete(
                get_llm_evaluation(
                    prompt_text, llm_provider, api_key, model_name, evaluation_mode
                )
            )
        finally:
            loop.close()
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={'current': 80, 'total': 100, 'status': 'Saving results...'}
        )
        
        # Cache the result
        cache.set_evaluation(
            prompt_text=prompt_text,
            evaluation_mode=evaluation_mode,
            provider=llm_provider,
            model=model_name or "default",
            evaluation_data=evaluation_data,
            ttl_hours=24
        )
        
        # Save to database
        db = get_db_client()
        
        evaluation_record = {
            "id": evaluation_id,
            "prompt_text": prompt_text,
            "llm_provider": llm_provider,
            "model_name": model_name,
            "evaluation_mode": evaluation_mode,
            "criteria_scores": evaluation_data['criteria_scores'],
            "total_score": evaluation_data['total_score'],
            "max_score": evaluation_data.get('max_score', 250),
            "refinement_suggestions": evaluation_data['refinement_suggestions'],
            "category_scores": evaluation_data.get('category_scores'),
            "provider_scores": evaluation_data.get('provider_scores'),
            "contradiction_analysis": evaluation_data.get('contradiction_analysis'),
            "cost": evaluation_data.get('cost'),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task_id": self.request.id
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                db.evaluations.insert_one(evaluation_record)
            )
        finally:
            loop.close()
        
        logger.info(f"✅ Evaluation {evaluation_id} completed successfully")
        
        # Return final state
        self.update_state(
            state='SUCCESS',
            meta={'current': 100, 'total': 100, 'status': 'Completed successfully'}
        )
        
        return evaluation_data
        
    except Exception as e:
        logger.error(f"❌ Evaluation {evaluation_id} failed: {str(e)}")
        self.update_state(
            state='FAILURE',
            meta={'current': 0, 'total': 100, 'status': f'Failed: {str(e)}'}
        )
        raise


@celery_app.task(name='celery_tasks.cleanup_old_evaluations')
def cleanup_old_evaluations():
    """
    Periodic task to cleanup old evaluations
    Runs daily to remove evaluations older than 90 days
    """
    try:
        db = get_db_client()
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=90)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                db.evaluations.delete_many({
                    "created_at": {"$lt": cutoff_date.isoformat()}
                })
            )
            
            deleted_count = result.deleted_count
            logger.info(f"🗑️  Cleaned up {deleted_count} old evaluations")
            return {"deleted": deleted_count, "cutoff_date": cutoff_date.isoformat()}
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {str(e)}")
        raise


@celery_app.task(bind=True, name='celery_tasks.batch_evaluate')
def batch_evaluate(self, prompts: list, llm_provider: str, api_key: str,
                   model_name: str = None, evaluation_mode: str = "standard"):
    """
    Batch evaluation of multiple prompts
    
    Args:
        prompts: List of prompt texts to evaluate
        llm_provider: LLM provider
        api_key: API key
        model_name: Optional model name
        evaluation_mode: Evaluation mode
        
    Returns:
        List of evaluation results
    """
    try:
        total = len(prompts)
        results = []
        
        for i, prompt_text in enumerate(prompts, 1):
            # Update progress
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': i,
                    'total': total,
                    'status': f'Evaluating prompt {i}/{total}...'
                }
            )
            
            # Create subtask
            from uuid import uuid4
            evaluation_id = str(uuid4())
            
            result = evaluate_prompt_async(
                evaluation_id, prompt_text, llm_provider,
                api_key, model_name, evaluation_mode
            )
            
            results.append({
                'prompt': prompt_text[:100] + '...' if len(prompt_text) > 100 else prompt_text,
                'evaluation_id': evaluation_id,
                'score': result.get('total_score'),
                'max_score': result.get('max_score', 250)
            })
        
        logger.info(f"✅ Batch evaluation completed: {total} prompts")
        
        return {
            'total': total,
            'completed': len(results),
            'results': results
        }
        
    except Exception as e:
        logger.error(f"❌ Batch evaluation failed: {str(e)}")
        raise
