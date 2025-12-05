#!/usr/bin/env python3
"""
Clear Redis cache to test new evaluation logic
"""

import redis
import os
from dotenv import load_dotenv

load_dotenv()

def clear_cache():
    """Clear all evaluation cache"""
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    try:
        r = redis.from_url(redis_url)
        
        # Get all evaluation cache keys
        keys = r.keys('eval:*')
        
        if keys:
            deleted = r.delete(*keys)
            print(f"✅ Cleared {deleted} cached evaluations")
        else:
            print("ℹ️  No cached evaluations found")
        
        # Also clear stats
        stats_keys = r.keys('cache_stats:*')
        if stats_keys:
            r.delete(*stats_keys)
            print("✅ Cleared cache statistics")
            
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")
        print(f"Make sure Redis is running at: {redis_url}")

if __name__ == "__main__":
    print("🧹 Clearing Redis cache...")
    clear_cache()
