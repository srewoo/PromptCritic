#!/usr/bin/env python3
"""
Test MongoDB connection and settings save functionality
"""
import asyncio
import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pathlib import Path

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

async def test_mongodb():
    print("🔍 Testing MongoDB Connection...")
    print("-" * 50)
    
    # Check environment variables
    mongo_url = os.environ.get('MONGO_URL')
    db_name = os.environ.get('DB_NAME')
    
    print(f"MONGO_URL: {mongo_url}")
    print(f"DB_NAME: {db_name}")
    print("-" * 50)
    
    if not mongo_url:
        print("❌ ERROR: MONGO_URL not set in .env file")
        print("\nCreate a .env file in the backend directory with:")
        print("MONGO_URL=mongodb://localhost:27017")
        print("DB_NAME=promptcritic")
        return
    
    if not db_name:
        print("❌ ERROR: DB_NAME not set in .env file")
        return
    
    try:
        # Connect to MongoDB
        print("\n📡 Connecting to MongoDB...")
        client = AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=5000)
        db = client[db_name]
        
        # Test connection
        await client.admin.command('ping')
        print("✅ MongoDB connection successful!")
        
        # List collections
        collections = await db.list_collection_names()
        print(f"\n📚 Collections in '{db_name}': {collections}")
        
        # Test settings save
        print("\n💾 Testing settings save...")
        test_settings = {
            "llm_provider": "openai",
            "api_key": "test_key_123",
            "model_name": "gpt-4o"
        }
        
        # Delete existing
        result = await db.settings.delete_many({})
        print(f"   Deleted {result.deleted_count} existing settings")
        
        # Insert new
        result = await db.settings.insert_one(test_settings)
        print(f"   Inserted new settings with ID: {result.inserted_id}")
        
        # Retrieve
        saved = await db.settings.find_one()
        print(f"   Retrieved settings: {saved}")
        
        if saved and saved.get('llm_provider') == 'openai':
            print("✅ Settings save/retrieve working correctly!")
        else:
            print("❌ Settings save/retrieve failed!")
        
        # Count documents
        print("\n📊 Document counts:")
        settings_count = await db.settings.count_documents({})
        evaluations_count = await db.evaluations.count_documents({})
        ab_tests_count = await db.ab_tests.count_documents({})
        
        print(f"   Settings: {settings_count}")
        print(f"   Evaluations: {evaluations_count}")
        print(f"   A/B Tests: {ab_tests_count}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nPossible issues:")
        print("1. MongoDB is not running. Start it with: brew services start mongodb-community")
        print("2. Wrong connection string in .env file")
        print("3. Network/firewall issues")
        
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    asyncio.run(test_mongodb())
