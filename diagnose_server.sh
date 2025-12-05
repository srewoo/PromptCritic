#!/bin/bash

echo "🔍 Diagnosing PromptCritic Server Issues"
echo "=========================================="
echo ""

cd backend

echo "1. Checking Python syntax..."
python -m py_compile server.py 2>&1 | head -20
if [ $? -eq 0 ]; then
    echo "✓ server.py syntax OK"
else
    echo "✗ server.py has syntax errors"
    exit 1
fi
echo ""

echo "2. Checking new modules..."
for file in models.py project_api.py best_practices_engine.py requirements_analyzer.py eval_generator.py dataset_generator.py prompt_rewriter.py; do
    if [ -f "$file" ]; then
        python -m py_compile "$file" 2>&1
        if [ $? -eq 0 ]; then
            echo "✓ $file syntax OK"
        else
            echo "✗ $file has errors"
        fi
    fi
done
echo ""

echo "3. Testing imports..."
python -c "
import sys
sys.path.insert(0, '.')
try:
    import project_api
    print('✓ project_api imports successfully')
except Exception as e:
    print(f'✗ project_api import failed: {e}')

try:
    import models
    print('✓ models imports successfully')
except Exception as e:
    print(f'✗ models import failed: {e}')

try:
    import best_practices_engine
    print('✓ best_practices_engine imports successfully')
except Exception as e:
    print(f'✗ best_practices_engine import failed: {e}')
"
echo ""

echo "4. Checking if server.py can start..."
timeout 5 python -c "
import sys
sys.path.insert(0, '.')
try:
    import server
    print('✓ server module loads')
except Exception as e:
    print(f'✗ server module failed: {e}')
    import traceback
    traceback.print_exc()
" 2>&1
echo ""

echo "5. Current server process..."
ps aux | grep uvicorn | grep -v grep
echo ""

echo "6. Checking MongoDB connection..."
python -c "
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
import os
from dotenv import load_dotenv

load_dotenv('.env')

async def test_mongo():
    try:
        client = AsyncIOMotorClient(os.environ.get('MONGO_URL', 'mongodb://localhost:27017'))
        await client.admin.command('ping')
        print('✓ MongoDB connection successful')
        client.close()
    except Exception as e:
        print(f'✗ MongoDB connection failed: {e}')

asyncio.run(test_mongo())
"
echo ""

echo "=========================================="
echo "Diagnosis complete!"
echo ""
echo "Recommendation:"
echo "  Kill the current server and restart it:"
echo "  pkill -f uvicorn"
echo "  ./start.sh"
