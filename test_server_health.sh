#!/bin/bash

# Server Health Check Script
# Checks if the PromptCritic server is running and responsive

echo "🏥 PromptCritic Server Health Check"
echo "===================================="
echo ""

BASE_URL="http://localhost:8000"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if server is running
echo -e "${BLUE}1. Checking if server is running...${NC}"
if lsof -ti:8000 > /dev/null 2>&1; then
    PID=$(lsof -ti:8000)
    echo -e "${GREEN}✓ Server is running (PID: $PID)${NC}"
else
    echo -e "${RED}✗ Server is NOT running on port 8000${NC}"
    echo ""
    echo "To start the server, run:"
    echo "  cd backend"
    echo "  source venv/bin/activate"
    echo "  python server.py"
    echo ""
    echo "Or use the start script:"
    echo "  ./start.sh"
    exit 1
fi
echo ""

# Check root endpoint
echo -e "${BLUE}2. Checking root endpoint...${NC}"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/")
if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "404" ]; then
    echo -e "${GREEN}✓ Server is responding (HTTP $HTTP_CODE)${NC}"
else
    echo -e "${RED}✗ Server returned HTTP $HTTP_CODE${NC}"
fi
echo ""

# Check API docs
echo -e "${BLUE}3. Checking API documentation...${NC}"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/docs")
if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✓ API docs available at $BASE_URL/docs${NC}"
else
    echo -e "${YELLOW}⚠ API docs returned HTTP $HTTP_CODE${NC}"
fi
echo ""

# Check new project endpoints
echo -e "${BLUE}4. Testing new project endpoints...${NC}"

# Test GET /api/projects
PROJECTS_RESPONSE=$(curl -s "$BASE_URL/api/projects")
if echo "$PROJECTS_RESPONSE" | jq empty 2>/dev/null; then
    PROJECT_COUNT=$(echo "$PROJECTS_RESPONSE" | jq '. | length')
    echo -e "${GREEN}✓ GET /api/projects works (found $PROJECT_COUNT projects)${NC}"
else
    echo -e "${RED}✗ GET /api/projects failed${NC}"
    echo "Response: $PROJECTS_RESPONSE"
fi
echo ""

# Check legacy endpoints
echo -e "${BLUE}5. Testing legacy endpoints...${NC}"

# Test /api/settings
SETTINGS_RESPONSE=$(curl -s "$BASE_URL/api/settings")
if echo "$SETTINGS_RESPONSE" | jq empty 2>/dev/null; then
    echo -e "${GREEN}✓ GET /api/settings works${NC}"
else
    echo -e "${YELLOW}⚠ GET /api/settings returned: ${SETTINGS_RESPONSE:0:50}...${NC}"
fi

# Test /api/evaluations
EVALS_RESPONSE=$(curl -s "$BASE_URL/api/evaluations")
if echo "$EVALS_RESPONSE" | jq empty 2>/dev/null; then
    EVAL_COUNT=$(echo "$EVALS_RESPONSE" | jq '. | length')
    echo -e "${GREEN}✓ GET /api/evaluations works (found $EVAL_COUNT evaluations)${NC}"
else
    echo -e "${YELLOW}⚠ GET /api/evaluations returned: ${EVALS_RESPONSE:0:50}...${NC}"
fi
echo ""

# Check MongoDB connection
echo -e "${BLUE}6. Checking database connection...${NC}"
# Try to create a test project to verify MongoDB
TEST_PROJECT=$(curl -s -X POST "$BASE_URL/api/projects" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Health Check Test",
    "use_case": "Testing",
    "key_requirements": ["test"],
    "target_provider": "openai",
    "initial_prompt": "test"
  }')

if echo "$TEST_PROJECT" | jq -e '.id' > /dev/null 2>&1; then
    TEST_ID=$(echo "$TEST_PROJECT" | jq -r '.id')
    echo -e "${GREEN}✓ MongoDB connection working (created test project: $TEST_ID)${NC}"

    # Clean up test project
    curl -s -X DELETE "$BASE_URL/api/projects/$TEST_ID" > /dev/null
    echo -e "${GREEN}✓ Cleaned up test project${NC}"
else
    echo -e "${RED}✗ MongoDB connection may have issues${NC}"
    echo "Response: $(echo $TEST_PROJECT | jq -r '.detail' 2>/dev/null || echo $TEST_PROJECT)"
fi
echo ""

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Server Health Check Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "📍 Server URL: $BASE_URL"
echo "📚 API Docs: $BASE_URL/docs"
echo "🎨 Frontend: http://localhost:3000"
echo "⚙️  Optimizer: http://localhost:3000/optimizer"
echo ""
echo "To run tests:"
echo "  ./test_project_workflow.sh  # Test new project workflow"
echo "  ./test_ai_features.sh       # Test legacy AI features"
echo ""
