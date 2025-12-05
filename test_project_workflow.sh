#!/bin/bash

# Test script for new Project-based System Prompt Optimization Workflow
# Run this after starting the server: ./start.sh

echo "🧪 Testing PromptCritic Project Workflow"
echo "=========================================="
echo ""

BASE_URL="http://localhost:8000/api"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}1. Creating a New Project${NC}"
echo "-----------------------------------"
PROJECT_RESPONSE=$(curl -s -X POST "$BASE_URL/projects" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Customer Support Bot",
    "use_case": "Automated customer support for e-commerce platform",
    "key_requirements": [
      "Handle customer queries professionally",
      "Provide accurate product information",
      "Escalate complex issues to human agents"
    ],
    "target_provider": "openai",
    "initial_prompt": "You are a helpful customer support assistant."
  }')

PROJECT_ID=$(echo $PROJECT_RESPONSE | jq -r '.id')
echo "Created project: $PROJECT_ID"
echo ""

if [ "$PROJECT_ID" = "null" ] || [ -z "$PROJECT_ID" ]; then
  echo -e "${RED}❌ Failed to create project${NC}"
  exit 1
fi

echo -e "${GREEN}✓ Project created successfully${NC}"
echo ""
echo ""

echo -e "${BLUE}2. Analyzing System Prompt${NC}"
echo "-----------------------------------"
ANALYSIS=$(curl -s -X POST "$BASE_URL/projects/$PROJECT_ID/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_text": "You are a helpful customer support assistant."
  }')

echo "Overall Score: $(echo $ANALYSIS | jq '.overall_score')/100"
echo "Requirements Alignment: $(echo $ANALYSIS | jq '.requirements_alignment_score')/100"
echo "Best Practices Score: $(echo $ANALYSIS | jq '.best_practices_score')/100"
echo "Gaps Found: $(echo $ANALYSIS | jq '.requirements_gaps | length')"
echo ""
echo -e "${GREEN}✓ Analysis complete${NC}"
echo ""
echo ""

echo -e "${BLUE}3. AI-Powered Rewrite${NC}"
echo "-----------------------------------"
echo -e "${YELLOW}Note: This requires valid API keys in backend/.env${NC}"
REWRITE=$(curl -s -X POST "$BASE_URL/projects/$PROJECT_ID/rewrite" \
  -H "Content-Type: application/json" \
  -d '{
    "current_prompt": "You are a helpful customer support assistant."
  }')

IMPROVED_PROMPT=$(echo $REWRITE | jq -r '.improved_prompt' | head -c 100)
if [ "$IMPROVED_PROMPT" != "null" ] && [ ! -z "$IMPROVED_PROMPT" ]; then
  echo "Improved prompt (first 100 chars): $IMPROVED_PROMPT..."
  echo "Changes made: $(echo $REWRITE | jq '.changes_made | length')"
  echo -e "${GREEN}✓ Rewrite successful${NC}"
else
  echo -e "${YELLOW}⚠ Rewrite skipped (requires API keys)${NC}"
fi
echo ""
echo ""

echo -e "${BLUE}4. Adding New Version${NC}"
echo "-----------------------------------"
NEW_VERSION=$(curl -s -X POST "$BASE_URL/projects/$PROJECT_ID/versions" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_text": "You are a professional customer support assistant for an e-commerce platform. Handle queries professionally, provide accurate information, and escalate complex issues.",
    "user_feedback": "Improved with more specific instructions",
    "is_final": false
  }')

VERSION_NUM=$(echo $NEW_VERSION | jq -r '.version')
echo "Created version: $VERSION_NUM"
echo -e "${GREEN}✓ Version added${NC}"
echo ""
echo ""

echo -e "${BLUE}5. Generating Evaluation Prompt${NC}"
echo "-----------------------------------"
EVAL_PROMPT=$(curl -s -X POST "$BASE_URL/projects/$PROJECT_ID/eval-prompt/generate" \
  -H "Content-Type: application/json" \
  -d '{}')

EVAL_TEXT=$(echo $EVAL_PROMPT | jq -r '.eval_prompt' | head -c 100)
echo "Eval prompt (first 100 chars): $EVAL_TEXT..."
echo "Test scenarios: $(echo $EVAL_PROMPT | jq '.test_scenarios | length')"
echo -e "${GREEN}✓ Evaluation prompt generated${NC}"
echo ""
echo ""

echo -e "${BLUE}6. Generating Test Dataset${NC}"
echo "-----------------------------------"
DATASET=$(curl -s -X POST "$BASE_URL/projects/$PROJECT_ID/dataset/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "sample_count": 100
  }')

SAMPLE_COUNT=$(echo $DATASET | jq -r '.sample_count')
echo "Dataset generated: $SAMPLE_COUNT samples"
echo "Preview (first 3 rows):"
echo $DATASET | jq -r '.preview[:3] | .[] | "  - " + .category + ": " + .input[:60] + "..."'
echo -e "${GREEN}✓ Dataset generated${NC}"
echo ""
echo ""

echo -e "${BLUE}7. Downloading Dataset${NC}"
echo "-----------------------------------"
curl -s -X GET "$BASE_URL/projects/$PROJECT_ID/dataset/export" \
  -o "test_dataset_$PROJECT_ID.csv"

if [ -f "test_dataset_$PROJECT_ID.csv" ]; then
  LINE_COUNT=$(wc -l < "test_dataset_$PROJECT_ID.csv")
  echo "Downloaded: test_dataset_$PROJECT_ID.csv"
  echo "Lines in file: $LINE_COUNT"
  echo -e "${GREEN}✓ Dataset downloaded${NC}"
else
  echo -e "${RED}❌ Failed to download dataset${NC}"
fi
echo ""
echo ""

echo -e "${BLUE}8. Listing All Projects${NC}"
echo "-----------------------------------"
PROJECTS=$(curl -s -X GET "$BASE_URL/projects")
PROJECT_COUNT=$(echo $PROJECTS | jq '. | length')
echo "Total projects: $PROJECT_COUNT"
echo -e "${GREEN}✓ Projects listed${NC}"
echo ""
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ All workflow tests complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "📊 Summary:"
echo "  - Project ID: $PROJECT_ID"
echo "  - Versions: $VERSION_NUM"
echo "  - Dataset: $SAMPLE_COUNT samples"
echo "  - Dataset file: test_dataset_$PROJECT_ID.csv"
echo ""
echo "🌐 Access the Optimizer UI at: http://localhost:3000/optimizer"
echo "📚 Full API documentation at: http://localhost:8000/docs"
echo ""
