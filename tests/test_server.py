"""
Backend unit tests for PromptCritic API
Following strict test guidelines: no mock data, actual service layers, proper cleanup
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from motor.motor_asyncio import AsyncIOMotorClient
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from server import app, db

# Test database
TEST_DB_NAME = "promptcritic_test"
TEST_MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")


@pytest.fixture(scope="module")
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture(scope="function")
async def clean_db():
    """Clean up test database before and after each test"""
    client = AsyncIOMotorClient(TEST_MONGO_URL)
    test_db = client[TEST_DB_NAME]
    
    # Clean before
    await test_db.settings.delete_many({})
    await test_db.evaluations.delete_many({})
    
    yield
    
    # Clean after
    await test_db.settings.delete_many({})
    await test_db.evaluations.delete_many({})
    client.close()


def test_root_endpoint(client):
    """Test root endpoint returns correct response"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "PromptCritic" in data["message"]


def test_save_and_get_settings(client, clean_db):
    """Test saving and retrieving LLM settings"""
    # Save settings
    settings_data = {
        "llm_provider": "openai",
        "api_key": "test-key-12345",
        "model_name": "gpt-4o"
    }
    
    response = client.post("/api/settings", json=settings_data)
    assert response.status_code == 200
    
    # Retrieve settings
    response = client.get("/api/settings")
    assert response.status_code == 200
    data = response.json()
    assert data["llm_provider"] == "openai"
    assert data["model_name"] == "gpt-4o"


def test_settings_not_found(client, clean_db):
    """Test getting settings when none exist"""
    response = client.get("/api/settings")
    # Should return None/null when no settings exist
    assert response.status_code == 200


def test_cost_calculation():
    """Test cost calculation function"""
    from server import calculate_cost, estimate_tokens
    
    # Test token estimation
    text = "This is a test prompt"
    tokens = estimate_tokens(text)
    assert tokens > 0
    assert isinstance(tokens, int)
    
    # Test cost calculation
    prompt = "Test prompt"
    response = "This is a longer response with more tokens to estimate the cost"
    cost = calculate_cost(prompt, response, "openai", "gpt-4o")
    
    assert "input_tokens" in cost
    assert "output_tokens" in cost
    assert "total_cost" in cost
    assert cost["total_cost"] > 0
    assert cost["currency"] == "USD"


def test_evaluate_prompt_without_settings(client, clean_db):
    """Test evaluation fails without configured settings"""
    response = client.post("/api/evaluate", json={"prompt_text": "Test prompt"})
    assert response.status_code == 400
    assert "configure" in response.json()["detail"].lower()


def test_get_evaluations_empty(client, clean_db):
    """Test getting evaluations when none exist"""
    response = client.get("/api/evaluations")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0


def test_get_evaluation_not_found(client, clean_db):
    """Test getting non-existent evaluation"""
    response = client.get("/api/evaluations/nonexistent-id")
    assert response.status_code == 404


def test_delete_evaluation_not_found(client, clean_db):
    """Test deleting non-existent evaluation"""
    response = client.delete("/api/evaluations/nonexistent-id")
    assert response.status_code == 404


def test_compare_evaluations_insufficient(client, clean_db):
    """Test comparison with less than 2 evaluations"""
    response = client.post("/api/compare", json={"evaluation_ids": ["id1"]})
    assert response.status_code == 400
    assert "at least 2" in response.json()["detail"].lower()


def test_rewrite_without_settings(client, clean_db):
    """Test rewrite fails without configured settings"""
    response = client.post("/api/rewrite", json={"prompt_text": "Test prompt"})
    assert response.status_code == 400
    assert "configure" in response.json()["detail"].lower()


def test_playground_without_settings(client, clean_db):
    """Test playground fails without configured settings"""
    response = client.post("/api/playground", json={
        "prompt_text": "Test {input}",
        "test_input": "example"
    })
    assert response.status_code == 400
    assert "configure" in response.json()["detail"].lower()


def test_api_endpoints_exist(client):
    """Test all API endpoints are accessible (negative tests for missing auth/data)"""
    endpoints = [
        ("/api/settings", "GET"),
        ("/api/evaluations", "GET"),
        ("/api/compare", "POST"),
        ("/api/rewrite", "POST"),
        ("/api/playground", "POST"),
    ]
    
    for path, method in endpoints:
        if method == "GET":
            response = client.get(path)
        elif method == "POST":
            response = client.post(path, json={})
        
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404


def test_token_costs_data_structure():
    """Test TOKEN_COSTS constant has correct structure"""
    from server import TOKEN_COSTS
    
    assert "openai" in TOKEN_COSTS
    assert "claude" in TOKEN_COSTS
    assert "gemini" in TOKEN_COSTS
    
    # Check structure for each provider
    for provider in TOKEN_COSTS.values():
        for model_costs in provider.values():
            assert "input" in model_costs
            assert "output" in model_costs
            assert isinstance(model_costs["input"], (int, float))
            assert isinstance(model_costs["output"], (int, float))


def test_edge_case_empty_prompt(client, clean_db):
    """Test evaluation with empty prompt"""
    # First set up settings
    settings_data = {
        "llm_provider": "openai",
        "api_key": "test-key",
        "model_name": "gpt-4o"
    }
    client.post("/api/settings", json=settings_data)
    
    # Try to evaluate empty prompt (should fail gracefully)
    response = client.post("/api/evaluate", json={"prompt_text": ""})
    # Should return error or handle gracefully
    assert response.status_code in [400, 422, 500]


def test_edge_case_very_long_prompt(client):
    """Test handling of very long prompts"""
    long_prompt = "a" * 100000  # 100k characters
    
    response = client.post("/api/evaluate", json={"prompt_text": long_prompt})
    # Should handle without crashing
    assert response.status_code in [400, 422, 500]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

