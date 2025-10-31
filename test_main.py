"""
Unit tests for AI Code Review Agent
"""

import pytest
from fastapi.testclient import TestClient
from main import app, tasks_db, contexts_db
import json

client = TestClient(app)

# Test fixtures
@pytest.fixture(autouse=True)
def reset_db():
    """Reset databases before each test"""
    tasks_db.clear()
    contexts_db.clear()
    yield


def get_auth_header():
    """Get valid authorization header"""
    return {"Authorization": "Bearer demo-token-12345"}


class TestAgentCard:
    """Test agent card endpoint"""
    
    def test_agent_card_returns_valid_json(self):
        response = client.get("/.well-known/agent-card.json")
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "description" in data
        assert "url" in data
        assert "capabilities" in data
        assert "skills" in data
        
    def test_agent_card_has_required_capabilities(self):
        response = client.get("/.well-known/agent-card.json")
        data = response.json()
        
        assert data["capabilities"]["streaming"] == True
        assert data["capabilities"]["multiTurn"] == True
        assert data["capabilities"]["artifacts"] == True
        
    def test_agent_card_has_skills(self):
        response = client.get("/.well-known/agent-card.json")
        data = response.json()
        
        assert len(data["skills"]) > 0
        assert any(skill["name"] == "code_review" for skill in data["skills"])


class TestAuthentication:
    """Test authentication"""
    
    def test_missing_auth_header_returns_401(self):
        request = {
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {"taskId": "test"},
            "id": "1"
        }
        response = client.post("/a2a/v1", json=request)
        assert response.status_code == 401
        
    def test_invalid_token_returns_401(self):
        request = {
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {"taskId": "test"},
            "id": "1"
        }
        headers = {"Authorization": "Bearer invalid-token"}
        response = client.post("/a2a/v1", json=request, headers=headers)
        assert response.status_code == 401
        
    def test_valid_token_returns_200(self):
        request = {
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {"taskId": "nonexistent"},
            "id": "1"
        }
        response = client.post("/a2a/v1", json=request, headers=get_auth_header())
        assert response.status_code == 200


class TestMessageSend:
    """Test message/send method"""
    
    def test_message_send_creates_task(self):
        request = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "Review this code: def hello(): print('hi')"
                        }
                    ]
                }
            },
            "id": "1"
        }
        
        response = client.post("/a2a/v1", json=request, headers=get_auth_header())
        assert response.status_code == 200
        
        data = response.json()
        assert "result" in data
        assert "task" in data["result"]
        
        task = data["result"]["task"]
        assert "taskId" in task
        assert task["state"]["status"] == "submitted"
        assert "contextId" in task
        
    def test_message_send_stores_task_in_db(self):
        request = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "test"}]
                }
            },
            "id": "1"
        }
        
        response = client.post("/a2a/v1", json=request, headers=get_auth_header())
        task_id = response.json()["result"]["task"]["taskId"]
        
        assert task_id in tasks_db
        assert tasks_db[task_id]["input"]["parts"][0]["text"] == "test"


class TestTaskGet:
    """Test tasks/get method"""
    
    def test_get_nonexistent_task_returns_error(self):
        request = {
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {"taskId": "nonexistent"},
            "id": "1"
        }
        
        response = client.post("/a2a/v1", json=request, headers=get_auth_header())
        data = response.json()
        
        assert "error" in data
        assert data["error"]["code"] == -32602
        
    def test_get_existing_task_returns_task(self):
        # First create a task
        create_request = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "test"}]
                }
            },
            "id": "1"
        }
        
        create_response = client.post("/a2a/v1", json=create_request, headers=get_auth_header())
        task_id = create_response.json()["result"]["task"]["taskId"]
        
        # Now get it
        get_request = {
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {"taskId": task_id},
            "id": "2"
        }
        
        get_response = client.post("/a2a/v1", json=get_request, headers=get_auth_header())
        data = get_response.json()
        
        assert "result" in data
        assert data["result"]["task"]["taskId"] == task_id


class TestTaskCancel:
    """Test tasks/cancel method"""
    
    def test_cancel_task_updates_status(self):
        # Create task
        create_request = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "test"}]
                }
            },
            "id": "1"
        }
        
        create_response = client.post("/a2a/v1", json=create_request, headers=get_auth_header())
        task_id = create_response.json()["result"]["task"]["taskId"]
        
        # Cancel it
        cancel_request = {
            "jsonrpc": "2.0",
            "method": "tasks/cancel",
            "params": {"taskId": task_id},
            "id": "2"
        }
        
        cancel_response = client.post("/a2a/v1", json=cancel_request, headers=get_auth_header())
        data = cancel_response.json()
        
        assert data["result"]["task"]["state"]["status"] == "canceled"


class TestHealthCheck:
    """Test health check endpoint"""
    
    def test_health_check_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestInvalidMethod:
    """Test invalid JSON-RPC methods"""
    
    def test_invalid_method_returns_error(self):
        request = {
            "jsonrpc": "2.0",
            "method": "invalid/method",
            "params": {},
            "id": "1"
        }
        
        response = client.post("/a2a/v1", json=request, headers=get_auth_header())
        data = response.json()
        
        assert "error" in data
        assert data["error"]["code"] == -32601


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=main", "--cov-report=html"])