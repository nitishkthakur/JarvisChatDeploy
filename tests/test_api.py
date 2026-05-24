"""Integration tests for the FastAPI endpoints."""

import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from backend.main import app, conversations, todo_states


@pytest.fixture(autouse=True)
def clear_state():
    """Clear in-memory state before each test."""
    conversations.clear()
    todo_states.clear()
    yield
    conversations.clear()
    todo_states.clear()


@pytest.fixture
def client():
    return TestClient(app)


# ---- Mock helpers ----

def _mock_chat_result(content: str) -> ChatResult:
    return ChatResult(
        generations=[ChatGeneration(message=AIMessage(content=content))]
    )


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "JarvisChat"


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------


class TestChatEndpoint:
    def test_empty_message_returns_400(self, client):
        resp = client.post("/chat", json={"message": "", "model": "test-model"})
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_empty_model_returns_400(self, client):
        resp = client.post("/chat", json={"message": "hello", "model": ""})
        assert resp.status_code == 400
        assert "model" in resp.json()["detail"].lower()

    def test_whitespace_only_message_returns_400(self, client):
        resp = client.post("/chat", json={"message": "   ", "model": "test-model"})
        assert resp.status_code == 400

    @patch("backend.main.OllamaCloudLLM._generate")
    def test_trivial_message_skips_agent(self, mock_generate, client):
        mock_generate.return_value = _mock_chat_result("Hello! How can I help?")
        resp = client.post(
            "/chat", json={"message": "hello", "model": "test-model"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "Hello! How can I help?"
        assert "conversation_id" in data
        mock_generate.assert_called_once()

    @patch("backend.main.OllamaCloudLLM._generate")
    def test_conversation_id_persists(self, mock_generate, client):
        mock_generate.return_value = _mock_chat_result("Hi!")
        resp1 = client.post(
            "/chat", json={"message": "hello", "model": "test-model"}
        )
        conv_id = resp1.json()["conversation_id"]

        mock_generate.return_value = _mock_chat_result("I'm fine!")
        resp2 = client.post(
            "/chat",
            json={
                "message": "how are you",
                "model": "test-model",
                "conversation_id": conv_id,
            },
        )
        assert resp2.json()["conversation_id"] == conv_id

    @patch("backend.main.OllamaCloudLLM._generate")
    def test_conversation_history_stored(self, mock_generate, client):
        mock_generate.return_value = _mock_chat_result("Hi!")
        resp = client.post(
            "/chat", json={"message": "hello", "model": "test-model"}
        )
        conv_id = resp.json()["conversation_id"]
        assert conv_id in conversations
        assert len(conversations[conv_id]) == 2
        assert conversations[conv_id][0]["role"] == "user"
        assert conversations[conv_id][1]["role"] == "assistant"

    @patch("backend.main.OllamaCloudLLM._generate")
    def test_nontrivial_message_uses_agent(self, mock_generate, client):
        # For non-trivial messages the agent executor is used.
        # Mock the LLM to produce a direct Final Answer (skip tool usage).
        mock_generate.return_value = _mock_chat_result(
            "I need to think.\nFinal Answer: The answer is 42."
        )
        resp = client.post(
            "/chat",
            json={"message": "Explain quantum computing in detail", "model": "test"},
        )
        # The agent should run and return something
        assert resp.status_code == 200
        assert "conversation_id" in resp.json()


# ---------------------------------------------------------------------------
# Chat stream endpoint
# ---------------------------------------------------------------------------


class TestChatStreamEndpoint:
    def test_stream_empty_message_returns_400(self, client):
        resp = client.post(
            "/chat/stream", json={"message": "", "model": "test-model"}
        )
        assert resp.status_code == 400

    def test_stream_empty_model_returns_400(self, client):
        resp = client.post(
            "/chat/stream", json={"message": "hello", "model": ""}
        )
        assert resp.status_code == 400

    @patch("backend.main.OllamaCloudLLM._generate")
    def test_stream_trivial_sends_sse_events(self, mock_generate, client):
        mock_generate.return_value = _mock_chat_result("Hi there!")
        resp = client.post(
            "/chat/stream",
            json={"message": "hello", "model": "test-model"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        body = resp.text
        # Should have step and answer events
        assert "event: step" in body
        assert "event: answer" in body
        assert "Hi there!" in body

    @patch("backend.main.OllamaCloudLLM._generate")
    def test_stream_stores_conversation(self, mock_generate, client):
        mock_generate.return_value = _mock_chat_result("Response!")
        resp = client.post(
            "/chat/stream",
            json={"message": "hi", "model": "test-model"},
        )
        body = resp.text
        # Extract conversation_id from the answer event
        import json
        for line in body.split("\n"):
            if line.startswith("data: ") and "conversation_id" in line:
                data = json.loads(line[6:])
                assert data["conversation_id"] in conversations
                break
