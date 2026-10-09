import json

from fastapi.testclient import TestClient

from api import server
from core.memory import Memory
from core.ron import Ron


client = TestClient(server.app)


def test_health_does_not_require_model_or_api_key(monkeypatch):
    monkeypatch.delenv("RON_API_KEY", raising=False)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "assistant": "Ron-1"}


def test_chat_fails_closed_when_api_key_is_not_configured(monkeypatch):
    monkeypatch.delenv("RON_API_KEY", raising=False)
    response = client.post(
        "/chat",
        json={"message": "hello", "user_id": "user-1"},
    )
    assert response.status_code == 503


def test_chat_rejects_invalid_api_key(monkeypatch):
    monkeypatch.setenv("RON_API_KEY", "expected-secret")
    response = client.post(
        "/chat",
        headers={"X-Ron-API-Key": "wrong-secret"},
        json={"message": "hello", "user_id": "user-1"},
    )
    assert response.status_code == 401


def test_chat_passes_verified_identity_and_conversation(monkeypatch):
    monkeypatch.setenv("RON_API_KEY", "expected-secret")
    monkeypatch.setattr(
        server.ron,
        "chat",
        lambda message, user_id, conversation_id: (
            f"{user_id}:{conversation_id}:{message}"
        ),
    )
    response = client.post(
        "/chat",
        headers={"X-Ron-API-Key": "expected-secret"},
        json={
            "message": " hello ",
            "user_id": "user-1",
            "conversation_id": "chat-9",
        },
    )
    assert response.status_code == 200
    assert response.json()["response"] == "user-1:chat-9:hello"


def test_blank_message_is_rejected(monkeypatch):
    monkeypatch.setenv("RON_API_KEY", "expected-secret")
    response = client.post(
        "/chat",
        headers={"X-Ron-API-Key": "expected-secret"},
        json={"message": "   ", "user_id": "user-1"},
    )
    assert response.status_code == 422


def test_memory_save_is_atomic_and_filters_invalid_entries(tmp_path):
    memory = Memory(str(tmp_path / "user" / "conversation.json"))
    memory.save([
        {"role": "user", "content": "مرحبا"},
        {"role": "system", "content": "do not retain"},
        {"role": "assistant", "content": "أهلا"},
        {"role": "assistant", "content": 42},
    ])
    assert memory.load() == [
        {"role": "user", "content": "مرحبا"},
        {"role": "assistant", "content": "أهلا"},
    ]
    assert not memory.path.with_name(f".{memory.path.name}.tmp").exists()


def test_conversation_memory_is_isolated_by_user(tmp_path):
    ron = Ron()
    ron._memory_root = tmp_path
    seen_messages = []

    def fake_generate(messages, **kwargs):
        seen_messages.append(messages)
        return "answer"

    ron.backend.generate = fake_generate
    ron.chat("private message", user_id="user-A", conversation_id="same-chat")
    ron.chat("new conversation", user_id="user-B", conversation_id="same-chat")

    assert any(m.get("content") == "private message" for m in seen_messages[0])
    assert not any(m.get("content") == "private message" for m in seen_messages[1])
