"""FastAPI V1.0 integration tests with an offline Agent stub.

No network/API calls: Anthropic is mocked and Session writes are redirected
into pytest's temporary directory. This tests HTTP behavior, not LLM quality.
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock, Thread
from time import sleep
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import agent
import api


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-test-key")
    monkeypatch.setattr(api, "Anthropic", MagicMock())
    # Per-test registry cleanup. No concurrent tests share this fixture.
    with api._locks_guard:
        api._session_locks.clear()
    with TestClient(api.app) as test_client:
        yield test_client


def stub_agent(client, session, system_text, meter, user_input, **kwargs):
    assert kwargs["allowed_tool_names"] == {"retrieve_knowledge"}
    assert kwargs["max_turns"] == 6
    assert kwargs["raise_on_error"] is True

    session.append_user(user_input)
    answer = "Echo: " + user_input
    on_text = kwargs.get("on_text")
    if on_text is not None:
        on_text(answer)
    session.append_assistant([{"type": "text", "text": answer}])
    return answer


def reopened(session_id):
    return agent.Session(session_id=session_id, cwd=api.PROJECT_ROOT)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_chat_creates_and_resumes_persisted_session(client, monkeypatch):
    seen = []

    def fake(*args, **kwargs):
        session = kwargs["session"]
        seen.append(len(session.messages))
        return stub_agent(*args, **kwargs)

    monkeypatch.setattr(api.agent, "agent_turn", fake)
    first = client.post("/chat", json={"message": "hello"})
    assert first.status_code == 200, first.text
    sid = first.json()["session_id"]
    assert len(sid) == 12
    assert first.json()["answer"] == "Echo: hello"

    second = client.post("/chat", json={"message": "second", "session_id": sid})
    assert second.status_code == 200, second.text
    assert second.json() == {"session_id": sid, "answer": "Echo: second"}
    assert seen == [0, 2]

    messages = reopened(sid).messages
    assert len(messages) == 4
    assert messages[0] == {"role": "user", "content": "hello"}
    assert messages[-1]["content"][0]["text"] == "Echo: second"


def test_chat_failure_rolls_back_and_sanitizes_503(client, monkeypatch):
    monkeypatch.setattr(api.agent, "agent_turn", stub_agent)
    r = client.post("/chat", json={"message": "good", "session_id": "rollback-001"})
    assert r.status_code == 200
    before = reopened("rollback-001").messages

    def failed(client, session, system_text, meter, user_input, **kwargs):
        session.append_user(user_input)
        raise agent.AgentTurnError("private model details")

    monkeypatch.setattr(api.agent, "agent_turn", failed)
    r = client.post("/chat", json={"message": "fail", "session_id": "rollback-001"})
    assert r.status_code == 503
    assert r.json() == {"detail": "Agent could not complete the request."}
    assert "private model details" not in r.text
    assert reopened("rollback-001").messages == before


def test_chat_unexpected_exception_returns_500_without_details(client, monkeypatch):
    def failed(client, session, system_text, meter, user_input, **kwargs):
        session.append_user(user_input)
        raise ValueError("sensitive internal details")

    monkeypatch.setattr(api.agent, "agent_turn", failed)
    r = client.post("/chat", json={"message": "hello", "session_id": "error-test-001"})
    assert r.status_code == 500
    assert r.json() == {"detail": "Internal agent error."}
    assert "sensitive internal details" not in r.text
    assert reopened("error-test-001").messages == []


def test_stream_creates_and_resumes_persisted_session(client, monkeypatch):
    seen = []

    def fake(client, session, system_text, meter, user_input, **kwargs):
        seen.append(len(session.messages))
        assert kwargs["allowed_tool_names"] == {"retrieve_knowledge"}
        assert kwargs["max_turns"] == 6
        assert kwargs["raise_on_error"] is True
        session.append_user(user_input)
        kwargs["on_text"]("hello ")
        kwargs["on_text"]("world")
        session.append_assistant([{"type": "text", "text": "hello world"}])
        return "hello world"

    monkeypatch.setattr(api.agent, "agent_turn", fake)
    r1 = client.post("/chat/stream", json={"message": "first"})
    assert r1.status_code == 200
    assert r1.text == "hello world"
    assert r1.headers["content-type"].startswith("text/plain")
    sid = r1.headers["x-session-id"]

    r2 = client.post("/chat/stream", json={"message": "second", "session_id": sid})
    assert r2.status_code == 200
    assert r2.headers["x-session-id"] == sid
    assert r2.text == "hello world"
    assert seen == [0, 2]
    assert len(reopened(sid).messages) == 4


def test_stream_failure_marks_body_and_rolls_back(client, monkeypatch):
    monkeypatch.setattr(api.agent, "agent_turn", stub_agent)
    first = client.post("/chat", json={"message": "previous", "session_id": "stream-rollback01"})
    assert first.status_code == 200
    before = reopened("stream-rollback01").messages

    def failed(client, session, system_text, meter, user_input, **kwargs):
        session.append_user(user_input)
        kwargs["on_text"]("partial")
        raise agent.AgentTurnError("simulated model failure")

    monkeypatch.setattr(api.agent, "agent_turn", failed)
    r = client.post("/chat/stream", json={"message": "fail", "session_id": "stream-rollback01"})
    assert r.status_code == 200  # Headers already sent; error is in the body.
    assert r.text == "partial\n[stream error: agent could not complete the request]"
    assert reopened("stream-rollback01").messages == before


@pytest.mark.parametrize("endpoint", ["/chat", "/chat/stream"])
def test_invalid_session_ids_rejected(client, endpoint):
    for sid in ("../escape", "..\\escape", "short", "", "x" * 65):
        r = client.post(endpoint, json={"message": "hi", "session_id": sid})
        assert r.status_code == 422, (sid, r.text)


@pytest.mark.parametrize("endpoint", ["/chat", "/chat/stream"])
def test_blank_or_overlong_messages_rejected(client, endpoint):
    for message in ("", "   ", "x" * 4001):
        r = client.post(endpoint, json={"message": message})
        assert r.status_code == 422, (message[:30], r.text)


def test_missing_model_configuration_returns_503(client, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    for endpoint in ("/chat", "/chat/stream"):
        r = client.post(endpoint, json={"message": "hi"})
        assert r.status_code == 503
        assert r.json()["detail"] == "Model service is not configured."


def test_same_session_requests_are_serialized(client, monkeypatch):
    """Single-process lock test, not a multi-worker load test."""
    guard = Lock()
    state = {"active": 0, "peak": 0}
    failures = []

    def slow_agent(*args, **kwargs):
        with guard:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        try:
            sleep(0.03)
            return stub_agent(*args, **kwargs)
        finally:
            with guard:
                state["active"] -= 1

    monkeypatch.setattr(api.agent, "agent_turn", slow_agent)

    def request():
        try:
            r = client.post("/chat", json={"message": "hello", "session_id": "parallel-001"})
            assert r.status_code == 200, r.text
        except Exception as e:
            failures.append(e)

    threads = [Thread(target=request) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert all(not t.is_alive() for t in threads), "request threads timed out"
    assert not failures, failures
    assert state["peak"] == 1
    assert len(reopened("parallel-001").messages) == 4
