import pytest
from fastapi.testclient import TestClient

import api


client = TestClient(api.app)


@pytest.fixture(autouse=True)
def clear_sessions():
    api.SESSIONS.clear()

    yield

    api.SESSIONS.clear()


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
    }


def test_chat_creates_session(monkeypatch):
    def fake_run_agent_with_history(
        prompt: str,
        history=None,
    ):
        assert prompt == "hello"
        assert history == []

        updated_history = [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": "fake answer",
            },
        ]

        return "fake answer", updated_history

    monkeypatch.setattr(
        api,
        "run_agent_with_history",
        fake_run_agent_with_history,
    )

    response = client.post(
        "/chat",
        json={
            "message": "hello",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "fake answer"
    assert data["session_id"]

    session_id = data["session_id"]

    assert session_id in api.SESSIONS
    assert api.SESSIONS[session_id][-1] == {
        "role": "assistant",
        "content": "fake answer",
    }


def test_chat_reuses_existing_session(monkeypatch):
    calls = []

    def fake_run_agent_with_history(
        prompt: str,
        history=None,
    ):
        history = history or []

        calls.append({
            "prompt": prompt,
            "history": list(history),
        })

        answer = f"reply to: {prompt}"

        updated_history = history + [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]

        return answer, updated_history

    monkeypatch.setattr(
        api,
        "run_agent_with_history",
        fake_run_agent_with_history,
    )

    first_response = client.post(
        "/chat",
        json={
            "message": "first message",
        },
    )

    assert first_response.status_code == 200

    session_id = first_response.json()["session_id"]

    second_response = client.post(
        "/chat",
        json={
            "message": "second message",
            "session_id": session_id,
        },
    )

    assert second_response.status_code == 200
    assert second_response.json()["session_id"] == session_id

    assert len(calls) == 2

    assert calls[0]["history"] == []

    assert calls[1]["history"] == [
        {
            "role": "user",
            "content": "first message",
        },
        {
            "role": "assistant",
            "content": "reply to: first message",
        },
    ]

def test_chat_stream_creates_session(monkeypatch):
    def fake_stream_agent_with_history(
        prompt: str,
        history=None,
    ):
        assert prompt == "stream hello"
        assert history == []

        yield "hello "
        yield "world"

    monkeypatch.setattr(
        api,
        "stream_agent_with_history",
        fake_stream_agent_with_history,
    )

    response = client.post(
        "/chat/stream",
        json={
            "message": "stream hello",
        },
    )

    assert response.status_code == 200
    assert response.text == "hello world"

    session_id = response.headers.get("x-session-id")

    assert session_id
    assert session_id in api.SESSIONS

    assert api.SESSIONS[session_id] == [
        {
            "role": "user",
            "content": "stream hello",
        },
        {
            "role": "assistant",
            "content": "hello world",
        },
    ]


def test_chat_stream_reuses_existing_session(monkeypatch):
    calls = []

    def fake_stream_agent_with_history(
        prompt: str,
        history=None,
    ):
        history = history or []

        calls.append({
            "prompt": prompt,
            "history": list(history),
        })

        yield f"reply to: {prompt}"

    monkeypatch.setattr(
        api,
        "stream_agent_with_history",
        fake_stream_agent_with_history,
    )

    first_response = client.post(
        "/chat/stream",
        json={
            "message": "first stream message",
        },
    )

    assert first_response.status_code == 200

    session_id = first_response.headers.get("x-session-id")

    assert session_id

    second_response = client.post(
        "/chat/stream",
        json={
            "message": "second stream message",
            "session_id": session_id,
        },
    )

    assert second_response.status_code == 200
    assert second_response.headers.get("x-session-id") == session_id

    assert len(calls) == 2

    assert calls[0]["history"] == []

    assert calls[1]["history"] == [
        {
            "role": "user",
            "content": "first stream message",
        },
        {
            "role": "assistant",
            "content": "reply to: first stream message",
        },
    ]

def test_chat_handles_agent_runtime_error(monkeypatch):
    def fake_run_agent_with_history(
        prompt: str,
        history=None,
    ):
        raise RuntimeError(
            "Agent exceeded maximum steps."
        )

    monkeypatch.setattr(
        api,
        "run_agent_with_history",
        fake_run_agent_with_history,
    )

    response = client.post(
        "/chat",
        json={
            "message": "cause a loop",
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Agent could not complete the request.",
    }

    assert api.SESSIONS == {}


def test_chat_handles_unexpected_error(monkeypatch):
    def fake_run_agent_with_history(
        prompt: str,
        history=None,
    ):
        raise ValueError("sensitive internal details")

    monkeypatch.setattr(
        api,
        "run_agent_with_history",
        fake_run_agent_with_history,
    )

    response = client.post(
        "/chat",
        json={
            "message": "hello",
        },
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Internal agent error.",
    }

    assert "sensitive internal details" not in response.text
    assert api.SESSIONS == {}

def test_chat_rejects_message_that_is_too_long():
    response = client.post(
        "/chat",
        json={
            "message": "x" * (api.MAX_MESSAGE_LENGTH + 1),
        },
    )

    assert response.status_code == 422