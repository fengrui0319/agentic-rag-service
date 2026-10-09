"""Tests for the new append-only JSONL Session, not old memory trimming."""
from __future__ import annotations

import pytest

import agent


@pytest.fixture
def session_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "SESSION_DIR", tmp_path / "sessions")
    return tmp_path


def test_session_persists_and_replays_messages(session_dir):
    sid = "history-test-001"
    s1 = agent.Session(sid, cwd=session_dir)
    s1.append_user("hello")
    s1.append_assistant([{"type": "text", "text": "hi"}])
    s2 = agent.Session(sid, cwd=session_dir)
    assert s2.messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    ]


def test_compact_snapshot_restores_after_reopen(session_dir):
    sid = "compact-test-001"
    s = agent.Session(sid, cwd=session_dir)
    s.append_user("obsolete question")
    s.append_assistant([{"type": "text", "text": "obsolete response"}])
    compacted = [
        {"role": "user", "content": "<conversation_summary>prior facts</conversation_summary>"},
        {"role": "assistant", "content": [{"type": "text", "text": "recent answer"}]},
    ]
    s.replace_messages(compacted, reason="test_compaction")
    assert agent.Session(sid, cwd=session_dir).messages == compacted


def test_restore_previous_snapshot_on_request_failure(session_dir):
    sid = "rollback-test-001"
    s = agent.Session(sid, cwd=session_dir)
    s.append_user("original")
    before = list(s.messages)
    s.append_user("failed request")
    s.replace_messages(before, reason="http_request_failed")
    assert agent.Session(sid, cwd=session_dir).messages == before


def test_session_clear_persists(session_dir):
    sid = "clear-test-001"
    s = agent.Session(sid, cwd=session_dir)
    s.append_user("old")
    s.clear()
    assert agent.Session(sid, cwd=session_dir).messages == []


def test_orphan_user_can_be_removed_on_resume(session_dir):
    sid = "orphan-test-001"
    s = agent.Session(sid, cwd=session_dir)
    s.append_user("pending")
    reopened = agent.Session(sid, cwd=session_dir)
    reopened.truncate_orphan_user()
    assert agent.Session(sid, cwd=session_dir).messages == []


@pytest.mark.parametrize("bad_id", ["../escape", "..\\escape", "", "x" * 65, "short"])
def test_invalid_session_id_rejected(session_dir, bad_id):
    with pytest.raises(ValueError):
        agent.Session(bad_id, cwd=session_dir)


def test_generated_session_id_is_valid(session_dir):
    s = agent.Session(cwd=session_dir)
    assert len(s.id) == 12
    assert s.path.exists()
