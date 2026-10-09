"""Test current Agent tool dispatch and HTTP allowlist without model API calls."""
from unittest.mock import Mock, patch

import agent


def test_http_tool_schema_only_exposes_rag():
    api_tools = agent._build_tools_param({"retrieve_knowledge"})
    assert [t["name"] for t in api_tools] == ["retrieve_knowledge"]
    assert "retrieve_knowledge" in agent.DISPATCH


def test_cli_tool_schema_includes_core_tools():
    tool_names = {t["name"] for t in agent._build_tools_param()}
    assert {"Read", "Write", "Edit", "Bash", "retrieve_knowledge"} <= tool_names


def test_local_read_tool_reports_missing_file(tmp_path):
    out = agent.tool_read(str(tmp_path / "not-there.txt"))
    assert out.startswith("ERROR: no such file:")


def test_local_read_tool_reads_existing_file(tmp_path):
    p = tmp_path / "hello.txt"
    p.write_text("hello from the test", encoding="utf-8")
    assert agent.tool_read(str(p)) == "hello from the test"


def test_forbidden_bash_is_rejected_without_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "SESSION_DIR", tmp_path / "sessions")
    session = agent.Session("tool-denied-001", cwd=tmp_path)
    calls = []

    def fake_stream(client, system, messages, **kwargs):
        assert kwargs["allowed_tool_names"] == {"retrieve_knowledge"}
        calls.append(1)
        if len(calls) == 1:
            return ([{"type": "tool_use", "id": "blocked_1", "name": "Bash",
                     "input": {"command": "echo SHOULD_NOT_EXECUTE"}}], "tool_use", None)
        return ([{"type": "text", "text": "blocked"}], "end_turn", None)

    fake_bash = Mock()
    fake_permission = Mock(return_value=True)
    with patch.object(agent, "USE_STREAM", True), \
         patch.object(agent, "stream_one_turn", side_effect=fake_stream), \
         patch.object(agent, "ask_permission", fake_permission), \
         patch.dict(agent.DISPATCH, {"Bash": fake_bash}):
        answer = agent.agent_turn(
            object(), session, "system", agent.Meter(), "test",
            allowed_tool_names={"retrieve_knowledge"}, max_turns=2,
            raise_on_error=True,
        )

    fake_bash.assert_not_called()
    fake_permission.assert_not_called()
    assert answer == "blocked"
    assert len(calls) == 2
    result = session.messages[2]["content"][0]
    assert result["is_error"] is True
    assert result["content"] == "ERROR: tool not permitted: Bash"


def test_unknown_tool_does_not_crash_agent(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "SESSION_DIR", tmp_path / "sessions")
    session = agent.Session("tool-unknown001", cwd=tmp_path)
    calls = []

    def fake_stream(client, system, messages, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return ([{"type": "tool_use", "id": "bad_1", "name": "not_registered", "input": {}}],
                    "tool_use", None)
        return ([{"type": "text", "text": "handled"}], "end_turn", None)

    with patch.object(agent, "USE_STREAM", True), \
         patch.object(agent, "stream_one_turn", side_effect=fake_stream):
        answer = agent.agent_turn(
            object(), session, "system", agent.Meter(), "test",
            max_turns=2, raise_on_error=True,
        )
    assert answer == "handled"
    assert session.messages[2]["content"][0]["is_error"] is True
    assert "unknown tool" in session.messages[2]["content"][0]["content"]
