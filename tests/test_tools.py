from mini_agent import execute_tool


def test_execute_tool_runs_calculator():
    result = execute_tool(
        "calculator",
        {
            "expression": "2 + 3",
        },
    )

    assert result == "5"


def test_execute_tool_returns_error_instead_of_raising():
    result = execute_tool(
        "calculator",
        {
            "expression": "open(1)",
        },
    )

    assert result.startswith(
        "Tool error: ValueError:"
    )

    assert "Unsupported expression: Call" in result


def test_execute_tool_handles_unknown_tool():
    result = execute_tool(
        "does_not_exist",
        {},
    )

    assert result == (
        "Tool error: unknown tool 'does_not_exist'."
    )