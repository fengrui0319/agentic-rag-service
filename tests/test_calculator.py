"""Offline end-to-end test of the CURRENT local MCP calculator server.

Unlike the old mini_agent calculator, this is a real stdio MCP client/server
interaction. It uses no LLM API; the server must be present in mcp_servers/.
"""
from pathlib import Path
import sys

import pytest

from mcp_client import MCPClient


@pytest.fixture(scope="module")
def calculator_client():
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "mcp_servers" / "calculator_server.py"
    assert script.is_file(), f"Missing current MCP calculator server: {script}"
    client = MCPClient("calc", [sys.executable, str(script)])
    client.initialize()
    try:
        assert "calculator" in [tool["name"] for tool in client.tools]
        yield client
    finally:
        client.close()


@pytest.mark.parametrize("expression,expected", [
    ("2+3*4", "14"),
    ("(10+2)*3-5", "31"),
    ("17*23+1234", "1625"),
    ("-15+7", "-8"),
])
def test_calculator_arithmetic(calculator_client, expression, expected):
    result = str(calculator_client.call_tool("calculator", {"expression": expression})).strip()
    assert result == expected


@pytest.mark.parametrize("expression", ["open(1)", "abs(1)", "x+1"])
def test_calculator_rejects_python_expressions(calculator_client, expression):
    result = str(calculator_client.call_tool("calculator", {"expression": expression}))
    assert result.startswith("ERROR:"), result
