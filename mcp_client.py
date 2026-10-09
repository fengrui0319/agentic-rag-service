from __future__ import annotations

import json
import subprocess
import sys


PROTOCOL_VERSION = "2024-11-05"


class MCPProcess:
    """Manage one stdio MCP server process using JSON-RPC 2.0."""

    def __init__(self, command: list[str]):
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
        )
        self.next_id = 1

    def call(self, method: str, params: dict | None = None) -> dict:
        if self.proc.poll() is not None:
            raise RuntimeError("MCP server is not running")

        request = {
            "jsonrpc": "2.0",
            "id": self.next_id,
            "method": method,
        }

        if params is not None:
            request["params"] = params

        self.next_id += 1

        assert self.proc.stdin is not None
        assert self.proc.stdout is not None

        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()

        while True:
            raw = self.proc.stdout.readline()

            if not raw:
                raise RuntimeError("MCP server exited unexpectedly")

            message = json.loads(raw)

            # Notifications have no id. Ignore them here.
            if message.get("id") != request["id"]:
                continue

            if "error" in message:
                raise RuntimeError(f"MCP error: {message['error']}")

            return message.get("result", {})

    def notify(self, method: str, params: dict | None = None) -> None:
        if self.proc.poll() is not None:
            raise RuntimeError("MCP server is not running")

        message = {
            "jsonrpc": "2.0",
            "method": method,
        }

        if params is not None:
            message["params"] = params

        assert self.proc.stdin is not None

        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def close(self) -> None:
        if self.proc.poll() is not None:
            return

        if self.proc.stdin is not None:
            self.proc.stdin.close()

        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class MCPClient:
    """High-level wrapper around one MCP server."""

    def __init__(self, server_id: str, command: list[str]):
        self.server_id = server_id
        self.process = MCPProcess(command)
        self.tools: list[dict] = []
        self.initialized = False

    def initialize(self) -> None:
        self.process.call(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "agentic-rag-service",
                    "version": "0.1.0",
                },
            },
        )

        self.process.notify("notifications/initialized")

        result = self.process.call("tools/list")
        self.tools = result.get("tools", [])
        self.initialized = True

    def call_tool(self, name: str, arguments: dict) -> str:
        if not self.initialized:
            raise RuntimeError(
                f"MCP server '{self.server_id}' has not been initialized"
            )

        result = self.process.call(
            "tools/call",
            {
                "name": name,
                "arguments": arguments,
            },
        )

        texts = [
            block.get("text", "")
            for block in result.get("content", [])
            if block.get("type") == "text"
        ]

        output = "\n".join(texts)

        if result.get("isError"):
            return f"ERROR: {output}"

        return output or "(no output)"

    def close(self) -> None:
        self.process.close()


def to_model_tool(server_id: str, mcp_tool: dict) -> dict:
    """
    Convert an MCP tool schema into the tool schema expected by our LLM client.

    Example:
        calculator
        ->
        mcp__calc__calculator
    """
    return {
        "name": f"mcp__{server_id}__{mcp_tool['name']}",
        "description": (
            f"[MCP:{server_id}] "
            f"{mcp_tool.get('description', '')}"
        ),
        "input_schema": mcp_tool["inputSchema"],
    }


def parse_mcp_tool_name(name: str) -> tuple[str, str]:
    """
    mcp__calc__calculator
    ->
    ("calc", "calculator")
    """
    if not name.startswith("mcp__"):
        raise ValueError(f"not an MCP tool name: {name}")

    parts = name.split("__", 2)

    if len(parts) != 3 or not parts[1] or not parts[2]:
        raise ValueError(f"invalid MCP tool name: {name}")

    return parts[1], parts[2]