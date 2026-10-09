"""
A minimal MCP server in pure stdlib python. ~80 lines.

speaks JSON-RPC 2.0 over stdin/stdout. exposes one tool: `calculator`.
chapter 13 / 14 talk to this. you can also point claude desktop or cursor
at it — set in their MCP config:

  {
    "mcpServers": {
      "calculator": {"command": "python",
                     "args": ["/path/to/agent-zero-to-hero/mcp_servers/calculator_server.py"]}
    }
  }

we deliberately do NOT use FastMCP or any SDK. you should see the bytes.
"""

import ast
import json
import operator
import sys


PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "calculator", "version": "0.1.0"}


# ---- the actual tool implementation -------------------------------------

BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_expr(node):
    if isinstance(node, ast.Expression):
        return _eval_expr(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numeric literals are allowed")

    if isinstance(node, ast.BinOp):
        op = BINARY_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(
                f"operator not allowed: {type(node.op).__name__}"
            )
        return op(
            _eval_expr(node.left),
            _eval_expr(node.right),
        )

    if isinstance(node, ast.UnaryOp):
        op = UNARY_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(
                f"operator not allowed: {type(node.op).__name__}"
            )
        return op(_eval_expr(node.operand))

    raise ValueError(
        f"expression type not allowed: {type(node).__name__}"
    )

def calculator(args: dict) -> str:
    expression = args["expression"]

    if len(expression) > 200:
        raise ValueError("expression is too long")

    tree = ast.parse(expression, mode="eval")
    result = _eval_expr(tree)

    return str(result)


TOOLS = {
    "calculator": {
        "schema": {
            "name": "calculator",
            "description": "Evaluate a math expression like '2 + 3 * 4'.",
            "inputSchema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
        "fn": calculator,
    },
}


# ---- JSON-RPC 2.0 method handlers ---------------------------------------

def handle_initialize(params: dict) -> dict:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"tools": {}},      # we support tools, no resources/prompts
        "serverInfo": SERVER_INFO,
    }


def handle_tools_list(params: dict) -> dict:
    return {"tools": [t["schema"] for t in TOOLS.values()]}


def handle_tools_call(params: dict) -> dict:
    name = params["name"]
    if name not in TOOLS:
        # MCP error responses are content blocks with isError=True
        return {"content": [{"type": "text", "text": f"unknown tool: {name}"}],
                "isError": True}
    try:
        result = TOOLS[name]["fn"](params.get("arguments", {}))
        return {"content": [{"type": "text", "text": result}], "isError": False}
    except Exception as e:
        return {"content": [{"type": "text",
                             "text": f"{type(e).__name__}: {e}"}],
                "isError": True}


HANDLERS = {
    "initialize":  handle_initialize,
    "tools/list":  handle_tools_list,
    "tools/call":  handle_tools_call,
}


# ---- the main loop: read JSON lines from stdin, write JSON lines to stdout

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue                # malformed json: ignore (per JSON-RPC spec)

        # notifications have no id and expect no response.
        if "id" not in msg:
            continue

        method = msg.get("method", "")
        handler = HANDLERS.get(method)
        if handler is None:
            response = {"jsonrpc": "2.0", "id": msg["id"],
                        "error": {"code": -32601, "message": f"method not found: {method}"}}
        else:
            try:
                result = handler(msg.get("params") or {})
                response = {"jsonrpc": "2.0", "id": msg["id"], "result": result}
            except Exception as e:
                response = {"jsonrpc": "2.0", "id": msg["id"],
                            "error": {"code": -32000, "message": str(e)}}

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
