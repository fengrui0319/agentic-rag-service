import json
import sys


PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "weather", "version": "0.1.0"}


# ---- actual tool ---------------------------------------------------------

WEATHER_DATA = {
    "beijing": "Sunny, 22°C",
    "shanghai": "Cloudy, 24°C",
    "london": "Rainy, 15°C",
    "tokyo": "Clear, 20°C",
}


def weather(args: dict) -> str:
    city = args["city"].strip().lower()

    if city not in WEATHER_DATA:
        return f"No demo weather data for {city}."

    return WEATHER_DATA[city]


TOOLS = {
    "weather": {
        "schema": {
            "name": "weather",
            "description": (
                "Return demo weather for a city from a built-in table. "
                "This is not live weather data."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"],
            },
        },
        "fn": weather,
    },
}


# ---- MCP handlers --------------------------------------------------------

def handle_initialize(params: dict) -> dict:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "serverInfo": SERVER_INFO,
    }


def handle_tools_list(params: dict) -> dict:
    return {
        "tools": [t["schema"] for t in TOOLS.values()]
    }


def handle_tools_call(params: dict) -> dict:
    name = params["name"]

    if name not in TOOLS:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"unknown tool: {name}"
                }
            ],
            "isError": True,
        }

    try:
        result = TOOLS[name]["fn"](
            params.get("arguments", {})
        )

        return {
            "content": [
                {
                    "type": "text",
                    "text": result
                }
            ],
            "isError": False,
        }

    except Exception as e:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"{type(e).__name__}: {e}"
                }
            ],
            "isError": True,
        }


HANDLERS = {
    "initialize": handle_initialize,
    "tools/list": handle_tools_list,
    "tools/call": handle_tools_call,
}


# ---- main MCP loop -------------------------------------------------------

def main():
    for line in sys.stdin:
        line = line.strip()

        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        # notification: no response
        if "id" not in msg:
            continue

        method = msg.get("method", "")
        handler = HANDLERS.get(method)

        if handler is None:
            response = {
                "jsonrpc": "2.0",
                "id": msg["id"],
                "error": {
                    "code": -32601,
                    "message": f"method not found: {method}",
                },
            }

        else:
            try:
                result = handler(
                    msg.get("params") or {}
                )

                response = {
                    "jsonrpc": "2.0",
                    "id": msg["id"],
                    "result": result,
                }

            except Exception as e:
                response = {
                    "jsonrpc": "2.0",
                    "id": msg["id"],
                    "error": {
                        "code": -32000,
                        "message": str(e),
                    },
                }

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()