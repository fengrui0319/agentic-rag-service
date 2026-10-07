import ast
import operator
import os
from anthropic import Anthropic
from dotenv import load_dotenv
from rag import retrieve_knowledge

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

MAX_EXPRESSION_LENGTH = 100
MAX_ABS_NUMBER = 1_000_000_000_000
MAX_EXPONENT = 10


def _eval_math_node(node):
    if isinstance(node, ast.Expression):
        return _eval_math_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise ValueError("Boolean values are not supported.")

        if not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric values are supported.")

        if abs(node.value) > MAX_ABS_NUMBER:
            raise ValueError("Number is too large.")

        return node.value

    if isinstance(node, ast.UnaryOp):
        operator_fn = _UNARY_OPERATORS.get(type(node.op))

        if operator_fn is None:
            raise ValueError("Unsupported unary operator.")

        return operator_fn(
            _eval_math_node(node.operand)
        )

    if isinstance(node, ast.BinOp):
        operator_fn = _BINARY_OPERATORS.get(type(node.op))

        if operator_fn is None:
            raise ValueError("Unsupported binary operator.")

        left = _eval_math_node(node.left)
        right = _eval_math_node(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError("Exponent is too large.")

        result = operator_fn(left, right)

        if isinstance(result, (int, float)) and abs(result) > MAX_ABS_NUMBER:
            raise ValueError("Result is too large.")

        return result

    raise ValueError(
        f"Unsupported expression: {type(node).__name__}"
    )


def calculator(expression: str) -> str:
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("Expression is too long.")

    try:
        tree = ast.parse(
            expression,
            mode="eval",
        )
    except SyntaxError as exc:
        raise ValueError("Invalid mathematical expression.") from exc

    result = _eval_math_node(tree)

    return str(result)

DISPATCH = {
    "calculator": calculator,
    "retrieve_knowledge": retrieve_knowledge,
}

def execute_tool(
    tool_name: str,
    tool_input: dict,
) -> str:
    handler = DISPATCH.get(tool_name)

    if handler is None:
        return f"Tool error: unknown tool '{tool_name}'."

    try:
        return str(handler(**tool_input))
    except Exception as exc:
        return (
            f"Tool error: {type(exc).__name__}: {exc}"
        )

CALCULATOR_TOOL = {
    "name": "calculator",
    "description": "Evaluate a math expression.",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string"
            }
        },
        "required": ["expression"]
    }
}

RETRIEVE_TOOL = {
    "name": "retrieve_knowledge",
    "description": (
        "Search the local project knowledge base for information relevant "
        "to the user's question. Use this for questions about this project's "
        "API, architecture, capabilities, or documentation."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string"
            }
        },
        "required": ["query"]
    }
}

load_dotenv(override=True)

MODEL = os.environ["AZH_MODEL"]

client = Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url=os.environ["ANTHROPIC_BASE_URL"],
)

MAX_STEPS = 6
MAX_TOKENS = 512
MAX_HISTORY_MESSAGES = 20

def trim_history(history: list[dict]) -> list[dict]:
    return history[-MAX_HISTORY_MESSAGES:]

def run_agent_with_history(
    prompt: str,
    history: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    if history is None:
        history = []

    history = trim_history(history)

    messages = history + [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    for step in range(MAX_STEPS):
        r = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=(
                "You are a tool-using assistant. "
                "For arithmetic questions, use the calculator tool. "
                "For questions about this project, its API, architecture, capabilities, "
                "or local documentation, use the retrieve_knowledge tool. "
                "Use retrieved information to answer accurately."
                "Treat retrieved context as the source of truth for project-specific facts. "
                "Do not invent project details that are not present in the retrieved context. "
                "If the information is missing, say so."
            ),
            tools=[
                CALCULATOR_TOOL,
                RETRIEVE_TOOL,
            ],
            messages=messages,
        )

        if r.stop_reason == "max_tokens":
            raise RuntimeError(
                f"Model response was truncated at {MAX_TOKENS} tokens."
            )

        messages.append({
            "role": "assistant",
            "content": [
                b.model_dump(exclude_none=True)
                for b in r.content
            ]
        })

        if r.stop_reason != "tool_use":
            text_block = next(
                b for b in r.content
                if b.type == "text"
            )
            answer = text_block.text

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

            return answer, trim_history(updated_history)

        tool_calls = [
            b for b in r.content
            if b.type == "tool_use"
        ]

        tool_results = []

        for tool_call in tool_calls:
            print("tool call:", tool_call.name, tool_call.input)

            result = execute_tool(
                tool_call.name,
                tool_call.input,
            )

            print("tool result:", result)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": str(result),
            })

        messages.append({
            "role": "user",
            "content": tool_results,
        })

    raise RuntimeError(
        f"Agent exceeded maximum steps ({MAX_STEPS})."
    )

def stream_agent_with_history(
    prompt: str,
    history: list[dict] | None = None,
):
    if history is None:
        history = []

    history = trim_history(history)

    messages = history + [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    for step in range(MAX_STEPS):
        with client.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=(
                "You are a tool-using assistant. "
                "For arithmetic questions, use the calculator tool. "
                "For questions about this project, its API, architecture, capabilities, "
                "or local documentation, use the retrieve_knowledge tool. "
                "Use retrieved information to answer accurately. "
                "Treat retrieved context as the source of truth for project-specific facts. "
                "Do not invent project details that are not present in the retrieved context. "
                "If the information is missing, say so."
            ),
            tools=[
                CALCULATOR_TOOL,
                RETRIEVE_TOOL,
            ],
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text

            r = stream.get_final_message()

        if r.stop_reason == "max_tokens":
            raise RuntimeError(
                f"Model response was truncated at {MAX_TOKENS} tokens."
            )

        messages.append({
            "role": "assistant",
            "content": [
                b.model_dump(exclude_none=True)
                for b in r.content
            ]
        })

        if r.stop_reason != "tool_use":
            return

        tool_calls = [
            b for b in r.content
            if b.type == "tool_use"
        ]

        tool_results = []

        for tool_call in tool_calls:
            print("tool call:", tool_call.name, tool_call.input)

            result = execute_tool(
                tool_call.name,
                tool_call.input,
            )

            print("tool result:", result)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": str(result),
            })

        messages.append({
            "role": "user",
            "content": tool_results,
        })

    raise RuntimeError(
        f"Agent exceeded maximum steps ({MAX_STEPS})."
    )

def run_agent(prompt: str) -> str:
    answer, _ = run_agent_with_history(prompt)
    return answer

if __name__ == "__main__":
    history = []

    answer1, history = run_agent_with_history(
        "The temporary test label for this session is ALPHA-27. Remember it.",
        history,
    )
    print("\n--- turn 1 ---")
    print(answer1)

    answer2, history = run_agent_with_history(
        "What temporary test label did I just give you?",
        history,
    )
    print("\n--- turn 2 ---")
    print(answer2)

