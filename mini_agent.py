import os
from anthropic import Anthropic
from dotenv import load_dotenv
from rag import retrieve_knowledge

def calculator(expression: str) -> str:
    return str(eval(expression, {"__builtins__": {}}))

DISPATCH = {
    "calculator": calculator,
    "retrieve_knowledge": retrieve_knowledge,
}

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

def run_agent(prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    while True:
        r = client.messages.create(
            model=MODEL,
            max_tokens=512,
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
            return text_block.text

        tool_calls = [
            b for b in r.content
            if b.type == "tool_use"
        ]

        tool_results = []

        for tool_call in tool_calls:
            print("tool call:", tool_call.name, tool_call.input)

            handler = DISPATCH[tool_call.name]
            result = handler(**tool_call.input)

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

if __name__ == "__main__":
    answer = run_agent(
        "How can I check whether this project service is running?"
    )
    print(answer)

