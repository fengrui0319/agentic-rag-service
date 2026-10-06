import os
from anthropic import Anthropic

def calculator(expression: str) -> str:
    return str(eval(expression, {"__builtins__": {}}))

DISPATCH = {
    "calculator": calculator,
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

client = Anthropic()
MODEL = os.environ["AZH_MODEL"]


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
            system="For every arithmetic question, you MUST use the calculator tool. Never calculate arithmetic yourself.",
            tools=[CALCULATOR_TOOL],
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

        tool_call = next(
            b for b in r.content
            if b.type == "tool_use"
        )

        print("tool call:", tool_call.name, tool_call.input)

        handler = DISPATCH[tool_call.name]
        result = handler(**tool_call.input)

        print("tool result:", result)

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": result,
                }
            ],
        })

if __name__ == "__main__":
    answer = run_agent("what is 9876543 * 1234567?")
    print(answer)

