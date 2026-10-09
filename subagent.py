from __future__ import annotations

from typing import Callable


def run_subagent(
    client,
    task: str,
    *,
    model: str,
    tools: list[dict],
    dispatch: dict[str, Callable],
    system: str = "",
    max_turns: int = 20,
):
    """
    Run a fresh child-agent loop.

    The child gets its own messages array.
    Only its final answer is returned to the parent.

    Returns:
        final_answer, usage_records
    """
    messages = [
        {
            "role": "user",
            "content": task,
        }
    ]

    usage_records = []
    final_text = ""

    for _ in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=(
                system
                or (
                    "You are a focused read-only research subagent. "
                    "Inspect the available files carefully. "
                    "Do not invent facts. "
                    "Return a concise final answer to the parent."
                )
            ),
            tools=tools,
            messages=messages,
        )

        if response.usage is not None:
            usage_records.append(response.usage)

        # Important for DeepSeek thinking:
        # preserve every returned block, including thinking/signature.
        assistant_content = [
            block.model_dump()
            if hasattr(block, "model_dump")
            else block
            for block in response.content
        ]

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
            }
        )

        text_parts = [
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
            and getattr(block, "text", "").strip()
        ]

        if text_parts:
            final_text = "\n".join(text_parts)

        stop_reason = response.stop_reason

        if stop_reason == "tool_use":
            # Continue below and execute the requested tools.
            pass

        elif stop_reason in ("end_turn", "stop_sequence"):
            return (
                final_text or "(subagent had no final answer)",
                usage_records,
            )

        elif stop_reason == "max_tokens":
            return (
                "ERROR: subagent response was truncated by max_tokens",
                usage_records,
            )

        elif stop_reason == "refusal":
            return (
                "ERROR: subagent model stopped with stop_reason='refusal'",
                usage_records,
            )

        elif stop_reason == "pause_turn":
            return (
                "ERROR: subagent model stopped with stop_reason='pause_turn'; "
                "server-side continuation is not implemented",
                usage_records,
            )

        elif stop_reason == "model_context_window_exceeded":
            return (
                "ERROR: subagent context window was exceeded",
                usage_records,
            )

        else:
            return (
                f"ERROR: unexpected subagent stop_reason: {stop_reason!r}",
                usage_records,
            )

        tool_results = []

        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue

            name = block.name
            args = block.input

            handler = dispatch.get(name)

            if handler is None:
                result = f"ERROR: subagent cannot use tool: {name}"
                is_error = True
            else:
                try:
                    result = handler(**args)
                    is_error = str(result).startswith("ERROR")
                except Exception as exc:
                    result = (
                        f"ERROR: {type(exc).__name__}: {exc}"
                    )
                    is_error = True

            tool_result = {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(result),
            }

            if is_error:
                tool_result["is_error"] = True

            tool_results.append(tool_result)

        messages.append(
            {
                "role": "user",
                "content": tool_results,
            }
        )

    return (
        f"ERROR: subagent exceeded {max_turns} turns",
        usage_records,
    )