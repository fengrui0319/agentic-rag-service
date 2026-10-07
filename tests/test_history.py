from mini_agent import MAX_HISTORY_MESSAGES, trim_history


def test_trim_history_keeps_only_recent_messages():
    history = [
        {
            "role": "user",
            "content": str(i),
        }
        for i in range(30)
    ]

    result = trim_history(history)

    assert len(result) == MAX_HISTORY_MESSAGES
    assert result[0]["content"] == "10"
    assert result[-1]["content"] == "29"


def test_short_history_is_unchanged():
    history = [
        {
            "role": "user",
            "content": "hello",
        },
        {
            "role": "assistant",
            "content": "hi",
        },
    ]

    result = trim_history(history)

    assert result == history


def test_empty_history_stays_empty():
    assert trim_history([]) == []