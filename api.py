from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mini_agent import (
    run_agent_with_history,
    stream_agent_with_history,
    trim_history,
)


app = FastAPI()

SESSIONS: dict[str, list[dict]] = {}

MAX_MESSAGE_LENGTH = 4000

class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
    )
    session_id: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid4())

    history = SESSIONS.get(session_id, [])

    try:
        answer, updated_history = run_agent_with_history(
            req.message,
            history,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail="Agent could not complete the request.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Internal agent error.",
        ) from exc

    SESSIONS[session_id] = updated_history

    return {
        "session_id": session_id,
        "answer": answer,
    }

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    session_id = req.session_id or str(uuid4())

    history = SESSIONS.get(session_id, [])

    def generate():
        chunks = []

        for text in stream_agent_with_history(
            req.message,
            history,
        ):
            chunks.append(text)
            yield text

        answer = "".join(chunks)

        updated_history = history + [
            {
                "role": "user",
                "content": req.message,
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]

        SESSIONS[session_id] = trim_history(updated_history)

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "X-Session-ID": session_id,
        },
    )