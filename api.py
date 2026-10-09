import os
import re
from copy import deepcopy
from pathlib import Path
from threading import Event, Lock, Thread
from queue import Queue
from fastapi.responses import StreamingResponse
from uuid import uuid4

from anthropic import Anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import agent

app = FastAPI(title="Agentic RAG Service")

PROJECT_ROOT = Path(__file__).resolve().parent

API_ALLOWED_TOOLS = {"retrieve_knowledge"}
API_MAX_TURNS = 6

SESSION_ID_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_-]{7,63}"
)

# V1.0: single-process, per-session synchronization.
_session_locks: dict[str, Lock] = {}
_locks_guard = Lock()

API_SYSTEM = (
    "You are a technical documentation assistant. "
    "Use retrieve_knowledge when answering project-specific "
    "questions. Base factual claims on retrieved evidence. "
    "Cite source document names. If evidence is insufficient, "
    "say so. Do not invent project details."
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None


def get_session_lock(session_id: str) -> Lock:
    with _locks_guard:
        if session_id not in _session_locks:
            _session_locks[session_id] = Lock()
        return _session_locks[session_id]


def resolve_session_id(session_id: str | None) -> str:
    if session_id is None:
        return uuid4().hex[:12]

    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise HTTPException(
            status_code=422,
            detail="Invalid session_id.",
        )

    return session_id


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(
            status_code=422,
            detail="Message must not be blank.",
        )

    session_id = resolve_session_id(req.session_id)

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Model service is not configured.",
        )

    session_lock = get_session_lock(session_id)

    with session_lock:
        session = agent.Session(
            session_id=session_id,
            cwd=PROJECT_ROOT,
        )

        before = deepcopy(session.messages)

        try:
            with Anthropic(
                api_key=api_key,
                base_url=agent.DEEPSEEK_BASE_URL,
            ) as client:
                answer = agent.agent_turn(
                    client=client,
                    session=session,
                    system_text=API_SYSTEM,
                    meter=agent.Meter(),
                    user_input=req.message,
                    on_text=lambda chunk: None,
                    allowed_tool_names=API_ALLOWED_TOOLS,
                    max_turns=API_MAX_TURNS,
                    raise_on_error=True,
                )

            if not isinstance(answer, str) or not answer.strip():
                raise agent.AgentTurnError(
                    "Agent returned no final answer."
                )

        except agent.AgentTurnError as exc:
            session.replace_messages(
                before,
                reason="http_request_failed",
            )
            raise HTTPException(
                status_code=503,
                detail="Agent could not complete the request.",
            ) from exc

        except Exception as exc:
            session.replace_messages(
                before,
                reason="http_request_failed",
            )
            raise HTTPException(
                status_code=500,
                detail="Internal agent error.",
            ) from exc

        return {
            "session_id": session_id,
            "answer": answer,
        }


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """Stream text produced by the SAME Agent core used by /chat.

    V1.0 local/single-process adapter: an Agent worker writes into a Queue,
    while StreamingResponse consumes its output on demand.
    """
    if not req.message.strip():
        raise HTTPException(status_code=422, detail="Message must not be blank.")

    session_id = resolve_session_id(req.session_id)
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Model service is not configured.")

    session_lock = get_session_lock(session_id)

    def generate():
        events = Queue()
        cancelled = Event()

        def on_text(chunk: str):
            if cancelled.is_set():
                raise agent.AgentTurnError("HTTP stream cancelled.")
            if chunk:
                events.put(("text", chunk))

        def worker():
            try:
                with session_lock:
                    # Cancellation might happen while waiting for this lock.
                    if cancelled.is_set():
                        return

                    session = agent.Session(session_id=session_id, cwd=PROJECT_ROOT)
                    before = deepcopy(session.messages)
                    try:
                        with Anthropic(
                            api_key=api_key,
                            base_url=agent.DEEPSEEK_BASE_URL,
                        ) as client:
                            answer = agent.agent_turn(
                                client=client,
                                session=session,
                                system_text=API_SYSTEM,
                                meter=agent.Meter(),
                                user_input=req.message,
                                on_text=on_text,
                                allowed_tool_names=API_ALLOWED_TOOLS,
                                max_turns=API_MAX_TURNS,
                                raise_on_error=True,
                            )

                        if not isinstance(answer, str) or not answer.strip():
                            raise agent.AgentTurnError("Agent returned no final answer.")
                        if cancelled.is_set():
                            raise agent.AgentTurnError("HTTP stream cancelled.")
                    except Exception:
                        # Persist the pre-request snapshot so Session._replay()
                        # also restores the previous state after a failure.
                        session.replace_messages(before, reason="http_stream_failed")
                        if not cancelled.is_set():
                            events.put((
                                "error",
                                "\n[stream error: agent could not complete the request]",
                            ))
            except Exception:
                if not cancelled.is_set():
                    events.put(("error", "\n[stream error: internal agent error]"))
            finally:
                events.put(("end", None))

        Thread(target=worker, daemon=True).start()
        try:
            while True:
                kind, payload = events.get()
                if kind == "end":
                    break
                if payload is not None:
                    yield payload
        finally:
            # Best-effort cancellation; an in-flight SDK call may not stop
            # immediately. The worker will release its lock when it exits.
            cancelled.set()

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"X-Session-ID": session_id, "Cache-Control": "no-cache"},
    )
