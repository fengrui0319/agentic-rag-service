# API

This document describes the **V1.0 local FastAPI adapter** in `api.py`, which uses the current `agent.py` runtime (not the legacy `mini_agent.py`). It is a single-process development service, not an authenticated multi-user deployment.

## GET /health

Returns `{"status": "ok"}` with HTTP 200 when the HTTP application is responding. This is a liveness check; it does **not** verify DeepSeek availability or that the local RAG index can be loaded.

## POST /chat

Accepts JSON:

```json
{
  "message": "What does GET /health do?",
  "session_id": "optional-valid-session-id"
}
```

- `message` is required, non-blank and at most 4,000 characters.
- `session_id` is optional. If omitted, the server creates a new one and returns it. If present, it must be 8–64 characters long, start with an ASCII letter or digit, and otherwise contain only letters, digits, `_` or `-`.
- The response is JSON containing `session_id` and the completed `answer`.
- HTTP execution uses the **current Agent Core** with only `retrieve_knowledge` exposed as a tool and a limit of **6 model turns per request**. CLI permissions and tools are different.
- Invalid input or session IDs result in HTTP 422. Missing model configuration returns HTTP 503. An Agent execution failure returns an HTTP error (typically 503; unexpected internal failures are mapped to 500).

## POST /chat/stream

Accepts the same JSON request and session ID rules as `/chat`.

- The response is a `text/plain` HTTP stream, **not** Server-Sent Events (SSE).
- Its `X-Session-ID` response header supplies the session ID.
- Text is emitted as the Agent produces it. This can include short interim statements before tool calls as well as the final answer; it is not guaranteed to contain only the final answer.
- If an error occurs **after streaming has begun**, the HTTP status may remain 200. The stream may instead end with an in-band marker such as `[stream error: agent could not complete the request]`; clients must not interpret HTTP 200 alone as task success.
- Client disconnect cancellation is best-effort: a DeepSeek request already in progress may continue briefly.

## Conversation persistence

Both routes use `agent.Session`, which appends session events to JSONL files under the configured user-level Session directory (see `agent.py`). **Sessions are not stored only in memory.** A later request using the same valid session ID can reload prior conversation history from disk, including after a normal service restart.

For a handled failed request, the HTTP adapter restores the pre-request message snapshot using `replace_messages()`. Earlier failed events can still be present in the append-only JSONL file. Abrupt process termination is not a fully transactional rollback. A per-session lock prevents concurrent modifications **within one server process**, not across multiple workers or machines.

A session ID is an identifier, **not authentication**. Run this development server bound to `127.0.0.1`; do not expose it to the public Internet without additional security work.

## Knowledge sources

`retrieve_knowledge` searches the Markdown files currently present in `knowledge/`. Results cite source filenames. The in-process index is cached; after editing these documents, restart the FastAPI process to rebuild the index.
