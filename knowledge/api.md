# API

The FastAPI service currently provides three endpoints.

## GET /health

Checks whether the service is running.

Example response:

```json
{
  "status": "ok"
}
```

## POST /chat

Accepts a user message and returns the agent's complete answer.

The request contains:

- `message`: the user's message
- `session_id`: optional session ID for continuing a conversation

If no session ID is provided, the server creates one.

The response contains:

- `session_id`
- `answer`

Sessions are stored in memory, so restarting the server clears existing session history.

## POST /chat/stream

Provides a streaming version of the chat endpoint.

The response body is streamed incrementally as text.

For streaming responses, the session ID is returned in the `X-Session-ID` response header.

The same session ID can be supplied in a later request to continue the conversation.

Sessions are stored in memory, so restarting the server clears existing session history.