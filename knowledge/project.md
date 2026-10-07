# Agentic RAG Service

Agentic RAG Service is a lightweight Python service for building an LLM agent.

The agent uses DeepSeek through an Anthropic-compatible API.

The current agent supports:
- calculator tool calling
- retrieval-augmented generation using local documents
- multi-turn conversations with session history
- streaming responses

The project exposes the agent through a FastAPI backend.