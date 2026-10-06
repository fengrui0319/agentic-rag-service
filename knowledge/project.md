# Agentic RAG Service

Agentic RAG Service is a lightweight Python project for building an LLM agent.

The agent uses DeepSeek as its language model provider.

The current agent supports a calculator tool. The model decides when to call the tool, while Python executes the actual function.

The project exposes the agent through a FastAPI backend.

The API currently provides two endpoints:
- GET /health checks whether the service is running.
- POST /chat accepts a user message and returns the agent's answer.

The project will also support retrieval-augmented generation, allowing the agent to answer questions using local documents.