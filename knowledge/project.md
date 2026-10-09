# Agentic RAG Service

Agentic RAG Service is a Python project that combines a DeepSeek-backed tool-using Agent runtime with a local Markdown retrieval tool and a FastAPI development service.

## Implemented in the current Agent core

- Streaming and non-streaming model interactions through DeepSeek's Anthropic-compatible endpoint.
- Local coding/search tools (`Read`, `Write`, `Edit`, `Bash`, `Glob`, `Grep`), TODO tracking, Skills loaded on demand, and a read-only `Task` subagent.
- Example MCP tools discovered from calculator and demo-weather servers when running the CLI.
- JSONL-backed conversation sessions, resume, conversation compaction, retry/error handling, and estimated token cost reporting in CNY.
- A local `retrieve_knowledge` tool backed by Sentence Transformer embeddings of Markdown files in `knowledge/`.

## Current FastAPI scope

- `GET /health`: application liveness.
- `POST /chat`: full-answer JSON with a session ID.
- `POST /chat/stream`: incremental **plain-text** output with `X-Session-ID` header.
- Both chat endpoints use the new `agent.py` execution core but expose **only the RAG retrieval tool**, not local shell, write, MCP, Skills, or Subagent tools. Requests run in a single-process local service with a per-session lock and JSONL message history.

The legacy `mini_agent.py` remains in the repository but is not the live HTTP Agent. HTTP session history survives normal restarts because it is stored in JSONL files, not only in RAM. The retrieval **index**, by contrast, is an in-memory cache rebuilt after restarting the process.

## Scope and limits

The current RAG corpus is a small Markdown documentation collection, **not** a Python codebase index. This V1.0 service is local-only, unauthenticated, and not production-ready. AST-aware code retrieval, Hybrid Search/RRF, Reranking, broad benchmarks, and multi-hop code evidence gathering are future upgrades, not current capabilities.
