# Architecture

## Runtime and entry points

The current system has **one Agent runtime** in `agent.py` with two entry points:

- **CLI (`python agent.py`)**: a terminal coding Agent with local file/search/shell tools, task tracking, Skills, a read-only `Task` subagent, and dynamically discovered MCP tools when its MCP servers are initialized.
- **FastAPI (`api.py`)**: `GET /health`, `POST /chat`, and `POST /chat/stream`. Both chat routes call the same `agent_turn()` function. For this local V1.0 HTTP adapter, the model can use **only** `retrieve_knowledge`, with a maximum of **6 model turns per request**; the API does not expose shell/file-editing tools, `Task`, Skills, or MCP tools.

The legacy `mini_agent.py` is still in the repository for historical code/tests but is **not** the Agent invoked by the current FastAPI adapter.

## Agent loop and tools

`agent_turn()` sends messages and tool schemas to DeepSeek through the Anthropic-compatible API, processes model stop reasons, dispatches tool calls in Python, appends tool results to the conversation, and continues until completion or failure.

The local tools defined in `agent.py` are `Read`, `Write`, `Edit`, `Bash`, `Glob`, `Grep`, `TodoWrite`, `read_skill`, `Task`, and `retrieve_knowledge`. MCP tools are discovered separately on CLI startup; the included examples are a calculator server and a **demonstration** weather server (not a live weather source).

The CLI asks for approval before `Write`, `Edit`, or `Bash` by default (configurable). This is not a filesystem sandbox: other local tools may read project or host files. HTTP uses a separate tool allowlist enforced both in model-visible schemas and before dispatch.

`read_skill` loads a discovered Skill's full instructions on demand. `Task` starts a read-only child Agent using a separate message history; only its result is returned to the parent. These capabilities have been exercised in CLI tests, not exposed by the current HTTP routes.

## RAG pipeline — current V1.0 baseline

The current `rag.py` **only indexes immediate `*.md` files** in `knowledge/`:

1. Split Markdown at headings, using overlapping fixed-size windows for longer sections (200 characters, 50-character overlap).
2. Embed the chunks with local Sentence Transformers model `paraphrase-multilingual-MiniLM-L12-v2`.
3. Embed the query with the same model, normalize embeddings, and rank via cosine similarity.
4. Return up to 3 passages above the configured similarity threshold (`0.35`) as text with `[Source: filename]` labels.

The embedding model and document index load lazily and are cached **in process**. They are not automatically refreshed after Markdown edits; restart the service to re-index. No vector database, BM25, RRF, Reranker, Python AST indexing or function/line-number retrieval is implemented in this V1.0 baseline.

## Session history, compaction, and failures

`agent.Session` stores messages in an **append-only JSONL event log on disk**, and replays it when reopening a session ID. It supports clearing the session and persisting compaction snapshots. The CLI can trigger conversation summarization and resume a previous session; compaction itself invokes a model API and consumes tokens.

The HTTP adapter validates session ID syntax, serializes requests with the same ID using an **in-process** lock, and restores the pre-request message snapshot after a handled Agent failure. This is not an atomic transaction against abrupt process termination. The service does not implement identity-based authorization, distributed locking, or a production multi-user session store.

The `text/plain` stream is backed by a worker thread and queue. It may contain interim Agent text; once streaming begins, later errors are signaled in the response body rather than by changing the HTTP status code. Client disconnect cancellation is best-effort.

## Current limits and planned evaluation

The deployed HTTP surface is restricted to knowledge-base questions. Coding tools and MCP are available only through the current CLI path. The small Markdown corpus and current Dense retriever are the V1.0 baseline; Code-aware chunking, Hybrid BM25/Dense with RRF, Reranking, and richer retrieval evaluation are **planned for V1.5**, not yet implemented.
