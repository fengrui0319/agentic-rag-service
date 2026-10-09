# Agentic RAG Service

**A local-first LLM Agent system combining tool orchestration, persistent conversations, retrieval-augmented generation, and a streaming HTTP interface.**

`Python` · `FastAPI` · `DeepSeek API` · `MCP` · `Skills` · `Subagents` · `Sentence Transformers` · `pytest`

The project brings together a coding-oriented Agent runtime and a document-grounded question-answering service. It uses a **shared Agent execution core** for both the terminal interface and FastAPI, with different tool-access policies for each entry point.

## Engineering Highlights

- **Unified Agent execution:** Multi-turn tool-use loop, streamed text and reasoning-block handling, explicit stop-reason processing, bounded retries, and structured failure propagation.
- **Extensible tool ecosystem:** Local filesystem/search/shell tools, stdio-based **MCP** tool discovery and dispatch, on-demand **Skills** loading, and a read-only **Subagent** with separate conversation context.
- **Persistent conversation state:** Append-only **JSONL** event log, session resume and replay, context compaction snapshots, and request-level rollback for handled HTTP failures.
- **HTTP integration and execution boundaries:** FastAPI `/chat` and `/chat/stream` share `agent_turn()`. HTTP requests expose only the `retrieve_knowledge` tool, with per-session serialization and a bounded model-turn budget.
- **Retrieval pipeline with measurable baseline:** Markdown-aware chunking, local embeddings, cosine-similarity retrieval, source attribution, out-of-domain thresholding, and reproducible offline evaluation.

**Verification:** The Windows regression suite completed **41/41 tests in 8.17 seconds** after a Session path-compatibility fix. Separately, the CLI RAG workflow and both HTTP chat endpoints were exercised using real DeepSeek requests.

## System Architecture

```text
                  +--------------------------------------+
                  |           Agent Runtime              |
                  |    tool loop / streaming / retries   |
                  |     permissions / stop handling      |
                  +------------------+-------------------+
                                     |
              +----------------------+----------------------+
              |                                             |
       CLI: agent.py                                   FastAPI: api.py
       interactive / one-shot                         GET  /health
              |                                       POST /chat
              |                                       POST /chat/stream
     +--------+-------------------+                        |
     |        |        |          |               retrieve_knowledge only
  Local     MCP      Skills    Task /                      |
  tools    servers  on-demand  Subagent                rag.py
     |        |        |          |                        |
     +--------+--------+----------+             Markdown chunks + local
              |                                    embeddings / Top-K
         JSONL sessions                                |
         resume / compaction                       knowledge/*.md
```

**One runtime, different capabilities:** The CLI can access the coding-tool ecosystem, while the HTTP interface intentionally uses a narrow retrieval-only allowlist. The service does **not** expose remote shell or file-write access through the chat endpoints.

## Technical Design

### 1. Agent Runtime and Tool Routing

The core `agent_turn()` handles the model's response cycle: submit messages and tool schemas, process streamed blocks, route tool calls, append tool results, and terminate according to explicit stop reasons. A configurable upper bound prevents indefinitely continuing the model-tool loop.

Tools are dispatched through a local registry or the MCP namespace. The CLI prompts for approval before `Write`, `Edit`, or `Bash` by default. The HTTP adapter applies a separate `retrieve_knowledge`-only allowlist.

### 2. MCP, Skills, and Subagent

- **MCP:** Initializes included stdio servers, discovers their tool schemas, and routes calls through namespaced tools. Included servers demonstrate calculator and example-weather integrations; the weather server uses fixed sample data, not live forecasts.
- **Skills:** Builds a lightweight catalog of Skill descriptions and loads full instructions on demand via `read_skill`, avoiding the need to inject every Skill body into the initial context.
- **Subagent:** `Task` delegates bounded, read-only investigation to a child Agent with its own message history and a restricted tool set (`Read`, `Glob`, `Grep`, `read_skill`). Child usage is incorporated into the local cost meter.

### 3. Session Persistence and Recovery

`Session` writes conversation events to an append-only JSONL log. Replaying these events reconstructs the working message history on resume. Compaction writes a replacement message snapshot that can be replayed after reopening the session.

For HTTP requests, a per-session lock serializes calls inside a single process. On a handled failure, the API writes a snapshot of the pre-request message state. Session identifiers are validated, and Session-directory naming handles Windows paths.

This provides recoverable application state for the tested flows; it is **not** a transactional database or a cross-process concurrency mechanism.

### 4. Retrieval-Augmented HTTP Service

The current retrieval path loads Markdown documents from `knowledge/`, splits sections using heading-aware chunking and overlapping windows, embeds passages using `paraphrase-multilingual-MiniLM-L12-v2`, and ranks by cosine similarity. Results include source-document labels and are filtered by a minimum similarity threshold.

Both HTTP chat endpoints run against the same Agent core:

| Endpoint | Response | Notes |
|---|---|---|
| `GET /health` | JSON health response | Service liveness |
| `POST /chat` | JSON `session_id` and `answer` | Non-streaming HTTP response |
| `POST /chat/stream` | Incremental `text/plain` | Session ID in `X-Session-ID`; may include text before tool calls |

Both POST routes accept a `message` and an optional `session_id`. The HTTP Agent is limited to **6 model turns** per request.

## Verification and Evaluation

### Regression Tests

On Windows, after the Session path fix:

```text
41 passed in 8.17s
```

The automated suite exercises HTTP endpoints and error responses (using mock model calls), Session replay/rollback and same-session request serialization, MCP calculator behavior, tool permissions, and core RAG utilities. The **41 tests are not 41 paid model calls**.

### V1.0 Retrieval Baseline

The current corpus contains **three project Markdown documents**. The offline evaluation uses nine in-domain questions and two out-of-domain questions:

| Metric | Result |
|---|---:|
| Source Hit@1 | **4/9 (44.4%)** |
| Source Hit@3 | **7/9 (77.8%)** |
| Out-of-domain rejection | **2/2 (100.0%)** |

These results describe a **small, source-document retrieval benchmark**, not general answer accuracy. They provide a reproducible starting point for the planned code-retrieval evaluation; the tiny sample should not be read as production-level evidence.

```powershell
python -m pytest -v
python eval.py
```

## Getting Started

### Requirements and Configuration

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set your own key in the local `.env` file; never commit it.

```dotenv
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/anthropic
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_PRICE_TIER=peak
```

The runtime uses DeepSeek via its Anthropic-compatible interface. **Real model calls consume billed API tokens**. The local Sentence Transformer may download model weights on first use, but embeddings do not call the DeepSeek API. Usage and CNY cost displayed in the CLI are estimates.

### CLI

```powershell
python agent.py --help
python agent.py "Explain the available tools."
python agent.py
```

### HTTP Service

Start the local API from the project root:

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

Example request from a second PowerShell terminal:

```powershell
$body = @{
    message = "Use retrieve_knowledge to explain GET /health. Cite the source document."
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/chat" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

`/chat/stream` returns incremental plain text, not SSE. Knowledge-document edits require an API restart to rebuild the process-local retrieval index.

## Repository Guide

| Path | Responsibility |
|---|---|
| `agent.py` | Agent loop, CLI, tool dispatch, Session management |
| `api.py` | FastAPI adapter, request limits, Session locking, streaming |
| `rag.py` | Local Markdown retrieval baseline |
| `mcp_client.py`, `mcp_servers/` | MCP integration and included tool servers |
| `skills.py`, `skills/` | Skills catalog and on-demand loading |
| `subagent.py` | Read-only child-Agent execution |
| `knowledge/` | Project documentation used by the current RAG pipeline |
| `tests/`, `eval.py` | Regression tests and offline retrieval evaluation |

## Development Roadmap

The project is being evolved incrementally, with measurements attached to features **after implementation**.

- **Next — Code-aware RAG:** Index actual Python source and technical documentation; introduce AST-aware chunking, symbol/file/line metadata, BM25 + dense hybrid retrieval, reciprocal-rank fusion, reranker comparisons, and manually verified retrieval cases.
- **Further Agent engineering:** Explore multi-hop code evidence collection, MCP tool-description lazy loading, stricter execution boundaries, and latency/cost benchmarking where justified by test results.

These are planned directions; they are **not included in the current measured baseline**.

## Security and Scope

This is a **local development service**, not an authenticated multi-user deployment. The HTTP interface deliberately restricts tool access, but session IDs are not identity credentials. Locking applies within one process only; stream cancellation is best-effort, and errors after streaming starts may appear in the text body even when HTTP status is 200. CLI approval prompts are not a filesystem or process sandbox.

Session logs are stored on disk and should remain outside version control. The retrieval index is currently process-local and dense-only; code-level citations and hybrid retrieval belong to the next development stage.

## References and Acknowledgements

The Agent runtime was developed through studying and adapting implementation patterns and selected foundational components from [agent-zero-to-hero](https://github.com/KeWang0622/agent-zero-to-hero), alongside additional integration work in this repository. The current service combines the DeepSeek-backed Agent runtime with local RAG, FastAPI endpoints, session handling, integration tests, and project-specific fixes.

For reused upstream code, retain the applicable upstream license and copyright notice.
