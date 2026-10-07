# Agentic RAG Service

A lightweight Python service that combines an LLM agent, tool calling, retrieval-augmented generation (RAG), multi-turn sessions, and streaming responses behind a FastAPI API.

The project uses DeepSeek through an Anthropic-compatible API and a local Sentence Transformer model for document retrieval.

## Features

- Tool-using agent loop with configurable step and token limits
- Safe calculator tool using AST-based expression evaluation
- Multi-document RAG over local Markdown files
- Markdown-aware chunking with overlapping chunks
- Source metadata for retrieved context
- Cosine-similarity retrieval with a relevance threshold
- Multi-turn conversations with session history
- Bounded conversation history
- Streaming responses
- Lazy loading of the embedding model and knowledge index
- Tool-level and API-level error handling
- Input length validation
- Offline retrieval evaluation
- Automated regression tests with pytest

## Architecture

```text
User
  |
  v
FastAPI
  |
  +---- POST /chat
  |
  +---- POST /chat/stream
  |
  v
Agent Loop
  |
  +---- calculator
  |
  +---- retrieve_knowledge
           |
           v
     Local Markdown Documents
           |
           v
     Markdown-aware Chunking
           |
           v
     Sentence Transformer
           |
           v
     Cosine Similarity Search
```

The agent decides when to call a tool. Python executes the selected tool and returns the tool result to the model before the agent continues.

A maximum agent step count prevents unlimited tool-call loops.

## RAG Pipeline

Knowledge documents are stored in the `knowledge/` directory.

At retrieval time, the service:

1. loads all Markdown documents,
2. splits documents into Markdown-aware chunks,
3. preserves source metadata for every chunk,
4. embeds chunks using `paraphrase-multilingual-MiniLM-L12-v2`,
5. embeds the user query using the same model,
6. ranks chunks with cosine similarity,
7. removes results below a relevance threshold,
8. returns relevant context and its source to the agent.

The embedding model and knowledge index are initialized lazily, so functionality that does not require RAG does not pay the model-loading cost.

The current implementation performs retrieval in memory and does not require a vector database.

## Design Decisions

### Why AST instead of `eval()`?

The calculator accepts model-generated expressions, so executing them directly
with Python `eval()` would unnecessarily expose Python execution semantics.

The calculator instead parses expressions with `ast` and explicitly allows only
supported numeric constants and arithmetic operators. Expression length,
numeric magnitude, and exponent size are also bounded.

### Why a local Sentence Transformer?

The current knowledge base is small and local, so retrieval does not require
an external embedding API or a vector database.

`paraphrase-multilingual-MiniLM-L12-v2` provides lightweight local inference
and multilingual semantic embeddings suitable for the current project scope.

### Why use a relevance threshold?

Top-k retrieval always returns the nearest chunks even when none of them are
actually relevant.

A minimum similarity threshold allows the retriever to return no context for
out-of-domain questions instead of forcing unrelated documents into the agent
context.

The current threshold (`0.35`) works for the included small evaluation set and
should be recalibrated if the corpus or embedding model changes.

### Why use Markdown-aware overlapping chunks?

The current implementation uses a 200-character chunk size with a 50-character
overlap.

Markdown structure is preserved when possible, and fixed-size overlapping
windows are only used when a section exceeds the configured chunk size.

The current chunk size and overlap are practical defaults for this small corpus,
not globally optimized values.

### Why use both `max_steps` and `max_tokens`?

They protect against different failure modes.

`max_steps` limits how many agent/tool-call iterations can occur, preventing
unbounded tool loops.

`max_tokens` limits the size of a single model response, helping control
latency and API cost.

Using both provides separate safeguards for agent control flow and model output
size.

### Why lazy-load the embedding model?

The embedding model is only required when the RAG tool is actually used.

Lazy initialization prevents calculator-only or lightweight API paths from
paying the startup cost of loading the Sentence Transformer model and building
the knowledge index.

The initialized model and index are then reused within the same process.

### Why keep sessions and retrieval in memory?

The current project uses a small local knowledge base and is designed as a
lightweight single-process service.

For this scope, in-memory session history and retrieval avoid adding Redis,
a database, or a vector store before they are necessary.

This keeps the implementation easy to inspect and run locally. For a larger
or multi-instance deployment, persistent session storage and a persistent
retrieval index would be natural next steps.

## API

### `GET /health`

Checks whether the service is running.

### `POST /chat`

Returns a complete agent response.

The request supports an optional `session_id`. If none is provided, the server creates a new session.

### `POST /chat/stream`

Streams the agent response incrementally.

The session ID is returned in the `X-Session-ID` response header.

Sessions are currently stored in memory, so restarting the server clears session history.

## Retrieval Evaluation

The project includes a small offline retrieval evaluation set in `eval.py`.

Current results on the included 11-case evaluation set:

```text
Retrieval Hit@1:      9/9 (100.0%)
Retrieval Hit@3:      9/9 (100.0%)
Rejection accuracy:   2/2 (100.0%)
```

The evaluation set contains nine in-domain retrieval questions and two irrelevant questions used to verify relevance-based rejection.

These results describe only the included small offline evaluation set and are not intended as a general RAG accuracy benchmark.

Run the evaluation with:

```bash
python eval.py
```

## Tests

The project includes automated tests for:

- FastAPI endpoints and session behavior
- streaming session behavior
- calculator correctness and safety checks
- conversation history trimming
- RAG chunking and source metadata
- retrieval threshold behavior
- tool execution and error handling
- request length validation

Run all tests with:

```bash
python -m pytest -v
```

Current test status:

```text
27 passed
```

Tests use mocks where appropriate so normal unit tests do not require real LLM API calls or loading the embedding model.

## Setup

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Copy the example environment configuration:

```bash
copy .env.example .env
```

Fill in the required model/API configuration in `.env`.

The real `.env` file is intentionally excluded from Git.

## Run

Start the FastAPI service:

```bash
python -m uvicorn api:app --reload
```

Then open `http://127.0.0.1:8000/docs` to use the automatically generated Swagger UI.

to use the automatically generated Swagger UI.

## Project Structure

```text
agentic-rag-service/
├── knowledge/
│   ├── api.md
│   ├── architecture.md
│   └── project.md
├── tests/
│   ├── test_api.py
│   ├── test_calculator.py
│   ├── test_history.py
│   ├── test_rag.py
│   └── test_tools.py
├── api.py
├── eval.py
├── mini_agent.py
├── rag.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Current Limitations

Sessions and the retrieval index are stored in memory.

The current knowledge base is intentionally small and local.

The project does not currently use a persistent vector database, distributed session storage, subagents, or a production deployment layer.