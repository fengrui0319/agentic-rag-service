# Agentic RAG Service

A lightweight LLM agent service built with Python, FastAPI, tool calling, and retrieval-augmented generation (RAG).

The project combines an agent loop with local knowledge retrieval so the model can decide whether to answer directly, call a calculator, or retrieve relevant project documentation.

## Features

- LLM agent loop with tool calling
- Calculator tool for arithmetic tasks
- RAG tool for local knowledge retrieval
- Text chunking with overlap
- Multilingual embeddings with Sentence Transformers
- Cosine-similarity based top-k retrieval
- FastAPI backend
- Automatic Swagger API documentation
- Environment-based API key configuration

## Architecture

```text
User
  |
  v
FastAPI /chat
  |
  v
LLM Agent
  |
  +---- calculator
  |
  +---- retrieve_knowledge
             |
             v
      Local documents
             |
             v
      Chunking + Embedding
             |
             v
      Similarity Search
             |
             v
       Relevant Context