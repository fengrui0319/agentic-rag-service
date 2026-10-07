# Architecture

The service contains an LLM agent loop with tool calling.

The agent currently has two tools:

- `calculator`: evaluates supported arithmetic tasks
- `retrieve_knowledge`: retrieves relevant information from the local knowledge base

The agent can make one or more tool calls and receives the corresponding tool results before continuing.

The agent loop uses a maximum step limit to prevent unlimited tool-call loops.

Each model request also has a maximum token limit.

## Conversation History

The service supports multi-turn conversations.

Conversation history is associated with a session ID.

Only a limited number of recent messages are retained so conversation context does not grow without bound.

## Retrieval-Augmented Generation

Local Markdown documents are divided into overlapping chunks.

Each chunk is converted into an embedding using a multilingual Sentence Transformer model.

The user query is embedded using the same model.

Retrieval compares the normalized query embedding with document chunk embeddings using cosine similarity.

The most relevant chunks are returned to the agent as tool results.

The current implementation performs retrieval in memory and does not require a vector database.