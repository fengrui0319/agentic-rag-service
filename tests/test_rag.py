import numpy as np

import rag


def test_chunk_text_preserves_markdown_sections():
    text = """# Section A

Alpha content.

## Section B

Beta content.
"""

    chunks = rag.chunk_text(
        text,
        chunk_size=500,
        overlap=80,
    )

    assert len(chunks) == 2
    assert chunks[0].startswith("# Section A")
    assert "Alpha content." in chunks[0]

    assert chunks[1].startswith("## Section B")
    assert "Beta content." in chunks[1]


def test_chunk_documents_preserves_source_metadata():
    documents = [
        {
            "source": "first.md",
            "text": "# First\n\nFirst document.",
        },
        {
            "source": "second.md",
            "text": "# Second\n\nSecond document.",
        },
    ]

    chunks = rag.chunk_documents(
        documents,
        chunk_size=500,
        overlap=80,
    )

    assert chunks[0]["source"] == "first.md"
    assert chunks[1]["source"] == "second.md"

    assert "First document." in chunks[0]["text"]
    assert "Second document." in chunks[1]["text"]


def test_build_context_includes_sources():
    results = [
        {
            "score": 0.8,
            "source": "api.md",
            "text": "GET /health checks service status.",
        },
        {
            "score": 0.7,
            "source": "architecture.md",
            "text": "The agent uses tool calling.",
        },
    ]

    context = rag.build_context(results)

    assert "[Source: api.md]" in context
    assert "GET /health checks service status." in context

    assert "[Source: architecture.md]" in context
    assert "The agent uses tool calling." in context


def test_build_context_handles_no_results():
    context = rag.build_context([])

    assert context == (
        "No sufficiently relevant information was found "
        "in the local knowledge base."
    )


def test_retrieve_applies_similarity_threshold(monkeypatch):
    chunks = [
        {
            "source": "good.md",
            "text": "relevant",
        },
        {
            "source": "weak.md",
            "text": "weak",
        },
        {
            "source": "bad.md",
            "text": "irrelevant",
        },
    ]

    chunk_embeddings = np.array([
        [1.0, 0.0],
        [0.3, 0.0],
        [-1.0, 0.0],
    ])

    def fake_embed_texts(texts):
        return np.array([
            [1.0, 0.0],
        ])

    monkeypatch.setattr(
        rag,
        "embed_texts",
        fake_embed_texts,
    )

    results = rag.retrieve(
        "test query",
        chunks,
        chunk_embeddings,
        top_k=3,
        min_similarity=0.35,
    )

    assert len(results) == 1
    assert results[0]["source"] == "good.md"
    assert results[0]["text"] == "relevant"