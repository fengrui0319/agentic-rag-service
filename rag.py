from pathlib import Path
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

MIN_SIMILARITY = 0.35

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 50

_EMBED_MODEL = None
_DOCUMENTS = None
_CHUNKS = None
_CHUNK_EMBEDDINGS = None

def load_documents(directory: Path) -> list[dict[str, str]]:
    documents = []

    for path in sorted(directory.glob("*.md")):
        documents.append({
            "source": path.name,
            "text": path.read_text(encoding="utf-8"),
        })

    return documents

def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    sections = []
    current_section = []

    for line in text.splitlines():
        if line.startswith("#") and current_section:
            sections.append(
                "\n".join(current_section).strip()
            )
            current_section = [line]
        else:
            current_section.append(line)

    if current_section:
        sections.append(
            "\n".join(current_section).strip()
        )

    chunks = []

    for section in sections:
        if len(section) <= chunk_size:
            if section:
                chunks.append(section)
            continue

        step = chunk_size - overlap

        for i in range(0, len(section), step):
            chunk = section[i:i + chunk_size].strip()

            if chunk:
                chunks.append(chunk)

    return chunks

def chunk_documents(
    documents: list[dict[str, str]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, str]]:
    chunks = []

    for document in documents:
        source = document["source"]
        text = document["text"]

        text_chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for text_chunk in text_chunks:
            chunks.append({
                "source": source,
                "text": text_chunk,
            })

    return chunks

def get_embed_model():
    global _EMBED_MODEL

    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(
            EMBED_MODEL_NAME
        )

    return _EMBED_MODEL

def embed_texts(texts: list[str]):
    model = get_embed_model()

    return model.encode(
        texts,
        normalize_embeddings=True,
    )

def retrieve(
    query: str,
    chunks: list[dict[str, str]],
    chunk_embeddings,
    top_k: int = 3,
    min_similarity: float = MIN_SIMILARITY,
):
    query_embedding = embed_texts([query])[0]

    scores = chunk_embeddings @ query_embedding

    top_indices = scores.argsort()[::-1][:top_k]

    results = []

    for i in top_indices:
        score = float(scores[i])

        if score < min_similarity:
            continue

        results.append({
            "score": score,
            "source": chunks[i]["source"],
            "text": chunks[i]["text"],
        })

    return results

def build_context(results: list[dict]) -> str:
    if not results:
        return (
            "No sufficiently relevant information was found "
            "in the local knowledge base."
        )

    context_parts = []

    for result in results:
        context_parts.append(
            f"[Source: {result['source']}]\n"
            f"{result['text']}"
        )

    return "\n\n".join(context_parts)

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"


def get_knowledge_index():
    global _DOCUMENTS
    global _CHUNKS
    global _CHUNK_EMBEDDINGS

    if _CHUNKS is None or _CHUNK_EMBEDDINGS is None:
        _DOCUMENTS = load_documents(KNOWLEDGE_DIR)
        _CHUNKS = chunk_documents(_DOCUMENTS)

        chunk_texts = [
            chunk["text"]
            for chunk in _CHUNKS
        ]

        _CHUNK_EMBEDDINGS = embed_texts(chunk_texts)

    return _CHUNKS, _CHUNK_EMBEDDINGS

def retrieve_knowledge(query: str) -> str:
    chunks, chunk_embeddings = get_knowledge_index()

    results = retrieve(
        query,
        chunks,
        chunk_embeddings,
        top_k=3,
    )

    return build_context(results)

if __name__ == "__main__":
    context = retrieve_knowledge(
        "Who won the football World Cup?"
    )

    print("\n--- context sent to agent ---")
    print(context)