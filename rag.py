from pathlib import Path
from sentence_transformers import SentenceTransformer

EMBED_MODEL = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

def load_document(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def chunk_text(
    text: str,
    chunk_size: int = 200,
    overlap: int = 50,
) -> list[str]:
    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks

def embed_texts(texts: list[str]):
    return EMBED_MODEL.encode(
        texts,
        normalize_embeddings=True,
    )

def retrieve(
    query: str,
    chunks: list[str],
    embeddings,
    top_k: int = 2,
) -> list[tuple[float, str]]:
    query_embedding = embed_texts([query])[0]

    scores = embeddings @ query_embedding

    top_indices = scores.argsort()[::-1][:top_k]

    return [
        (float(scores[i]), chunks[i])
        for i in top_indices
    ]

def build_context(results: list[tuple[float, str]]) -> str:
    return "\n\n".join(
        chunk
        for score, chunk in results
    )

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_PATH = BASE_DIR / "knowledge" / "project.md"

DOCUMENT_TEXT = load_document(KNOWLEDGE_PATH)
CHUNKS = chunk_text(DOCUMENT_TEXT)
CHUNK_EMBEDDINGS = embed_texts(CHUNKS)

def retrieve_knowledge(query: str, top_k: int = 2) -> str:
    results = retrieve(
        query,
        CHUNKS,
        CHUNK_EMBEDDINGS,
        top_k=top_k,
    )

    return build_context(results)

if __name__ == "__main__":
    result = retrieve_knowledge(
        "How can I check whether the service is running?"
    )

    print("\n--- retrieved context ---")
    print(result)