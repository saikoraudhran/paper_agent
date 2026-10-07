import chromadb
from openai import OpenAI
from backend.config import (
    NVIDIA_API_KEY, NVIDIA_BASE_URL,
    EMBEDDING_MODEL, CHROMA_PATH, TOP_K_RETRIEVAL
)

nvidia_client = OpenAI(
    base_url=NVIDIA_BASE_URL,
    api_key=NVIDIA_API_KEY
)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))


def _embed_query(query: str) -> list[float]:
    """Embed a user query for similarity search."""
    response = nvidia_client.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL,
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "END"}
    )
    return response.data[0].embedding


def _get_active_collections(active_paper_ids: list[str]) -> list:
    """Get ChromaDB collections for active papers only."""
    collections = []
    for paper_id in active_paper_ids:
        try:
            col = chroma_client.get_collection(f"paper_{paper_id}")
            collections.append(col)
        except Exception:
            pass
    return collections


def retrieve_chunks(
    query: str,
    active_paper_ids: list[str],
    top_k: int = TOP_K_RETRIEVAL,
    section_filter: str = None
) -> list[dict]:
    """
    Embed query and retrieve top-k chunks from active papers.
    Optionally filter by section name.
    Returns list of chunk dicts with text + metadata.
    """
    if not active_paper_ids:
        return []

    query_embedding = _embed_query(query)
    collections = _get_active_collections(active_paper_ids)

    if not collections:
        return []

    all_results = []

    for collection in collections:
        where_filter = None
        if section_filter:
            where_filter = {
                "section_name": {
                    "$contains": section_filter.lower()
                }
            }

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count()),
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

            for i, doc in enumerate(results["documents"][0]):
                all_results.append({
                    "text": doc,
                    "filename": results["metadatas"][0][i]["filename"],
                    "paper_id": results["metadatas"][0][i]["paper_id"],
                    "section_name": results["metadatas"][0][i]["section_name"],
                    "page_num": results["metadatas"][0][i]["page_num"],
                    "chunk_index": results["metadatas"][0][i]["chunk_index"],
                    "distance": results["distances"][0][i]
                })
        except Exception:
            continue

    # Sort all results by similarity distance (lower = more similar)
    all_results.sort(key=lambda x: x["distance"])

    return all_results[:top_k]


def format_chunks_for_prompt(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into the context block
    injected into the LLM prompt.
    """
    if not chunks:
        return "No relevant content found in the uploaded papers."

    formatted = []
    for i, chunk in enumerate(chunks, start=1):
        formatted.append(
            f"[CHUNK {i}]\n"
            f"Source: {chunk['filename']} | "
            f"Page: {chunk['page_num']} | "
            f"Section: {chunk['section_name']}\n"
            f"{chunk['text']}\n"
        )

    return "\n---\n".join(formatted)
