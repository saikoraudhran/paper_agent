import fitz  # PyMuPDF
import tiktoken
import chromadb
import uuid
from pathlib import Path
from openai import OpenAI
from backend.config import (
    NVIDIA_API_KEY, NVIDIA_BASE_URL, EMBEDDING_MODEL,
    CHROMA_PATH, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS,
    SECTION_HEADINGS, UPLOADS_PATH
)

# NVIDIA client for embeddings
nvidia_client = OpenAI(
    base_url=NVIDIA_BASE_URL,
    api_key=NVIDIA_API_KEY
)

# ChromaDB client
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

# Tokenizer for chunk sizing
tokenizer = tiktoken.get_encoding("cl100k_base")


def _get_collection(paper_id: str):
    """Get or create a ChromaDB collection for a specific paper."""
    return chroma_client.get_or_create_collection(
        name=f"paper_{paper_id}",
        metadata={"hnsw:space": "cosine"}
    )


def _detect_section(line: str) -> str | None:
    """Detect if a line is a section heading."""
    cleaned = line.strip().lower().rstrip(".")
    for heading in SECTION_HEADINGS:
        if cleaned == heading or cleaned.startswith(f"{heading} "):
            return line.strip()
    # Detect numbered sections like "1. Introduction" or "2.1 Methods"
    import re
    if re.match(r'^\d+(\.\d+)?\s+[A-Z]', line.strip()):
        return line.strip()
    return None


def _extract_sections(pdf_path: Path) -> list[dict]:
    """
    Extract text from PDF organized by section.
    Returns list of: {section_name, page_num, text}
    """
    doc = fitz.open(str(pdf_path))
    sections = []
    current_section = "Preamble"
    current_text = []
    current_page = 1

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")
        for block in blocks:
            text = block[4].strip()
            if not text:
                continue

            # Check each line for section heading
            lines = text.split("\n")
            for line in lines:
                detected = _detect_section(line)
                if detected:
                    # Save previous section
                    if current_text:
                        sections.append({
                            "section_name": current_section,
                            "page_num": current_page,
                            "text": " ".join(current_text).strip()
                        })
                    current_section = detected
                    current_text = []
                    current_page = page_num
                else:
                    current_text.append(line)

    # Save last section
    if current_text:
        sections.append({
            "section_name": current_section,
            "page_num": current_page,
            "text": " ".join(current_text).strip()
        })

    doc.close()
    return [s for s in sections if len(s["text"]) > 50]


def _chunk_section(section: dict) -> list[dict]:
    """
    Split a section into overlapping token-sized chunks.
    Each chunk keeps section metadata.
    """
    tokens = tokenizer.encode(section["text"])
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + CHUNK_SIZE_TOKENS, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)

        chunks.append({
            "text": chunk_text,
            "section_name": section["section_name"],
            "page_num": section["page_num"],
            "chunk_index": len(chunks)
        })

        if end == len(tokens):
            break
        start += CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS

    return chunks


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using NVIDIA NIM."""
    # Batch in groups of 20 to avoid rate limits
    all_embeddings = []
    batch_size = 20

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = nvidia_client.embeddings.create(
            input=batch,
            model=EMBEDDING_MODEL,
            encoding_format="float",
            extra_body={"input_type": "passage", "truncate": "END"}
        )
        all_embeddings.extend([e.embedding for e in response.data])

    return all_embeddings


def ingest_pdf(pdf_path: Path, filename: str, paper_id: str) -> dict:
    """
    Full ingestion pipeline:
    PDF -> sections -> chunks -> embeddings -> ChromaDB
    Returns summary stats.
    """
    # 1. Extract sections
    sections = _extract_sections(pdf_path)

    # 2. Chunk each section
    all_chunks = []
    for section in sections:
        chunks = _chunk_section(section)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("No text could be extracted from this PDF")

    # 3. Embed all chunks
    texts = [c["text"] for c in all_chunks]
    embeddings = _embed_texts(texts)

    # 4. Store in ChromaDB
    collection = _get_collection(paper_id)

    ids = [str(uuid.uuid4()) for _ in all_chunks]
    metadatas = [
        {
            "filename": filename,
            "paper_id": paper_id,
            "section_name": c["section_name"],
            "page_num": c["page_num"],
            "chunk_index": c["chunk_index"]
        }
        for c in all_chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    return {
        "paper_id": paper_id,
        "filename": filename,
        "sections_found": len(sections),
        "chunks_stored": len(all_chunks)
    }


def delete_paper_from_chroma(paper_id: str):
    """Remove a paper's ChromaDB collection entirely."""
    try:
        chroma_client.delete_collection(f"paper_{paper_id}")
    except Exception:
        pass


def reingest_all_papers(papers: list[dict]) -> list[dict]:
    """
    Re-ingest all papers from disk.
    Used as recovery if ChromaDB gets corrupted.
    """
    results = []
    for paper in papers:
        pdf_path = UPLOADS_PATH / paper["filename"]
        if pdf_path.exists():
            try:
                result = ingest_pdf(pdf_path, paper["filename"], paper["paper_id"])
                results.append({"status": "success", **result})
            except Exception as e:
                results.append({
                    "status": "error",
                    "filename": paper["filename"],
                    "error": str(e)
                })
    return results
