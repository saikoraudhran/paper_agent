import json
import re
import uuid
import shutil
from pathlib import Path
from fastapi import (
    FastAPI, Request, UploadFile, File,
    HTTPException, Response, Depends
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

from backend.auth import (
    verify_login, create_session_cookie,
    get_current_user, require_admin,
    add_user, list_users
)
from backend.ingest import ingest_pdf, delete_paper_from_chroma, reingest_all_papers
from backend.retriever import retrieve_chunks, format_chunks_for_prompt
from backend.chat import build_messages, stream_response
from backend.history import (
    append_message, get_recent_for_llm,
    clear_history, export_history_markdown
)
from backend.config import UPLOADS_PATH, DATA_PATH, HUMANIZE_MAX_WORDS

app = FastAPI(title="ResearchAI")

# ── Paper registry (simple JSON file) ──────────────────────────

PAPERS_FILE = DATA_PATH / "papers.json"


def load_papers() -> list[dict]:
    if not PAPERS_FILE.exists():
        return []
    with open(PAPERS_FILE) as f:
        return json.load(f)


def save_papers(papers: list[dict]):
    with open(PAPERS_FILE, "w") as f:
        json.dump(papers, f, indent=2)


# ── Serve Frontend ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path("frontend/index.html")
    if html_path.exists():
        return html_path.read_text()
    return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)


# ── Auth Routes ─────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    pin: str


@app.post("/auth/login")
async def login(body: LoginRequest, response: Response):
    user = verify_login(body.username, body.pin)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or PIN")
    create_session_cookie(response, user["username"])
    return {"username": user["username"], "display_name": user["display_name"],
            "role": user["role"]}


@app.post("/auth/logout")
async def logout(response: Response):
    response.delete_cookie("session")
    return {"message": "Logged out"}


@app.get("/auth/me")
async def me(request: Request):
    return get_current_user(request)


# ── Admin: User Management ──────────────────────────────────────

class AddUserRequest(BaseModel):
    username: str
    pin: str
    display_name: str
    role: str = "user"


@app.post("/admin/users")
async def create_user(body: AddUserRequest, request: Request):
    require_admin(request)
    add_user(body.username, body.pin, body.display_name, body.role)
    return {"message": f"User {body.username} created"}


@app.get("/admin/users")
async def get_users(request: Request):
    require_admin(request)
    return list_users()


# ── Papers Routes ───────────────────────────────────────────────

@app.get("/papers")
async def get_papers(request: Request):
    get_current_user(request)  # any logged-in user
    return load_papers()


@app.post("/papers/upload")
async def upload_paper(request: Request, file: UploadFile = File(...)):
    require_admin(request)

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    paper_id = str(uuid.uuid4())[:8]
    save_path = UPLOADS_PATH / file.filename

    # Save PDF to disk
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ingest into ChromaDB
    try:
        stats = ingest_pdf(save_path, file.filename, paper_id)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    # Register paper
    papers = load_papers()
    papers.append({
        "paper_id": paper_id,
        "filename": file.filename,
        "active": True,
        "sections_found": stats["sections_found"],
        "chunks_stored": stats["chunks_stored"]
    })
    save_papers(papers)

    return {"message": "Paper uploaded and indexed", **stats}


@app.delete("/papers/{paper_id}")
async def delete_paper(paper_id: str, request: Request):
    require_admin(request)

    papers = load_papers()
    paper = next((p for p in papers if p["paper_id"] == paper_id), None)

    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    # Delete from ChromaDB
    delete_paper_from_chroma(paper_id)

    # Delete PDF file
    pdf_path = UPLOADS_PATH / paper["filename"]
    pdf_path.unlink(missing_ok=True)

    # Remove from registry
    papers = [p for p in papers if p["paper_id"] != paper_id]
    save_papers(papers)

    return {"message": "Paper deleted"}


@app.patch("/papers/{paper_id}/toggle")
async def toggle_paper(paper_id: str, request: Request):
    get_current_user(request)
    papers = load_papers()
    for p in papers:
        if p["paper_id"] == paper_id:
            p["active"] = not p["active"]
            save_papers(papers)
            return {"paper_id": paper_id, "active": p["active"]}
    raise HTTPException(status_code=404, detail="Paper not found")


@app.post("/admin/reingest")
async def reingest_all(request: Request):
    require_admin(request)
    papers = load_papers()
    results = reingest_all_papers(papers)
    return {"results": results}


# ── Chat Routes ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    mode: str = "qa"
    active_paper_ids: list[str] = []
    section_filter: str | None = None
    continuation_meta: dict | None = None
    humanize_options: dict | None = None


@app.post("/chat")
async def chat(body: ChatRequest, request: Request):
    user = get_current_user(request)

    is_humanize = body.mode == "humanize"

    # Word-count guard for humanize mode
    if is_humanize and len(body.message.split()) > HUMANIZE_MAX_WORDS:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long for humanize (max {HUMANIZE_MAX_WORDS} words). "
                   "Please split into smaller sections."
        )

    # 1. Retrieve relevant chunks (skip for humanize mode)
    if is_humanize:
        # No retrieval needed; use empty context and pass humanize options
        chunks = []
        chunks_text = ""
    else:
        chunks = retrieve_chunks(
            query=body.message,
            active_paper_ids=body.active_paper_ids,
            section_filter=body.section_filter
        )
        chunks_text = format_chunks_for_prompt(chunks)

    # 2. Get user's recent history (skip history for humanize mode)
    history = [] if is_humanize else get_recent_for_llm(user["username"])

    # 3. Build LLM messages, including humanize options when needed
    messages = build_messages(
        mode=body.mode,
        user_message=body.message,
        chunks_text=chunks_text,
        history=history,
        continuation_meta=body.continuation_meta,
        humanize_options=body.humanize_options
    )

    # 4. Save user message to history
    append_message(user["username"], "user", body.message, mode=body.mode)

    # 5. Stream response via SSE
    async def event_generator():
        full_reasoning = []
        full_content = []
        if is_humanize:
            # No citations for humanize mode
            citations = []
            token_iter = stream_response(messages, thinking=False, temperature=0.8)
        else:
            citations = [
                {
                    "filename": c["filename"],
                    "page_num": c["page_num"],
                    "section_name": c["section_name"]
                }
                for c in chunks
            ]
            token_iter = stream_response(messages)
        try:
            for token in token_iter:
                if token["type"] == "reasoning":
                    full_reasoning.append(token["text"])
                    yield {
                        "event": "reasoning",
                        "data": json.dumps({"text": token["text"]})
                    }
                elif token["type"] == "content":
                    full_content.append(token["text"])
                    yield {
                        "event": "content",
                        "data": json.dumps({"text": token["text"]})
                    }
                elif token["type"] == "done":
                    # Save complete AI message to history
                    append_message(
                        user["username"],
                        "assistant",
                        "".join(full_content),
                        reasoning="".join(full_reasoning),
                        mode=body.mode,
                        citations=citations
                    )

                    # Citation validation for humanize mode
                    done_payload = {"citations": citations}
                    if is_humanize:
                        cite_re = re.compile(r"\[[^\]]*\.pdf[^\]]*\]")
                        before = len(cite_re.findall(body.message))
                        after = len(cite_re.findall("".join(full_content)))
                        if before > 0 and before != after:
                            done_payload["warning"] = (
                                "Some citations may have been altered during "
                                "humanization. Please verify."
                            )

                    yield {
                        "event": "done",
                        "data": json.dumps(done_payload)
                    }
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)})
            }

    return EventSourceResponse(event_generator())


@app.post("/chat/clear")
async def clear_chat(request: Request):
    user = get_current_user(request)
    clear_history(user["username"])
    return {"message": "Chat history cleared"}


@app.get("/chat/export")
async def export_chat(request: Request):
    user = get_current_user(request)
    markdown = export_history_markdown(user["username"])
    return Response(
        content=markdown,
        media_type="text/markdown",
        headers={
            "Content-Disposition": f"attachment; filename=chat_{user['username']}.md"
        }
    )


# ── Health Check ────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}
