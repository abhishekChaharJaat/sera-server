import os
import uuid
import base64
import json
import asyncio
from bson import ObjectId
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Optional
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma
from app.db.mongo import get_db
from app.db.vector_store import get_thread_chroma_client, get_company_chroma_client, run_in_chroma_thread
from app.auth import get_auth_user, AuthUser
from app.prompts import AGENT_SYSTEM_PROMPT
from app.helpers.ingets import ingest_file, read_file_content
from app.helpers.chat_helpers import generate_title

router = APIRouter()

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0
)

_embeddings = None
_thread_vector_db = None
_company_vector_db = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        )
    return _embeddings


def _get_thread_vector_db():
    global _thread_vector_db
    if _thread_vector_db is None:
        _thread_vector_db = Chroma(
            client=get_thread_chroma_client(),
            embedding_function=_get_embeddings()
        )
    return _thread_vector_db


def _get_company_vector_db():
    global _company_vector_db
    if _company_vector_db is None:
        _company_vector_db = Chroma(
            client=get_company_chroma_client(),
            embedding_function=_get_embeddings()
        )
    return _company_vector_db


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


class AttachmentInput(BaseModel):
    filename: str
    content_type: str
    data: str


class ChatRequest(BaseModel):
    message: str
    attachments: Optional[List[AttachmentInput]] = []


@router.post("/{thread_id}/send-message")
async def send_message(thread_id: str, body: ChatRequest, auth_user: AuthUser = Depends(get_auth_user)):
    db = get_db()

    thread = await db.threads.find_one({"thread_id": thread_id})
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    if thread.get("user_id") != auth_user.user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    MAX_FILES_PER_THREAD = 4

    existing_files = thread.get("attached_files", 0)
    incoming = len(body.attachments or [])
    if existing_files + incoming > MAX_FILES_PER_THREAD:
        raise HTTPException(
            status_code=400,
            detail=f"File limit reached. A thread can have at most {MAX_FILES_PER_THREAD} attached files."
        )

    attachments = []
    for att in (body.attachments or []):
        file_id = str(uuid.uuid4())
        ext = os.path.splitext(att.filename)[1]
        saved_name = f"{file_id}{ext}"
        file_path = os.path.join(UPLOAD_DIR, saved_name)
        content = base64.b64decode(att.data)

        os.makedirs(UPLOAD_DIR, exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(content)

        if len(content) > 5 * 1024 * 1024:
            status, reason = "failed", "File size limit exceeded (max 5MB)"
        else:
            status, reason = "success", ""
            # Submit ingestion to dedicated ChromaDB thread (fire & forget)
            from app.db.vector_store import get_chroma_executor
            get_chroma_executor().submit(ingest_file, file_path, file_id, thread_id)

        attachments.append({
            "file_id": file_id,
            "filename": att.filename,
            "content_type": att.content_type,
            "size": len(content),
            "path": file_path,
            "status": status,
            "reason": reason
        })

    now = datetime.now(timezone.utc)
    user_msg = {
        "_id": ObjectId(),
        "role": "user",
        "content": body.message,
        "timestamp": now,
        **({"attachments": attachments} if attachments else {}),
    }

    successful_uploads = sum(1 for a in attachments if a["status"] == "success")
    await db.threads.update_one(
        {"thread_id": thread_id},
        {
            "$push": {"messages": user_msg},
            "$inc": {"attached_files": successful_uploads},
        },
    )

    async def stream():
        yield sse({
            "type": "user_message",
            "id": str(user_msg["_id"]),
            "content": body.message,
            "attachments": [{"file_id": a["file_id"], "filename": a["filename"], "content_type": a["content_type"], "size": a["size"]} for a in attachments],
        })

    return StreamingResponse(stream(), media_type="text/event-stream")


# ========================== Generate response =========================

@router.post("/{thread_id}/generate-response")
async def generate_response(thread_id: str, auth_user: AuthUser = Depends(get_auth_user)):
    db = get_db()

    thread = await db.threads.find_one({"thread_id": thread_id})
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    if thread.get("user_id") != auth_user.user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    messages = thread.get("messages", [])

    last_user_msg = next((m for m in reversed(messages) if m["role"] == "user"), None)
    if not last_user_msg:
        raise HTTPException(status_code=400, detail="No user message found in thread")

    is_first_message = not any(m["role"] == "assistant" for m in messages)

    user_input = last_user_msg["content"].strip()
    current_attachments = last_user_msg.get("attachments", [])
    if not user_input and current_attachments:
        user_input = "Summarize the uploaded document."

    # ── Document context ──
    # Current message has attachments → read file content directly (instant, no RAG needed)
    # Follow-up question → RAG: similarity search across all docs in this thread
    doc_context = ""
    if current_attachments:
        parts = []
        for att in current_attachments:
            if att.get("status") == "success" and att.get("path"):
                content = await asyncio.to_thread(read_file_content, att["path"])
                if content:
                    parts.append(f"[{att['filename']}]\n{content}")
        doc_context = "\n\n---\n\n".join(parts)
    else:
        # RAG: similarity search across all docs ingested in this thread
        chroma_filter = {"thread_id": {"$eq": thread_id}}
        thread_docs = await run_in_chroma_thread(
            lambda: _get_thread_vector_db().similarity_search(user_input, k=5, filter=chroma_filter)
        )
        doc_context = "\n\n".join(d.page_content for d in thread_docs)

    # ── Company knowledge base (RAG) ──
    company_docs = await asyncio.to_thread(
        lambda: _get_company_vector_db().similarity_search(user_input, k=3)
    )

    # ── System prompt ──
    system_content = AGENT_SYSTEM_PROMPT
    if doc_context:
        system_content += f"\n\n[Uploaded Document Context]\n{doc_context}"
    if company_docs:
        company_context = "\n\n".join(d.page_content for d in company_docs)
        system_content += f"\n\n[Company Information]\n{company_context}"

    # ── Message history ──
    all_messages = [SystemMessage(content=system_content)]
    for msg in messages[:-1][-12:]:
        if msg["role"] == "user":
            all_messages.append(HumanMessage(content=msg["content"] or "📎 [file uploaded]"))
        elif msg["role"] == "assistant":
            all_messages.append(AIMessage(content=msg["content"]))
    all_messages.append(HumanMessage(content=user_input))

    async def stream():
        if is_first_message:
            title = await generate_title(last_user_msg["content"] or user_input)
            await db.threads.update_one({"thread_id": thread_id}, {"$set": {"title": title}})
            yield sse({"type": "title", "title": title})

        assistant_id = str(ObjectId())
        full_response = ""

        async for chunk in llm.astream(all_messages):
            delta = chunk.content
            if delta:
                full_response += delta
                yield sse({"type": "chunk", "content": delta})

        assistant_msg = {
            "_id": ObjectId(assistant_id),
            "role": "assistant",
            "content": full_response,
            "timestamp": datetime.now(timezone.utc),
        }
        await db.threads.update_one(
            {"thread_id": thread_id},
            {"$push": {"messages": assistant_msg}},
        )
        yield sse({"type": "done", "id": assistant_id})

    return StreamingResponse(stream(), media_type="text/event-stream")
