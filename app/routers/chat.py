import os
import uuid
import base64
import json
import asyncio
from bson import ObjectId
from datetime import datetime, timezone
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Optional
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma
from app.db.mongo import get_db
from app.auth import get_auth_user, AuthUser
from app.prompts import AGENT_SYSTEM_PROMPT
from app.helpers.ingets import ingest_file
from app.helpers.chat_helpers import generate_title

router = APIRouter()

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads")
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_db")
COMPANY_CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_company_db")

os.makedirs(UPLOAD_DIR, exist_ok=True)

llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0
)

_embeddings = None
_thread_vector_db = None
_company_retriever = None


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
            persist_directory=CHROMA_DIR,
            embedding_function=_get_embeddings()
        )
    return _thread_vector_db


def _get_company_retriever():
    global _company_retriever
    if _company_retriever is None:
        company_vector_db = Chroma(
            persist_directory=COMPANY_CHROMA_DIR,
            embedding_function=_get_embeddings()
        )
        _company_retriever = company_vector_db.as_retriever(search_kwargs={"k": 3})
    return _company_retriever


class AttachmentInput(BaseModel):
    filename: str
    content_type: str
    data: str  # base64 encoded


class ChatRequest(BaseModel):
    message: str
    attachments: Optional[List[AttachmentInput]] = []


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@router.post("/{thread_id}/send-message")
async def send_message(thread_id: str, body: ChatRequest, background_tasks: BackgroundTasks, auth_user: AuthUser = Depends(get_auth_user)):
    db = get_db()

    thread = await db.threads.find_one({"thread_id": thread_id})
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    if thread.get("user_id") != auth_user.user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Save uploaded files
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
            status, reason = "processing", ""
            background_tasks.add_task(ingest_file, file_path, file_id, thread_id)

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

    await db.threads.update_one(
        {"thread_id": thread_id},
        {"$push": {"messages": user_msg}},
    )

    async def stream():
        yield sse({
            "type": "user_message",
            "id": str(user_msg["_id"]),
            "content": body.message,
            "attachments": [{"file_id": a["file_id"], "filename": a["filename"], "content_type": a["content_type"], "size": a["size"]} for a in attachments],
        })

    return StreamingResponse(stream(), media_type="text/event-stream")


# ========================== Generate resopnse =========================

@router.post("/{thread_id}/generate-response")
async def generate_response(thread_id: str, auth_user: AuthUser = Depends(get_auth_user)):
    db = get_db()

    # Fetch thread and validate ownership
    thread = await db.threads.find_one({"thread_id": thread_id})
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    if thread.get("user_id") != auth_user.user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    messages = thread.get("messages", [])

    # Get the last user message to generate a response for
    last_user_msg = next(
        (m for m in reversed(messages) if m["role"] == "user"), None
    )
    if not last_user_msg:
        raise HTTPException(status_code=400, detail="No user message found in thread")

    # Check if this is the first message (no assistant reply yet) — used for title generation
    is_first_message = not any(m["role"] == "assistant" for m in messages)

    # Use message content as query; if empty (file only), default to summarize
    user_input = last_user_msg["content"].strip()
    if not user_input and last_user_msg.get("attachments"):
        user_input = "Summarize the uploaded document briefly."

    # ── RAG: Retrieve relevant chunks from uploaded documents ──
    # If current message has attachments, search only those specific files by file_id
    # Otherwise search all files uploaded in this thread by thread_id
    thread_vector_db = _get_thread_vector_db()
    current_attachments = last_user_msg.get("attachments", [])
    if current_attachments:
        file_ids = [a["file_id"] for a in current_attachments]
       # $in ka matlab: match any of these file_ids. - $eq ka matlab: exactly match this file_id.
        chroma_filter = {"file_id": {"$in": file_ids}} if len(file_ids) > 1 else {"file_id": {"$eq": file_ids[0]}}
    else:
        chroma_filter = {"thread_id": {"$eq": thread_id}}

    thread_docs = await asyncio.to_thread(
        lambda: thread_vector_db.similarity_search(user_input, k=5, filter=chroma_filter)
    )

    # ── RAG: Retrieve relevant chunks from company knowledge base ──
    company_docs = await asyncio.to_thread(
        lambda: _get_company_retriever().invoke(user_input)
    )

    # ── Build system prompt — inject retrieved context ──
    # Start with base agent prompt, then append doc/company context if found
    system_content = AGENT_SYSTEM_PROMPT
    if thread_docs:
        doc_context = "\n\n".join(d.page_content for d in thread_docs)
        system_content += f"\n\n[Uploaded Document Context]\n{doc_context}"
    if company_docs:
        company_context = "\n\n".join(d.page_content for d in company_docs)
        system_content += f"\n\n[Company Information]\n{company_context}"

    # ── Build full message list for LLM ──
    # System message first, then last 12 conversation messages, then current user message
    all_messages = [SystemMessage(content=system_content)]
    for msg in messages[:-1][-12:]:
        if msg["role"] == "user":
            all_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            all_messages.append(AIMessage(content=msg["content"]))
    all_messages.append(HumanMessage(content=user_input))

    async def stream():

        # Generate and send thread title on the first message
        if is_first_message:
            title = await generate_title(last_user_msg["content"] or user_input)
            await db.threads.update_one(
                {"thread_id": thread_id},
                {"$set": {"title": title}},
            )
            yield sse({"type": "title", "title": title})

        assistant_id = str(ObjectId())
        full_response = ""

        # Stream LLM response chunk by chunk
        async for chunk in llm.astream(all_messages):
            delta = chunk.content
            if delta:
                full_response += delta
                yield sse({"type": "chunk", "content": delta})

        # Save complete assistant message to MongoDB
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

        # Signal stream completion
        yield sse({"type": "done", "id": assistant_id})

    return StreamingResponse(stream(), media_type="text/event-stream")
