import os
import json
from bson import ObjectId
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import AsyncGroq
from app.db.mongo import get_db
from app.auth import get_auth_user, AuthUser
from app.prompts import SYSTEM_PROMPT

router = APIRouter()

# Async Groq client for non-blocking streaming
client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

async def generate_title(message: str) -> str:
    completion = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": (
                    f"Generate a short chat title (max 50 characters, no quotes, no punctuation at end) "
                    f"that is specific to this exact message. Use key terms or the actual subject from the message — "
                    f"do NOT use generic labels like 'Math problem' or 'Question'. "
                    f"Message: {message}"
                ),
            }
        ],
        model="llama-3.1-8b-instant",
    )
    return completion.choices[0].message.content.strip()[:50]


class ChatRequest(BaseModel):
    message: str


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@router.post("/{thread_id}/send-message")
async def send_message(thread_id: str, body: ChatRequest, auth_user: AuthUser = Depends(get_auth_user)):
    db = get_db()

    thread = await db.threads.find_one({"thread_id": thread_id})
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    if thread.get("user_id") != auth_user.user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    now = datetime.now(timezone.utc)
    user_msg = {"_id": ObjectId(), "role": "user", "content": body.message, "timestamp": now}

    await db.threads.update_one(
        {"thread_id": thread_id},
        {"$push": {"messages": user_msg}},
    )

    async def stream():
        yield sse({"type": "user_message", "id": str(user_msg["_id"]), "content": body.message})

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.post("/{thread_id}/generate-response")
async def generate_response(thread_id: str, auth_user: AuthUser = Depends(get_auth_user)):
    db = get_db()

    thread = await db.threads.find_one({"thread_id": thread_id})
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    if thread.get("user_id") != auth_user.user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    messages = thread.get("messages", [])

    # Find the last user message
    last_user_msg = next(
        (m for m in reversed(messages) if m["role"] == "user"), None
    )
    if not last_user_msg:
        raise HTTPException(status_code=400, detail="No user message found in thread")

    # First message = no assistant messages yet
    is_first_message = not any(m["role"] == "assistant" for m in messages)

    # Build conversation history
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in messages:
        if msg["role"] in ("user", "assistant"):
            history.append({"role": msg["role"], "content": msg["content"]})

    async def stream():
        # Generate and send title on first message
        if is_first_message:
            title = await generate_title(last_user_msg["content"])
            await db.threads.update_one(
                {"thread_id": thread_id},
                {"$set": {"title": title}},
            )
            yield sse({"type": "title", "title": title})

        assistant_id = str(ObjectId())
        full_response = ""

        async with await client.chat.completions.create(
            messages=history,
            model="llama-3.1-8b-instant",
            stream=True,
        ) as stream_response:
            async for chunk in stream_response:
                delta = chunk.choices[0].delta.content
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


    #Langlain or langgraph
