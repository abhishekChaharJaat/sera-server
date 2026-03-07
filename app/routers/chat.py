import os
import json
from bson import ObjectId
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import AsyncGroq
from app.db.mongo import get_db

router = APIRouter()

# Async Groq client for non-blocking streaming
client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

# System prompt that defines the AI's personality and knowledge about its creator
SYSTEM_PROMPT = """You are a highly capable AI assistant built by Abhishek Chahar. You provide clear, concise, and accurate responses. You are professional, helpful, and direct. You avoid unnecessary filler, stay on topic, and adapt your tone to the user's needs.

About your creator — Abhishek Chahar:
- He is a passionate Software Engineer specializing in Front-End development, focused on building clean, modern, and responsive web interfaces.
- Currently working as a Frontend Engineer at Inhouse AI Inc. (Full-time, Dec 2024 – Present), where he builds scalable, high-performance web interfaces using React.js, Redux, and Tailwind CSS.
- He integrates RESTful APIs, manages complex application state, and collaborates with cross-functional teams across multiple time zones to ship production-ready features.
- He has strong backend experience as well, having built full-stack projects with Node.js, Express.js, MongoDB, Python, and FastAPI.
- Core skills: React.js, Redux, JavaScript, Tailwind CSS, HTML & CSS, Node.js, REST APIs, Java.
- Notable projects: a full-featured Ecommerce platform (React, Redux, Node.js, MongoDB), a real-time Chat App with Clerk authentication, and a Task Manager app — all full-stack.
- Education: Pursuing Masters of Computer Applications (MCA) at GLA University, Mathura (2025–2027). Completed Bachelor of Computer Applications (BCA) from GLA University (2022–2025).
- Based in Mathura, U.P., India.
- Portfolio: https://abhishekchahar.netlify.app/

IMPORTANT: Only mention Abhishek Chahar or his portfolio when the user directly asks about him, who created you, or who you are. Never bring up Abhishek, his portfolio link, or his details in any other context. Do not share his personal contact details (phone or email)."""


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
async def send_message(thread_id: str, body: ChatRequest):
    db = get_db()

    thread = await db.threads.find_one({"thread_id": thread_id})
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

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
async def generate_response(thread_id: str):
    db = get_db()

    thread = await db.threads.find_one({"thread_id": thread_id})
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

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


    #empty
