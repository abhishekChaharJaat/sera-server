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
    # Ask AI to generate a short, relevant title based on the first user message
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


# Request body schema
class ChatRequest(BaseModel):
    message: str


def sse(data: dict) -> str:
    # Format a dict as an SSE data event
    return f"data: {json.dumps(data)}\n\n"


@router.post("/{thread_id}/chat")
async def chat(thread_id: str, body: ChatRequest):
    db = get_db()

    # Verify the thread exists
    thread = await db.threads.find_one({"thread_id": thread_id})
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    now = datetime.now(timezone.utc)

    # Check if this is the first message in the thread (used for title generation)
    is_first_message = len(thread.get("messages", [])) == 0

    # Save user message to DB immediately before streaming
    user_msg = {"_id": ObjectId(), "role": "user", "content": body.message, "timestamp": now}
    await db.threads.update_one(
        {"thread_id": thread_id},
        {"$push": {"messages": user_msg}},
    )

    # Build full conversation history to give the AI context of previous messages
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in thread.get("messages", []):
        if msg["role"] in ("user", "assistant"):
            history.append({"role": msg["role"], "content": msg["content"]})
    # Append the current user message at the end
    history.append({"role": "user", "content": body.message})

    async def stream():
        # Send the user message instantly so frontend can render it right away
        yield sse({"type": "user_message", "id": str(user_msg["_id"]), "content": body.message})

        # On first message — generate an AI title and send it before chat stream starts
        if is_first_message:
            title = await generate_title(body.message)
            # Save the AI-generated title to DB
            await db.threads.update_one(
                {"thread_id": thread_id},
                {"$set": {"title": title}},
            )
            yield sse({"type": "title", "title": title})

        # Pre-generate the assistant message ID before streaming starts
        assistant_id = str(ObjectId())
        full_response = ""

        # Stream AI response from Groq token by token
        async with await client.chat.completions.create(
            messages=history,
            model="llama-3.1-8b-instant",
            stream=True,
        ) as stream_response:
            async for chunk in stream_response:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    # Send each token chunk to the frontend
                    yield sse({"type": "chunk", "content": delta})

        # Save the complete assistant response to DB after streaming finishes
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

        # Notify frontend that streaming is complete with the final message ID
        yield sse({"type": "done", "id": assistant_id})

    return StreamingResponse(stream(), media_type="text/event-stream")
