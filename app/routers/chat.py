import os
from bson import ObjectId
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from groq import Groq
from app.db.mongo import get_db

router = APIRouter()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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


def generate_title(message: str) -> str:
    title = message[:20].strip()
    if len(message) > 20:
        title += "..."
    return title


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    thread_id: str
    id: str
    message: str


@router.post("/{thread_id}/chat", response_model=ChatResponse)
async def chat(thread_id: str, body: ChatRequest):
    db = get_db()

    # Verify thread exists
    thread = await db.threads.find_one({"thread_id": thread_id})
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Build conversation history from thread
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in thread.get("messages", []):
        if msg["role"] in ("user", "assistant"):
            history.append({"role": msg["role"], "content": msg["content"]})
    history.append({"role": "user", "content": body.message})

    # Get AI response
    chat_completion = client.chat.completions.create(
        messages=history,
        model="llama-3.1-8b-instant",
    )
    response = chat_completion.choices[0].message.content

    now = datetime.now(timezone.utc)

    user_msg = {"role": "user", "content": body.message, "timestamp": now}
    assistant_msg = {"_id": ObjectId(), "role": "assistant", "content": response, "timestamp": now}

    # Build update — generate title from AI on first message
    is_first_message = len(thread.get("messages", [])) == 0
    update: dict = {"$push": {"messages": {"$each": [user_msg, assistant_msg]}}}
    if is_first_message:
        update["$set"] = {"title": generate_title(body.message)}

    await db.threads.update_one({"thread_id": thread_id}, update)

    return {"thread_id": thread_id, "id": str(assistant_msg["_id"]), "message": response}
