import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from app.db.mongo import get_db

router = APIRouter()


@router.post("/create-thread")
async def create_thread():
    db = get_db()

    thread = {
        "thread_id": str(uuid.uuid4()),
        "title": "New Thread",
        "messages": [],
        "created_at": datetime.now(timezone.utc),
    }

    await db.threads.insert_one(thread)

    return {
        "thread_id": thread["thread_id"],
        "title": thread["title"],
        "messages": thread["messages"],
        "created_at": thread["created_at"],
    }


@router.get("/{thread_id}/list-messages")
async def list_messages(thread_id: str):
    db = get_db()

    thread = await db.threads.find_one({"thread_id": thread_id})
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Convert ObjectId to string in messages
    messages = []
    for msg in thread.get("messages", []):
        msg["_id"] = str(msg["_id"]) if "_id" in msg else None
        messages.append(msg)

    return {
        "thread_id": thread["thread_id"],
        "title": thread["title"],
        "messages": messages,
        "created_at": thread["created_at"],
    }
