import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from app.db.mongo import get_db
from app.auth import get_auth_user, AuthUser

router = APIRouter()


@router.post("/create-thread")
async def create_thread(user: AuthUser = Depends(get_auth_user)):
    db = get_db()

    thread = {
        "thread_id": str(uuid.uuid4()),
        "user_id": user.user_id,
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

# ========================== List threads =========================

@router.get("/list-threads")
async def list_threads(user: AuthUser = Depends(get_auth_user)):
    if user.auth_type != "auth_user":
        return []

    db = get_db()

    cursor = db.threads.find(
        {"user_id": user.user_id},
        {"_id": 0, "thread_id": 1, "title": 1, "created_at": 1},
    ).sort("created_at", -1)

    threads = await cursor.to_list(length=None)
    return threads

# ========================== Dlete thread =========================

@router.delete("/{thread_id}/delete-thread")
async def delete_thread(thread_id: str, user: AuthUser = Depends(get_auth_user)):
    if user.auth_type == "anon_user":
        raise HTTPException(status_code=403, detail="Anonymous users cannot delete threads")

    db = get_db()

    thread = await db.threads.find_one({"thread_id": thread_id})
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    if thread.get("user_id") != user.user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    await db.threads.delete_one({"thread_id": thread_id})

    return {"message": "Thread deleted successfully"}

# ========================== List messages =========================

@router.get("/{thread_id}/list-messages")
async def list_messages(thread_id: str, user: AuthUser = Depends(get_auth_user)):
    db = get_db()

    thread = await db.threads.find_one({"thread_id": thread_id})
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    if thread.get("user_id") != user.user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    messages = []
    for msg in thread.get("messages", []):
        msg["_id"] = str(msg["_id"]) if "_id" in msg else None
        messages.append(msg)

    return {
        "thread_id": thread["thread_id"],
        "title": thread["title"],
        "messages": messages,
        "created_at": thread["created_at"],
        "attached_files": thread.get("attached_files", 0),
    }

# ========================== Shared messages =========================

@router.get("/{thread_id}/get-shared-messages")
async def list_messages(thread_id: str):
    db = get_db()

    thread = await db.threads.find_one({"thread_id": thread_id})
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    messages = []
    for msg in thread.get("messages", []):
        msg["_id"] = str(msg["_id"]) if "_id" in msg else None
        messages.append(msg)

    return {
        "thread_id": thread["thread_id"],
        "title": thread["title"],
        "messages": messages,
        "created_at": thread["created_at"],
        "attached_files": thread.get("attached_files", 0),
    }
