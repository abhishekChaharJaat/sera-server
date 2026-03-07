import os
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

_client: AsyncIOMotorClient = None


def connect():
    global _client
    _client = AsyncIOMotorClient(os.environ.get("MONGO_URL", "mongodb://localhost:27017"))
    print("MongoDB connected")


def disconnect():
    global _client
    if _client:
        _client.close()
        print("MongoDB disconnected")


def get_db() -> AsyncIOMotorDatabase:
    return _client[os.environ.get("MONGO_DB", "sera")]
