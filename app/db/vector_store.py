import os
import asyncio
import concurrent.futures
import chromadb

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_db")
COMPANY_CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_company_db")

# Single-threaded executor — all ChromaDB reads/writes happen in this one thread.
# This avoids Rust backend cross-thread issues entirely.
_chroma_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="chroma")

_thread_client = None
_company_client = None


def get_thread_chroma_client() -> chromadb.PersistentClient:
    """Must only be called from within _chroma_executor thread."""
    global _thread_client
    if _thread_client is None:
        _thread_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _thread_client


def get_company_chroma_client() -> chromadb.PersistentClient:
    """Must only be called from within _chroma_executor thread."""
    global _company_client
    if _company_client is None:
        _company_client = chromadb.PersistentClient(path=COMPANY_CHROMA_DIR)
    return _company_client


def get_chroma_executor() -> concurrent.futures.ThreadPoolExecutor:
    return _chroma_executor


async def run_in_chroma_thread(func):
    return await asyncio.get_running_loop().run_in_executor(_chroma_executor, func)
