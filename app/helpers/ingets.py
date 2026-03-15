import os
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma
from app.db.vector_store import get_thread_chroma_client

SUPPORTED_EXTENSIONS = {".docx", ".pdf", ".txt"}

_embeddings = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        )
    return _embeddings


def read_file_content(file_path: str, max_chars: int = 15000) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return ""
    if ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    docs = loader.load()
    return "\n\n".join(d.page_content for d in docs)[:max_chars]


def ingest_file(file_path: str, file_id: str, thread_id: str):
    """Sync — must be called via get_chroma_executor() to stay on the ChromaDB thread."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return

    if ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["file_id"] = file_id
        chunk.metadata["thread_id"] = thread_id

    vector_db = Chroma(
        client=get_thread_chroma_client(),
        embedding_function=_get_embeddings()
    )
    vector_db.add_documents(chunks)
    print(f"[DEBUG] ingest_file done: file_id={file_id} chunks={len(chunks)}")
