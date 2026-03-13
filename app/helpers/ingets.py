import os
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_db")

SUPPORTED_EXTENSIONS = {".docx", ".pdf", ".txt"}

# Initialize once at module level to avoid reloading on every call
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def ingest_file(file_path: str, file_id: str, thread_id: str):
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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=25
    )
    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["file_id"] = file_id
        chunk.metadata["thread_id"] = thread_id

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"[DEBUG] ingest_file done: file_id={file_id} thread_id={thread_id} chunks={len(chunks)}")
