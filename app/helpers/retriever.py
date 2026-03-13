import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_db")
COMPANY_CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_company_db")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def retrieve_thread_docs(query: str, thread_id: str, k: int = 5) -> list[str]:
    vector_db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    results = vector_db.similarity_search(
        query,
        k=k,
        filter={"thread_id": thread_id}
    )
    return [doc.page_content for doc in results]


def retrieve_company_docs(query: str, k: int = 5) -> list[str]:
    vector_db = Chroma(
        persist_directory=COMPANY_CHROMA_DIR,
        embedding_function=embeddings
    )
    results = vector_db.similarity_search(query, k=k)
    return [doc.page_content for doc in results]
