import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma

COMPANY_CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_company_db")
FILE_PATH = os.path.join(os.path.dirname(__file__), "company_docs", "sera_company.txt")

# Clear existing data so re-running doesn't create duplicates
if os.path.exists(COMPANY_CHROMA_DIR):
    shutil.rmtree(COMPANY_CHROMA_DIR)
    print("Cleared existing company knowledge base.")

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)

loader = TextLoader(FILE_PATH)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=25)
chunks = text_splitter.split_documents(documents)

Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=COMPANY_CHROMA_DIR
)

print(f"Company knowledge ingested: {len(chunks)} chunks saved to {COMPANY_CHROMA_DIR}")
