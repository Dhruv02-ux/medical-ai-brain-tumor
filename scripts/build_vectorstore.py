"""Run ONCE locally: python scripts/build_vectorstore.py
Then commit the vectorstore/ folder to git — the live app never rebuilds
the index, so it never pays repeated embedding-API cost at runtime."""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from app.genai.embeddings import get_embeddings
from app.core.config import VECTORSTORE_PATH

def build_index(source_path: str = "data/who_cns_guidelines.txt") -> None:
    docs = TextLoader(source_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    FAISS.from_documents(chunks, get_embeddings()).save_local(VECTORSTORE_PATH)
    print(f"Indexed {len(chunks)} chunks -> {VECTORSTORE_PATH}")

if __name__ == "__main__":
    build_index()