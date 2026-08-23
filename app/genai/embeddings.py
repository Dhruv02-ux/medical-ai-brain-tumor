"""Embeddings — uses local sentence-transformers model for reliable offline operation."""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from app.core.config import EMBEDDING_MODEL, VECTORSTORE_PATH

def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

_vectorstore: FAISS | None = None

def get_vectorstore() -> FAISS:
    """Loads the prebuilt FAISS index (built offline — see scripts/build_vectorstore.py)."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = FAISS.load_local(VECTORSTORE_PATH, get_embeddings(), allow_dangerous_deserialization=True)
    return _vectorstore