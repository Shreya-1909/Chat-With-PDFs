from typing import List
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document


def build_vectorstore(
    docs: List[Document],
    embedding_model: Embeddings,
    persist_directory: str,
):
    """
    Build an in-memory vector store from the given documents.

    Note:
        We ignore persist_directory here and keep everything in memory
        to avoid heavy dependencies like chromadb/pyarrow.
    """
    vectorstore = DocArrayInMemorySearch.from_documents(
        documents=docs,
        embedding=embedding_model,
    )
    return vectorstore
