from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Split documents into overlapping chunks for better retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)

    # Ensure metadata contains a chunk id
    for i, chunk in enumerate(chunks):
        chunk.metadata = chunk.metadata or {}
        chunk.metadata.setdefault("chunk_id", i)

    return chunks
