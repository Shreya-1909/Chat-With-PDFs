from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import streamlit as st


def save_and_load_pdfs(uploaded_files, upload_dir: Path) -> List[Document]:
    """
    Save uploaded PDFs to disk and load them into LangChain Document objects.
    """
    upload_dir.mkdir(parents=True, exist_ok=True)
    all_docs: List[Document] = []

    for uploaded_file in uploaded_files:
        file_path = upload_dir / uploaded_file.name

        # Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load with PyPDFLoader
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()

        # Attach source metadata
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata.setdefault("source", uploaded_file.name)

        all_docs.extend(docs)

    st.success(f"Loaded {len(all_docs)} pages from {len(uploaded_files)} file(s).")
    return all_docs
