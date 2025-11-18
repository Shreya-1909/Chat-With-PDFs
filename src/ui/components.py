from typing import List

import streamlit as st
from langchain_core.documents import Document


def chat_message(role: str, content: str):
    """
    Render a single chat message in Streamlit.
    role: "user" or "assistant"
    """
    with st.chat_message(role):
        st.markdown(content)


def show_sources(source_docs: List[Document]):
    """
    Show a list of source documents (chunks) used to answer the question.
    """
    if not source_docs:
        return

    with st.expander("📚 View sources"):
        for i, doc in enumerate(source_docs, start=1):
            metadata = doc.metadata or {}
            source = metadata.get("source", "Unknown source")
            page = metadata.get("page", "N/A")

            st.markdown(f"**Source {i}:** `{source}` (page {page})")
            st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
            st.markdown("---")
