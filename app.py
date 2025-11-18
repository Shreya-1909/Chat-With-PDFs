import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.loaders.pdf_loader import save_and_load_pdfs
from src.processing.chunker import chunk_documents
from src.processing.embeddings import get_embedding_model
from src.vectorstore.chroma_store import build_vectorstore
from src.rag.qa_chain import get_conversational_qa_chain
from src.ui.layout import render_sidebar, render_header
from src.ui.components import chat_message, show_sources
from src.utils.session import init_session_state

# Load environment variables (OPENAI_API_KEY, etc.)
load_dotenv()

DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_DIR = DATA_DIR / "vectorstore"


def main():
    st.set_page_config(
        page_title="Chat with Your PDFs",
        page_icon="📄",
        layout="wide",
    )

    init_session_state()

    render_header()

    with st.sidebar:
        uploaded_files, chunk_size, chunk_overlap, k = render_sidebar()

    # Process documents
    if uploaded_files and st.sidebar.button("⚙️ Process Documents"):
        with st.spinner("Processing PDFs (upload → parse → chunk → embed)…"):
            docs = save_and_load_pdfs(uploaded_files, UPLOAD_DIR)
            st.session_state["num_pages"] = len(docs)

            chunks = chunk_documents(
                docs,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            st.session_state["num_chunks"] = len(chunks)

            embedding = get_embedding_model()
            vectorstore = build_vectorstore(
                chunks,
                embedding_model=embedding,
                persist_directory=str(VECTOR_DIR),
            )

            st.session_state["vectorstore"] = vectorstore
            st.session_state["chat_ready"] = True
            st.success(
                f"Indexed {len(uploaded_files)} file(s), "
                f"{st.session_state['num_pages']} pages, "
                f"{st.session_state['num_chunks']} chunks."
            )

    # If vectorstore exists, build QA chain
    if st.session_state.get("chat_ready") and st.session_state.get("vectorstore") is not None:
        vectorstore = st.session_state["vectorstore"]
        qa_chain = get_conversational_qa_chain(
            vectorstore=vectorstore,
            k=st.session_state.get("top_k", 4),
        )
        st.session_state["qa_chain"] = qa_chain

    # Chat area
    st.divider()
    st.subheader("💬 Chat")

    if not st.session_state.get("qa_chain"):
        st.info("Upload and process at least one PDF from the sidebar to start chatting.")
        return

    # Display previous chat history
    for message in st.session_state["chat_history"]:
        chat_message(message["role"], message["content"])

    # User input
    user_query = st.chat_input("Ask a question about your PDFs…")
    if user_query:
        # Add user message to history
        st.session_state["chat_history"].append({"role": "user", "content": user_query})
        chat_message("user", user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                qa_chain = st.session_state["qa_chain"]

                response = qa_chain(
                    {
                        "question": user_query,
                        "chat_history": [
                            (m["content"], n["content"])
                            for m, n in zip(
                                st.session_state["chat_history"][::2],
                                st.session_state["chat_history"][1::2],
                            )
                            if m["role"] == "user" and n["role"] == "assistant"
                        ],
                    }
                )

                answer = response["answer"]
                source_docs = response.get("source_documents", [])

                st.markdown(answer)
                show_sources(source_docs)

                # Save assistant answer to history
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": answer}
                )


if __name__ == "__main__":
    main()
