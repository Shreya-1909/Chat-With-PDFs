import streamlit as st


def render_header():
    st.title("📄 Chat with Your PDFs")
    st.caption(
        "Upload PDFs, index them with embeddings, and chat using Retrieval-Augmented Generation (RAG)."
    )


def render_sidebar():
    st.header("⚙️ Settings")

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    st.markdown("---")
    st.subheader("Chunking")
    chunk_size = st.slider("Chunk size (characters)", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, 150, step=50)

    st.markdown("---")
    st.subheader("Retrieval")
    top_k = st.slider("Top-k documents to retrieve", 2, 8, 4, step=1)
    st.session_state["top_k"] = top_k

    st.markdown("---")
    st.subheader("Index Info")
    num_pages = st.session_state.get("num_pages", 0)
    num_chunks = st.session_state.get("num_chunks", 0)
    st.metric("Pages indexed", num_pages)
    st.metric("Chunks indexed", num_chunks)

    return uploaded_files, chunk_size, chunk_overlap, top_k
