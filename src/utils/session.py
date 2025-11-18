import streamlit as st


def init_session_state():
    """
    Initialize Streamlit session state variables.
    """
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None

    if "qa_chain" not in st.session_state:
        st.session_state["qa_chain"] = None

    if "chat_ready" not in st.session_state:
        st.session_state["chat_ready"] = False

    if "num_pages" not in st.session_state:
        st.session_state["num_pages"] = 0

    if "num_chunks" not in st.session_state:
        st.session_state["num_chunks"] = 0

    if "top_k" not in st.session_state:
        st.session_state["top_k"] = 4
