from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings


def get_embedding_model() -> Embeddings:
    """
    Return a local embedding model instance (no OpenAI / API calls).

    Uses the sentence-transformers/all-MiniLM-L6-v2 model,
    which runs locally on CPU via sentence-transformers.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings
