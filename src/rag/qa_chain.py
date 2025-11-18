from typing import Any, List, Tuple

from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from .prompts import SYSTEM_PROMPT


# Create a global text-generation pipeline so the model is loaded only once
_model_name = "google/flan-t5-small"
_tokenizer = AutoTokenizer.from_pretrained(_model_name)
_model = AutoModelForSeq2SeqLM.from_pretrained(_model_name)
_local_llm = pipeline(
    "text2text-generation",
    model=_model,
    tokenizer=_tokenizer,
)


class SimpleConversationalQA:
    """
    A minimal conversational RAG wrapper that behaves like a LangChain chain:
    - It is callable: qa_chain({"question": ..., "chat_history": [...]})
    - It returns: {"answer": str, "source_documents": List[Document]}
    """

    def __init__(self, vectorstore: VectorStore, k: int = 4):
        # New-style retriever (Runnable in LangChain 0.3+)
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def __call__(self, inputs: dict) -> dict:
        question: str = inputs.get("question", "")
        chat_history: List[Tuple[str, str]] = inputs.get("chat_history", [])

        # 1. Retrieve relevant documents (use .invoke in new LangChain)
        docs: List[Document] = self.retriever.invoke(question)

        # 2. Build context text from retrieved docs
        context_text = "\n\n".join(
            f"[{i+1}] (source={d.metadata.get('source', '?')}, page={d.metadata.get('page', '?')})\n{d.page_content}"
            for i, d in enumerate(docs)
        )

        # 3. Format previous conversation (user/assistant pairs)
        history_text = "\n".join(
            f"User: {u}\nAssistant: {a}" for (u, a) in chat_history
        )

        # 4. Build a single prompt for the local model
        user_prompt = f"""
{SYSTEM_PROMPT}

Context:
{context_text}

Conversation so far:
{history_text}

User question: {question}

Answer based only on the context. If the answer is not in the context, say you don't know.
""".strip()

        # 5. Call the local HuggingFace model
        result = _local_llm(
            user_prompt,
            max_new_tokens=256,
        )[0]["generated_text"]

        return {
            "answer": result,
            "source_documents": docs,
        }


def get_conversational_qa_chain(
    vectorstore: VectorStore,
    k: int = 4,
) -> Any:
    """
    Factory function used by app.py to get a QA 'chain'.
    """
    return SimpleConversationalQA(vectorstore=vectorstore, k=k)
