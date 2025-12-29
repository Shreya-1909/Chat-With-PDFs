# 📄 Chat with Your PDFs (RAG + LangChain + Streamlit)

A simple **"Chat with your PDFs"** app built with **Python, LangChain, Chroma, and Streamlit**.  
Upload one or more PDF documents and ask natural-language questions about their content.  
Under the hood, the app uses **Retrieval-Augmented Generation (RAG)** with text embeddings and a vector store.

## ✨ Features

- Upload multiple PDF files from the sidebar
- Automatic **text extraction, chunking, and embedding**
- Vector search with **Chroma** as a persistent vector store
- **Conversational** Q&A with memory using LangChain’s `ConversationalRetrievalChain`
- Answers include **source snippets** and document metadata (file name, page number)

---

## 🧱 Tech Stack

- **Language / Framework**: Python, Streamlit
- **LLM / RAG Framework**: LangChain
- **Embeddings & Chat Model**: OpenAI (via `langchain-openai`)
- **Vector Store**: ChromaDB
- **PDF Parsing**: `pypdf`
- **Config**: `python-dotenv`

---

## 📂 Project Structure

```bash
chat-with-pdfs/
├── app.py
├── requirements.txt
├── README.md
├── .env              # (create this) holds your API keys
├── data/
│   ├── uploads/      # uploaded PDFs
│   └── vectorstore/  # Chroma persistent index
└── src/
    ├── loaders/
    │   └── pdf_loader.py
    ├── processing/
    │   ├── chunker.py
    │   └── embeddings.py
    ├── vectorstore/
    │   └── chroma_store.py
    ├── rag/
    │   ├── qa_chain.py
    │   └── prompts.py
    ├── ui/
    │   ├── layout.py
    │   └── components.py
    └── utils/
        ├── helpers.py
        └── session.py
```

---

## ⚙️ Setup Instructions

### 1️⃣ Create and activate a virtual environment (recommended)

```bash
cd chat-with-pdfs

# Create venv
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS / Linux)
source .venv/bin/activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Set your OpenAI API key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

Or set it directly in your shell:

```bash
export OPENAI_API_KEY=your_openai_api_key_here  # macOS / Linux
setx OPENAI_API_KEY "your_openai_api_key_here"  # Windows (PowerShell)
```

### 4️⃣ Run the app

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

---

## 🧠 How It Works (High Level)

1. **Upload PDFs**  
   PDF files are uploaded via Streamlit and saved under `data/uploads/`.

2. **Load & Chunk**  
   - `pdf_loader.py` extracts text from each PDF.  
   - `chunker.py` splits text into overlapping chunks (configurable size & overlap).

3. **Embeddings + Vector Store**  
   - `embeddings.py` uses OpenAI embeddings to convert chunks to vectors.  
   - `chroma_store.py` stores these vectors in a persistent **Chroma** database.

4. **RAG-based Q&A**  
   - `qa_chain.py` builds a `ConversationalRetrievalChain` using:
     - A retriever from the vector store.
     - A chat model from OpenAI.
   - On each question, the app:
     - Retrieves the top-k most relevant chunks.
     - Passes them + the question to the LLM.
     - Returns an answer grounded in those chunks.

5. **UI & Conversation**  
   - Chat-style interface with `st.chat_input` and `st.chat_message`.
   - Conversation history is stored in `st.session_state`.

---

## 🧪 Ideas for Extensions

- Add support for:
  - File type filters (e.g., `docx`, `txt`)
  - Confidence scores for answers
  - Document selection / filtering in sidebar
- Display retrieved chunk snippets inline with citations
- Add authentication and per-user document spaces
- Deploy to Streamlit Community Cloud / Azure / GCP

---

