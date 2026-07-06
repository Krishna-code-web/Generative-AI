import hashlib
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-book-assistant")

st.set_page_config(page_title="RAG Book Assistant", page_icon="📚", layout="wide")

BASE_DB_DIR = Path(os.environ.get("CHROMA_BASE_DIR", "chroma-db"))
MAX_FILE_MB = int(os.environ.get("MAX_FILE_MB", "50"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-small-2506")

SYSTEM_PROMPT = """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context, say exactly:
"I could not find the answer in the document."

Cite the page number(s) you used when possible.
"""

# --------------------------------------------------------------------------
# Guard: required API key
# --------------------------------------------------------------------------

if not os.environ.get("MISTRAL_API_KEY"):
    st.error(
        "MISTRAL_API_KEY is not set. Add it to a `.env` file or your "
        "environment/secrets before running this app."
    )
    st.stop()

# --------------------------------------------------------------------------
# Cached resources (created once per process, not per rerun)
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return MistralAIEmbeddings()


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatMistralAI(model=MISTRAL_MODEL, temperature=0)


@st.cache_resource(show_spinner=False)
def get_prompt():
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}\n"),
        ]
    )


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def build_vectorstore(file_bytes: bytes, collection_dir: Path):
    """Load, split, and embed a PDF into a fresh Chroma collection."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        if not docs:
            raise ValueError("No extractable text found in this PDF.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            persist_directory=str(collection_dir),
        )
        return vectorstore, len(docs), len(chunks)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]
if "processed_hash" not in st.session_state:
    st.session_state.processed_hash = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer, sources)

session_db_dir = BASE_DB_DIR / st.session_state.session_id

# --------------------------------------------------------------------------
# UI - header
# --------------------------------------------------------------------------

st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions about its contents.")

with st.sidebar:
    st.subheader("Settings")
    st.caption(f"Model: `{MISTRAL_MODEL}`")
    st.caption(f"Session: `{st.session_state.session_id}`")
    k = st.slider("Chunks to retrieve (k)", 2, 10, 4)
    if st.button("🗑️ Clear this session's data"):
        if session_db_dir.exists():
            shutil.rmtree(session_db_dir, ignore_errors=True)
        st.session_state.processed_hash = None
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.rerun()

# --------------------------------------------------------------------------
# Upload + process
# --------------------------------------------------------------------------

uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    size_mb = len(file_bytes) / (1024 * 1024)

    if size_mb > MAX_FILE_MB:
        st.error(f"File is {size_mb:.1f} MB, which exceeds the {MAX_FILE_MB} MB limit.")
        st.stop()

    current_hash = file_hash(file_bytes)
    st.success(f"'{uploaded_file.name}' uploaded ({size_mb:.1f} MB).")

    already_processed = current_hash == st.session_state.processed_hash

    if already_processed:
        st.info("This document is already indexed for this session.")
    elif st.button("Create Vector Database", type="primary"):
        try:
            with st.spinner("Processing document..."):
                collection_dir = session_db_dir / current_hash
                vectorstore, num_pages, num_chunks = build_vectorstore(
                    file_bytes, collection_dir
                )
            st.session_state.vectorstore = vectorstore
            st.session_state.processed_hash = current_hash
            st.session_state.chat_history = []
            st.success(f"Indexed {num_pages} page(s) into {num_chunks} chunk(s).")
        except Exception as exc:
            logger.exception("Failed to build vector database")
            st.error(f"Could not process this PDF: {exc}")

# --------------------------------------------------------------------------
# Q&A
# --------------------------------------------------------------------------

if st.session_state.vectorstore is not None:
    st.divider()
    st.subheader("Ask Questions From the Book")

    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(10, k * 2), "lambda_mult": 0.5},
    )

    for question, answer, sources in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
            if sources:
                st.caption("Sources: " + ", ".join(sources))

    query = st.chat_input("Enter your question")

    if query:
        with st.chat_message("user"):
            st.write(query)

        try:
            with st.spinner("Thinking..."):
                docs = retriever.invoke(query)

                if not docs:
                    answer = "I could not find the answer in the document."
                    sources = []
                else:
                    context = "\n\n".join(doc.page_content for doc in docs)
                    sources = sorted(
                        {
                            f"p.{doc.metadata.get('page', '?')}"
                            for doc in docs
                            if doc.metadata
                        }
                    )
                    final_prompt = get_prompt().invoke(
                        {"context": context, "question": query}
                    )
                    response = get_llm().invoke(final_prompt)
                    answer = response.content

            with st.chat_message("assistant"):
                st.write(answer)
                if sources:
                    st.caption("Sources: " + ", ".join(sources))

            st.session_state.chat_history.append((query, answer, sources))

        except Exception as exc:
            logger.exception("Failed to answer query")
            st.error(f"Something went wrong while answering: {exc}")
else:
    st.info("Upload a PDF and click **Create Vector Database** to get started.")
