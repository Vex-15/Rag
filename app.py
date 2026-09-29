"""


Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import io
import requests
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss


# Config
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
CHAT_MODEL_NAME = "openai/gpt-oss-20b"
CHUNK_SIZE = 800                          # characters per chunk
CHUNK_OVERLAP = 150                       # overlap between chunks
TOP_K = 4                                 # number of chunks retrieved per question

st.set_page_config(page_title="RAG Agent (PDF + VectorDB)",
                   page_icon="📄", layout="wide")


#
# Cached resources
#
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


# Helpers
#
def extract_text_from_pdf(file_bytes: bytes, filename: str):
    """Return a list of (page_number, text) tuples for a PDF file."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append((i + 1, text))
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Simple fixed-size character chunker with overlap."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks


def build_chunks_for_file(file_bytes: bytes, filename: str):
    """Extract, chunk, and tag chunks with source metadata."""
    pages = extract_text_from_pdf(file_bytes, filename)
    records = []
    for page_num, page_text in pages:
        for chunk in chunk_text(page_text):
            records.append({
                "text": chunk,
                "source": filename,
                "page": page_num,
            })
    return records


def rebuild_index(all_records):
    """Embed all chunk records and build a fresh FAISS index."""
    embedder = get_embedder()
    if not all_records:
        return None, None
    texts = [r["text"] for r in all_records]
    embeddings = embedder.encode(
        texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")
    dim = embeddings.shape[1]
    # cosine similarity via normalized inner product
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings


def retrieve(query: str, index, records, top_k: int = TOP_K):
    if index is None or not records:
        return []
    embedder = get_embedder()
    q_emb = embedder.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype="float32")
    scores, idxs = index.search(q_emb, min(top_k, len(records)))
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        results.append({**records[idx], "score": float(score)})
    return results


SYSTEM_PROMPT = """You are a strict document Q&A assistant (RAG agent).

Rules you MUST follow:
1. Answer ONLY using the information contained in the provided context chunks below.
2. If the answer cannot be found in the context, or the question is unrelated to the
   uploaded documents, respond EXACTLY with:
   "I can only answer questions based on the content of the uploaded document(s). I couldn't find information about that in them."
3. Do not use outside/general knowledge, even if you know the answer.
4. When you do answer, cite the source file name and page number(s) you used, like: (source: filename.pdf, page 3).
5. Be concise and accurate. Do not make up information.
"""


def call_llm(question: str, contexts: list, model: str = CHAT_MODEL_NAME):
    if not contexts:
        return "I can only answer questions based on the content of the uploaded document(s). I couldn't find information about that in them."

    context_block = "\n\n".join(
        f"[Chunk {i+1} | source: {c['source']} | page: {c['page']} | relevance: {c['score']:.2f}]\n{c['text']}"
        for i, c in enumerate(contexts)
    )

    user_prompt = f"""Context chunks from the uploaded document(s):
---------------------
{context_block}
---------------------

Question: {question}

Follow the system rules strictly. If the context above does not contain the answer,
say you cannot answer based on the documents — do not guess."""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def list_groq_models():
    try:
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        response = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=10)
        response.raise_for_status()
        models = [m["id"] for m in response.json().get("data", [])]
        # Filter out audio models if we just want chat, but for now return all
        return models if models else [CHAT_MODEL_NAME]
    except Exception:
        return [CHAT_MODEL_NAME]


# Session state
if "records" not in st.session_state:
    st.session_state.records = []       # list of chunk dicts
if "index" not in st.session_state:
    st.session_state.index = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, content)


# Sidebar
with st.sidebar:
    st.header(" Settings")
    groq_models = list_groq_models()
    selected_model = st.selectbox("Groq Model", groq_models, index=(groq_models.index(CHAT_MODEL_NAME) if CHAT_MODEL_NAME in groq_models else 0))
    st.caption("Powered by Groq API")

    st.divider()
    st.header(" Upload documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        new_files = [
            f for f in uploaded_files if f.name not in st.session_state.indexed_files]
        if new_files:
            with st.spinner(f"Processing {len(new_files)} new file(s) and updating vector DB..."):
                for f in new_files:
                    file_bytes = f.read()
                    records = build_chunks_for_file(file_bytes, f.name)
                    st.session_state.records.extend(records)
                    st.session_state.indexed_files.add(f.name)
                st.session_state.index, _ = rebuild_index(
                    st.session_state.records)
            st.success(f"Indexed {len(new_files)} new file(s).")

    if st.session_state.indexed_files:
        st.subheader("Indexed files")
        for name in sorted(st.session_state.indexed_files):
            st.write(f" {name}")
        st.caption(
            f"Total chunks in vector DB: {len(st.session_state.records)}")

    if st.button(" Clear all documents"):
        st.session_state.records = []
        st.session_state.index = None
        st.session_state.indexed_files = set()
        st.session_state.chat_history = []
        st.rerun()


# ----------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------
st.title(" RAG Agent — Ask Your Documents")
st.caption(
    "Upload PDF(s) in the sidebar, then ask questions. The agent answers strictly "
    "from your documents (via a FAISS vector database) and will refuse anything else."
)

if not st.session_state.indexed_files:
    st.info(" Upload at least one PDF from the sidebar to get started.")

for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)

question = st.chat_input("Ask a question about your uploaded document(s)...")

if question:
    st.session_state.chat_history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not st.session_state.indexed_files:
            answer = " Please upload at least one PDF first."
            st.warning(answer)
        else:
            with st.spinner("Retrieving relevant chunks and thinking..."):
                contexts = retrieve(
                    question, st.session_state.index, st.session_state.records, TOP_K)
                try:
                    answer = call_llm(question, contexts, model=selected_model)
                except requests.exceptions.ConnectionError:
                    answer = " Could not connect to Groq API."
                    st.warning(answer)
                except Exception as e:
                    answer = f" Error calling the LLM: {e}"
            st.markdown(answer)

            with st.expander(" Retrieved context used for this answer"):
                if contexts:
                    for c in contexts:
                        st.markdown(
                            f"**{c['source']}** (page {c['page']}, score {c['score']:.2f})")
                        st.text(c["text"][:500] +
                                ("..." if len(c["text"]) > 500 else ""))
                else:
                    st.write("No relevant context found.")

    st.session_state.chat_history.append(("assistant", answer))
