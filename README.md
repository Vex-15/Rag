# 📄 RAG Agent (Streamlit + FAISS Vector DB)

A simple Retrieval-Augmented Generation (RAG) agent you can run locally with Streamlit.

- Upload **one or multiple PDF files**.
- Documents are chunked and embedded locally with `sentence-transformers`
  (`all-MiniLM-L6-v2` — free, no API key required for embeddings).
- Chunks are stored in an in-memory **FAISS** vector database.
- Ask questions in a chat UI. The agent retrieves the most relevant chunks
  and queries the Groq API (dynamically fetching available models) to answer **only**
  from that retrieved context.
- If your question isn't answerable from the uploaded documents, the agent
  refuses instead of making something up or answering from general knowledge.

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Setup Groq API Key

The app uses Groq for fast LLM inference. You must set your `GROQ_API_KEY` in the environment before launching:

```bash
export GROQ_API_KEY="your-api-key-here"         # Windows (PowerShell): $env:GROQ_API_KEY="your-api-key-here"
```

## 3. Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## 4. Use it

1. Select a Groq model in the sidebar. The available models are dynamically loaded
   from your API key's permissions.
2. Upload one or more PDFs.
3. Wait for "Indexed N new file(s)" — this means the text has been chunked,
   embedded, and added to the FAISS vector index.
4. Ask questions in the chat box at the bottom.
5. Expand "🔍 Retrieved context used for this answer" under any answer to
   see exactly which chunks were used.
6. Click "🗑️ Clear all documents" in the sidebar to reset and start fresh.

## How the "only answer from the PDFs" behavior works

Two layers enforce this:

1. **Retrieval gate** — if no reasonably relevant chunks are retrieved for a
   question, the app returns a fixed refusal message without even calling
   the LLM.
2. **System prompt** — the LLM is instructed to answer strictly from the
   provided context chunks and to output a specific refusal sentence when
   the answer isn't in them, and to cite source file + page for anything it
   does answer.

## Customizing

- **Swap the LLM provider**: replace the Groq request in `call_llm()` with
  your provider of choice — the retrieval/vector-DB part is provider-agnostic.
- **Chunk size / overlap**: tune `CHUNK_SIZE` and `CHUNK_OVERLAP` in
  `app.py`.
- **Number of retrieved chunks**: tune `TOP_K`.
- **Embedding model**: swap `EMBED_MODEL_NAME` for any Sentence-Transformers
  model (e.g. a larger, more accurate one) if you don't mind slower/larger
  downloads.
- **Persistent vector DB**: this demo keeps the FAISS index in memory for
  the session only. For persistence across restarts, swap FAISS for
  Chroma/Weaviate/Pinecone or serialize the FAISS index to disk.

## Notes

- Only PDFs are supported out of the box (easy to extend to .docx/.txt by
  adding another extractor and calling `build_chunks_for_file`-style logic).
- Embeddings run locally; text generation uses the Groq API.
