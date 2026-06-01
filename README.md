# RAG Document Q&A Bot

A command-line and web-based Q&A bot that lets you ask natural-language questions against a collection of documents and receive accurate, source-cited answers — powered by a full Retrieval-Augmented Generation (RAG) pipeline built from scratch.

---

## Tech Stack

| Component | Library / Tool | Version |
|-----------|----------------|---------|
| Language | Python | 3.11+ |
| PDF loading | PyMuPDF (`fitz`) | 1.24.5 |
| DOCX loading | python-docx | 1.1.2 |
| Tokenisation | tiktoken | 0.7.0 |
| Embeddings | OpenAI `text-embedding-3-small` | via openai 1.30.5 |
| Vector DB | ChromaDB (persisted) | 0.5.3 |
| LLM | OpenAI `gpt-4o-mini` | via openai 1.30.5 |
| Web UI (bonus) | Streamlit | 1.35.0 |
| Env management | python-dotenv | 1.0.1 |

---

## Architecture Overview

```
Documents (PDF / DOCX / TXT)
        │
        ▼
  [1] Document Loader          — PyMuPDF / python-docx / built-in
        │  (page-level text)
        ▼
  [2] Text Chunker             — token-aware sliding window (tiktoken)
        │  (chunks + metadata)
        ▼
  [3] Embedding Model          — text-embedding-3-small (batched, 96/call)
        │  (dense vectors)
        ▼
  [4] Vector Store             — ChromaDB (persisted to chroma_db/)
        │
        │  ◄─── User Query
        ▼
  [5] Retrieval                — cosine similarity, top-k chunks
        │
        ▼
  [6] Answer Generation        — gpt-4o-mini, context-only system prompt
        │
        ▼
  Answer + Source Citations
```

**Indexing** (run once):  `src/ingest.py`
**Querying** (run anytime): `src/query.py` or `app.py`

---

## Chunking Strategy

**Strategy chosen:** Token-aware fixed-size sliding window with overlap.

**Why:**
- Embedding models have a hard token limit (8 191 tokens for `text-embedding-3-small`). Character- or word-based splits can silently truncate long sentences, producing malformed embeddings. Tiktoken encodes the same way the model does, so every chunk is guaranteed to fit.
- A fixed window of **400 tokens** is small enough for the retrieved chunk to stay focused on a single topic, yet large enough to contain full sentences.
- An overlap of **80 tokens** (~20%) ensures that sentences spanning a chunk boundary are still retrievable from either neighbouring chunk, preventing context loss at the seams.

---

## Embedding Model and Vector Database

**Embedding model:** `text-embedding-3-small`
- 1 536-dimensional vectors; strong multilingual performance.
- Cheap ($0.00002 / 1K tokens) and fast — suitable for a local knowledge base of 4–5 documents.
- All calls are **batched** (up to 96 chunks per API call) to minimise latency and cost.

**Vector database:** ChromaDB
- Zero-infrastructure: runs embedded in-process, persists to a local folder (`chroma_db/`).
- Cosine similarity search out of the box.
- No Docker or external server needed — ideal for a self-contained intern project.
- Clear separation: `ingest.py` populates the store; `query.py` only reads from it.

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/rag-qa-bot.git
cd rag-qa-bot
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API key
```bash
cp .env.example .env
# Open .env and paste your key
```

### 5. Add your documents
Place 4–5 PDF / DOCX / TXT files in the `data/` folder.

### 6. Index the documents (run once)
```bash
python src/ingest.py
```
This loads, chunks, embeds, and stores all documents in `chroma_db/`.

### 7. Ask questions

**CLI (interactive loop):**
```bash
python src/query.py
```

**CLI (single question):**
```bash
python src/query.py --question "What is the main argument of document X?"
```

**Web UI (Streamlit):**
```bash
streamlit run app.py
```
Then open [http://localhost:8501](http://localhost:8501).

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | Your OpenAI API key — get one at [platform.openai.com](https://platform.openai.com) |

**Never commit your actual key.** Add it to `.env` (which is git-ignored).

---

## Example Queries

| Query | Expected answer theme |
|-------|-----------------------|
| "What is RAG and how does it work?" | Retrieval-Augmented Generation overview |
| "What are the main limitations of large language models?" | Hallucination, knowledge cutoff, context length |
| "Summarise the key points of the climate report." | Document-specific summary |
| "What methodology was used in the research paper?" | Research design details |
| "What is the author's conclusion about AI safety?" | Document conclusion / recommendations |

---

## Known Limitations

| Limitation | Reason |
|------------|--------|
| Scanned PDFs are not supported | PyMuPDF extracts embedded text; scanned images require OCR (e.g. Tesseract) |
| Very long answers may be cut off | `max_tokens=1024` in the generation call — increase if needed |
| Table and chart data may be lost | PDF text extraction is linear; complex layouts lose structure |
| No conversation memory | Each query is independent; follow-up questions need full context re-stated |
| Retrieval quality depends on chunk relevance | If the cosine distance threshold is too strict, borderline-relevant chunks are dropped |

---

## Project Structure

```
rag-qa-bot/
├── data/                  # Your documents (PDF, DOCX, TXT)
├── src/
│   ├── ingest.py          # Indexing pipeline (load → chunk → embed → store)
│   └── query.py           # Retrieval + generation pipeline
├── app.py                 # Streamlit web UI (bonus)
├── chroma_db/             # Persisted vector store (auto-created by ingest.py)
├── requirements.txt
├── .env.example
└── README.md
```
