"""
query.py — Interactive Q&A using Google Gemini (FREE).
Run AFTER ingest.py.
"""

import os
import sys
import textwrap
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from google import genai
from google.genai import types

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

CHROMA_PATH = Path("chroma_db")
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "models/gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"   # free tier model
TOP_K = 5

SYSTEM_PROMPT = """You are a precise document Q&A assistant.
Rules:
1. Answer ONLY using the information in the context provided.
2. If the answer is not in the context, say exactly:
   "I could not find an answer to that question in the provided documents."
3. At the end of every answer include a Sources section:
   Sources:
   - <filename>, page <N>
4. Be concise and factual."""

def get_collection():
    if not CHROMA_PATH.exists():
        print("Vector store not found. Run 'python src/ingest.py' first.")
        sys.exit(1)
    chroma = chromadb.PersistentClient(path=str(CHROMA_PATH), settings=Settings(anonymized_telemetry=False))
    return chroma.get_collection(COLLECTION_NAME)

def embed_query(query):
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return result.embeddings[0].values

def retrieve(collection, query_embedding, top_k=TOP_K):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        chunks.append({"text": doc, "source": meta["source"], "page": meta["page"], "distance": round(dist, 4)})
    return chunks

def generate_answer(query, chunks):
    if not chunks:
        return "I could not find an answer to that question in the provided documents."
    context = "\n\n".join(
        f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}"
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )
    return response.text.strip()

def interactive_loop():
    collection = get_collection()
    print(f"\n{'='*55}")
    print(" RAG Document Q&A Bot - Ready (Gemini Free)")
    print(f"{'='*55}")
    print(f" Chunks indexed: {collection.count()}")
    print(" Type your question. Type 'quit' to exit.")
    print(f"{'='*55}\n")

    while True:
        try:
            raw = input("❓ You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        print("\n⏳ Searching and generating answer...")
        embedding = embed_query(raw)
        chunks = retrieve(collection, embedding)
        answer = generate_answer(raw, chunks)

        print(f"\n{'='*55}\nANSWER\n{'='*55}")
        print(answer)
        print(f"\n{'─'*55}")
        print(f"Retrieved {len(chunks)} chunks:")
        for i, c in enumerate(chunks, 1):
            print(f"  [{i}] {c['source']} - page {c['page']} (dist={c['distance']})")
        print()

if __name__ == "__main__":
    interactive_loop()
