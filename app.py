"""
app.py — Streamlit web UI using Google Gemini (FREE).
Run with: streamlit run app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
import chromadb
from chromadb.config import Settings
import io
import time
import fitz  # PyMuPDF
import docx

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

CHROMA_PATH = Path("chroma_db")
COLLECTION_NAME = "documents"
GENERATION_MODEL = "llama-3.1-8b-instant"
TOP_K = 5
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 300

# --- IN-MEMORY INGESTION ENGINE ---
def load_pdf_stream(uploaded_file):
    pages = []
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"text": text, "source": uploaded_file.name, "page": page_num})
    return pages

def load_docx_stream(uploaded_file):
    doc = docx.Document(io.BytesIO(uploaded_file.read()))
    pages = []
    current_page = []
    page_num = 1
    
    for p in doc.paragraphs:
        xml_str = p._element.xml
        if 'w:lastRenderedPageBreak' in xml_str or 'w:type="page"' in xml_str:
            if current_page:
                text = "\n".join(current_page).strip()
                if text:
                    pages.append({"text": text, "source": uploaded_file.name, "page": page_num})
            current_page = []
            page_num += 1
            
        if p.text.strip():
            current_page.append(p.text)
            
    if current_page:
        text = "\n".join(current_page).strip()
        if text:
            pages.append({"text": text, "source": uploaded_file.name, "page": page_num})
            
    # Fallback if no page breaks were found but document is long
    if len(pages) == 1 and len(pages[0]["text"]) > 3500:
        full_text = pages[0]["text"]
        pages = []
        page_num = 1
        for i in range(0, len(full_text), 3500):
            pages.append({"text": full_text[i:i+3500], "source": uploaded_file.name, "page": page_num})
            page_num += 1
            
    return pages

def load_txt_stream(uploaded_file):
    text = uploaded_file.read().decode("utf-8", errors="ignore").strip()
    return [{"text": text, "source": uploaded_file.name, "page": 1}]

def chunk_documents(pages):
    all_chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk = text[start:end].strip()
            if chunk:
                all_chunks.append({"text": chunk, "source": page["source"], "page": page["page"], "chunk_index": idx})
                idx += 1
            if end == len(text):
                break
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return all_chunks

SYSTEM_PROMPT = """You are a helpful and detailed document Q&A assistant.
Rules:
1. Answer ONLY using the information in the context provided.
2. If the answer is not in the context, say: "I could not find an answer to that question in the provided documents."
3. Provide a clear, comprehensive answer with necessary details and explanations about the topic based on the context. Do not just return the page numbers.
4. Always end your response with a 'Sources:' section, listing the filenames and page numbers (e.g., Sources: - <filename>, page <N>)."""

st.set_page_config(page_title="RAG Q&A Bot", page_icon="📚", layout="wide")

# Inject premium CSS design styles
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Apply global font family and modern dark background */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    font-family: 'Outfit', sans-serif !important;
    background-color: #0b0f19 !important;
    color: #f1f5f9 !important;
}

/* Radical grid background */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: radial-gradient(rgba(99, 102, 241, 0.08) 1.5px, transparent 1.5px);
    background-size: 24px 24px;
    pointer-events: none;
    z-index: 0;
}

/* Beautiful Sidebar */
[data-testid="stSidebar"] {
    background-color: #060810 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.04) !important;
}

[data-testid="stSidebar"] h2 {
    color: #818cf8 !important;
    font-weight: 700 !important;
    font-size: 1.5rem !important;
}

/* Elegant Title and Gradient */
h1 {
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #c7d2fe 0%, #818cf8 50%, #4f46e5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-top: 10px !important;
    margin-bottom: 5px !important;
    letter-spacing: -0.5px;
}

.stCaption {
    color: #94a3b8 !important;
    text-align: center;
    font-size: 1rem !important;
    margin-bottom: 30px !important;
}

/* Form Container glassmorphism */
[data-testid="stForm"] {
    background: rgba(15, 23, 42, 0.65) !important;
    border: 1px solid rgba(99, 102, 241, 0.18) !important;
    border-radius: 20px !important;
    box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.4) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    padding: 30px !important;
    margin-bottom: 35px !important;
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

[data-testid="stForm"]:hover {
    border-color: rgba(99, 102, 241, 0.35) !important;
    box-shadow: 0 10px 40px 0 rgba(99, 102, 241, 0.08) !important;
}

/* Text Input container */
.stTextInput input {
    background-color: rgba(9, 13, 26, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    font-size: 1.05rem !important;
    transition: all 0.3s ease !important;
}

.stTextInput input:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.25) !important;
}

/* Glow submit button */
button[kind="primaryFormSubmit"] {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.3) !important;
    margin-top: 10px !important;
    width: 100% !important;
}

button[kind="primaryFormSubmit"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px 0 rgba(99, 102, 241, 0.45) !important;
    background: linear-gradient(135deg, #818cf8 0%, #4f46e5 100%) !important;
}

button[kind="primaryFormSubmit"]:active {
    transform: translateY(1px) !important;
}

/* Sidebar Action Buttons Custom Colors */
/* Clear Chat (Secondary) - Bright Blue */
button[data-testid="stBaseButton-secondary"] {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    padding: 8px 16px !important;
    width: 100% !important;
}

button[data-testid="stBaseButton-secondary"]:hover {
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4) !important;
    transform: translateY(-2px) !important;
    color: #ffffff !important;
}

/* Reset All & Process (Primary) - Bright Purple */
button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #a855f7 0%, #7e22ce 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    padding: 8px 16px !important;
}

button[data-testid="stBaseButton-primary"]:hover {
    box-shadow: 0 4px 15px rgba(168, 85, 247, 0.4) !important;
    transform: translateY(-2px) !important;
    color: #ffffff !important;
}

/* Custom Metrics Styling */
[data-testid="stMetric"] {
    background-color: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 15px !important;
    text-align: center !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
}

[data-testid="stMetricValue"] {
    color: #a855f7 !important; /* Beautiful purple numbers */
    font-size: 2.2rem !important;
    font-weight: 800 !important;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.95rem !important;
    justify-content: center !important;
}

/* Dotted File Uploader */
[data-testid="stFileUploadDropzone"] {
    background-color: rgba(255, 255, 255, 0.02) !important;
    border: 2px dashed rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploadDropzone"]:hover {
    border-color: #a855f7 !important;
    background-color: rgba(168, 85, 247, 0.05) !important;
}

/* Premium Chat Message styling */
[data-testid="stChatMessage"] {
    background-color: rgba(15, 23, 42, 0.45) !important;
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15) !important;
    backdrop-filter: blur(8px) !important;
    transition: all 0.3s ease !important;
}

[data-testid="stChatMessage"]:hover {
    border-color: rgba(99, 102, 241, 0.15) !important;
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.03) !important;
}

/* User Chat Style */
[data-testid="stChatMessage"]:has(span[data-testid="stChatMessageUserAvatar"]) {
    border-left: 5px solid #6366f1 !important;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.07) 0%, rgba(79, 70, 229, 0.02) 100%) !important;
}

/* Assistant Chat Style */
[data-testid="stChatMessage"]:has(span[data-testid="stChatMessageAssistantAvatar"]) {
    border-left: 5px solid #10b981 !important;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.07) 0%, rgba(6, 95, 70, 0.02) 100%) !important;
}

/* Expander sleek glass */
[data-testid="stExpander"] {
    background: rgba(9, 13, 26, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
    margin-top: 15px !important;
}

[data-testid="stExpander"] summary {
    font-weight: 500 !important;
    color: #a5b4fc !important;
}

[data-testid="stExpander"] p {
    color: #cbd5e1 !important;
}

/* Clean up divider styling */
hr {
    border: none !important;
    height: 1px !important;
    background: rgba(255, 255, 255, 0.06) !important;
    margin: 30px 0 !important;
}

/* Custom Success/Error banner styling */
[data-testid="stAlert"] {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 14px !important;
}

/* Sidebar history items */
.sidebar-history-item {
    background-color: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    margin-bottom: 10px !important;
    font-size: 0.92rem !important;
    color: #cbd5e1 !important;
    line-height: 1.4 !important;
    transition: all 0.3s ease !important;
}

.sidebar-history-item:hover {
    background-color: rgba(99, 102, 241, 0.08) !important;
    border-color: rgba(99, 102, 241, 0.2) !important;
    color: #ffffff !important;
}

/* --- PREMIUM HIGH-FIDELITY ANIMATIONS --- */

/* Floating emoji animation */
@keyframes emoji-float {
    0% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-8px) rotate(3deg); }
    100% { transform: translateY(0px) rotate(0deg); }
}

.floating-emoji {
    display: inline-block;
    font-size: 2.8rem;
    animation: emoji-float 3s ease-in-out infinite;
    margin-right: 12px;
    vertical-align: middle;
}

/* Gradient title with moving shift animation */
@keyframes gradient-shift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.gradient-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #c7d2fe 0%, #818cf8 25%, #4f46e5 50%, #818cf8 75%, #c7d2fe 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient-shift 6s linear infinite;
    vertical-align: middle;
    letter-spacing: -0.5px;
    display: inline-block;
}

/* Message entry fade-in and slide-up */
@keyframes message-slide-in {
    from {
        opacity: 0;
        transform: translateY(16px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

[data-testid="stChatMessage"] {
    animation: message-slide-in 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards !important;
}

/* Pulse glow for primary submit button on hover */
@keyframes pulse-glow {
    0% { box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.3); }
    50% { box-shadow: 0 4px 20px 8px rgba(99, 102, 241, 0.5); }
    100% { box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.3); }
}

button[kind="primaryFormSubmit"]:hover {
    animation: pulse-glow 1.5s infinite !important;
}

/* --- PREMIUM FLOATING CHAT INPUT BAR --- */

/* Pinned chat input styling */
[data-testid="stChatInput"] {
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(99, 102, 241, 0.22) !important;
    border-radius: 20px !important;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    padding: 8px 16px !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #818cf8 !important;
    box-shadow: 0 10px 40px rgba(99, 102, 241, 0.15) !important;
}

/* Chat input text field styling */
[data-testid="stChatInput"] textarea {
    color: #ffffff !important;
    font-size: 1.05rem !important;
    font-family: 'Outfit', sans-serif !important;
    line-height: 1.5 !important;
}

/* Send button glowing accent */
[data-testid="stChatInput"] button {
    background-color: #6366f1 !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    transition: all 0.3s ease !important;
}

[data-testid="stChatInput"] button:hover {
    background-color: #818cf8 !important;
    transform: scale(1.08) !important;
}

/* Fix Streamlit light mode default white backgrounds */
[data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {
    background-color: transparent !important;
}

[data-testid="stFileUploader"] section, 
[data-testid="stFileUploadDropzone"] {
    background-color: transparent !important;
    color: #f1f5f9 !important;
}

[data-testid="stFileUploader"] button {
    background-color: rgba(99, 102, 241, 0.2) !important;
    color: #ffffff !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
}

[data-testid="stFileUploader"] small, 
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #cbd5e1 !important;
}

/* Top Banner Styles */
.top-banner-card {
    background: linear-gradient(90deg, #8b5cf6 0%, #3b82f6 100%);
    padding: 30px;
    border-radius: 16px;
    margin-bottom: 30px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    text-align: left !important;
}

.top-banner-title {
    color: #ffffff !important;
    margin: 0;
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    background: none !important;
    display: block !important;
    letter-spacing: -0.5px;
}

.top-banner-icon {
    width: 48px;
    vertical-align: middle;
    margin-right: 12px;
    margin-bottom: 6px;
}

.top-banner-subtitle {
    color: #ffffff !important;
    margin: 10px 0 0 0;
    font-size: 1.05rem !important;
    opacity: 0.95;
    font-weight: 400 !important;
}

/* Responsive Design */
@media (max-width: 768px) {
    h1, .gradient-title, .top-banner-title {
        font-size: 1.8rem !important;
    }
    .top-banner-icon {
        width: 36px !important;
        margin-right: 8px !important;
        margin-bottom: 4px !important;
    }
    .top-banner-subtitle {
        font-size: 0.9rem !important;
    }
    .top-banner-card {
        padding: 20px !important;
    }
    .floating-emoji {
        font-size: 2rem !important;
    }
    [data-testid="stForm"] {
        padding: 15px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
    }
    [data-testid="stChatMessage"] {
        padding: 12px !important;
        margin-bottom: 12px !important;
    }
    [data-testid="stChatInput"] {
        padding: 6px 12px !important;
    }
    [data-testid="stSidebar"] h2 {
        font-size: 1.2rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# Initialize ChromaDB first so we can use its stats in the UI
@st.cache_resource(show_spinner="Connecting to vector store...")
def load_collection():
    chroma = chromadb.PersistentClient(path=str(CHROMA_PATH), settings=Settings(anonymized_telemetry=False))
    return chroma.get_collection(COLLECTION_NAME)

if not CHROMA_PATH.exists():
    collection = None
else:
    try:
        collection = load_collection()
        _ = collection.count()  # Ping the collection to ensure it exists
    except Exception:
        load_collection.clear()
        try:
            collection = load_collection()
        except Exception:
            collection = None

# Create the top wide gradient banner card matching the screenshot
st.markdown("""
<div class="top-banner-card">
    <div class="top-banner-title"><img src="https://em-content.zobj.net/source/apple/391/robot_1f916.png" class="top-banner-icon"> AI RAG Chatbot</div>
    <p class="top-banner-subtitle">Ask questions about your documents • Powered by Groq</p>
</div>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

# Build the custom sidebar matching the screenshot
with st.sidebar:
    st.markdown("### <img src='https://em-content.zobj.net/source/apple/391/robot_1f916.png' width='30' style='vertical-align: middle; margin-right: 8px; margin-bottom: 4px;'> AI RAG Chatbot\n**Powered by Groq**", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Metrics Row
    c1, c2 = st.columns(2)
    doc_count = 0
    chunk_count = 0
    if collection:
        chunk_count = collection.count()
        if chunk_count > 0:
            try:
                # Extract unique document names from metadata
                meta = collection.get(include=["metadatas"])["metadatas"]
                doc_count = len(set([m["source"] for m in meta if m and "source" in m]))
            except Exception:
                pass
    
    c1.metric("Documents", doc_count)
    c2.metric("Chunks", chunk_count)
    
    st.markdown("<hr style='margin: 15px 0; border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
    
    # File Uploader section
    st.markdown("### 📂 Upload Documents")
    uploaded_files = st.file_uploader("Drag and drop files here", accept_multiple_files=True, help="Limit 200MB per file • PDF, TXT, MD, DOCX, CSV")
    
    if st.button("🚀 Process Documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload files first.")
        else:
            with st.spinner("Extracting text from files..."):
                pages = []
                for f in uploaded_files:
                    ext = f.name.lower().split('.')[-1]
                    if ext == 'pdf':
                        pages.extend(load_pdf_stream(f))
                    elif ext == 'docx':
                        pages.extend(load_docx_stream(f))
                    else:
                        pages.extend(load_txt_stream(f))
                
                chunks = chunk_documents(pages)
            
            with st.spinner("Storing in ChromaDB (this will automatically generate embeddings locally)..."):
                chroma = chromadb.PersistentClient(path=str(CHROMA_PATH), settings=Settings(anonymized_telemetry=False))
                try:
                    chroma.delete_collection(COLLECTION_NAME)
                except Exception:
                    pass
                new_collection = chroma.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
                ids = [f"{c['source']}__p{c['page']}__c{c['chunk_index']}" for c in chunks]
                new_collection.add(
                    ids=ids,
                    documents=[c["text"] for c in chunks],
                    metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
                )
            
            st.success(f"✅ {len(chunks)} chunks stored!")
            time.sleep(1.5)
            load_collection.clear()  # Clear cache so it fetches the new collection ID!
            st.rerun()
            
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Action Buttons Side-by-Side
    bc1, bc2 = st.columns(2)
    if bc1.button("✨ New Chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    if bc2.button("🗑️ Reset All", type="primary", use_container_width=True):
        st.session_state.history = []
        try:
            chroma = chromadb.PersistentClient(path=str(CHROMA_PATH), settings=Settings(anonymized_telemetry=False))
            chroma.delete_collection(COLLECTION_NAME)
            load_collection.clear()
        except Exception:
            pass
        st.rerun()

    st.markdown("<hr style='margin: 15px 0; border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
    st.markdown("### 🕒 Chat History")
    if st.session_state.history:
        for entry in reversed(st.session_state.history):
            title = entry['question'][:40] + ("..." if len(entry['question']) > 40 else "")
            st.markdown(f"<div class='sidebar-history-item'>💬 {title}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color: #94a3b8; font-size: 0.9rem;'>No chat history yet.</div>", unsafe_allow_html=True)

if collection is None or collection.count() == 0:
    st.info("👋 **Welcome!** Your database is currently empty.\n\nPlease upload documents using the sidebar on the left and click **'Process Documents'** to begin.")
    st.stop()

# Show welcome message if chat is empty
if not st.session_state.history:
    with st.chat_message("assistant", avatar="😎"):
        st.markdown("""
        **👋 Welcome! I'm your AI RAG Chatbot powered by Groq.**
        
        📂 Upload documents in the sidebar (PDF, DOCX, TXT, CSV) and ask me anything about them.
        
        ⚡ Groq gives ultra-fast responses — try it!
        """)

# Render Chat History
for entry in st.session_state.history:
    with st.chat_message("user", avatar="❓"):
        st.markdown(entry["question"])
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(entry["answer"])
        if entry.get("chunks"):
            with st.expander(f"📄 Retrieved Chunks ({len(entry['chunks'])})"):
                for i, c in enumerate(entry["chunks"], 1):
                    st.markdown(f"**[{i}] {c['source']} — Page {c['page']}** (distance={c['distance']})")
                    st.text(c["text"][:400])
    st.markdown("<br>", unsafe_allow_html=True)

# Chat Input at the bottom
question = st.chat_input("Ask a question about your documents...")

if question and question.strip():
    with st.spinner("Thinking..."):
        results = collection.query(query_texts=[question], n_results=TOP_K, include=["documents", "metadatas", "distances"])
        chunks = [{"text": d, "source": m["source"], "page": m["page"], "distance": round(dist, 4)}
                  for d, m, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])]
        context = "\n\n".join(f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}" for c in chunks)
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()
        st.session_state.history.append({"question": question, "answer": answer, "chunks": chunks})
        st.rerun()
