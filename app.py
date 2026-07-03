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
from auth.auth_handler import register_user, authenticate_user
from streamlit_cookies_controller import CookieController

load_dotenv()

# Load API Key with fallback to Streamlit secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY and "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

if not GROQ_API_KEY:
    st.error("🔑 **GROQ_API_KEY is missing!** Please add it to your environment variables or Streamlit Secrets (under settings -> Secrets).")
    st.stop()

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

SYSTEM_PROMPT = """You are a helpful and detailed AI assistant.
Rules:
1. Prioritize answering using the information in the provided context if it is relevant.
2. If no context is provided, or if the answer is not found in the documents, use your general knowledge to answer the question. If you do this, politely mention that you are answering from general knowledge because the documents didn't contain the information.
3. Provide a clear, comprehensive answer with necessary details and explanations about the topic. Do not just return the page numbers.
4. If you used the provided documents, always end your response with a 'Sources:' section, listing the filenames and page numbers (e.g., Sources: - <filename>, page <N>)."""

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
/* Clear Chat (Secondary) - Bright Purple */
button[data-testid="stBaseButton-secondary"], button[kind="secondary"] {
    background: linear-gradient(135deg, #a855f7 0%, #7e22ce 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    padding: 8px 16px !important;
    width: 100% !important;
}

button[data-testid="stBaseButton-secondary"]:hover, button[kind="secondary"]:hover {
    box-shadow: 0 4px 15px rgba(168, 85, 247, 0.4) !important;
    transform: translateY(-2px) !important;
    color: #ffffff !important;
}

/* Reset All & Process (Primary) - Bright Purple */
button[data-testid="stBaseButton-primary"], button[kind="primary"] {
    background: linear-gradient(135deg, #a855f7 0%, #7e22ce 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    padding: 8px 16px !important;
}

button[data-testid="stBaseButton-primary"]:hover, button[kind="primary"]:hover {
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
.sidebar-history-item,
button[data-testid="stBaseButton-tertiary"] {
    background-color: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    margin-bottom: 10px !important;
    font-size: 0.92rem !important;
    color: #cbd5e1 !important;
    line-height: 1.4 !important;
    transition: all 0.3s ease !important;
    text-align: left !important;
    justify-content: flex-start !important;
}

.sidebar-history-item:hover,
button[data-testid="stBaseButton-tertiary"]:hover {
    background-color: rgba(99, 102, 241, 0.08) !important;
    border-color: rgba(99, 102, 241, 0.2) !important;
    color: #ffffff !important;
}

button[data-testid="stBaseButton-tertiary"] p {
    margin: 0 !important;
    font-size: 0.92rem !important;
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

# --- SESSION & AUTHENTICATION MANAGEMENT ---
controller = CookieController()


def logout():
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.history = []
    st.session_state.all_history = []
    st.session_state.logout_triggered = True
    st.query_params.pop("user", None)
    try:
        controller.remove('auth_username')
    except Exception:
        pass

# Try to auto-login using query parameters or persisted cookies
if "logout_triggered" not in st.session_state:
    st.session_state.logout_triggered = False

if not st.session_state.logout_triggered:
    # 1. Check query parameters first (instant, zero-flicker reconnect)
    query_user = st.query_params.get("user")
    if query_user:
        st.session_state.authenticated = True
        st.session_state.username = query_user
    
    # 2. Fall back to cookies if query params are not present
    if "authenticated" not in st.session_state or not st.session_state.authenticated:
        try:
            # 2a. Try synchronous check first
            persisted_username = st.context.cookies.get('auth_username')
            # 2b. Fall back to client-side cookie component check
            if not persisted_username:
                persisted_username = controller.get('auth_username')
                
            if persisted_username:
                st.session_state.authenticated = True
                st.session_state.username = persisted_username
                # Sync back to query params for future reconnects
                st.query_params["user"] = persisted_username
        except Exception:
            pass

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "auth_page" not in st.session_state:
    st.session_state.auth_page = "login"

if not st.session_state.authenticated:
    # Custom CSS for Auth Page (Highly compact layout)
    st.markdown("""
    <style>
    /* Styling for secondary buttons on Login/Register page */
    button[data-testid="stBaseButton-secondary"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        border-radius: 10px !important;
        padding: 6px 12px !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        font-size: 0.85rem !important;
    }
    button[data-testid="stBaseButton-secondary"]:hover {
        background: rgba(99, 102, 241, 0.1) !important;
        border-color: rgba(99, 102, 241, 0.3) !important;
        color: #818cf8 !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1) !important;
    }
    /* Reduce container padding for the auth screens specifically */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1.5rem !important;
    }
    /* Compact forms for auth screens */
    [data-testid="stForm"] {
        padding: 16px 20px !important;
        margin-bottom: 8px !important;
    }
    .stTextInput input {
        padding: 8px 12px !important;
        font-size: 0.9rem !important;
    }
    .stTextInput label {
        font-size: 0.8rem !important;
        margin-bottom: 1px !important;
    }
    /* Hide "Press Enter to submit form" helper text */
    [data-testid="InputInstructions"] {
        display: none !important;
    }
    button[kind="primaryFormSubmit"] {
        padding: 8px 16px !important;
        margin-top: 4px !important;
        font-size: 0.9rem !important;
    }
    /* Reduce vertical space between Streamlit elements in auth */
    [data-testid="stVerticalBlock"] > div {
        gap: 0.4rem !important;
    }
    /* Center and restrict width of the auth elements responsively */
    [data-testid="stForm"], 
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] {
        max-width: 420px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header block
    st.markdown("""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: 1vh; margin-bottom: 10px;">
        <div style="width: 50px; height: 50px; border-radius: 50%; background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); display: flex; align-items: center; justify-content: center; box-shadow: 0 0 12px rgba(99, 102, 241, 0.4); margin-bottom: 8px;">
            <span style="font-size: 1.5rem;">🤖</span>
        </div>
        <h2 style="color: #ffffff; font-weight: 700; margin: 0; font-size: 1.6rem; text-align: center; font-family: 'Outfit', sans-serif;">AI RAG Chatbot</h2>
        <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 2px; margin-bottom: 8px; text-align: center;">Your intelligent document assistant</p>
        <div style="display: flex; gap: 5px; justify-content: center; margin-bottom: 3px;">
            <span style="background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.3); color: #c7d2fe; font-size: 0.65rem; font-weight: 600; padding: 2px 6px; border-radius: 20px;">LLaMA</span>
            <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); color: #a7f3d0; font-size: 0.65rem; font-weight: 600; padding: 2px 6px; border-radius: 20px;">Groq</span>
            <span style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); color: #fde68a; font-size: 0.65rem; font-weight: 600; padding: 2px 6px; border-radius: 20px;">ChromaDB</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.auth_page == "login":
        with st.form("login_form"):
            st.markdown("<h3 style='text-align: center; color: #ffffff; margin-bottom: 10px; font-weight: 600;'>Sign In</h3>", unsafe_allow_html=True)
            username = st.text_input("Username", key="login_username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
            
            submit_btn = st.form_submit_button("Sign In", type="primary")
            
            if submit_btn:
                success, msg = authenticate_user(username, password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.logout_triggered = False
                    st.query_params["user"] = username
                    try:
                        # Set persistent cookie with 30-day expiration (in seconds)
                        controller.set('auth_username', username, max_age=2592000)
                    except Exception:
                        pass
                    st.success("🎉 Welcome back!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")
        
        st.markdown("<div style='height: 2px;'></div>", unsafe_allow_html=True)
        col_l, col_r = st.columns([1.6, 1])
        with col_l:
            st.markdown("<p style='margin-top: 4px; color: #94a3b8; font-size: 0.85rem;'>Don't have an account?</p>", unsafe_allow_html=True)
        with col_r:
            if st.button("Create Account", key="go_to_register", type="secondary", use_container_width=True):
                st.session_state.auth_page = "register"
                st.rerun()
    else:
        with st.form("register_form"):
            st.markdown("<h3 style='text-align: center; color: #ffffff; margin-bottom: 10px; font-weight: 600;'>Create Account</h3>", unsafe_allow_html=True)
            username = st.text_input("Username", key="reg_username", placeholder="Choose a username")
            password = st.text_input("Password", type="password", key="reg_password", placeholder="Choose a password")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm_password", placeholder="Confirm your password")
            
            submit_btn = st.form_submit_button("Register", type="primary")
            
            if submit_btn:
                if not username.strip() or not password:
                    st.error("❌ Username and password cannot be empty.")
                elif password != confirm_password:
                    st.error("❌ Passwords do not match.")
                else:
                    success, msg = register_user(username, password)
                    if success:
                        st.success("🎉 Registration successful! You can now sign in.")
                        st.session_state.auth_page = "login"
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")
                        
        st.markdown("<div style='height: 2px;'></div>", unsafe_allow_html=True)
        col_l, col_r = st.columns([1.8, 1])
        with col_l:
            st.markdown("<p style='margin-top: 4px; color: #94a3b8; font-size: 0.85rem;'>Already have an account?</p>", unsafe_allow_html=True)
        with col_r:
            if st.button("Sign In", key="go_to_login", type="secondary", use_container_width=True):
                st.session_state.auth_page = "login"
                st.rerun()
    st.stop()

# Create the top wide gradient banner card matching the screenshot
st.markdown("""
<div class="top-banner-card">
    <div class="top-banner-title"><img src="https://em-content.zobj.net/source/apple/391/robot_1f916.png" class="top-banner-icon"> AI RAG Chatbot</div>
    <p class="top-banner-subtitle">Ask questions about your documents • Powered by Groq</p>
</div>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "all_history" not in st.session_state:
    st.session_state.all_history = []

# Build the custom sidebar matching the screenshot
with st.sidebar:
    st.markdown("### <img src='https://em-content.zobj.net/source/apple/391/robot_1f916.png' width='30' style='vertical-align: middle; margin-right: 8px; margin-bottom: 4px;'> AI RAG Chatbot\n**Powered by Groq**", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # User Profile Block & Logout
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); padding: 12px 16px; border-radius: 12px; margin-bottom: 15px;">
        <div style="width: 38px; height: 38px; border-radius: 50%; background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); display: flex; align-items: center; justify-content: center; font-weight: 700; color: white; font-size: 1.1rem; font-family: 'Outfit', sans-serif;">
            {st.session_state.username[0].upper() if st.session_state.username else 'U'}
        </div>
        <div style="flex-grow: 1; min-width: 0;">
            <div style="font-size: 0.75rem; color: #94a3b8; font-family: 'Outfit', sans-serif;">Logged in as</div>
            <div style="font-weight: 600; color: #f1f5f9; font-size: 0.9rem; font-family: 'Outfit', sans-serif; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{st.session_state.username}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.button("🚪 Log Out", key="logout_btn", use_container_width=True, on_click=logout)
        
    st.markdown("<hr style='margin: 15px 0; border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
    
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
        st.session_state.all_history = []
        try:
            chroma = chromadb.PersistentClient(path=str(CHROMA_PATH), settings=Settings(anonymized_telemetry=False))
            chroma.delete_collection(COLLECTION_NAME)
            load_collection.clear()
        except Exception:
            pass
        st.rerun()

    st.markdown("<hr style='margin: 15px 0; border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
    st.markdown("### 🕒 Chat History")
    if st.session_state.all_history:
        for i, entry in enumerate(reversed(st.session_state.all_history)):
            title = entry['question'][:40] + ("..." if len(entry['question']) > 40 else "")
            original_idx = len(st.session_state.all_history) - 1 - i
            if st.button(f"💬 {title}", key=f"hist_{original_idx}", type="secondary", use_container_width=True):
                st.session_state.history = [st.session_state.all_history[original_idx]]
                st.rerun()
    else:
        st.markdown("<div style='color: #94a3b8; font-size: 0.9rem;'>No chat history yet.</div>", unsafe_allow_html=True)

if collection is None or collection.count() == 0:
    st.info("👋 **Welcome!** Your database is currently empty.\n\nPlease upload documents using the sidebar on the left and click **'Process Documents'** to begin.")

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
        chunks = []
        context = "No documents provided yet."
        if collection is not None and collection.count() > 0:
            n_res = min(TOP_K, collection.count())
            results = collection.query(query_texts=[question], n_results=n_res, include=["documents", "metadatas", "distances"])
            chunks = [{"text": d, "source": m["source"], "page": m["page"], "distance": round(dist, 4)}
                      for d, m, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])]
            context = "\n\n".join(f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}" for c in chunks)
            
        try:
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
            st.session_state.all_history.append({"question": question, "answer": answer, "chunks": chunks})
            st.rerun()
        except Exception as e:
            if "AuthenticationError" in str(type(e)) or "401" in str(e):
                st.error("🔑 **Groq Authentication Error:** The API key provided is invalid, expired, or revoked. Please verify the `GROQ_API_KEY` configured in your Streamlit Secrets.")
            else:
                st.error(f"❌ **API Error:** {str(e)}")
