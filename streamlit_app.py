# Dr. Stephen Dietrich-Kolokouris -- Portfolio RAG Interface
# Enhanced production build: premium UI + full backend + professional improvements.
#
# Self-healing: if a phone editor corrupted quotes, fix and restart.
import sys as _sys, os as _os, pathlib as _pl
_me = _pl.Path(__file__)
_raw = _me.read_bytes()
_lq = chr(8220).encode('utf-8')
_rq = chr(8221).encode('utf-8')
_la = chr(8216).encode('utf-8')
_ra = chr(8217).encode('utf-8')
_em = chr(8212).encode('utf-8')
_en = chr(8211).encode('utf-8')
if _lq in _raw or _rq in _raw or _la in _raw or _ra in _raw:
    _raw = _raw.replace(_lq, b'\x22').replace(_rq, b'\x22')
    _raw = _raw.replace(_la, b'\x27').replace(_ra, b'\x27')
    _raw = _raw.replace(_em, b'\x2d\x2d').replace(_en, b'\x2d')
    _me.write_bytes(_raw)
    _os.execv(_sys.executable, [_sys.executable] + _sys.argv)

import hashlib
import json
import logging
import os
import re
import tempfile
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

try:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & VIEWPORT CONTROL
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Dr. Stephen Dietrich-Kolokouris | Cybersecurity & AI Expert",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Force scroll to top on every load/rerun
components.html(
    """
    <script>
        const mainContent = window.parent.document.querySelector('.main');
        if (mainContent) {
            mainContent.scrollTo({ top: 0, left: 0, behavior: 'auto' });
        }
    </script>
    """,
    height=0,
)

# ═══════════════════════════════════════════════════════════════════════════════
# GUARDRAILS & ASSETS
# ═══════════════════════════════════════════════════════════════════════════════
_EXTERNAL_REF_REGEX = re.compile(
    r"(^\s*works\s+cited\b|^\s*references\s*$|^\s*bibliography\b|https?://|www\.)",
    flags=re.IGNORECASE | re.MULTILINE,
)

def enforce_no_external_refs(text: str) -> str:
    if not text: return text
    if _EXTERNAL_REF_REGEX.search(text):
        return "Response blocked: external citation detected. This system only cites the loaded corpus."
    return text

def safe_exists(path: Optional[str]) -> bool:
    return bool(path) and os.path.exists(path)

def first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if safe_exists(p): return p
    return None

HEADSHOT_PATH = first_existing(["assets/headshot.png", "assets/headshot.jpg", "headshot.jpg"])
LINKEDIN_URL = "https://www.linkedin.com/in/stephendietrich-kolokouris/"

# ═══════════════════════════════════════════════════════════════════════════════
# LLM & RAG BACKEND
# ═══════════════════════════════════════════════════════════════════════════════

def init_llm(model="gpt-4o"):
    return ChatOpenAI(model=model, temperature=0, api_key=st.secrets["OPENAI_API_KEY"])

def init_embeddings():
    return OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])

FAISS_DIR = "faiss_index"

def load_or_build_faiss():
    if not os.path.exists("data"):
        st.error("Missing /data directory for PDFs."); st.stop()
    
    embeddings = init_embeddings()
    if os.path.isdir(FAISS_DIR):
        return FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    
    docs = []
    for root, _, fnames in os.walk("data"):
        for fn in sorted(fnames):
            if fn.lower().endswith(".pdf"):
                docs.extend(PyPDFLoader(os.path.join(root, fn)).load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=160)
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(FAISS_DIR)
    return vs

@st.cache_resource
def get_retriever():
    vs = load_or_build_faiss()
    return vs.as_retriever(search_type="mmr", search_kwargs={"k": 10})

# ═══════════════════════════════════════════════════════════════════════════════
# CORE LOGIC: run_turn()
# ═══════════════════════════════════════════════════════════════════════════════
def run_turn(user_text: str, action_mode: str = "chat") -> str:
    llm = init_llm()
    retriever = get_retriever()
    docs = retriever.invoke(user_text)
    
    evidence = "\n\n".join([f"[SOURCE: {os.path.basename(d.metadata['source'])}]\n{d.page_content}" for d in docs])
    
    sys_prompt = f"""You represent Dr. Stephen Dietrich-Kolokouris. 
    Use ONLY this evidence: {evidence}.
    If unsure, acknowledge and invite specific questions. No invented facts. No external URLs."""
    
    try:
        placeholder = st.empty()
        full_response = ""
        for chunk in llm.stream([{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_text}]):
            full_response += (chunk.content or "")
            placeholder.markdown(full_response + " ▋")
        placeholder.markdown(full_response)
        return enforce_no_external_refs(full_response)
    except Exception as e:
        return f"Processing error: {e}"

# ═══════════════════════════════════════════════════════════════════════════════
# UI: CSS & STYLING
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;700&display=swap');
    :root { --accent: #1a5c3a; --surface: #f8f7f4; --text: #1a1d23; }
    html, body, [data-testid="stAppViewContainer"] { background-color: var(--surface) !important; font-family: 'DM Sans', sans-serif; color: var(--text); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d0f12 0%, #1a1f2e 100%) !important; }
    .main-header { max-width: 1000px; margin: 0 auto; padding: 3rem 1rem; }
    .main-greeting { font-family: 'DM Serif Display', serif; font-size: 2.6rem; line-height: 1.2; }
    .domain-card { background: white; border: 1px solid #e2e0db; border-radius: 12px; padding: 1.5rem; transition: 0.3s; }
    .domain-card:hover { border-color: var(--accent); transform: translateY(-3px); box-shadow: 0 8px 32px rgba(0,0,0,0.06); }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR & MAIN UI
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    if safe_exists(HEADSHOT_PATH):
        st.image(HEADSHOT_PATH)
    st.markdown(f"### Dr. Stephen Dietrich-Kolokouris")
    st.markdown("Cybersecurity Architect | AI Engineer")
    st.markdown("---")
    st.session_state.personal_mode = st.toggle("Conversational Mode", value=True)

st.markdown("""
<div class="main-header">
    <div style="color:var(--accent); font-weight:600; letter-spacing:1px; margin-bottom:1rem;">◆ AI-POWERED PORTFOLIO</div>
    <div class="main-greeting">Expertise Alignment Interface</div>
    <div style="font-size:1.1rem; color:#5a5f6b; margin-top:1rem; max-width:750px;">
        Explore project history, certifications, and research through this RAG-enabled assistant.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Chat Logic ──
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I assist with your review of Stephen's credentials today?"}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Ask about technical expertise or project history...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        response = run_turn(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
