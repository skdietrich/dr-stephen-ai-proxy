# streamlit_app.py — Professional Q&A Proxy for Dr. Stephen Dietrich-Kolokouris
# ═══════════════════════════════════════════════════════════════════════════════
# TARGET: GitHub → Streamlit Community Cloud (free tier)
# DESIGN: Clean professional tool. No AI demo aesthetics.
# ═══════════════════════════════════════════════════════════════════════════════

import os
import re
import json
import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import pandas as pd
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

try:
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    from langchain.prompts import ChatPromptTemplate

try:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG + STYLES
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Dr. Stephen Dietrich-Kolokouris, PhD",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap" rel="stylesheet">
<style>
:root {
  --bg:         #0B0F19;
  --bg-surface: #101624;
  --bg-card:    #151C2E;
  --border:     rgba(148,163,184,0.10);
  --border-lit: rgba(148,163,184,0.18);
  --txt:        #D8DEE9;
  --txt-dim:    #7B8599;
  --txt-faint:  #4E576B;
  --accent:     #C9A84C;
  --accent-dim: rgba(201,168,76,0.12);
  --font:       'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  --radius:     8px;
}

html, body, [class*="stApp"] {
  background: var(--bg) !important;
  color: var(--txt) !important;
  font-family: var(--font) !important;
}

.hdr {
  border-bottom: 1px solid var(--border);
  padding-bottom: 16px;
  margin-bottom: 12px;
}
.hdr-name {
  font-size: 1.45rem; font-weight: 700; color: #fff;
  margin: 0; line-height: 1.3;
}
.hdr-role {
  font-size: 0.88rem; color: var(--accent);
  font-weight: 500; letter-spacing: 0.03em;
  margin: 2px 0 10px;
}
.hdr-bio {
  font-size: 0.86rem; color: var(--txt-dim);
  line-height: 1.55; max-width: 640px;
}

[data-testid="stChatMessage"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--bg-card) !important;
  margin-bottom: 6px !important;
}

.stButton > button {
  font-family: var(--font) !important;
  font-size: 0.82rem !important;
  border-color: var(--border-lit) !important;
  color: var(--txt-dim) !important;
}

.stButton > button[kind="primary"],
.stDownloadButton > button {
  background: var(--accent) !important;
  color: #0B0F19 !important;
  font-weight: 600 !important;
  border: none !important;
}

.src-line {
  font-size: 0.74rem;
  color: var(--txt-faint);
  margin-top: 6px;
  padding-top: 6px;
  border-top: 1px solid var(--border);
  font-style: italic;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS & LLM ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def init_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o", temperature=0, api_key=st.secrets["OPENAI_API_KEY"])

def init_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_resource
def init_retriever():
    if not os.path.exists("data"):
        st.error("Missing `/data` directory.")
        st.stop()
    embeddings = init_embeddings()
    loader = PyPDFDirectoryLoader("data/")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=160)
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(chunks, embeddings)
    return vs.as_retriever(search_type="mmr", search_kwargs={"k": 8})

retriever = init_retriever()

# ═══════════════════════════════════════════════════════════════════════════════
# RECRUITER CONTEXT & UTILS
# ═══════════════════════════════════════════════════════════════════════════════
_EMPTY_RECRUITER_STATE = {
    "target_roles": [], "domains": [], "location": None,
    "onsite_remote": None, "must_haves": [], "nice_to_haves": [],
    "dealbreakers": [],
}

def _sanitize_for_reportlab(text: str) -> str:
    if not text: return text
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text.replace("\r", ""))

def build_qa_pdf_bytes(title: str, messages: List[Dict]) -> bytes:
    from io import BytesIO
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER
    y = height - 60
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, y, _sanitize_for_reportlab(title))
    y -= 40
    c.setFont("Helvetica", 10)
    for m in messages:
        role = m["role"].upper()
        content = m["content"]
        text = f"{role}: {content}"
        lines = [text[i:i+95] for i in range(0, len(text), 95)]
        for line in lines:
            if y < 50:
                c.showPage(); y = height - 60
            c.drawString(72, y, _sanitize_for_reportlab(line))
            y -= 15
        y -= 10
    c.save()
    return buf.getvalue()

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.recruiter_state = _EMPTY_RECRUITER_STATE.copy()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Professional Links")
    st.link_button("🔗  LinkedIn Profile", "https://www.linkedin.com/in/stephendietrich-kolokouris/", use_container_width=True)
    st.markdown("---")
    st.markdown("### Export Session")
    if REPORTLAB_OK and st.session_state.messages:
        pdf_data = build_qa_pdf_bytes("Q&A — Dr. Stephen Dietrich-Kolokouris", st.session_state.messages)
        st.download_button("Download PDF Transcript", data=pdf_data, file_name="QA_Stephen_DK.pdf", mime="application/pdf", use_container_width=True)
    else:
        st.caption("Complete a turn to enable PDF export.")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hdr">
  <p class="hdr-name">Dr. Stephen Dietrich-Kolokouris, PhD</p>
  <p class="hdr-role">Cybersecurity · AI/ML Systems · Data Engineering · Strategic Analysis</p>
  <p class="hdr-bio">
    Ask about Stephen's background, technical capabilities, or project experience. 
    Answers are based on his published work and professional documentation.
  </p>
</div>
""", unsafe_allow_html=True)

# Chat History
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# User Input
user_input = st.chat_input("Ask about skills, experience, or role fit…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        llm = init_llm()
        docs = retriever.invoke(user_input)
        context = "\n\n".join([d.page_content for d in docs])
        
        system_prompt = (
            "You are a professional representative for Dr. Stephen Dietrich-Kolokouris. "
            "Use the provided documentation strictly. If facts are missing, say so.\n\n"
            f"DOCUMENTATION:\n{context}"
        )
        
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ])
        
        st.markdown(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})
    
    st.rerun()
