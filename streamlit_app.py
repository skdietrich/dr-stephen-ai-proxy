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
# GUARDRAILS & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
_EXTERNAL_REF_REGEX = re.compile(
    r"(^\s*works\s+cited\b|^\s*references\s*$|^\s*bibliography\b|https?://|www\.)",
    flags=re.IGNORECASE | re.MULTILINE,
)

def enforce_no_external_refs(text: str) -> str:
    if not text:
        return text
    if _EXTERNAL_REF_REGEX.search(text):
        return "I can only answer based on documentation I have on file."
    return text

def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG + STYLES
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Dr. Stephen Dietrich-Kolokouris, PhD",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
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
  --font:       'DM Sans', sans-serif;
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
.hdr-name { font-size: 1.45rem; font-weight: 700; color: #fff; margin: 0; }
.hdr-role { font-size: 0.88rem; color: var(--accent); font-weight: 500; margin: 2px 0 10px; }
.hdr-bio { font-size: 0.86rem; color: var(--txt-dim); line-height: 1.55; max-width: 640px; }

[data-testid="stChatMessage"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--bg-card) !important;
}

.stButton > button { font-size: 0.82rem !important; border-color: var(--border-lit) !important; color: var(--txt-dim) !important; }
.stButton > button[kind="primary"], .stDownloadButton > button {
  background: var(--accent) !important; color: #0B0F19 !important; font-weight: 600 !important; border: none !important;
}

.src-line { font-size: 0.74rem; color: var(--txt-faint); margin-top: 6px; border-top: 1px solid var(--border); font-style: italic; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LLM & VECTOR STORE ENGINE
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
    loader = PyPDFDirectoryLoader("data/")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=160)
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(chunks, init_embeddings())
    return vs.as_retriever(search_type="mmr", search_kwargs={"k": 8})

retriever = init_retriever()

# ═══════════════════════════════════════════════════════════════════════════════
# PDF EXPORT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def _sanitize_rl(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text.replace("\r", ""))

def build_qa_pdf_bytes(title: str, messages: List[Dict]) -> bytes:
    from io import BytesIO
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    y = 750
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, y, _sanitize_rl(title))
    y -= 40
    c.setFont("Helvetica", 10)
    for m in messages:
        role = m["role"].upper()
        content = m["content"]
        lines = [content[i:i+95] for i in range(0, len(content), 95)]
        for line in lines:
            if y < 50: c.showPage(); y = 750
            c.drawString(72, y, f"{role}: {_sanitize_rl(line)}")
            y -= 15
        y -= 10
    c.save()
    return buf.getvalue()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []

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

st.markdown("""
<div class="hdr">
  <p class="hdr-name">Dr. Stephen Dietrich-Kolokouris, PhD</p>
  <p class="hdr-role">Cybersecurity · AI/ML Systems · Data Engineering · Strategic Analysis</p>
  <p class="hdr-bio">Answers are based on published work and professional documentation.</p>
</div>
""", unsafe_allow_html=True)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Ask about skills, experience, or role fit…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        docs = retriever.invoke(user_input)
        context = "\n\n".join([d.page_content for d in docs])
        response = init_llm().invoke([
            {"role": "system", "content": f"Answer using documentation strictly.\n\n{context}"},
            {"role": "user", "content": user_input}
        ])
        answer = enforce_no_external_refs(response.content)
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
