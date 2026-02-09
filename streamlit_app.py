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

from langchain_community.document_loaders import PyPDFLoader
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
# GUARDRAILS
# ═══════════════════════════════════════════════════════════════════════════════
_EXTERNAL_REF_REGEX = re.compile(
    r"(^\s*works\s+cited\b|^\s*references\s*$|^\s*bibliography\b|https?://|www\.)",
    flags=re.IGNORECASE | re.MULTILINE,
)

def enforce_no_external_refs(text: str) -> str:
    if not text:
        return text
    if _EXTERNAL_REF_REGEX.search(text):
        return (
            "I can only answer based on documentation I have on file. "
            "Could you rephrase your question?"
        )
    return text

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG + STYLES
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Dr. Stephen Dietrich-Kolokouris, PhD",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0B0F19; --bg-surface: #101624; --bg-card: #151C2E;
  --border: rgba(148,163,184,0.10); --border-lit: rgba(148,163,184,0.18);
  --txt: #D8DEE9; --txt-dim: #7B8599; --accent: #C9A84C;
  --font: 'DM Sans', sans-serif; --radius: 8px;
}
html, body, [class*="stApp"] { background: var(--bg) !important; color: var(--txt) !important; font-family: var(--font) !important; }
.hdr { border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 12px; }
.hdr-name { font-size: 1.45rem; font-weight: 700; color: #fff; margin: 0; }
.hdr-role { font-size: 0.88rem; color: var(--accent); font-weight: 500; margin: 2px 0 10px; }
[data-testid="stChatMessage"] { border: 1px solid var(--border) !important; border-radius: var(--radius) !important; background: var(--bg-card) !important; margin-bottom: 6px !important; }
.stButton > button { font-family: var(--font) !important; font-size: 0.82rem !important; border-color: var(--border-lit) !important; color: var(--txt-dim) !important; }
.stButton > button[kind="primary"], .stDownloadButton > button { background: var(--accent) !important; color: #0B0F19 !important; font-weight: 600 !important; border: none !important; }
.src-line { font-size: 0.74rem; color: var(--txt-faint); margin-top: 6px; border-top: 1px solid var(--border); font-style: italic; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ROBUST FAISS LOADING
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def init_retriever():
    data_dir = "data"
    if not os.path.exists(data_dir):
        st.error("Missing `/data` directory.")
        st.stop()
    
    all_docs = []
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    
    with st.status("Initializing Knowledge Base...", expanded=False) as status:
        for filename in pdf_files:
            try:
                loader = PyPDFLoader(os.path.join(data_dir, filename))
                all_docs.extend(loader.load())
            except Exception as e:
                st.warning(f"Skipping {filename}: Truncated or corrupted.")
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=160)
        chunks = splitter.split_documents(all_docs)
        embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
        vs = FAISS.from_documents(chunks, embeddings)
        status.update(label="Index Ready", state="complete")
    
    return vs.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 30, "lambda_mult": 0.55})

# ═══════════════════════════════════════════════════════════════════════════════
# RECRUITER & QUERY LOGIC
# ═══════════════════════════════════════════════════════════════════════════════
_EMPTY_STATE = {"target_roles": [], "domains": [], "location": None, "must_haves": []}

def extract_recruiter_constraints(llm, message):
    prompt = f"Extract recruiter constraints as JSON matching: {json.dumps(_EMPTY_STATE)}\n\nMSG: {message}\nJSON:"
    try:
        out = llm.invoke(prompt)
        text = re.sub(r"^```json\s*|\s*```$", "", out.content.strip())
        return json.loads(text)
    except: return {}

def rewrite_to_standalone(llm, history, user_input, state):
    prompt = f"Context: {json.dumps(state)}\nHistory: {history}\nRewrite '{user_input}' as a standalone search query for Stephen DK."
    try: return llm.invoke(prompt).content.strip()
    except: return user_input

# ═══════════════════════════════════════════════════════════════════════════════
# PDF EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
def build_qa_pdf_bytes(title, messages):
    from io import BytesIO
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    y = 750
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, y, title)
    y -= 40
    c.setFont("Helvetica", 10)
    for m in messages:
        role, content = m["role"].upper(), m["content"].replace("\r", "")
        text = f"{role}: {content}"
        lines = [text[i:i+95] for i in range(0, len(text), 95)]
        for line in lines:
            if y < 50: c.showPage(); y = 750
            c.drawString(72, y, line); y -= 15
        y -= 10
    c.save()
    return buf.getvalue()

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION INITIALIZATION & SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.recruiter_state = _EMPTY_STATE.copy()

with st.sidebar:
    st.markdown("### Professional Links")
    st.link_button("🔗 LinkedIn Profile", "https://www.linkedin.com/in/stephendietrich-kolokouris/", use_container_width=True)
    st.markdown("---")
    st.markdown("### Export Session")
    if REPORTLAB_OK and st.session_state.messages:
        pdf_data = build_qa_pdf_bytes("Q&A — Dr. Stephen Dietrich-Kolokouris", st.session_state.messages)
        st.download_button("Download PDF Copy", data=pdf_data, file_name="Stephen_DK_QA.pdf", mime="application/pdf", use_container_width=True)
    else: st.caption("Complete a turn to enable PDF export.")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hdr">
  <p class="hdr-name">Dr. Stephen Dietrich-Kolokouris, PhD</p>
  <p class="hdr-role">Cybersecurity · AI/ML Systems · Data Engineering · Strategic Analysis</p>
</div>
""", unsafe_allow_html=True)

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

user_input = st.chat_input("Ask about Stephen's background...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    with st.chat_message("assistant"):
        llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=st.secrets["OPENAI_API_KEY"])
        retriever = init_retriever()
        
        # Background logic
        new_bits = extract_recruiter_constraints(llm, user_input)
        st.session_state.recruiter_state.update(new_bits)
        
        standalone = rewrite_to_standalone(llm, st.session_state.messages[-5:], user_input, st.session_state.recruiter_state)
        docs = retriever.invoke(standalone)
        context = "\n\n".join([d.page_content for d in docs])
        
        response = llm.invoke([
            {"role": "system", "content": f"Answer as Stephen's proxy using docs only. Context: {st.session_state.recruiter_state}\n\nDocs: {context}"},
            {"role": "user", "content": user_input}
        ])
        
        answer = enforce_no_external_refs(response.content)
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
