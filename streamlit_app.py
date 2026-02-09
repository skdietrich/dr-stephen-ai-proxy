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
# ASSET HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
LINKEDIN_URL = "https://www.linkedin.com/in/stephendietrich-kolokouris/"

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
.main .block-container {
  padding-top: 0.8rem !important;
  padding-bottom: 2rem !important;
  max-width: 960px;
}

section[data-testid="stSidebar"] {
  background: #090D16 !important;
  border-right: 1px solid var(--border);
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

[data-testid="stChatMessage"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--bg-card) !important;
  margin-bottom: 6px !important;
}

.stButton > button {
  font-family: var(--font) !important;
  font-size: 0.82rem !important;
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
# LLM + EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════════
def init_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o", temperature=0, api_key=st.secrets["OPENAI_API_KEY"])

def init_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])


# ═══════════════════════════════════════════════════════════════════════════════
# FAISS LOGIC
# ═══════════════════════════════════════════════════════════════════════════════
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
    return vs.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 30, "lambda_mult": 0.55})

retriever = init_retriever()


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT & CONTEXT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def format_evidence_pack(docs) -> Tuple[str, List[str]]:
    parts = []
    labels = []
    for d in docs:
        src = os.path.basename(d.metadata.get("source", "unknown"))
        parts.append(f"[SOURCE: {src}]\n{d.page_content}")
        labels.append(src)
    return "\n\n".join(parts), labels

_EMPTY_RECRUITER_STATE = {"target_roles": [], "domains": [], "location": None}

def extract_recruiter_constraints(llm: ChatOpenAI, user_message: str) -> dict:
    prompt = f"Extract recruiter constraints as JSON matching this schema: {json.dumps(_EMPTY_RECRUITER_STATE)}\n\nMSG: {user_message}"
    try:
        out = llm.invoke(prompt)
        return json.loads(re.sub(r"^```json\s*|\s*```$", "", out.content.strip()))
    except: return {}

def rewrite_to_standalone(llm, chat_history, user_input, state):
    prompt = f"Rewrite '{user_input}' into a standalone search query for Stephen DK. History: {chat_history[-5:]}. Context: {state}"
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
    c.drawString(72, y, title.replace("\r", ""))
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
# SESSION STATE & SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.recruiter_state = _EMPTY_RECRUITER_STATE.copy()

with st.sidebar:
    st.markdown("### Professional Links")
    st.link_button("🔗  LinkedIn Profile", LINKEDIN_URL, use_container_width=True)
    st.markdown("---")
    st.markdown("### Export Transcript")
    if REPORTLAB_OK and st.session_state.messages:
        pdf_data = build_qa_pdf_bytes("Q&A — Dr. Stephen Dietrich-Kolokouris", st.session_state.messages)
        st.download_button("Download PDF", data=pdf_data, file_name="Stephen_DK_QA.pdf", mime="application/pdf", use_container_width=True)
    else: st.caption("Start a chat to enable PDF export.")


# ═══════════════════════════════════════════════════════════════════════════════
# CORE CHAT TURN
# ═══════════════════════════════════════════════════════════════════════════════
def run_turn(user_text: str, action_mode: str = "chat") -> str:
    llm = init_llm()
    new_bits = extract_recruiter_constraints(llm, user_text)
    st.session_state.recruiter_state.update(new_bits)

    standalone = rewrite_to_standalone(llm, st.session_state.messages, user_text, st.session_state.recruiter_state)
    docs = retriever.invoke(standalone)
    evidence, labels = format_evidence_pack(docs)

    prompt = f"You are Stephen's proxy. Answer strictly using these docs: {evidence}\nUser context: {st.session_state.recruiter_state}"
    try:
        out = llm.invoke([{"role": "system", "content": prompt}, {"role": "user", "content": user_text}])
        answer = enforce_no_external_refs(out.content)
        if labels:
            answer += f'\n\n<div class="src-line">Sources: {", ".join(list(set(labels)))}</div>'
        return answer
    except Exception as e: return f"Error: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hdr">
  <p class="hdr-name">Dr. Stephen Dietrich-Kolokouris, PhD</p>
  <p class="hdr-role">Cybersecurity · AI/ML Systems · Data Engineering · Strategic Analysis</p>
</div>
""", unsafe_allow_html=True)

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"], unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)
with col_a: do_verify = st.button("Check sources", use_container_width=True)
with col_b: do_fit = st.button("Summarize fit", use_container_width=True)
with col_c: do_outreach = st.button("Draft message", use_container_width=True)

if do_verify:
    with st.chat_message("assistant"):
        a = run_turn("Verify the previous answer.", action_mode="verify")
        st.markdown(a, unsafe_allow_html=True); st.session_state.messages.append({"role": "assistant", "content": a})
if do_fit:
    with st.chat_message("assistant"):
        a = run_turn("Summarize fit.", action_mode="fit")
        st.markdown(a, unsafe_allow_html=True); st.session_state.messages.append({"role": "assistant", "content": a})
if do_outreach:
    with st.chat_message("assistant"):
        a = run_turn("Draft outreach.", action_mode="outreach")
        st.markdown(a, unsafe_allow_html=True); st.session_state.messages.append({"role": "assistant", "content": a})

user_input = st.chat_input("Ask about skills, experience, projects...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)
    with st.chat_message("assistant"):
        a = run_turn(user_input)
        st.markdown(a, unsafe_allow_html=True); st.session_state.messages.append({"role": "assistant", "content": a})
