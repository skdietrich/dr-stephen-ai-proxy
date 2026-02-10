# Dr. Stephen Dietrich-Kolokouris -- Portfolio RAG Interface
# Production build: premium UI + full backend + viewport control.

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
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & VIEWPORT
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Dr. Stephen Dietrich-Kolokouris | Cybersecurity & AI Expert",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Return to top of page on every rerun
components.html(
    """
    <script>
        window.parent.document.querySelector('.main').scrollTo({top: 0, left: 0, behavior: 'auto'});
    </script>
    """,
    height=0,
)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
if "recruiter_state" not in st.session_state:
    st.session_state.recruiter_state = {"target_roles": [], "must_haves": []}

# ═══════════════════════════════════════════════════════════════════════════════
# RAG BACKEND
# ═══════════════════════════════════════════════════════════════════════════════
def init_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0, api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_resource
def get_retriever():
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    if os.path.isdir("faiss_index"):
        vs = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        docs = []
        for f in os.listdir("data"):
            if f.endswith(".pdf"):
                docs.extend(PyPDFLoader(f"data/{f}").load())
        splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=160)
        vs = FAISS.from_documents(splitter.split_documents(docs), embeddings)
        vs.save_local("faiss_index")
    return vs.as_retriever(search_type="mmr", search_kwargs={"k": 10})

def run_turn(query: str, action_mode: str = "chat"):
    llm = init_llm()
    retriever = get_retriever()
    docs = retriever.invoke(query)
    
    context = "\n\n".join([d.page_content for d in docs])
    sys_prompt = f"You are an assistant for Dr. Stephen Dietrich-Kolokouris. Mode: {action_mode}. Context: {context}"
    
    placeholder = st.empty()
    full_res = ""
    for chunk in llm.stream([{"role": "system", "content": sys_prompt}, {"role": "user", "content": query}]):
        full_res += (chunk.content or "")
        placeholder.markdown(full_res + " ▋")
    placeholder.markdown(full_res)
    return full_res

# ═══════════════════════════════════════════════════════════════════════════════
# UI STYLING
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] { background-color: #f8f7f4 !important; font-family: 'DM Sans', sans-serif; }
    .main-header { max-width: 1000px; margin: 0 auto; padding: 3rem 1rem 1rem; }
    .main-greeting { font-family: 'DM Serif Display', serif; font-size: 2.6rem; color: #1a1d23; }
    .stButton > button { border-radius: 8px; font-weight: 600; transition: 0.3s; }
    .stButton > button:hover { border-color: #1a5c3a; color: #1a5c3a; background: #e8f0ec; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-header"><div class="main-greeting">Stephen\'s Expertise Alignment</div></div>', unsafe_allow_html=True)

# ── Action Buttons ──
col_a, col_b = st.columns(2)
with col_a:
    if st.button("⊕ Analyze Role Fit", use_container_width=True):
        with st.chat_message("assistant"):
            res = run_turn("Analyze how Stephen fits the roles discussed.", "fit")
            st.session_state.messages.append({"role": "assistant", "content": res})
with col_b:
    if st.button("✎ Draft Outreach Message", use_container_width=True):
        with st.chat_message("assistant"):
            res = run_turn("Draft a professional outreach email.", "outreach")
            st.session_state.messages.append({"role": "assistant", "content": res})

st.markdown("---")

# ── Render History ──
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ── Starter Chips (Visible only at start) ──
if not any(m["role"] == "user" for m in st.session_state.messages):
    st.markdown("### Suggested Topics")
    chip_cols = st.columns(4)
    topics = ["Security Architecture", "AI & RAG Systems", "Intelligence Background", "Research & Books"]
    for i, topic in enumerate(topics):
        if chip_cols[i].button(topic, key=f"chip_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": topic})
            with st.chat_message("user"): st.markdown(topic)
            with st.chat_message("assistant"):
                res = run_turn(topic)
                st.session_state.messages.append({"role": "assistant", "content": res})
            st.rerun()

# ── Chat Input ──
user_input = st.chat_input("Ask about Stephen's experience...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)
    with st.chat_message("assistant"):
        res = run_turn(user_input)
        st.session_state.messages.append({"role": "assistant", "content": res})
