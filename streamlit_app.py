# Dr. Stephen Dietrich-Kolokouris -- Portfolio RAG Interface
# Final Production Build: Premium Layout + Full RAG Backend + Viewport Fix

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
# PAGE CONFIG & VIEWPORT CONTROL
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Dr. Stephen Dietrich-Kolokouris | Cybersecurity & AI Expert",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Critical: This ensures that on every interaction/load, the user is reset to the top
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
# ASSET HELPERS & UTILS
# ═══════════════════════════════════════════════════════════════════════════════
def safe_exists(path: str) -> bool:
    return os.path.exists(path)

HEADSHOT_PATH = "assets/headshot.jpg" if os.path.exists("assets/headshot.jpg") else None
LINKEDIN_URL = "https://www.linkedin.com/in/stephendietrich-kolokouris/"

# ═══════════════════════════════════════════════════════════════════════════════
# RAG BACKEND
# ═══════════════════════════════════════════════════════════════════════════════
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
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=st.secrets["OPENAI_API_KEY"])
    retriever = get_retriever()
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])
    sys_prompt = f"Represent Dr. Stephen Dietrich-Kolokouris using this evidence: {context}. Mode: {action_mode}"
    
    placeholder = st.empty()
    full_res = ""
    for chunk in llm.stream([{"role": "system", "content": sys_prompt}, {"role": "user", "content": query}]):
        full_res += (chunk.content or "")
        placeholder.markdown(full_res + " ▋")
    placeholder.markdown(full_res)
    return full_res

# ═══════════════════════════════════════════════════════════════════════════════
# PREMIUM CSS (The Original Layout)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;700&display=swap');

:root {
    --accent: #1a5c3a; --surface: #f8f7f4; --card: #ffffff; --text: #1a1d23;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--surface) !important; font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0f12 0%, #1a1f2e 100%) !important;
}

/* Sidebar Custom Elements */
.sidebar-photo {
    width: 150px; height: 150px; border-radius: 50%; margin: 2rem auto; 
    display: block; border: 4px solid rgba(74,222,128,0.3);
}
.sidebar-name { text-align: center; color: white; font-family: 'DM Serif Display'; font-size: 1.5rem; }
.cred-tag { background: rgba(74,222,128,0.1); color: #4ade80; padding: 4px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: 700; margin: 2px; display: inline-block; }

/* Main Header */
.main-header { max-width: 1000px; margin: 0 auto; padding: 4rem 1rem 2rem; }
.main-greeting { font-family: 'DM Serif Display'; font-size: 3rem; color: var(--text); line-height: 1.1; }

/* Domain Cards */
.domain-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; max-width: 1000px; margin: 2rem auto; }
.domain-card { background: var(--card); border: 1px solid #e2e0db; padding: 1.5rem; border-radius: 12px; transition: 0.3s ease; }
.domain-card:hover { transform: translateY(-5px); border-color: var(--accent); box-shadow: 0 10px 30px rgba(0,0,0,0.05); }
.domain-label { font-weight: 700; margin-top: 10px; font-size: 1.1rem; }
.domain-desc { font-size: 0.85rem; color: #5a5f6b; margin-top: 8px; }

#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR CONTENT
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    if HEADSHOT_PATH:
        st.markdown(f'<img src="data:image/jpeg;base64," class="sidebar-photo">', unsafe_allow_html=True)
    else:
        st.markdown('<div style="height:150px; width:150px; background:#333; border-radius:50%; margin:2rem auto;"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-name">Dr. Stephen Dietrich-Kolokouris</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; color:#888; font-size:0.8rem; margin-bottom:1rem;">Cybersecurity & AI Expert</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="text-align:center;"><span class="cred-tag">PhD</span><span class="cred-tag">CCIE</span><span class="cred-tag">Ex-CIA</span></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Core Expertise")
    st.markdown("- Security Architecture\n- RAG Systems\n- Intelligence Analysis")
    st.session_state.personal_mode = st.toggle("Conversational Mode", value=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <div style="color:var(--accent); font-weight:700; letter-spacing:1.5px; margin-bottom:0.5rem;">◆ PORTFOLIO ASSISTANT</div>
    <div class="main-greeting">Discover how Stephen's expertise aligns with your needs</div>
</div>
""", unsafe_allow_html=True)

# Domain Cards
st.markdown("""
<div class="domain-grid">
    <div class="domain-card">
        <div style="font-size:2rem;">🛡️</div>
        <div class="domain-label">Security Architecture</div>
        <div class="domain-desc">Designing resilient security frameworks and infrastructure.</div>
    </div>
    <div class="domain-card">
        <div style="font-size:2rem;">🤖</div>
        <div class="domain-label">AI & RAG Systems</div>
        <div class="domain-desc">Building production-grade retrieval-augmented generation.</div>
    </div>
    <div class="domain-card">
        <div style="font-size:2rem;">🕵️</div>
        <div class="domain-label">Intelligence Analysis</div>
        <div class="domain-desc">Strategic threat intelligence and defensive modeling.</div>
    </div>
    <div class="domain-card">
        <div style="font-size:2rem;">📚</div>
        <div class="domain-label">Research & Publishing</div>
        <div class="domain-desc">Author of 7 books on technology and history.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Action Buttons ──
st.markdown('<div style="max-width:1000px; margin:0 auto;">', unsafe_allow_html=True)
col_a, col_b = st.columns(2)
with col_a:
    if st.button("⊕ Analyze Role Fit", use_container_width=True):
        st.session_state.messages.append({"role": "assistant", "content": run_turn("Analyze fit.", "fit")})
with col_b:
    if st.button("✎ Draft Outreach Message", use_container_width=True):
        st.session_state.messages.append({"role": "assistant", "content": run_turn("Draft email.", "outreach")})
st.markdown('</div>', unsafe_allow_html=True)

# ── Chat Logic ──
if "messages" not in st.session_state:
    st.session_state.messages = []

# Layout wrapper for Chat
st.markdown('<div style="max-width:1000px; margin:2rem auto;">', unsafe_allow_html=True)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Starter Chips (Hidden after first message)
if not st.session_state.messages:
    st.markdown("### Quick Inquiries")
    c1, c2, c3 = st.columns(3)
    if c1.button("Security Background", use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": "Security Background"})
        st.session_state.messages.append({"role": "assistant", "content": run_turn("Security Background")})
        st.rerun()
    if c2.button("AI/ML Experience", use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": "AI/ML Experience"})
        st.session_state.messages.append({"role": "assistant", "content": run_turn("AI/ML Experience")})
        st.rerun()
    if c3.button("Intelligence Work", use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": "Intelligence Work"})
        st.session_state.messages.append({"role": "assistant", "content": run_turn("Intelligence Work")})
        st.rerun()

user_input = st.chat_input("Ask about Stephen's qualifications...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)
    with st.chat_message("assistant"):
        res = run_turn(user_input)
        st.session_state.messages.append({"role": "assistant", "content": res})
st.markdown('</div>', unsafe_allow_html=True)
