# streamlit_app.py — Evidence-Only Recruiter Proxy (Definitive Build)
# ═══════════════════════════════════════════════════════════════════════════════
# MERGED: V2 backend intelligence + V1 presentation layer + all bug fixes
# TARGET: GitHub → Streamlit Community Cloud (free tier)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Backend (from V2):
#   - Recruiter context extraction (auto per-turn)
#   - Action buttons: Verify / Fit Summary / Draft Outreach
#   - Proof snippets expander (retrieved chunk text)
#   - Evidence confidence badges (source-count heuristic)
#   - FAISS manifest with SHA-256 (rebuild only when PDFs change)
#   - Debug expander (standalone retrieval query)
#   - Unified run_turn() with action_mode dispatch
#
# Presentation (from V1):
#   - DM Sans + JetBrains Mono typography
#   - Gold / navy credential-grade color system
#   - Hero banner with credential badge pills
#   - Animated status strip
#   - Responsive CSS (mobile-safe)
#   - Styled chat, buttons, expanders, scrollbars
#
# Bug fixes applied:
#   - Anchored guardrail regex (no false positives on "references")
#   - .invoke() replaces deprecated .get_relevant_documents()
#   - Proper file handle management (no leaked handles)
#   - List[str] type hints (Python 3.8+ compatible)
#   - ReportLab sanitization (control chars stripped)
#   - Empty retrieval user warning
#   - Logged rewrite/extraction failures
#   - FAISS cache-clear admin button
#   - Manifest write uses context manager
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

try:
    from scoring import score_overall
    from mitigations import tier_from_score, mitigation_playbook
    SCORING_ENABLED = True
except Exception:
    SCORING_ENABLED = False

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# GUARDRAILS — anchored regex (fix: no false positives on "the doc references…")
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
            "⚠️ Response blocked: external citation/link pattern detected.\n\n"
            "This system is **Public-safe / Evidence-only** and can only cite the loaded corpus."
        )
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# ASSET HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def safe_exists(path: Optional[str]) -> bool:
    try:
        return bool(path) and os.path.exists(path)
    except Exception:
        return False

def first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if safe_exists(p):
            return p
    return None

def _read_file_safe(path: str) -> str:
    """Read text file with proper handle management."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

LOGO_PATH = first_existing([
    os.path.join("assets", "logo.png"), os.path.join("assets", "logo.jpg"),
    os.path.join("assets", "logo.jpeg"), "logo.png", "logo.jpg", "logo.jpeg",
])
HEADSHOT_PATH = first_existing([
    os.path.join("assets", "headshot.png"), os.path.join("assets", "headshot.jpg"),
    os.path.join("assets", "headshot.jpeg"), "headshot.png", "headshot.jpg", "headshot.jpeg",
])
ABOUT_MD_PATH = first_existing([os.path.join("assets", "about_stephen.md"), "about_stephen.md"])
THINK_MD_PATH = first_existing([os.path.join("assets", "how_i_think.md"), "how_i_think.md"])
PUBS_CSV_PATH = first_existing([os.path.join("assets", "publications.csv"), "publications.csv"])
LINKEDIN_URL = "https://www.linkedin.com/in/stephendietrich-kolokouris/"


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT CONFIG + PRESENTATION LAYER (V1 design system)
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Dr. Stephen Dietrich-Kolokouris | Strategic Technical Proxy",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
/* ── FOUNDATION ────────────────────────────────────────────── */
:root {
  --bg-deep:    #04060C;
  --bg-surface: #0A0F1E;
  --bg-card:    #0E1529;
  --bg-raised:  #131B33;
  --border:     rgba(203,213,225,0.08);
  --border-lit: rgba(203,213,225,0.14);
  --txt:        #E2E8F0;
  --txt-dim:    #8892A8;
  --txt-faint:  #5A6478;
  --gold:       #D4A843;
  --gold-dim:   rgba(212,168,67,0.15);
  --blue:       #5B9CF5;
  --blue-dim:   rgba(91,156,245,0.12);
  --green:      #34D399;
  --green-dim:  rgba(52,211,153,0.10);
  --red:        #EF4444;
  --red-dim:    rgba(239,68,68,0.10);
  --amber:      #FBBF24;
  --font:       'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  --mono:       'JetBrains Mono', 'SF Mono', Consolas, monospace;
  --radius:     10px;
  --radius-lg:  16px;
}

html, body, [class*="stApp"] {
  background: var(--bg-deep) !important;
  color: var(--txt) !important;
  font-family: var(--font) !important;
}
.main .block-container {
  padding-top: 0.6rem !important;
  padding-bottom: 2rem !important;
  max-width: 1100px;
}

/* ── SIDEBAR ───────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #080D1A 0%, #050811 100%) !important;
  border-right: 1px solid var(--border);
  font-family: var(--font);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
section[data-testid="stSidebar"] [data-testid="stMarkdown"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdown"] li {
  font-size: 0.88rem; color: var(--txt-dim);
}
section[data-testid="stSidebar"] h3 {
  font-size: 1.05rem !important; letter-spacing: 0.01em;
  color: var(--txt) !important; margin-bottom: 2px !important;
}

/* ── HERO BANNER ───────────────────────────────────────────── */
.sdk-hero {
  position: relative;
  background: linear-gradient(135deg, rgba(212,168,67,0.06) 0%, rgba(91,156,245,0.04) 40%, transparent 70%), var(--bg-surface);
  border: 1px solid var(--border-lit);
  border-radius: var(--radius-lg);
  padding: 28px 32px 24px;
  margin-bottom: 8px;
  overflow: hidden;
}
.sdk-hero::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--gold) 0%, transparent 60%);
}
.sdk-name {
  font-size: 1.55rem; font-weight: 700; letter-spacing: -0.01em;
  color: #fff; margin: 0 0 2px; line-height: 1.25;
}
.sdk-title-line {
  font-size: 0.92rem; color: var(--gold); font-weight: 500;
  letter-spacing: 0.04em; margin: 0 0 14px;
}
.sdk-desc {
  font-size: 0.88rem; color: var(--txt-dim);
  line-height: 1.55; max-width: 680px; margin: 0;
}

/* ── CREDENTIAL BADGES ─────────────────────────────────────── */
.sdk-badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.sdk-badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 12px; border-radius: 6px;
  font-size: 0.78rem; font-weight: 500; letter-spacing: 0.02em;
  border: 1px solid var(--border); background: var(--bg-card);
  color: var(--txt-dim); transition: border-color 0.2s;
}
.sdk-badge:hover { border-color: var(--border-lit); }
.sdk-badge .badge-dot {
  width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0;
}
.dot-gold { background: var(--gold); }
.dot-blue { background: var(--blue); }
.dot-green { background: var(--green); }
.dot-red { background: var(--red); }

/* ── EVIDENCE CONFIDENCE BADGES ────────────────────────────── */
.ev-badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 12px; border-radius: 6px;
  font-size: 0.76rem; font-weight: 600; letter-spacing: 0.03em;
  margin-bottom: 8px;
}
.ev-high  { background: var(--green-dim); color: var(--green); border: 1px solid rgba(52,211,153,0.25); }
.ev-med   { background: var(--blue-dim);  color: var(--blue);  border: 1px solid rgba(91,156,245,0.25); }
.ev-low   { background: rgba(251,191,36,0.10); color: var(--amber); border: 1px solid rgba(251,191,36,0.25); }
.ev-none  { background: var(--red-dim);   color: var(--red);   border: 1px solid rgba(239,68,68,0.25); }

/* ── INFO CARDS ────────────────────────────────────────────── */
.sdk-card {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 14px 16px;
}
.sdk-card b {
  font-size: 0.82rem; color: var(--txt);
  letter-spacing: 0.03em; text-transform: uppercase;
}
.sdk-card-body {
  font-size: 0.84rem; color: var(--txt-dim); margin-top: 8px; line-height: 1.5;
}

/* ── CHAT MESSAGES ─────────────────────────────────────────── */
[data-testid="stChatMessage"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--bg-card) !important;
  padding: 14px 16px !important;
  margin-bottom: 8px !important;
  font-family: var(--font) !important;
}
[data-testid="stChatMessage"] p {
  font-size: 0.92rem !important; line-height: 1.6 !important;
}

/* ── CHAT INPUT ────────────────────────────────────────────── */
[data-testid="stChatInput"] {
  border-color: var(--border-lit) !important;
  background: var(--bg-surface) !important;
}
[data-testid="stChatInput"] textarea {
  font-family: var(--font) !important;
  font-size: 0.9rem !important;
  color: var(--txt) !important;
}

/* ── STATUS STRIP ──────────────────────────────────────────── */
.sdk-status-strip {
  display: flex; align-items: center; gap: 20px;
  padding: 10px 16px;
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius); margin: 6px 0 14px;
  font-size: 0.78rem; color: var(--txt-dim);
}
.sdk-status-strip .strip-item { display: flex; align-items: center; gap: 6px; }
.sdk-status-strip .strip-dot {
  width: 7px; height: 7px; border-radius: 50%;
  animation: pulse-dot 2.5s ease-in-out infinite;
}
@keyframes pulse-dot {
  0%, 100% { opacity: 0.6; }
  50% { opacity: 1; }
}

/* ── EXPANDERS ─────────────────────────────────────────────── */
details[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--bg-card) !important;
}

/* ── SCROLLBAR ─────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.14); }

/* ── BUTTONS ───────────────────────────────────────────────── */
.stButton > button[kind="primary"],
.stDownloadButton > button {
  background: linear-gradient(135deg, var(--gold) 0%, #C49B38 100%) !important;
  color: #0A0F1E !important; font-weight: 600 !important;
  border: none !important; font-family: var(--font) !important;
}
.stButton > button {
  font-family: var(--font) !important;
  font-size: 0.82rem !important;
  border-color: var(--border-lit) !important;
}
.stLinkButton > a {
  color: var(--gold) !important;
  border-color: var(--border-lit) !important;
  font-family: var(--font) !important;
}

/* ── PROOF SNIPPET STYLING ─────────────────────────────────── */
.proof-label {
  font-family: var(--mono); font-size: 0.75rem;
  color: var(--gold); font-weight: 500;
  margin-bottom: 2px;
}
.proof-text {
  font-size: 0.82rem; color: var(--txt-dim);
  line-height: 1.45; padding-left: 8px;
  border-left: 2px solid var(--border-lit);
  margin-bottom: 10px;
}

/* ── RESPONSIVE ────────────────────────────────────────────── */
@media (max-width: 768px) {
  .sdk-hero { padding: 20px 18px 18px; }
  .sdk-name { font-size: 1.25rem; }
  .sdk-badges { gap: 6px; }
  .sdk-badge { font-size: 0.72rem; padding: 4px 9px; }
  .sdk-status-strip { flex-wrap: wrap; gap: 10px; }
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM + EMBEDDINGS INIT
# ═══════════════════════════════════════════════════════════════════════════════
def init_llm() -> ChatOpenAI:
    try:
        return ChatOpenAI(model="gpt-4o", temperature=0, api_key=st.secrets["OPENAI_API_KEY"])
    except TypeError:
        return ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])

def init_embeddings() -> OpenAIEmbeddings:
    try:
        return OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    except TypeError:
        return OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])


# ═══════════════════════════════════════════════════════════════════════════════
# FAISS PERSISTENCE + SHA-256 MANIFEST (from V2 — rebuild only when PDFs change)
# ═══════════════════════════════════════════════════════════════════════════════
FAISS_DIR = "faiss_index"
MANIFEST_PATH = os.path.join(FAISS_DIR, "manifest.json")

def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _build_manifest(data_dir: str) -> dict:
    files = []
    for root, _, fnames in os.walk(data_dir):
        for fn in fnames:
            if fn.lower().endswith(".pdf"):
                p = os.path.join(root, fn)
                try:
                    files.append({
                        "path": p.replace("\\", "/"),
                        "sha256": _file_sha256(p),
                        "size": os.path.getsize(p),
                    })
                except Exception:
                    continue
    files = sorted(files, key=lambda x: x["path"])
    return {"data_dir": data_dir.replace("\\", "/"), "files": files}

def _manifest_changed(new_manifest: dict) -> bool:
    if not os.path.exists(MANIFEST_PATH):
        return True
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            old = json.loads(f.read())
        return old != new_manifest
    except Exception:
        return True

def load_or_build_faiss() -> FAISS:
    if not os.path.exists("data"):
        st.error("Missing **/data** directory. Commit/upload your PDFs into `/data`.")
        st.stop()

    embeddings = init_embeddings()
    new_manifest = _build_manifest("data")

    # Load existing index if manifest matches
    if os.path.isdir(FAISS_DIR) and not _manifest_changed(new_manifest):
        try:
            return FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.warning(f"FAISS index load failed; rebuilding. Reason: {e}")

    # Build / rebuild
    loader = PyPDFDirectoryLoader("data/")
    docs = loader.load()
    if not docs:
        st.error("No documents found in `/data`.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=160)
    chunks = splitter.split_documents(docs)

    with st.status("Indexing corpus (FAISS)…", expanded=False) as status:
        vs = FAISS.from_documents(chunks, embeddings)
        status.update(label="✅ Corpus indexed (FAISS ready)", state="complete")

    # Persist index + manifest
    try:
        os.makedirs(FAISS_DIR, exist_ok=True)
        vs.save_local(FAISS_DIR)
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            f.write(json.dumps(new_manifest, indent=2))
    except Exception as e:
        st.warning(f"FAISS index could not be saved (non-fatal): {e}")

    return vs

@st.cache_resource
def init_retriever():
    vs = load_or_build_faiss()
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 30, "lambda_mult": 0.55},
    )

retriever = init_retriever()


# ═══════════════════════════════════════════════════════════════════════════════
# EVIDENCE PACK + PROOF SNIPPETS + CONFIDENCE (from V2, with fixes)
# ═══════════════════════════════════════════════════════════════════════════════
def _page_label(meta: dict) -> Optional[str]:
    for key in ("page", "page_number", "loc.page_number"):
        if key in meta:
            try:
                return f"p.{int(meta[key]) + 1}"
            except Exception:
                pass
    return None

def format_evidence_pack(docs) -> Tuple[str, List[str], List[str], List[Tuple[str, str]]]:
    """
    Returns:
      evidence_pack_text  — for LLM (source-tagged)
      evidence_labels     — list like "file.pdf (p.2)"
      evidence_files_only — filenames for PDF export
      proof_snippets      — list of (label, short_snippet) for UI
    """
    seen = set()
    labels: List[str] = []
    files_only: List[str] = []
    parts: List[str] = []
    proof: List[Tuple[str, str]] = []

    for d in docs:
        meta = d.metadata or {}
        src_full = meta.get("source", "") or ""
        src = os.path.basename(src_full) if src_full else "unknown"
        page = _page_label(meta)
        label = f"{src} ({page})" if page else src

        text = (d.page_content or "").strip()
        if not text:
            continue

        uniq = (src, page, text[:120])
        if uniq in seen:
            continue
        seen.add(uniq)

        pack_text = text
        if len(pack_text) > 2200:
            pack_text = pack_text[:2200].rstrip() + "…"

        parts.append(f"[SOURCE: {label}]\n{pack_text}")
        labels.append(label)
        files_only.append(src)

        # Shorter snippet for UI proof expander
        snip = text.replace("\n", " ").strip()
        if len(snip) > 420:
            snip = snip[:420].rstrip() + "…"
        proof.append((label, snip))

    return "\n\n".join(parts), labels, sorted(set(files_only)), proof

def evidence_confidence(evidence_labels: List[str]) -> str:
    if not evidence_labels:
        return "none"
    distinct_files = set(re.split(r"\s+\(p\.\d+\)$", x)[0] for x in evidence_labels)
    n = len(distinct_files)
    if n >= 3:
        return "high"
    if n >= 1:
        return "medium"
    return "none"

def render_confidence_badge(level: str) -> str:
    m = {
        "high":   ("<span class='ev-badge ev-high'>● Evidence: High</span>", "3+ distinct sources"),
        "medium": ("<span class='ev-badge ev-med'>● Evidence: Medium</span>", "1–2 sources"),
        "low":    ("<span class='ev-badge ev-low'>● Evidence: Low</span>", "weak match"),
        "none":   ("<span class='ev-badge ev-none'>● Evidence: None</span>", "no relevant chunks"),
    }
    html, _ = m.get(level, m["none"])
    return html


# ═══════════════════════════════════════════════════════════════════════════════
# RECRUITER CONTEXT STATE MACHINE (from V2)
# ═══════════════════════════════════════════════════════════════════════════════
_EMPTY_RECRUITER_STATE = {
    "target_roles": [],
    "domains": [],
    "location": None,
    "onsite_remote": None,
    "must_haves": [],
    "nice_to_haves": [],
    "dealbreakers": [],
}

def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items or []:
        x = (x or "").strip()
        if not x:
            continue
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out

def extract_recruiter_constraints(llm: ChatOpenAI, user_message: str) -> dict:
    schema = _EMPTY_RECRUITER_STATE.copy()
    prompt = (
        "Extract recruiter constraints from the message if present.\n"
        "Return JSON only, matching this schema exactly:\n"
        f"{json.dumps(schema)}\n\n"
        "Rules:\n"
        "- If not mentioned, use empty lists or null.\n"
        "- onsite_remote must be one of: onsite, hybrid, remote, null.\n"
        "- Keep items short (e.g., 'Okta', 'PowerScale', 'IR', 'SOC2', 'onsite Dallas').\n\n"
        f"MESSAGE:\n{user_message}\n\nJSON:"
    )
    try:
        out = llm.invoke(prompt)
        text = (getattr(out, "content", None) or str(out)).strip()
        # Strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("Recruiter constraint extraction failed (non-fatal): %s", e)
        return {}

def update_recruiter_state(new_bits: dict):
    if not new_bits:
        return
    s = st.session_state.recruiter_state
    for k in ("target_roles", "domains", "must_haves", "nice_to_haves", "dealbreakers"):
        if k in new_bits and isinstance(new_bits[k], list):
            s[k] = _dedupe_keep_order((s.get(k) or []) + new_bits[k])
    loc = new_bits.get("location")
    if isinstance(loc, str) and loc.strip():
        s["location"] = loc.strip()
    o = new_bits.get("onsite_remote")
    if o in ("onsite", "hybrid", "remote"):
        s["onsite_remote"] = o


# ═══════════════════════════════════════════════════════════════════════════════
# HISTORY + CONTEXT-AWARE QUERY REWRITE (merged: V2 recruiter context + V1 logging)
# ═══════════════════════════════════════════════════════════════════════════════
def rewrite_to_standalone(
    llm: ChatOpenAI,
    chat_history: List[Dict],
    user_input: str,
    recruiter_state: dict,
    max_turns: int = 8,
) -> str:
    hist_lines = []
    for m in chat_history[-max_turns:]:
        r = (m.get("role") or "").lower()
        if r in ("user", "assistant"):
            content = (m.get("content") or "").strip()
            if content:
                hist_lines.append(f"{r.upper()}: {content}")

    history_text = "\n".join(hist_lines).strip()
    state_text = json.dumps(recruiter_state or {}, ensure_ascii=False)

    prompt = (
        "Rewrite the user's last message into a fully standalone search query "
        "for retrieving evidence about Dr. Stephen Dietrich-Kolokouris.\n"
        "Use conversation history and recruiter context to resolve pronouns and implied constraints.\n"
        "Do NOT add new facts. Keep it short and retrieval-friendly.\n\n"
        f"RECRUITER CONTEXT JSON:\n{state_text}\n\n"
        f"CONVERSATION:\n{history_text}\n\n"
        f"USER MESSAGE:\n{user_input}\n\n"
        "STANDALONE QUERY:"
    )

    try:
        out = llm.invoke(prompt)
        text = (getattr(out, "content", None) or str(out)).strip()
        return text if text else user_input
    except Exception as e:
        logger.warning("Query rewrite failed (falling back to raw input): %s", e)
        return user_input


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT BUILDER (from V2 — action_mode dispatch)
# ═══════════════════════════════════════════════════════════════════════════════
def build_system_prompt(
    personal_mode: bool,
    recruiter_state: dict,
    evidence_pack: str,
    vendor_block: str,
    action_mode: str,
) -> str:
    state_text = json.dumps(recruiter_state or {}, ensure_ascii=False)

    tone_line = (
        "TONE MODE: Personal Mode.\n"
        "- You may include brief career context and lessons learned ONLY if supported by the evidence pack.\n"
        "- Keep it recruiter-friendly; avoid hype; keep it precise.\n"
    ) if personal_mode else (
        "TONE MODE: Technical-only.\n"
        "- Direct, systems-focused, implementation-oriented.\n"
    )

    action_map = {
        "verify": (
            "TASK MODE: VERIFY.\n"
            "- Produce a short checklist:\n"
            "  1) Supported claims (each must cite a [SOURCE: ...] from the evidence pack)\n"
            "  2) Unsupported / Not in corpus\n"
            "  3) What document line(s) would be needed to support the missing items\n"
        ),
        "fit": (
            "TASK MODE: FIT SUMMARY.\n"
            "- Use recruiter context JSON to tailor the fit.\n"
            "- Output:\n"
            "  - Fit summary (3–5 sentences)\n"
            "  - Evidence-backed strengths (3–6 bullets, each must cite a [SOURCE: ...])\n"
            "  - Risks / gaps (if any) labeled Not in corpus when unsupported\n"
            "  - Suggested next questions to validate fit\n"
        ),
        "outreach": (
            "TASK MODE: OUTREACH.\n"
            "- Draft a recruiter outreach message (100–160 words) based on recruiter context JSON.\n"
            "- Use only evidence-backed claims.\n"
            "- If key context is missing (role/location), ask ONE short question at the end.\n"
        ),
    }
    action_instructions = action_map.get(action_mode, (
        "TASK MODE: CHAT.\n"
        "- Answer the user question recruiter-grade.\n"
        "- If something is not supported, say **Not in corpus** and suggest what to add.\n"
    ))

    return (
        "You are an evidence-only technical proxy representing Dr. Stephen Dietrich-Kolokouris.\n\n"
        "MANDATORY CONSTRAINTS:\n"
        "1) Use ONLY the EVIDENCE PACK below (and deterministic vendor block if present).\n"
        "2) Do NOT invent facts, dates, employers, credentials, project details, or external references.\n"
        "3) If the answer cannot be supported, say **Not in corpus**.\n"
        "4) Do NOT include URLs or bibliography headings.\n"
        "5) When making a key claim, cite the supporting source by referencing its label.\n\n"
        f"RECRUITER CONTEXT JSON:\n{state_text}\n\n"
        + tone_line + "\n"
        + action_instructions + "\n\n"
        "EVIDENCE PACK:\n"
        + evidence_pack
        + vendor_block
    )

def build_vendor_block(vendor_ctx: Optional[dict]) -> str:
    if not vendor_ctx:
        return ""
    return (
        "\n\nSelected Vendor Context (deterministic):\n"
        f"- Vendor: {vendor_ctx.get('vendor_name')}\n"
        f"- Component: {vendor_ctx.get('product_or_component')}\n"
        f"- Class: {vendor_ctx.get('component_class')}\n"
        f"- Origin/Jurisdiction: {vendor_ctx.get('origin_jurisdiction')}\n"
        f"- Criticality: {vendor_ctx.get('criticality')}\n"
        f"- Tier: {vendor_ctx.get('tier')}\n"
        f"- Scores: REE={vendor_ctx.get('ree_risk')}, FW={vendor_ctx.get('firmware_risk')}, Overall={vendor_ctx.get('overall_risk')}\n"
        "Mitigation priorities (deterministic):\n"
        + "\n".join(f"- {m}" for m in vendor_ctx.get("mitigations", []))
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PDF EXPORT (merged: V2 logic + V1 sanitization fix)
# ═══════════════════════════════════════════════════════════════════════════════
def _sanitize_for_reportlab(text: str) -> str:
    if not text:
        return text
    text = text.replace("\r", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text

def _wrap_text_lines(text: str, max_chars: int = 95) -> List[str]:
    text = _sanitize_for_reportlab(text or "")
    out = []
    for raw in text.split("\n"):
        s = raw.strip()
        if not s:
            out.append("")
            continue
        while len(s) > max_chars:
            cut = s.rfind(" ", 0, max_chars)
            if cut == -1:
                cut = max_chars
            out.append(s[:cut].rstrip())
            s = s[cut:].lstrip()
        out.append(s)
    return out

def build_qa_pdf_bytes(title: str, messages: List[Dict], evidence_files: List[str]) -> bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("reportlab not installed")
    from io import BytesIO

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER
    left = 0.85 * inch
    right = 0.85 * inch
    top = 0.85 * inch
    bottom = 0.75 * inch
    y = height - top

    def new_page():
        nonlocal y
        c.showPage()
        y = height - top

    c.setFont("Helvetica-Bold", 14)
    c.drawString(left, y, _sanitize_for_reportlab(title))
    y -= 18
    c.setFont("Helvetica", 9)
    stamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    c.drawString(left, y, f"Generated: {stamp}   |   Mode: Public-safe / Evidence-only")
    y -= 14
    c.setLineWidth(0.5)
    c.line(left, y, width - right, y)
    y -= 12

    for m in messages:
        role = (m.get("role") or "").lower()
        content = m.get("content") or ""
        label = "Q:" if role == "user" else "A:"
        c.setFont("Helvetica-Bold", 10)
        if y < bottom + 40:
            new_page()
        c.drawString(left, y, label)
        y -= 12
        c.setFont("Helvetica", 10)
        for ln in _wrap_text_lines(content, max_chars=105):
            if y < bottom + 14:
                new_page()
                c.setFont("Helvetica", 10)
            c.drawString(left + 18, y, ln)
            y -= 12
        y -= 8

    if evidence_files:
        if y < bottom + 80:
            new_page()
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left, y, "Evidence (corpus files referenced)")
        y -= 16
        c.setFont("Helvetica", 10)
        for f in sorted(set(evidence_files)):
            if y < bottom + 14:
                new_page()
                c.setFont("Helvetica", 10)
            c.drawString(left, y, f"- {f}")
            y -= 12

    c.save()
    buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE (merged: V1 base + V2 recruiter/proof/debug state)
# ═══════════════════════════════════════════════════════════════════════════════
_SESSION_DEFAULTS = {
    "messages": [],
    "personal_mode": False,
    "pinned_opening": True,
    "qa_evidence_files": [],
    "selected_vendor_context": None,
    "recruiter_state": _EMPTY_RECRUITER_STATE.copy(),
    "last_proof": [],
    "last_evidence_labels": [],
    "last_standalone_query": "",
}
for k, v in _SESSION_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED run_turn() (from V2 — single entry point for all answer modes)
# ═══════════════════════════════════════════════════════════════════════════════
def run_turn(user_text: str, action_mode: str = "chat") -> str:
    llm = init_llm()

    # 0) Extract recruiter constraints from this message (non-fatal)
    new_bits = extract_recruiter_constraints(llm, user_text)
    update_recruiter_state(new_bits)

    # 1) Context-aware rewrite
    standalone_query = rewrite_to_standalone(
        llm, st.session_state.messages, user_text,
        st.session_state.recruiter_state, max_turns=8,
    )
    st.session_state.last_standalone_query = standalone_query

    # 2) Retrieve (fix: .invoke() not deprecated .get_relevant_documents())
    docs = retriever.invoke(standalone_query)

    # 3) Handle empty retrieval
    if not docs:
        st.warning("⚠️ Retrieval returned zero chunks. The answer may be limited.")

    evidence_pack, evidence_labels, evidence_files, proof_snips = format_evidence_pack(docs)
    st.session_state.last_proof = proof_snips
    st.session_state.last_evidence_labels = evidence_labels

    # Track files for PDF export
    if evidence_files:
        existing = set(st.session_state.get("qa_evidence_files", []) or [])
        st.session_state.qa_evidence_files = sorted(existing.union(set(evidence_files)))

    # 4) Build prompt + answer
    vendor_block = build_vendor_block(st.session_state.get("selected_vendor_context"))
    system_prompt = build_system_prompt(
        personal_mode=st.session_state.personal_mode,
        recruiter_state=st.session_state.recruiter_state,
        evidence_pack=evidence_pack,
        vendor_block=vendor_block,
        action_mode=action_mode,
    )

    try:
        out = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ])
        answer = (out.content or "").strip()
    except Exception as e:
        answer = f"⚠️ Model error: {e}"

    answer = enforce_no_external_refs(answer)

    # 5) Confidence badge + evidence labels
    conf = evidence_confidence(evidence_labels)
    badge_html = render_confidence_badge(conf)
    st.markdown(badge_html, unsafe_allow_html=True)

    if evidence_labels:
        answer += "\n\n**Evidence:** " + ", ".join(evidence_labels)
    else:
        answer += "\n\n**Evidence:** Not in corpus (no relevant excerpts retrieved)."

    return answer


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR (V1 design + V2 recruiter context panel)
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    if safe_exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    if safe_exists(HEADSHOT_PATH):
        st.image(HEADSHOT_PATH, width=180)

    st.markdown("### Dr. Stephen Dietrich-Kolokouris")
    st.caption("PhD · CCIE · Applied Security · Strategic Modeling")
    st.markdown("---")
    st.link_button("🔗  LinkedIn Profile", LINKEDIN_URL, use_container_width=True)
    st.markdown("---")

    st.session_state.personal_mode = st.toggle(
        "🎙️ Personal Mode",
        value=st.session_state.personal_mode,
        help="Adds career narrative tone. Evidence-only constraint stays enforced.",
    )

    st.markdown(
        "<div style='background:rgba(212,168,67,0.08); border:1px solid rgba(212,168,67,0.18); "
        "border-radius:8px; padding:10px 12px; margin:8px 0; font-size:0.8rem; color:#D4A843;'>"
        "🔐 <b>Evidence-Only Mode Active</b><br>"
        "<span style='color:#8892A8;'>All answers cite only the loaded corpus. "
        "No external data. No hallucination.</span></div>",
        unsafe_allow_html=True,
    )

    with st.expander("🧩 Recruiter Context (auto)", expanded=False):
        st.json(st.session_state.recruiter_state)
        if st.button("Clear recruiter context"):
            st.session_state.recruiter_state = _EMPTY_RECRUITER_STATE.copy()
            st.success("Cleared.")

    with st.expander("🧾 About", expanded=False):
        if safe_exists(ABOUT_MD_PATH):
            st.markdown(_read_file_safe(ABOUT_MD_PATH))
        else:
            st.markdown("_Create `assets/about_stephen.md` with a career narrative._")

    with st.expander("🧠 How I Think", expanded=False):
        if safe_exists(THINK_MD_PATH):
            st.markdown(_read_file_safe(THINK_MD_PATH))
        else:
            st.markdown("_Create `assets/how_i_think.md` with decision cadence._")

    with st.expander("📚 Publications", expanded=False):
        if safe_exists(PUBS_CSV_PATH):
            try:
                pubs = pd.read_csv(PUBS_CSV_PATH)
                st.dataframe(pubs, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Unable to read publications.csv: {e}")
        else:
            st.markdown("_Create `assets/publications.csv` — columns: type, title, venue_or_outlet, year, link_")

    with st.expander("⚙️ System Architecture", expanded=False):
        st.markdown(
            "**Pipeline:** PDF ingest → SHA-256 manifest → chunk → embed → FAISS (MMR) → "
            "recruiter context extraction → context-aware rewrite → evidence-only GPT-4o → guardrail filter\n\n"
            "**Stack:** LangChain · OpenAI · FAISS · Streamlit · ReportLab"
        )

    with st.expander("🔧 Admin", expanded=False):
        if st.button("🔄 Rebuild FAISS Index", help="Clear cached retriever and rebuild from /data PDFs"):
            # Also remove manifest to force full rebuild
            try:
                if os.path.exists(MANIFEST_PATH):
                    os.remove(MANIFEST_PATH)
            except Exception:
                pass
            init_retriever.clear()
            st.rerun()

    if SCORING_ENABLED:
        with st.expander("📊 Supply Chain Risk", expanded=False):
            st.caption("Deterministic scoring → mitigations")
            weight_fw = st.slider("Weight: Firmware integrity", 0.0, 1.0, 0.55, 0.05, key="weight_fw")
            uploaded_csv = st.file_uploader("Vendor CSV", type=["csv"], key="vendor_csv_uploader")
            st.caption(
                "Required columns: vendor_name, product_or_component, component_class, "
                "origin_jurisdiction, criticality, contains_ree_magnets, firmware_ota, "
                "firmware_signing, secure_boot_attestation, sbom_available, remote_admin_access, "
                "telemetry_logging, alt_supplier_available, known_single_point_of_failure"
            )
            if uploaded_csv is not None:
                try:
                    df = pd.read_csv(uploaded_csv)
                except Exception as e:
                    st.error(f"Unable to read CSV: {e}")
                    df = None
                if df is not None:
                    required_cols = {
                        "vendor_name", "product_or_component", "component_class",
                        "origin_jurisdiction", "criticality", "contains_ree_magnets",
                        "firmware_ota", "firmware_signing", "secure_boot_attestation",
                        "sbom_available", "remote_admin_access", "telemetry_logging",
                        "alt_supplier_available", "known_single_point_of_failure",
                    }
                    missing = sorted(list(required_cols - set(df.columns)))
                    if missing:
                        st.error(f"CSV missing required columns: {missing}")
                    else:
                        def _score_row(r):
                            s = score_overall(r.to_dict(), weight_fw=weight_fw)
                            return pd.Series(s)
                        scores = df.apply(_score_row, axis=1)
                        out_df = pd.concat([df, scores], axis=1)
                        out_df["tier"] = out_df["overall_risk"].apply(tier_from_score)
                        st.dataframe(out_df.sort_values("overall_risk", ascending=False).head(10), use_container_width=True)
                        idx = st.number_input(
                            "Select row index (0-based)", min_value=0,
                            max_value=max(0, len(out_df) - 1), value=0, step=1, key="vendor_row_idx",
                        )
                        row = out_df.iloc[int(idx)].to_dict()
                        if st.button("Use selected vendor in chat", key="use_vendor_ctx"):
                            st.session_state.selected_vendor_context = {
                                "vendor_name": row.get("vendor_name"),
                                "product_or_component": row.get("product_or_component"),
                                "component_class": row.get("component_class"),
                                "origin_jurisdiction": row.get("origin_jurisdiction"),
                                "criticality": row.get("criticality"),
                                "contains_ree_magnets": row.get("contains_ree_magnets"),
                                "ree_risk": float(row.get("ree_risk", 0.0)),
                                "firmware_risk": float(row.get("firmware_risk", 0.0)),
                                "overall_risk": float(row.get("overall_risk", 0.0)),
                                "tier": row.get("tier"),
                                "mitigations": mitigation_playbook(float(row.get("overall_risk", 0.0))),
                            }
                            st.success("Vendor context stored.")

    with st.expander("🖨️ Export Q&A (PDF)", expanded=False):
        if not REPORTLAB_OK:
            st.warning("PDF export unavailable — add `reportlab` to requirements.txt")
        else:
            export_title = st.text_input(
                "PDF title",
                value="Q&A — Dr. Stephen Dietrich-Kolokouris (Evidence-Only)",
            )
            if st.button("Generate PDF", type="primary"):
                export_msgs = [
                    {"role": (m.get("role") or "").lower(), "content": m.get("content") or ""}
                    for m in st.session_state.messages
                    if (m.get("role") or "").lower() in ("user", "assistant")
                ]
                evidence_files = sorted(set(st.session_state.get("qa_evidence_files", []) or []))
                pdf_bytes = build_qa_pdf_bytes(export_title, export_msgs, evidence_files)
                st.download_button(
                    "📥 Download PDF", data=pdf_bytes,
                    file_name="QA_Transcript_Stephen_Dietrich_Kolokouris.pdf",
                    mime="application/pdf",
                )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT — HERO + STATUS + ACTION BAR + PROOF + CHAT
# ═══════════════════════════════════════════════════════════════════════════════

# ── Hero Banner ──
st.markdown("""
<div class="sdk-hero">
  <p class="sdk-name">Dr. Stephen Dietrich-Kolokouris, PhD</p>
  <p class="sdk-title-line">CYBERSECURITY &nbsp;·&nbsp; AI / ML SYSTEMS &nbsp;·&nbsp; STRATEGIC ANALYSIS &nbsp;·&nbsp; DATA ENGINEERING</p>
  <p class="sdk-desc">
    Evidence-only AI proxy backed by a curated corpus of publications, research, and project documentation.
    Ask about capabilities, architecture decisions, or domain expertise — every answer is sourced and verifiable.
  </p>
  <div class="sdk-badges">
    <span class="sdk-badge"><span class="badge-dot dot-gold"></span>PhD — Goethe University Frankfurt</span>
    <span class="sdk-badge"><span class="badge-dot dot-blue"></span>CCIE Certified</span>
    <span class="sdk-badge"><span class="badge-dot dot-green"></span>Production RAG Systems</span>
    <span class="sdk-badge"><span class="badge-dot dot-red"></span>Critical Infrastructure Security</span>
    <span class="sdk-badge"><span class="badge-dot dot-blue"></span>Supply Chain Analysis</span>
    <span class="sdk-badge"><span class="badge-dot dot-gold"></span>DoD Simulation Modeling</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Live Status Strip ──
st.markdown("""
<div class="sdk-status-strip">
  <div class="strip-item">
    <span class="strip-dot" style="background:#34D399;"></span>
    <span>FAISS Index Active</span>
  </div>
  <div class="strip-item">
    <span class="strip-dot" style="background:#D4A843;"></span>
    <span>Evidence-Only Guardrails</span>
  </div>
  <div class="strip-item">
    <span class="strip-dot" style="background:#5B9CF5;"></span>
    <span>GPT-4o + Recruiter Context</span>
  </div>
  <div class="strip-item">
    <span style="color:#5A6478;">MMR · k=8 · SHA-256 manifest</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Scope / Mode Cards ──
col1, col2 = st.columns([2.2, 1.0], gap="large")
with col1:
    st.markdown("""
<div class="sdk-card">
  <b>SCOPE BOUNDARY</b>
  <div class="sdk-card-body">
    Public-safe, evidence-only responses. Every claim is backed by file + page citation.
    No external links. No fabricated credentials. Recruiter context is extracted automatically.
  </div>
</div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""
<div class="sdk-card">
  <b>MODE</b>
  <div class="sdk-card-body">
    Personal Mode adds narrative context.<br>
    Evidence-only rule stays enforced.
  </div>
</div>""", unsafe_allow_html=True)

st.write("")

# ── Pinned Opening ──
if st.session_state.pinned_opening and not st.session_state.messages:
    pinned = (
        "Welcome. I'm the evidence-only proxy for **Dr. Stephen Dietrich-Kolokouris**. "
        "Every answer I give is sourced from his curated corpus — no hallucination, no external data.\n\n"
        "**To get started, pick a direction:**\n\n"
        "1. What role(s) are you hiring for, and where (onsite / hybrid / remote)?\n"
        "2. What are the top 3 must-haves and any dealbreakers?\n"
        "3. What would success look like in the first 90 days?\n\n"
        "_You can pivot roles mid-conversation — I'll adapt._"
    )
    st.session_state.messages.append({"role": "assistant", "content": pinned})

# ── Chat History ──
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ── Action Buttons (from V2, styled for V1 design) ──
act_col1, act_col2, act_col3, act_col4 = st.columns([1, 1, 1, 1.2])
with act_col1:
    do_verify = st.button("✓ Verify last answer", use_container_width=True)
with act_col2:
    do_fit = st.button("📋 Summarize fit", use_container_width=True)
with act_col3:
    do_outreach = st.button("✉️ Draft outreach", use_container_width=True)
with act_col4:
    st.caption("Actions use your accumulated recruiter context + latest evidence.")

# ── Proof Snippets Expander ──
with st.expander("📎 Proof snippets (last retrieval)", expanded=False):
    if not st.session_state.last_proof:
        st.markdown("<span style='color:#5A6478; font-size:0.85rem;'>No proof snippets yet. Ask a question first.</span>", unsafe_allow_html=True)
    else:
        for label, snip in st.session_state.last_proof[:10]:
            st.markdown(
                f"<div class='proof-label'>{label}</div>"
                f"<div class='proof-text'>{snip}</div>",
                unsafe_allow_html=True,
            )

# ── Debug Expander ──
with st.expander("🔍 Debug (standalone retrieval query)", expanded=False):
    if st.session_state.last_standalone_query:
        st.code(st.session_state.last_standalone_query, language="text")
    else:
        st.markdown("<span style='color:#5A6478; font-size:0.85rem;'>No retrieval query yet.</span>", unsafe_allow_html=True)

# ── Handle Action Buttons ──
if do_verify:
    if not st.session_state.messages:
        st.toast("No messages yet.", icon="⚠️")
    else:
        last_assistant = None
        for m in reversed(st.session_state.messages):
            if (m.get("role") or "").lower() == "assistant":
                last_assistant = (m.get("content") or "").strip()
                break
        if not last_assistant:
            st.toast("No assistant answer to verify.", icon="⚠️")
        else:
            with st.chat_message("assistant"):
                verify_prompt = "Verify the last assistant answer:\n\n" + last_assistant
                answer = run_turn(verify_prompt, action_mode="verify")
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

if do_fit:
    with st.chat_message("assistant"):
        fit_prompt = "Summarize fit using the recruiter context for role/domain/location constraints."
        answer = run_turn(fit_prompt, action_mode="fit")
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

if do_outreach:
    with st.chat_message("assistant"):
        outreach_prompt = "Draft an outreach message using the recruiter context."
        answer = run_turn(outreach_prompt, action_mode="outreach")
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Normal Chat Flow ──
user_input = st.chat_input("Ask about systems, portfolio, experience, or domain expertise…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        answer = run_turn(user_input, action_mode="chat")
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
