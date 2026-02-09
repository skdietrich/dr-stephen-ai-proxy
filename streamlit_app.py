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

try:
    from scoring import score_overall
    from mitigations import tier_from_score, mitigation_playbook
    SCORING_ENABLED = True
except Exception:
    SCORING_ENABLED = False

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

/* ── Sidebar ─────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: #090D16 !important;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
section[data-testid="stSidebar"] [data-testid="stMarkdown"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdown"] li {
  font-size: 0.88rem; color: var(--txt-dim);
}
section[data-testid="stSidebar"] h3 {
  font-size: 1.05rem !important; color: var(--txt) !important;
  margin-bottom: 2px !important;
}

/* ── Header ──────────────────────────────────────────────── */
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

/* ── Chat ────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--bg-card) !important;
  margin-bottom: 6px !important;
  font-family: var(--font) !important;
}
[data-testid="stChatMessage"] p {
  font-size: 0.9rem !important; line-height: 1.6 !important;
}
[data-testid="stChatInput"] {
  border-color: var(--border-lit) !important;
  background: var(--bg-surface) !important;
}
[data-testid="stChatInput"] textarea {
  font-family: var(--font) !important;
  font-size: 0.88rem !important;
  color: var(--txt) !important;
}

/* ── Source citations (subtle) ───────────────────────────── */
.src-line {
  font-size: 0.74rem;
  color: var(--txt-faint);
  margin-top: 6px;
  padding-top: 6px;
  border-top: 1px solid var(--border);
  font-style: italic;
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
  font-family: var(--font) !important;
  font-size: 0.82rem !important;
  border-color: var(--border-lit) !important;
  color: var(--txt-dim) !important;
}
.stButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}
.stButton > button[kind="primary"],
.stDownloadButton > button {
  background: var(--accent) !important;
  color: #0B0F19 !important;
  font-weight: 600 !important;
  border: none !important;
}
.stLinkButton > a {
  color: var(--accent) !important;
  border-color: var(--border-lit) !important;
}

/* ── Expanders ───────────────────────────────────────────── */
[data-testid="stExpander"],
[data-testid="stExpander"] > details,
[data-testid="stExpander"] details {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--bg-card) !important;
}

/* ── Scrollbar ───────────────────────────────────────────── */
/* Firefox */
* { scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.18) transparent; }
/* Chromium / WebKit */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.10); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.16); }

/* ── Responsive ──────────────────────────────────────────── */
@media (max-width: 768px) {
  .hdr-name { font-size: 1.2rem !important; }
}


# ═══════════════════════════════════════════════════════════════════════════════
# LLM + EMBEDDINGS
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
# FAISS + SHA-256 MANIFEST
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
    return {"files": sorted(files, key=lambda x: x["path"])}

def _manifest_changed(new_manifest: dict) -> bool:
    if not os.path.exists(MANIFEST_PATH):
        return True
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.loads(f.read()) != new_manifest
    except Exception:
        return True

def load_or_build_faiss() -> FAISS:
    if not os.path.exists("data"):
        st.error("Missing `/data` directory — add your PDFs there.")
        st.stop()

    embeddings = init_embeddings()
    new_manifest = _build_manifest("data")

    if os.path.isdir(FAISS_DIR) and not _manifest_changed(new_manifest):
        try:
            return FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.warning(f"Index reload failed, rebuilding. ({e})")

    loader = PyPDFDirectoryLoader("data/")
    docs = loader.load()
    if not docs:
        st.error("No PDFs found in `/data`.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=160)
    chunks = splitter.split_documents(docs)

    with st.status("Building index…", expanded=False) as status:
        vs = FAISS.from_documents(chunks, embeddings)
        status.update(label="Ready", state="complete")

    try:
        os.makedirs(FAISS_DIR, exist_ok=True)
        vs.save_local(FAISS_DIR)
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            f.write(json.dumps(new_manifest, indent=2))
    except Exception:
        pass

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
# EVIDENCE PACK
# ═══════════════════════════════════════════════════════════════════════════════
def _page_label(meta: dict) -> Optional[str]:
    for key in ("page", "page_number", "loc.page_number"):
        if key in meta:
            try:
                return f"p.{int(meta[key]) + 1}"
            except Exception:
                pass
    return None

def format_evidence_pack(docs) -> Tuple[str, List[str], List[str]]:
    seen = set()
    labels: List[str] = []
    files_only: List[str] = []
    parts: List[str] = []

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

        pack_text = text[:2200].rstrip() + "…" if len(text) > 2200 else text
        parts.append(f"[SOURCE: {label}]\n{pack_text}")
        labels.append(label)
        files_only.append(src)

    return "\n\n".join(parts), labels, sorted(set(files_only))


# ═══════════════════════════════════════════════════════════════════════════════
# RECRUITER CONTEXT (silent — works in background)
# ═══════════════════════════════════════════════════════════════════════════════
_EMPTY_RECRUITER_STATE = {
    "target_roles": [], "domains": [], "location": None,
    "onsite_remote": None, "must_haves": [], "nice_to_haves": [],
    "dealbreakers": [],
}

def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items or []:
        x = (x or "").strip()
        if x and x.lower() not in seen:
            seen.add(x.lower())
            out.append(x)
    return out

def extract_recruiter_constraints(llm: ChatOpenAI, user_message: str) -> dict:
    prompt = (
        "Extract recruiter constraints from the message if present.\n"
        "Return JSON only matching this schema:\n"
        f"{json.dumps(_EMPTY_RECRUITER_STATE)}\n\n"
        "If not mentioned, use empty lists or null.\n"
        "onsite_remote: onsite | hybrid | remote | null.\n"
        "Keep items short.\n\n"
        f"MESSAGE:\n{user_message}\n\nJSON:"
    )
    try:
        out = llm.invoke(prompt)
        text = (getattr(out, "content", None) or str(out)).strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("Constraint extraction failed: %s", e)
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
# QUERY REWRITE
# ═══════════════════════════════════════════════════════════════════════════════
def rewrite_to_standalone(
    llm: ChatOpenAI, chat_history: List[Dict],
    user_input: str, recruiter_state: dict, max_turns: int = 8,
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
        "Rewrite the user's last message into a standalone search query about "
        "Dr. Stephen Dietrich-Kolokouris. Resolve pronouns using history and context. "
        "Don't add facts. Keep it short.\n\n"
        f"CONTEXT:\n{state_text}\n\n"
        f"HISTORY:\n{history_text}\n\n"
        f"MESSAGE:\n{user_input}\n\nQUERY:"
    )

    try:
        out = llm.invoke(prompt)
        text = (getattr(out, "content", None) or str(out)).strip()
        return text if text else user_input
    except Exception as e:
        logger.warning("Rewrite failed: %s", e)
        return user_input


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
def build_system_prompt(
    personal_mode: bool, recruiter_state: dict,
    evidence_pack: str, vendor_block: str, action_mode: str,
) -> str:
    state_text = json.dumps(recruiter_state or {}, ensure_ascii=False)

    tone = (
        "Include brief career context and lessons learned when supported by the evidence. "
        "Be conversational but precise."
    ) if personal_mode else (
        "Be direct and implementation-focused. Lead with specifics."
    )

    action_map = {
        "verify": (
            "Review the previous answer and produce:\n"
            "1) Claims that are supported (cite the source)\n"
            "2) Claims that are not supported by documentation on file\n"
            "3) What documentation would be needed to fill gaps\n"
        ),
        "fit": (
            "Using the recruiter context, produce:\n"
            "- A fit summary (3–5 sentences)\n"
            "- Key strengths backed by documentation (cite sources)\n"
            "- Any gaps or risks (note when something isn't documented)\n"
            "- 2–3 suggested follow-up questions\n"
        ),
        "outreach": (
            "Draft a professional outreach message (100–160 words) using the recruiter context. "
            "Only include claims backed by documentation. "
            "If key details are missing (role, location), ask one clarifying question at the end.\n"
        ),
    }
    action_text = action_map.get(action_mode, (
        "Answer the question professionally. "
        "If something isn't covered in the documentation, say so plainly.\n"
    ))

    return (
        "You are a professional Q&A assistant representing Dr. Stephen Dietrich-Kolokouris. "
        "You answer questions based strictly on the documentation provided below.\n\n"
        "Rules:\n"
        "- Only use information from the DOCUMENTATION section.\n"
        "- Never invent facts, credentials, dates, or employers.\n"
        "- If something isn't documented, say \"That's not covered in my current documentation\" "
        "and suggest what would help.\n"
        "- Don't include URLs or external references.\n"
        "- Write like a knowledgeable colleague, not a chatbot.\n"
        "- Don't use phrases like \"based on the evidence pack\" or \"according to the corpus\" — "
        "just state the information naturally.\n\n"
        f"Tone: {tone}\n\n"
        f"Task: {action_text}\n"
        f"Recruiter context: {state_text}\n\n"
        f"DOCUMENTATION:\n{evidence_pack}{vendor_block}"
    )

def build_vendor_block(vendor_ctx: Optional[dict]) -> str:
    if not vendor_ctx:
        return ""
    return (
        "\n\nVendor Context (deterministic scoring):\n"
        f"- Vendor: {vendor_ctx.get('vendor_name')}\n"
        f"- Component: {vendor_ctx.get('product_or_component')}\n"
        f"- Class: {vendor_ctx.get('component_class')}\n"
        f"- Origin: {vendor_ctx.get('origin_jurisdiction')}\n"
        f"- Criticality: {vendor_ctx.get('criticality')}\n"
        f"- Tier: {vendor_ctx.get('tier')}\n"
        f"- Scores: REE={vendor_ctx.get('ree_risk')}, FW={vendor_ctx.get('firmware_risk')}, "
        f"Overall={vendor_ctx.get('overall_risk')}\n"
        "Mitigations:\n"
        + "\n".join(f"- {m}" for m in vendor_ctx.get("mitigations", []))
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PDF EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
def _sanitize_for_reportlab(text: str) -> str:
    if not text:
        return text
    text = text.replace("\r", "")
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

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
    left, right, top, bottom = 0.85 * inch, 0.85 * inch, 0.85 * inch, 0.75 * inch
    y = height - top

    def new_page():
        nonlocal y
        c.showPage()
        y = height - top

    c.setFont("Helvetica-Bold", 14)
    c.drawString(left, y, _sanitize_for_reportlab(title))
    y -= 18
    c.setFont("Helvetica", 9)
    c.drawString(left, y, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
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
        c.drawString(left, y, "Sources referenced")
        y -= 16
        c.setFont("Helvetica", 10)
        for f in sorted(set(evidence_files)):
            if y < bottom + 14:
                new_page()
                c.setFont("Helvetica", 10)
            c.drawString(left, y, f"  {f}")
            y -= 12

    c.save()
    buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
_SESSION_DEFAULTS = {
    "messages": [],
    "personal_mode": False,
    "pinned_opening": True,
    "qa_evidence_files": [],
    "selected_vendor_context": None,
    "recruiter_state": _EMPTY_RECRUITER_STATE.copy(),
}
for k, v in _SESSION_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════════
# CORE: run_turn()
# ═══════════════════════════════════════════════════════════════════════════════
def run_turn(user_text: str, action_mode: str = "chat") -> str:
    llm = init_llm()

    # Silent recruiter context extraction
    new_bits = extract_recruiter_constraints(llm, user_text)
    update_recruiter_state(new_bits)

    # Context-aware rewrite
    standalone_query = rewrite_to_standalone(
        llm, st.session_state.messages, user_text,
        st.session_state.recruiter_state, max_turns=8,
    )

    # Retrieve
    docs = retriever.invoke(standalone_query)
    evidence_pack, evidence_labels, evidence_files = format_evidence_pack(docs)

    # Track for PDF export
    if evidence_files:
        existing = set(st.session_state.get("qa_evidence_files", []) or [])
        st.session_state.qa_evidence_files = sorted(existing.union(set(evidence_files)))

    # Build prompt + get answer
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
        answer = f"Sorry, I hit an error processing that. ({e})"

    answer = enforce_no_external_refs(answer)

    # Subtle source line — only file names, no page numbers, no bold label
    if evidence_labels:
        file_names = sorted(set(
            re.split(r"\s+\(p\.\d+\)$", lbl)[0] for lbl in evidence_labels
        ))
        if file_names:
            sources = ", ".join(file_names)
            answer += f'\n\n<div class="src-line">Sources: {sources}</div>'

    return answer


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    if safe_exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    if safe_exists(HEADSHOT_PATH):
        st.image(HEADSHOT_PATH, width=180)

    st.markdown("### Dr. Stephen Dietrich-Kolokouris")
    st.caption("PhD · CCIE · Cybersecurity · AI Systems · Data Engineering")
    st.markdown("---")
    st.link_button("🔗  LinkedIn", LINKEDIN_URL, use_container_width=True)
    st.markdown("---")

    st.session_state.personal_mode = st.toggle(
        "Conversational style",
        value=st.session_state.personal_mode,
        help="Adds career narrative and context to answers.",
    )

    with st.expander("About", expanded=False):
        if safe_exists(ABOUT_MD_PATH):
            st.markdown(_read_file_safe(ABOUT_MD_PATH))
        else:
            st.markdown("_Career narrative available when `assets/about_stephen.md` is added._")

    with st.expander("Approach & Methodology", expanded=False):
        if safe_exists(THINK_MD_PATH):
            st.markdown(_read_file_safe(THINK_MD_PATH))
        else:
            st.markdown("_Decision-making approach available when `assets/how_i_think.md` is added._")

    with st.expander("Publications", expanded=False):
        if safe_exists(PUBS_CSV_PATH):
            try:
                st.dataframe(pd.read_csv(PUBS_CSV_PATH), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Could not load publications: {e}")
        else:
            st.markdown("_Publication list available when `assets/publications.csv` is added._")

    if SCORING_ENABLED:
        with st.expander("Supply Chain Risk Scoring", expanded=False):
            weight_fw = st.slider("Firmware weight", 0.0, 1.0, 0.55, 0.05, key="weight_fw")
            uploaded_csv = st.file_uploader("Upload vendor CSV", type=["csv"], key="vendor_csv_uploader")
            if uploaded_csv is not None:
                try:
                    df = pd.read_csv(uploaded_csv)
                except Exception as e:
                    st.error(f"Could not read CSV: {e}")
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
                        st.error(f"Missing columns: {missing}")
                    else:
                        scores = df.apply(lambda r: pd.Series(score_overall(r.to_dict(), weight_fw=weight_fw)), axis=1)
                        out_df = pd.concat([df, scores], axis=1)
                        out_df["tier"] = out_df["overall_risk"].apply(tier_from_score)
                        st.dataframe(out_df.sort_values("overall_risk", ascending=False).head(10), use_container_width=True)
                        idx = st.number_input("Row", min_value=0, max_value=max(0, len(out_df)-1), value=0, step=1, key="vendor_row_idx")
                        row = out_df.iloc[int(idx)].to_dict()
                        if st.button("Load into conversation", key="use_vendor_ctx"):
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
                            st.success("Loaded.")

    with st.expander("Export conversation (PDF)", expanded=False):
        if not REPORTLAB_OK:
            st.caption("PDF export requires `reportlab` in requirements.txt")
        else:
            if st.button("Download PDF transcript", type="primary"):
                export_msgs = [
                    {"role": (m.get("role") or "").lower(), "content": m.get("content") or ""}
                    for m in st.session_state.messages
                    if (m.get("role") or "").lower() in ("user", "assistant")
                ]
                evidence_files = sorted(set(st.session_state.get("qa_evidence_files", []) or []))
                pdf_bytes = build_qa_pdf_bytes(
                    "Q&A — Dr. Stephen Dietrich-Kolokouris",
                    export_msgs, evidence_files,
                )
                st.download_button(
                    "Save PDF", data=pdf_bytes,
                    file_name="QA_Stephen_Dietrich_Kolokouris.pdf",
                    mime="application/pdf",
                )

    # Admin (collapsed, out of the way)
    with st.expander("Admin", expanded=False):
        if st.button("Rebuild index"):
            try:
                if os.path.exists(MANIFEST_PATH):
                    os.remove(MANIFEST_PATH)
            except Exception:
                pass
            init_retriever.clear()
            st.rerun()
        if st.button("Clear conversation context"):
            st.session_state.recruiter_state = _EMPTY_RECRUITER_STATE.copy()
            st.success("Context cleared.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

# ── Header (clean, professional — no badges, no status strip, no cards) ──
st.markdown("""
<div class="hdr">
  <p class="hdr-name">Dr. Stephen Dietrich-Kolokouris, PhD</p>
  <p class="hdr-role">Cybersecurity · AI/ML Systems · Data Engineering · Strategic Analysis</p>
  <p class="hdr-bio">
    Ask me anything about Stephen's background, technical capabilities, project experience,
    or fit for a role you're hiring for. Answers are based on his published work and project documentation.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Pinned Opening ──
if st.session_state.pinned_opening and not st.session_state.messages:
    pinned = (
        "Hi — thanks for stopping by. I can answer questions about Stephen's "
        "background, skills, and project experience.\n\n"
        "A few ways to get started:\n\n"
        "1. Tell me what role you're hiring for and I'll walk you through relevant experience\n"
        "2. Ask about a specific domain — security architecture, RAG systems, supply chain, IR\n"
        "3. Describe what success looks like in 90 days and I'll map it to his track record"
    )
    st.session_state.messages.append({"role": "assistant", "content": pinned})

# ── Chat History ──
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"], unsafe_allow_html=True)

# ── Action Bar (human labels, no emoji overload) ──
col_a, col_b, col_c = st.columns(3)
with col_a:
    do_verify = st.button("Check sources", use_container_width=True)
with col_b:
    do_fit = st.button("Summarize fit", use_container_width=True)
with col_c:
    do_outreach = st.button("Draft message", use_container_width=True)

# ── Handle Action Buttons ──
if do_verify:
    last_assistant = None
    for m in reversed(st.session_state.messages):
        if (m.get("role") or "").lower() == "assistant":
            last_assistant = (m.get("content") or "").strip()
            break
    if last_assistant:
        with st.chat_message("assistant"):
            answer = run_turn("Verify the previous answer:\n\n" + last_assistant, action_mode="verify")
            st.markdown(answer, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.toast("Nothing to check yet.", icon="💬")

if do_fit:
    with st.chat_message("assistant"):
        answer = run_turn(
            "Summarize fit for the role and constraints discussed so far.",
            action_mode="fit",
        )
        st.markdown(answer, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": answer})

if do_outreach:
    with st.chat_message("assistant"):
        answer = run_turn(
            "Draft an outreach message based on what we've discussed.",
            action_mode="outreach",
        )
        st.markdown(answer, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Normal Chat ──
user_input = st.chat_input("Ask about skills, experience, projects, or role fit…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        answer = run_turn(user_input, action_mode="chat")
        st.markdown(answer, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": answer})

