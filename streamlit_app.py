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
from typing import List, Dict, Tuple, Optional

import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

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

LOGO_PATH = first_existing([
    os.path.join("assets", "logo.png"), os.path.join("assets", "logo.jpg"),
    os.path.join("assets", "logo.jpeg"), "logo.png", "logo.jpg", "logo.jpeg",
])
HEADSHOT_PATH = first_existing([
    os.path.join("assets", "headshot.png"), os.path.join("assets", "headshot.jpg"),
    os.path.join("assets", "headshot.jpeg"), "headshot.png", "headshot.jpg", "headshot.jpeg",
])
LINKEDIN_URL = "https://www.linkedin.com/in/stephendietrich-kolokouris/"


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG + STYLES
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Stephen Dietrich-Kolokouris",
    page_icon="◆",
    layout="wide",
)

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');
:root {
  --bg:         #0B0F19;
  --bg-surface: #101624;
  --bg-card:    #131A2B;
  --border:     rgba(148,163,184,0.08);
  --border-lit: rgba(148,163,184,0.15);
  --txt:        #D8DEE9;
  --txt-dim:    #8B95A8;
  --txt-faint:  #555F73;
  --accent:     #C9A84C;
  --font:       'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  --radius:     8px;
}

html, body, [class*="stApp"] {
  background: var(--bg) !important;
  color: var(--txt) !important;
  font-family: var(--font) !important;
}
.main .block-container {
  padding-top: 1rem !important;
  padding-bottom: 2rem !important;
  max-width: 880px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: #080C15 !important;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
section[data-testid="stSidebar"] [data-testid="stMarkdown"] p {
  font-size: 0.88rem; color: var(--txt-dim);
}
section[data-testid="stSidebar"] h3 {
  font-size: 1.02rem !important; color: var(--txt) !important;
  font-weight: 600 !important; margin-bottom: 0 !important;
}

/* Header */
.hdr {
  border-bottom: 1px solid var(--border-lit);
  padding-bottom: 14px;
  margin-bottom: 16px;
}
.hdr-name {
  font-size: 1.35rem; font-weight: 700; color: #fff;
  margin: 0; line-height: 1.3; letter-spacing: -0.01em;
}
.hdr-role {
  font-size: 0.84rem; color: var(--txt-dim);
  font-weight: 400; letter-spacing: 0.02em;
  margin: 4px 0 0;
}

/* Chat — no borders, subtle background shift only */
[data-testid="stChatMessage"] {
  border: none !important;
  border-radius: var(--radius) !important;
  background: transparent !important;
  padding: 10px 4px !important;
  margin-bottom: 2px !important;
  font-family: var(--font) !important;
}
[data-testid="stChatMessage"] p {
  font-size: 0.9rem !important; line-height: 1.65 !important;
  color: var(--txt) !important;
}

/* Chat input */
[data-testid="stChatInput"] {
  border-color: var(--border-lit) !important;
  background: var(--bg-surface) !important;
}
[data-testid="stChatInput"] textarea {
  font-family: var(--font) !important;
  font-size: 0.88rem !important;
  color: var(--txt) !important;
}

/* Buttons */
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
.stLinkButton > a {
  color: var(--accent) !important;
  border-color: var(--border-lit) !important;
  font-family: var(--font) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.05); border-radius: 3px; }

/* Responsive */
@media (max-width: 768px) {
  .hdr-name { font-size: 1.15rem; }
  .main .block-container { max-width: 100%; }
}
</style>
""", unsafe_allow_html=True)


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
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
_SESSION_DEFAULTS = {
    "messages": [],
    "personal_mode": True,
    "pinned_opening": True,
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

    return answer


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    if safe_exists(HEADSHOT_PATH):
        st.image(HEADSHOT_PATH, use_container_width=True)
    elif safe_exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)

    st.markdown("### Stephen Dietrich-Kolokouris, PhD")
    st.caption("CCIE · Cybersecurity · AI/ML Systems · Data Engineering")
    st.link_button("🔗  LinkedIn", LINKEDIN_URL, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hdr">
  <p class="hdr-name">Stephen Dietrich-Kolokouris, PhD</p>
  <p class="hdr-role">Cybersecurity · AI/ML Systems · Data Engineering · Strategic Analysis</p>
</div>
""", unsafe_allow_html=True)

# ── Pinned Opening ──
if st.session_state.pinned_opening and not st.session_state.messages:
    pinned = (
        "Good to meet you. Tell me a bit about the role you're looking to fill "
        "and I'll walk you through how Stephen's experience lines up."
    )
    st.session_state.messages.append({"role": "assistant", "content": pinned})

# ── Chat History ──
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ── Chat Input ──
user_input = st.chat_input("Type a question…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        answer = run_turn(user_input, action_mode="chat")
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
