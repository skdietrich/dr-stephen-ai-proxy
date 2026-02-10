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
from langchain_community.document_loaders import PyPDFDirectoryLoader
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

try:
    from scoring import score_overall
    from mitigations import tier_from_score, mitigation_playbook
    SCORING_ENABLED = True
except Exception:
    SCORING_ENABLED = False

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
        window.parent.document.querySelector('.main').scrollTo({
            top: 0,
            left: 0,
            behavior: 'auto'
        });
    </script>
    """,
    height=0,
)


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
            "Response blocked: external citation or link pattern detected. "
            "This system can only cite the loaded corpus."
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
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

LOGO_PATH = first_existing([
    os.path.join("assets", "logo.png"), os.path.join("assets", "logo.jpg"),
    "logo.png", "logo.jpg",
])
HEADSHOT_PATH = first_existing([
    os.path.join("assets", "headshot.png"), os.path.join("assets", "headshot.jpg"),
    os.path.join("assets", "headshot.jpeg"),
    "headshot.png", "headshot.jpg", "headshot.jpeg",
])
ABOUT_MD_PATH = first_existing([os.path.join("assets", "about_stephen.md"), "about_stephen.md"])
THINK_MD_PATH = first_existing([os.path.join("assets", "how_i_think.md"), "how_i_think.md"])
PUBS_CSV_PATH = first_existing([os.path.join("assets", "publications.csv"), "publications.csv"])
LINKEDIN_URL = "https://www.linkedin.com/in/stephendietrich-kolokouris/"


# ═══════════════════════════════════════════════════════════════════════════════
# LLM + EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════════
def init_llm() -> ChatOpenAI:
    try:
        return ChatOpenAI(model="gpt-4o", temperature=0, api_key=st.secrets["OPENAI_API_KEY"])
    except TypeError:
        return ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])

def init_llm_mini() -> ChatOpenAI:
    try:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=st.secrets["OPENAI_API_KEY"])
    except TypeError:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])

def init_embeddings() -> OpenAIEmbeddings:
    try:
        return OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    except TypeError:
        return OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])


# ═══════════════════════════════════════════════════════════════════════════════
# FAISS PERSISTENCE + SHA-256 MANIFEST
# ═══════════════════════════════════════════════════════════════════════════════
FAISS_DIR = "faiss_index"
MANIFEST_PATH = os.path.join(FAISS_DIR, "manifest.json")

# One-time rebuild: delete stale index, then leave a flag so it doesn't repeat
_REBUILD_FLAG = os.path.join(FAISS_DIR, ".rebuilt_v3")
if os.path.isdir(FAISS_DIR) and not os.path.exists(_REBUILD_FLAG):
    import shutil
    shutil.rmtree(FAISS_DIR, ignore_errors=True)

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

    if os.path.isdir(FAISS_DIR) and not _manifest_changed(new_manifest):
        try:
            return FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.warning(f"FAISS index load failed; rebuilding. Reason: {e}")
            import shutil
            shutil.rmtree(FAISS_DIR, ignore_errors=True)

    from langchain_community.document_loaders import PyPDFLoader
    docs = []
    skipped = []
    for root, _, fnames in os.walk("data"):
        for fn in sorted(fnames):
            if not fn.lower().endswith(".pdf"):
                continue
            fpath = os.path.join(root, fn)
            try:
                docs.extend(PyPDFLoader(fpath).load())
            except Exception as e:
                skipped.append(fn)
                logger.warning("Skipping corrupt PDF %s: %s", fn, e)
    if skipped:
        st.warning(f"Skipped {len(skipped)} corrupt PDF(s): {', '.join(skipped)}")
    if not docs:
        st.error("No readable documents found in `/data`.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=160)
    chunks = splitter.split_documents(docs)

    vs = FAISS.from_documents(chunks, embeddings)

    os.makedirs(FAISS_DIR, exist_ok=True)
    try:
        vs.save_local(FAISS_DIR)
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(new_manifest, f, indent=2)
        with open(_REBUILD_FLAG, "w") as f:
            f.write("ok")
    except Exception as e:
        st.warning(f"FAISS index could not be saved (non-fatal): {e}")

    return vs

def _corpus_fingerprint() -> str:
    m = _build_manifest("data") if os.path.exists("data") else {}
    return hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()[:16]

@st.cache_resource
def init_retriever(corpus_fingerprint: str = ""):
    vs = load_or_build_faiss()
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 15, "fetch_k": 40, "lambda_mult": 0.4},
    )

retriever = init_retriever(corpus_fingerprint=_corpus_fingerprint())


# ═══════════════════════════════════════════════════════════════════════════════
# EVIDENCE PACK + CITATIONS
# ═══════════════════════════════════════════════════════════════════════════════
def _page_label(meta: dict) -> Optional[str]:
    for key in ("page", "page_number", "loc.page_number"):
        if key in meta:
            try:
                p = int(meta[key])
                return f"p.{p+1}"
            except Exception:
                pass
    return None

def format_evidence_pack(docs) -> Tuple[str, List[str], List[str]]:
    seen = set()
    labels: List[str] = []
    files_only: List[str] = []
    parts: List[str] = []

    for d in docs:
        src_full = d.metadata.get("source", "") or ""
        src = os.path.basename(src_full) if src_full else "unknown"
        page = _page_label(d.metadata or {})
        label = f"{src} ({page})" if page else src

        uniq = (src, page, (d.page_content or "")[:80])
        if uniq in seen:
            continue
        seen.add(uniq)

        text = (d.page_content or "").strip()
        if not text:
            continue
        if len(text) > 3000:
            text = text[:3000].rstrip() + "..."

        parts.append(f"[SOURCE: {label}]\n{text}")
        labels.append(label)
        files_only.append(src)

    return "\n\n".join(parts), labels, sorted(set(files_only))


# ═══════════════════════════════════════════════════════════════════════════════
# RECRUITER CONTEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════
_EMPTY_RECRUITER_STATE = {
    "target_roles": [], "domains": [], "location": None,
    "onsite_remote": None, "must_haves": [], "nice_to_haves": [],
    "dealbreakers": [],
}

def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
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

def rewrite_to_standalone(
    llm: ChatOpenAI, chat_history: List[Dict],
    user_input: str, recruiter_state: dict, max_turns: int = 8,
) -> str:
    hist_lines: List[str] = []
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
        "Don't add facts.\n\n"
        f"RECRUITER CONTEXT: {state_text}\n\n"
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
# SYSTEM PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
def build_system_prompt(
    personal_mode: bool,
    recruiter_state: dict,
    evidence_pack: str,
    vendor_block: str,
    action_mode: str = "chat",
) -> str:
    state_text = json.dumps(recruiter_state or {}, ensure_ascii=False)

    if personal_mode:
        tone_line = (
            "TONE MODE: Conversational.\n"
            "- You may include brief career context and lessons learned ONLY if supported by the evidence.\n"
            "- Keep it recruiter-friendly and precise. No hype.\n"
            "- Don't use phrases like 'based on the evidence pack' or 'according to the corpus' -- "
            "just state the information naturally.\n"
        )
    else:
        tone_line = (
            "TONE MODE: Technical-only.\n"
            "- Direct, systems-focused, implementation-oriented.\n"
            "- Don't reference the evidence system -- just state information naturally.\n"
        )

    action_map = {
        "verify": (
            "TASK MODE: VERIFY.\n"
            "- Cross-check claims in the previous answer against the evidence.\n"
            "- Flag any claim not directly supported. Be honest.\n"
        ),
        "fit": (
            "TASK MODE: FIT SUMMARY.\n"
            "- Using the recruiter context JSON, produce a structured fit summary:\n"
            "  Strengths, Gaps or unknowns, Suggested next questions.\n"
            "- Use only evidence-backed claims.\n"
        ),
        "outreach": (
            "TASK MODE: OUTREACH.\n"
            "- Draft a recruiter outreach message (100-160 words) based on recruiter context.\n"
            "- Use only evidence-backed claims.\n"
            "- If key context is missing (role/location), ask ONE short question at the end.\n"
        ),
    }
    action_instructions = action_map.get(action_mode, (
        "TASK MODE: CHAT.\n"
        "- Answer the question in a recruiter-grade, professional manner.\n"
        "- If the evidence pack lacks detail, acknowledge the topic and suggest a more specific question.\n"
    ))

    return (
        "You are a professional assistant representing Dr. Stephen Dietrich-Kolokouris.\n\n"
        "CORPUS OVERVIEW (topics covered in the full documentation):\n"
        "- Published books: 'The American Paranormal' (2025, consciousness/spiritualism research),\n"
        "  'Chicago Ripper Crew: Reboot' (true crime), 'Behind the Mask: Hitler the Socialite'\n"
        "  (historical analysis), plus four additional published works (seven total).\n"
        "- Academic: PhD from Goethe University Frankfurt (History), German language fluency.\n"
        "- Research papers: 'Silent Weapons: Sleeper Malware and the Future of Cyber Warfare',\n"
        "  'Embedded Persistent Threats and the 2026 Venezuela Cyber-Kinetic Operation',\n"
        "  'AI Chatbots as National Security Risks' (WarSim), NAMECOMMS whitepaper.\n"
        "- Cybersecurity: CCIE certification, penetration testing, supply chain risk, IR.\n"
        "- Intelligence: Former CIA contractor, counterterrorism (Al-Qaeda/ISIS theaters).\n"
        "- AI/ML: Production RAG systems, LangChain, FAISS, agent frameworks.\n"
        "- Compliance: NIST CSF, 800-53, 800-171, NERC CIP, ISO 27001, FISMA, FedRAMP.\n"
        "- Media: Fox 4 Dallas appearances, YouTube content, Skinwalker Ranch fieldwork.\n\n"
        "MANDATORY CONSTRAINTS:\n"
        "1) Use ONLY the EVIDENCE PACK below (and vendor block if present).\n"
        "2) Do NOT invent facts, dates, employers, credentials, or project details.\n"
        "3) If the evidence pack lacks detail on a topic listed in the CORPUS OVERVIEW,\n"
        "   acknowledge the topic exists and invite a more specific follow-up question.\n"
        "4) Do NOT include URLs or bibliography headings.\n"
        "5) Never reference 'the corpus', 'evidence pack', or 'the system' -- speak naturally.\n\n"
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
        f"- Scores: REE={vendor_ctx.get('ree_risk')}, FW={vendor_ctx.get('firmware_risk')}, "
        f"Overall={vendor_ctx.get('overall_risk')}\n"
        "Mitigation priorities (deterministic):\n"
        + "\n".join(f"- {m}" for m in vendor_ctx.get("mitigations", []))
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PDF EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
def _sanitize_for_reportlab(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

def _wrap_text_lines(text: str, max_chars: int = 95) -> List[str]:
    lines: List[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            continue
        while len(paragraph) > max_chars:
            split_at = paragraph.rfind(" ", 0, max_chars)
            if split_at <= 0:
                split_at = max_chars
            lines.append(paragraph[:split_at])
            paragraph = paragraph[split_at:].strip()
        lines.append(paragraph)
    return lines

def build_qa_pdf_bytes(messages: List[Dict], evidence_files: List[str]) -> Optional[bytes]:
    if not REPORTLAB_OK:
        return None
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    w, h = LETTER
    margin = 0.75 * inch
    y = h - margin

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "Q&A Transcript -- Dr. Stephen Dietrich-Kolokouris")
    y -= 24
    c.setFont("Helvetica", 9)
    c.drawString(margin, y, f"Evidence files: {', '.join(evidence_files) if evidence_files else 'None'}")
    y -= 20
    c.line(margin, y, w - margin, y)
    y -= 16

    for m in messages:
        role = (m.get("role") or "").upper()
        content = _sanitize_for_reportlab(m.get("content") or "")
        if not content:
            continue

        c.setFont("Helvetica-Bold", 10)
        if y < margin + 40:
            c.showPage()
            y = h - margin
        c.drawString(margin, y, f"{role}:")
        y -= 14

        c.setFont("Helvetica", 9)
        for line in _wrap_text_lines(content, 95):
            if y < margin + 20:
                c.showPage()
                y = h - margin
                c.setFont("Helvetica", 9)
            c.drawString(margin + 10, y, line)
            y -= 12
        y -= 8

    c.save()
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# CORE: run_turn()
# ═══════════════════════════════════════════════════════════════════════════════
def run_turn(user_text: str, action_mode: str = "chat") -> str:
    llm_mini = init_llm_mini()
    llm = init_llm()

    new_bits = extract_recruiter_constraints(llm_mini, user_text)
    update_recruiter_state(new_bits)

    standalone_query = rewrite_to_standalone(
        llm_mini, st.session_state.messages, user_text,
        st.session_state.recruiter_state, max_turns=8,
    )

    docs = retriever.invoke(standalone_query)

    if not docs:
        st.warning("Retrieval returned zero chunks for this query.")

    evidence_pack, evidence_labels, evidence_files = format_evidence_pack(docs)

    if evidence_files:
        existing = set(st.session_state.get("qa_evidence_files", []) or [])
        st.session_state.qa_evidence_files = sorted(existing.union(set(evidence_files)))

    vendor_block = build_vendor_block(st.session_state.get("selected_vendor_context"))
    system_prompt = build_system_prompt(
        personal_mode=st.session_state.personal_mode,
        recruiter_state=st.session_state.recruiter_state,
        evidence_pack=evidence_pack,
        vendor_block=vendor_block,
        action_mode=action_mode,
    )

    try:
        chunks = []
        stream = llm.stream([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ])
        placeholder = st.empty()
        for chunk in stream:
            token = chunk.content or ""
            chunks.append(token)
            placeholder.markdown("".join(chunks) + " ▋", unsafe_allow_html=True)
        answer = "".join(chunks).strip()
        placeholder.empty()
    except Exception as e:
        answer = f"Sorry, I hit an error processing that. ({e})"

    answer = enforce_no_external_refs(answer)

    answer += (
        '\n\n<div style="margin-top:1.5rem;padding-top:1rem;'
        'border-top:1px solid #e2e0db;font-size:0.88rem;color:#5a5f6b;">'
        '<span style="color:#1a5c3a;font-weight:600;">▸</span> Want to discuss further? '
        '<a href="' + LINKEDIN_URL + '" target="_blank" '
        'style="color:#1a5c3a;font-weight:600;text-decoration:none;">'
        'Connect on LinkedIn →</a></div>'
    )

    return answer


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pinned_opening" not in st.session_state:
    st.session_state.pinned_opening = True
if "recruiter_state" not in st.session_state:
    st.session_state.recruiter_state = dict(_EMPTY_RECRUITER_STATE)
if "personal_mode" not in st.session_state:
    st.session_state.personal_mode = True
if "qa_evidence_files" not in st.session_state:
    st.session_state.qa_evidence_files = []


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED PREMIUM CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --ink: #0d0f12;
    --surface: #f8f7f4;
    --surface-warm: #f2f0eb;
    --accent: #1a5c3a;
    --accent-light: #e8f0ec;
    --accent-glow: #2d8a5e;
    --text-primary: #1a1d23;
    --text-secondary: #5a5f6b;
    --text-muted: #8b909e;
    --border: #e2e0db;
    --border-light: #eceae5;
    --card-bg: #ffffff;
    --gold: #c9a84c;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 16px rgba(0,0,0,0.06);
    --shadow-lg: 0 8px 32px rgba(0,0,0,0.08);
    --radius: 12px;
    --radius-sm: 8px;
    --transition: 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--surface) !important;
    font-family: 'DM Sans', -apple-system, sans-serif !important;
    color: var(--text-primary);
}
.stApp { background-color: var(--surface) !important; }

/* ── Sidebar Enhanced ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0f12 0%, #1a1f2e 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}

@media (min-width: 769px) {
    [data-testid="stSidebar"] {
        min-width: 320px !important;
        width: 320px !important;
        transform: none !important;
        position: relative !important;
    }
    [data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 320px !important;
        width: 320px !important;
        transform: none !important;
        margin-left: 0 !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
}

@media (max-width: 768px) {
    [data-testid="stSidebar"] {
        min-width: 280px !important;
        width: 85vw !important;
        max-width: 320px !important;
    }
    [data-testid="collapsedControl"] {
        display: block !important;
    }
}

[data-testid="stSidebar"] * { color: #c8cad0 !important; }
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #ffffff !important;
    font-family: 'DM Serif Display', Georgia, serif !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
    margin: 1.5rem 0 !important;
}

.sidebar-photo {
    width: 150px; height: 150px; border-radius: 50%;
    margin: 2.5rem auto 1.2rem; display: block;
    border: 4px solid rgba(74,222,128,0.35);
    box-shadow: 0 0 30px rgba(74,222,128,0.15), 0 4px 16px rgba(0,0,0,0.3);
    object-fit: cover;
    transition: var(--transition);
}

.sidebar-photo:hover {
    box-shadow: 0 0 40px rgba(74,222,128,0.25), 0 6px 24px rgba(0,0,0,0.4);
    transform: scale(1.02);
}

.sidebar-photo-placeholder {
    width: 150px; height: 150px; border-radius: 50%;
    margin: 2.5rem auto 1.2rem;
    display: flex; align-items: center; justify-content: center;
    border: 4px solid rgba(74,222,128,0.35);
    box-shadow: 0 0 30px rgba(74,222,128,0.15), 0 4px 16px rgba(0,0,0,0.3);
    background: linear-gradient(135deg, #1a2840, #0d1520);
    font-size: 3rem; font-family: 'DM Serif Display', serif;
    color: #4ade80; letter-spacing: -2px;
    transition: var(--transition);
}

.sidebar-photo-placeholder:hover {
    box-shadow: 0 0 40px rgba(74,222,128,0.25), 0 6px 24px rgba(0,0,0,0.4);
    transform: scale(1.02);
}

.sidebar-name {
    text-align: center;
    font-family: 'DM Serif Display', Georgia, serif !important;
    font-size: 1.5rem; color: #ffffff !important;
    margin-bottom: 0.2rem; letter-spacing: -0.4px;
    font-weight: 400;
}

.sidebar-title-line {
    text-align: center; font-size: 0.82rem;
    color: #9ca3af !important; letter-spacing: 0.6px;
    text-transform: uppercase; margin-bottom: 1.8rem;
    font-weight: 500;
}

.cred-row {
    display: flex; flex-wrap: wrap; gap: 8px;
    justify-content: center; margin-bottom: 1.5rem; padding: 0 0.8rem;
}
.cred-tag {
    display: inline-block; padding: 5px 12px; border-radius: 100px;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.5px;
    font-family: 'JetBrains Mono', monospace;
    transition: var(--transition);
}
.cred-tag:hover {
    transform: translateY(-1px);
}
.cred-tag.green { background: rgba(74,222,128,0.15); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
.cred-tag.gold  { background: rgba(201,168,76,0.15); color: #e5c468; border: 1px solid rgba(201,168,76,0.3); }
.cred-tag.blue  { background: rgba(96,165,250,0.15); color: #60a5fa; border: 1px solid rgba(96,165,250,0.3); }

.sb-section-title {
    font-size: 0.72rem !important; font-weight: 600 !important;
    letter-spacing: 1.4px !important; text-transform: uppercase !important;
    color: #6b7280 !important; margin-bottom: 0.8rem !important; padding-left: 3px;
}
.sb-item { 
    font-size: 0.88rem; color: #d1d5db; padding: 6px 0; line-height: 1.6;
    transition: color 0.2s ease;
}
.sb-item:hover { color: #e5e7eb; }
.sb-item strong { color: #f3f4f6 !important; font-weight: 600; }

.sb-link {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 10px 18px; border-radius: 10px;
    background: rgba(74,222,128,0.1); border: 1px solid rgba(74,222,128,0.2);
    color: #4ade80 !important; text-decoration: none !important;
    font-size: 0.88rem; font-weight: 600; transition: var(--transition);
    margin: 5px 0;
}
.sb-link:hover { 
    background: rgba(74,222,128,0.18); 
    border-color: rgba(74,222,128,0.4);
    transform: translateX(3px);
}

.link-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    border-radius: 4px;
    background: rgba(74,222,128,0.2);
    font-size: 0.75rem;
    font-weight: 700;
    font-family: 'Georgia', serif;
    font-style: italic;
}

.stat-icon {
    color: #4ade80;
    font-weight: 700;
    margin-right: 4px;
}

/* ── Main Content Enhanced ── */
.main-header { 
    max-width: 1000px; margin: 0 auto; padding: 3rem 1rem 1.5rem;
    animation: fadeUp 0.6s ease-out;
}

.main-tagline {
    font-size: 0.85rem; font-weight: 600; letter-spacing: 1px;
    text-transform: uppercase; color: var(--accent); margin-bottom: 0.8rem;
    display: flex; align-items: center; gap: 8px;
}

.tagline-icon {
    font-size: 1rem;
    color: var(--accent);
    font-weight: 700;
}

.main-greeting {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: 2.6rem; color: var(--text-primary);
    letter-spacing: -1px; line-height: 1.15; margin-bottom: 0.5rem;
    font-weight: 400;
}

.main-subtitle {
    font-size: 1.1rem; color: var(--text-secondary);
    line-height: 1.65; max-width: 700px; margin-bottom: 2.5rem;
}

.domain-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 16px; margin-bottom: 3rem; max-width: 1000px; 
    margin-left: auto; margin-right: auto;
    animation: fadeUp 0.6s ease-out 0.15s both;
}

.domain-card {
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.5rem 1.5rem;
    transition: var(--transition); position: relative; overflow: hidden;
    cursor: default;
}

.domain-card:hover {
    border-color: var(--accent); 
    box-shadow: var(--shadow-lg); 
    transform: translateY(-3px);
}

.domain-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
    border-radius: var(--radius) var(--radius) 0 0;
    transition: height 0.3s ease;
}

.domain-card:hover::before {
    height: 6px;
}

.domain-card.cyber::before    { background: linear-gradient(90deg, #1a5c3a, #2d8a5e); }
.domain-card.rag::before      { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
.domain-card.intel::before    { background: linear-gradient(90deg, #c9a84c, #e5c468); }
.domain-card.research::before { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }

.domain-icon { 
    font-size: 1.8rem; margin-bottom: 0.8rem; display: flex;
    align-items: center; justify-content: flex-start;
    transition: transform 0.3s ease;
}

.domain-icon svg {
    color: var(--accent);
    transition: all 0.3s ease;
}

.domain-card.cyber .domain-icon svg { color: #1a5c3a; }
.domain-card.rag .domain-icon svg { color: #3b82f6; }
.domain-card.intel .domain-icon svg { color: #c9a84c; }
.domain-card.research .domain-icon svg { color: #8b5cf6; }

.domain-card:hover .domain-icon {
    transform: scale(1.1);
}

.domain-card:hover .domain-icon svg {
    filter: drop-shadow(0 2px 8px currentColor);
}

.domain-label { 
    font-size: 1rem; font-weight: 700; color: var(--text-primary); 
    margin-bottom: 0.4rem; letter-spacing: -0.2px;
}

.domain-desc { 
    font-size: 0.85rem; color: var(--text-muted); line-height: 1.5;
}

.chat-section-label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 1.4px;
    text-transform: uppercase; color: var(--text-muted);
    margin-bottom: 1rem; padding-left: 3px;
    max-width: 1000px; margin-left: auto; margin-right: auto;
}

/* ── Action Buttons Enhanced ── */
.action-buttons-row {
    display: flex; gap: 12px; margin-bottom: 1.5rem;
    max-width: 1000px; margin-left: auto; margin-right: auto;
}

/* ── Chat Messages Enhanced ── */
[data-testid="stChatMessage"] {
    max-width: 1000px !important; margin-left: auto !important; margin-right: auto !important;
    border: none !important; background: transparent !important; padding: 1rem 0 !important;
}

[data-testid="stChatMessage"] p {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.98rem !important; line-height: 1.75 !important;
    color: var(--text-primary) !important;
}

/* ── Chat Input Enhanced ── */
[data-testid="stChatInput"] {
    max-width: 1000px !important; margin-left: auto !important; margin-right: auto !important;
}

[data-testid="stChatInput"] textarea {
    font-family: 'DM Sans', sans-serif !important; font-size: 0.98rem !important;
    border-radius: var(--radius) !important; border: 2px solid var(--border) !important;
    background: var(--card-bg) !important; padding: 1.1rem 1.3rem !important;
    color: var(--text-primary) !important; caret-color: var(--accent) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
    transition: var(--transition);
    box-shadow: var(--shadow-sm);
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
    -webkit-text-fill-color: var(--text-muted) !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important; 
    box-shadow: 0 0 0 4px var(--accent-light), var(--shadow-md) !important;
    outline: none !important;
}

[data-testid="stChatInput"] button {
    color: var(--accent) !important;
    transition: var(--transition);
}

[data-testid="stChatInput"] button:hover {
    transform: scale(1.1);
}

[data-testid="stChatInput"] button svg {
    fill: var(--accent) !important; stroke: var(--accent) !important;
}

/* ── Buttons Enhanced ── */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important; font-size: 0.88rem !important;
    font-weight: 600 !important; border-radius: var(--radius-sm) !important;
    border: 2px solid var(--border) !important; background: var(--card-bg) !important;
    color: var(--text-secondary) !important; padding: 0.65rem 1.2rem !important;
    transition: var(--transition);
    box-shadow: var(--shadow-sm);
}

.stButton > button:hover {
    border-color: var(--accent) !important; color: var(--accent) !important;
    background: var(--accent-light) !important;
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}

.stDownloadButton > button {
    font-family: 'DM Sans', sans-serif !important; font-size: 0.85rem !important;
    border-radius: var(--radius-sm) !important;
    border: 2px solid var(--border) !important; background: var(--card-bg) !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    transition: var(--transition);
}

.stDownloadButton > button:hover {
    border-color: var(--accent) !important;
    background: var(--accent-light) !important;
    color: var(--accent) !important;
}

/* ── Toggle Switch Enhanced ── */
[data-testid="stSidebar"] .stCheckbox {
    padding: 0.5rem 0;
}

/* ── Expanders Enhanced ── */
[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important; 
    background: rgba(255,255,255,0.03) !important;
    transition: var(--transition);
}

[data-testid="stExpander"]:hover {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(255,255,255,0.12) !important;
}

/* ── Hide chrome ── */
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Starter Chips Enhanced ── */
.chip-row { 
    display: flex; flex-wrap: wrap; gap: 10px; 
    margin: 0.8rem auto 2rem; max-width: 1000px;
}

.chip-btn { 
    background: #fff; border: 2px solid #e2e0db; border-radius: 100px;
    padding: 10px 20px; font-size: 0.88rem; color: #1a1d23; cursor: pointer;
    font-family: 'DM Sans', sans-serif; transition: var(--transition);
    font-weight: 500;
    box-shadow: var(--shadow-sm);
    position: relative;
    padding-left: 14px;
}

.chip-btn::before {
    content: '';
    position: absolute;
    left: 10px;
    top: 50%;
    transform: translateY(-50%);
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
}

.chip-btn:hover { 
    border-color: #1a5c3a; color: #1a5c3a; background: #e8f0ec;
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── Mobile Responsiveness ── */
@media (max-width: 768px) {
    .domain-grid { grid-template-columns: 1fr; gap: 12px; }
    .main-greeting { font-size: 2rem; }
    .main-subtitle { font-size: 1rem; }
    .main-header { padding: 2rem 1rem 1rem; }
    .chip-row { flex-direction: column; }
    .chip-btn { text-align: center; }
}

@media (min-width: 769px) and (max-width: 1024px) {
    .domain-grid { grid-template-columns: repeat(2, 1fr); }
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR CONTENT
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Photo ──
    if safe_exists(HEADSHOT_PATH):
        import base64 as _b64
        with open(HEADSHOT_PATH, "rb") as _img_f:
            _b64_str = _b64.b64encode(_img_f.read()).decode()
        _ext = HEADSHOT_PATH.rsplit(".", 1)[-1].lower()
        _mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(_ext, "jpeg")
        st.markdown(
            f'<img src="data:image/{_mime};base64,{_b64_str}" class="sidebar-photo" '
            f'alt="Dr. Stephen Dietrich-Kolokouris" />',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="sidebar-photo-placeholder">SDK</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-name">Dr. Stephen Dietrich-Kolokouris</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title-line">Cybersecurity Architect · AI/RAG Engineer · Intelligence Analyst</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="cred-row">
        <span class="cred-tag green">PhD</span>
        <span class="cred-tag green">CCIE</span>
        <span class="cred-tag gold">Ex-CIA Contractor</span>
        <span class="cred-tag blue">Published Author</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sb-section-title">Connect</div>', unsafe_allow_html=True)
    st.markdown(f'<a href="{LINKEDIN_URL}" target="_blank" class="sb-link"><span class="link-icon">in</span> LinkedIn Profile</a>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sb-section-title">Quick Stats</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-item"><span class="stat-icon">▸</span> <strong>20+ years</strong> in cybersecurity</div>
    <div class="sb-item"><span class="stat-icon">▸</span> <strong>7 published books</strong></div>
    <div class="sb-item"><span class="stat-icon">▸</span> <strong>PhD in History</strong> (Goethe Univ.)</div>
    <div class="sb-item"><span class="stat-icon">▸</span> <strong>CCIE certified</strong></div>
    <div class="sb-item"><span class="stat-icon">▸</span> <strong>Top Secret clearance</strong></div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sb-section-title">Core Expertise</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-item"><strong>Security Architecture</strong><br/>Infrastructure protection and incident response.</div>
    <div class="sb-item"><strong>AI / RAG Systems</strong><br/>Agentic frameworks and vector retrieval.</div>
    <div class="sb-item"><strong>Intelligence Analysis</strong><br/>Former CIA contractor operations.</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.session_state.personal_mode = st.toggle("◆ Conversational Mode", value=st.session_state.personal_mode)

    if REPORTLAB_OK and st.session_state.messages:
        pdf_bytes = build_qa_pdf_bytes(st.session_state.messages, st.session_state.get("qa_evidence_files", []))
        if pdf_bytes:
            st.download_button(label="⬇ Download Transcript (PDF)", data=pdf_bytes, file_name="conversation.pdf", mime="application/pdf", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

# ── Header ──
st.markdown("""
<div class="main-header">
    <div class="main-tagline"><span class="tagline-icon">◆</span> AI-POWERED PORTFOLIO ASSISTANT</div>
    <div class="main-greeting">Discover how Stephen's expertise aligns with your needs</div>
    <div class="main-subtitle">Description mapping qualifications with source-backed precision.</div>
</div>
""", unsafe_allow_html=True)

# ── Domain Cards ──
st.markdown("""
<div class="domain-grid">
    <div class="domain-card cyber">
        <div class="domain-icon"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg></div>
        <div class="domain-label">Security Architecture</div>
        <div class="domain-desc">Designing resilient security frameworks and infrastructure.</div>
    </div>
    <div class="domain-card rag">
        <div class="domain-icon"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg></div>
        <div class="domain-label">AI & RAG Systems</div>
        <div class="domain-desc">Building production-grade retrieval-augmented generation pipelines.</div>
    </div>
    <div class="domain-card intel">
        <div class="domain-icon"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg></div>
        <div class="domain-label">Intelligence & Analysis</div>
        <div class="domain-desc">Strategic threat intelligence and defensive modeling.</div>
    </div>
    <div class="domain-card research">
        <div class="domain-icon"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg></div>
        <div class="domain-label">Research & Publishing</div>
        <div class="domain-desc">Published author of 7 books and PhD in History.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Interactive Chat Section ──
st.markdown('<div class="chat-section-label">▸ Interactive Conversation</div>', unsafe_allow_html=True)

# ── Action Buttons ──
col_a, col_b = st.columns(2)
with col_a:
    if st.button("⊕ Analyze Role Fit", use_container_width=True):
        with st.chat_message("assistant"):
            answer = run_turn("Analyze fit.", action_mode="fit")
            st.session_state.messages.append({"role": "assistant", "content": answer})
with col_b:
    if st.button("✎ Draft Outreach Message", use_container_width=True):
        with st.chat_message("assistant"):
            answer = run_turn("Draft outreach.", action_mode="outreach")
            st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Chat Logic ──
if st.session_state.pinned_opening and not st.session_state.messages:
    pinned = "▸ Welcome! I'm here to help you explore Stephen's professional background."
    st.session_state.messages.append({"role": "assistant", "content": pinned})

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"], unsafe_allow_html=True)

# ── Starter Chips ──
if not any(m.get("role") == "user" for m in st.session_state.messages):
    chip_cols = st.columns(4)
    topics = ["Security Architecture", "AI & RAG Portfolio", "Intelligence Background", "Publications & Research"]
    for i, topic in enumerate(topics):
        if chip_cols[i].button(topic, key=f"chip_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": topic})
            with st.chat_message("user"): st.markdown(topic)
            with st.chat_message("assistant"):
                ans = run_turn(topic, action_mode="chat")
                st.session_state.messages.append({"role": "assistant", "content": ans})
            st.rerun()

# ── Chat Input ──
user_input = st.chat_input("Ask about Stephen's qualifications...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)
    with st.chat_message("assistant"):
        ans = run_turn(user_input, action_mode="chat")
        st.session_state.messages.append({"role": "assistant", "content": ans})
