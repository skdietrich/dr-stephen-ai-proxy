# streamlit_app.py  — Option B (Personal Mode + About/How-I-Think + Publications + PDF Export)
import os
import re
from datetime import datetime

import pandas as pd
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Prompt template import compatibility
try:
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    from langchain.prompts import ChatPromptTemplate

# PDF export (Streamlit Cloud compatible if reportlab in requirements)
try:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# -------------------------
# Optional deterministic modules (do NOT hard-fail if missing)
# -------------------------
try:
    from scoring import score_overall
    from mitigations import tier_from_score, mitigation_playbook
    SCORING_ENABLED = True
except Exception:
    SCORING_ENABLED = False


# =========================
# Guardrails (minimal / non-destructive)
# =========================
_EXTERNAL_REF_REGEX = re.compile(
    r"(\bet al\.\b|\bworks cited\b|\breferences\b|\bbibliography\b|\[\d+\]|\([A-Z][A-Za-z\-]+,\s*\d{4}\))",
    flags=re.IGNORECASE,
)

def enforce_no_external_refs(text: str) -> str:
    """Block bibliography-style external citations. Allow normal standards mention."""
    if not text:
        return text
    if _EXTERNAL_REF_REGEX.search(text):
        return (
            "⚠️ Response blocked: bibliography-style external citations detected.\n\n"
            "This system is **Public-safe / Evidence-only** and can only cite the loaded corpus."
        )
    return text


# =========================
# Assets helpers (logo + headshot + optional panels)
# =========================
def safe_exists(path: str | None) -> bool:
    try:
        return bool(path) and os.path.exists(path)
    except Exception:
        return False

def first_existing(paths: list[str]) -> str | None:
    for p in paths:
        if safe_exists(p):
            return p
    return None

LOGO_PATH = first_existing([
    os.path.join("assets", "logo.png"),
    os.path.join("assets", "logo.jpg"),
    os.path.join("assets", "logo.jpeg"),
    "logo.png", "logo.jpg", "logo.jpeg",
])

HEADSHOT_PATH = first_existing([
    os.path.join("assets", "headshot.png"),
    os.path.join("assets", "headshot.jpg"),
    os.path.join("assets", "headshot.jpeg"),
    "headshot.png", "headshot.jpg", "headshot.jpeg",
])

ABOUT_MD_PATH = first_existing([
    os.path.join("assets", "about_stephen.md"),
    "about_stephen.md",
])

THINK_MD_PATH = first_existing([
    os.path.join("assets", "how_i_think.md"),
    "how_i_think.md",
])

PUBS_CSV_PATH = first_existing([
    os.path.join("assets", "publications.csv"),
    "publications.csv",
])

LINKEDIN_URL = "https://www.linkedin.com/in/stephendietrich-kolokouris/"


# =========================
# Streamlit config + high-end UI skin (CSS)
# =========================
st.set_page_config(
    page_title="Dr. Stephen Dietrich-Kolokouris, PhD | Strategic Proxy",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
<style>
:root{
  --bg0:#070A12;
  --bg1:#0B1020;
  --card:#0F172A;
  --line:rgba(148,163,184,0.18);
  --txt:#E5E7EB;
  --muted:#A1A1AA;
  --accent:#60A5FA;
  --accent2:#22C55E;
}

html, body, [class*="stApp"]{
  background: radial-gradient(1200px 900px at 20% 0%, #0B1020 0%, #070A12 50%, #05060A 100%) !important;
  color: var(--txt) !important;
}
.main .block-container { padding-top: 1.0rem; max-width: 1200px; }

section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(11,16,32,0.92) 0%, rgba(5,6,10,0.92) 100%) !important;
  border-right: 1px solid var(--line);
}

.dk-hero{
  background: linear-gradient(135deg, rgba(96,165,250,0.14) 0%, rgba(34,197,94,0.08) 55%, rgba(15,23,42,0.65) 100%);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 18px 20px;
  box-shadow: 0 12px 34px rgba(0,0,0,0.42);
}
.dk-title{
  font-size: 1.35rem;
  font-weight: 760;
  letter-spacing: 0.15px;
  margin: 0;
}
.dk-subtitle{
  color: var(--muted);
  margin-top: 4px;
  margin-bottom: 0;
}

.dk-card{
  background: linear-gradient(180deg, rgba(15,23,42,0.90) 0%, rgba(2,6,23,0.70) 100%);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}

hr.dk-hr{
  border:0;
  border-top:1px solid rgba(148,163,184,0.18);
  margin:10px 0;
}

[data-testid="stChatMessage"]{
  border: 1px solid var(--line);
  border-radius: 14px;
  background: rgba(15,23,42,0.55);
}

.small-muted { color: var(--muted); font-size: 0.92rem; }
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# Knowledge base (RAG)
# =========================
@st.cache_resource
def init_knowledge_base():
    if not os.path.exists("data"):
        st.error("Missing **/data** directory. Commit/upload your PDFs into `/data`.")
        st.stop()

    loader = PyPDFDirectoryLoader("data/")
    docs = loader.load()
    if not docs:
        st.error("No documents found in `/data`.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=160)
    chunks = splitter.split_documents(docs)

    # Embeddings compatibility (api_key vs openai_api_key)
    try:
        embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    except TypeError:
        embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

    with st.status("Indexing corpus (FAISS)…", expanded=False) as status:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        status.update(label="✅ Corpus indexed (FAISS ready)", state="complete")

    return vectorstore.as_retriever(search_kwargs={"k": 7})

retriever = init_knowledge_base()


# =========================
# PDF export (Q&A transcript)
# =========================
def _wrap_text_lines(text: str, max_chars: int = 95) -> list[str]:
    text = (text or "").replace("\r", "")
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

def build_qa_pdf_bytes(title: str, messages: list[dict], evidence_files: list[str]) -> bytes:
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

    # Header
    c.setFont("Helvetica-Bold", 14)
    c.drawString(left, y, title)
    y -= 18
    c.setFont("Helvetica", 9)
    stamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    c.drawString(left, y, f"Generated: {stamp}   |   Mode: Public-safe / Evidence-only")
    y -= 14
    c.setLineWidth(0.5)
    c.line(left, y, width - right, y)
    y -= 12

    # Body
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

    # Evidence page
    if evidence_files:
        if y < bottom + 80:
            new_page()
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left, y, "Evidence (corpus files referenced)")
        y -= 16
        c.setFont("Helvetica", 10)
        for f in evidence_files:
            if y < bottom + 14:
                new_page()
                c.setFont("Helvetica", 10)
            c.drawString(left, y, f"- {f}")
            y -= 12

    c.save()
    buf.seek(0)
    return buf.read()


# =========================
# Session state
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "personal_mode" not in st.session_state:
    st.session_state.personal_mode = False

if "pinned_opening" not in st.session_state:
    st.session_state.pinned_opening = True

if "qa_evidence_files" not in st.session_state:
    st.session_state.qa_evidence_files = []


# =========================
# Sidebar (Professional, clutter behind drawers)
# =========================
with st.sidebar:
    if safe_exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)

    if safe_exists(HEADSHOT_PATH):
        st.image(HEADSHOT_PATH, width=190)

    st.markdown("### Dr. Stephen Dietrich-Kolokouris")
    st.caption("Applied Security • Systems Analysis • Data Engineering • Strategic Modeling")
    st.link_button("LinkedIn", LINKEDIN_URL)

    st.info(
        "🔒 **Public-safe / Evidence-only**\n\n"
        "Answers are generated exclusively from the loaded corpus. "
        "If a claim cannot be supported, it will be labeled **Not in corpus**.",
        icon="🔐",
    )

    st.session_state.personal_mode = st.toggle(
        "Personal Mode (stories + background)",
        value=st.session_state.personal_mode,
        help="Tone only. Evidence-only constraint remains enforced.",
    )

    # 🧾 About Stephen drawer
    with st.expander("🧾 About Stephen", expanded=False):
        if safe_exists(ABOUT_MD_PATH):
            st.markdown(open(ABOUT_MD_PATH, "r", encoding="utf-8").read())
        else:
            st.markdown(
                "**Add an About sheet (optional):**\n\n"
                "Create `assets/about_stephen.md` with a career narrative, scope boundaries, and highlights.\n\n"
                "_No file found yet._"
            )

    # 🧠 How I think panel
    with st.expander("🧠 How I think", expanded=False):
        if safe_exists(THINK_MD_PATH):
            st.markdown(open(THINK_MD_PATH, "r", encoding="utf-8").read())
        else:
            st.markdown(
                "**Add a decision-making sheet (optional):**\n\n"
                "Create `assets/how_i_think.md` with:\n"
                "- decision cadence (triage → evidence → options → tradeoffs)\n"
                "- validation habits (tests, guardrails, traceability)\n"
                "- failure lessons (what changed afterward)\n\n"
                "_No file found yet._"
            )

    # 🎓 Publications / talks / appearances section
    with st.expander("🎓 Publications / Talks / Appearances", expanded=False):
        if safe_exists(PUBS_CSV_PATH):
            try:
                pubs = pd.read_csv(PUBS_CSV_PATH)
                st.dataframe(pubs, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Unable to read publications.csv: {e}")
        else:
            st.markdown(
                "**Optional:** create `assets/publications.csv` with columns like:\n\n"
                "`type,title,venue_or_outlet,year,link`\n\n"
                "_No file found yet._"
            )

    # Optional tools behind drawers
    with st.expander("How this system works", expanded=False):
        st.markdown(
            "- **Ingest:** PDFs in `/data/` are chunked.\n"
            "- **Index:** embeddings stored in **FAISS**.\n"
            "- **Retrieve:** top-k chunks per question.\n"
            "- **Answer:** strictly from retrieved text.\n"
            "- **Evidence:** filenames appended to responses."
        )

    if SCORING_ENABLED:
        with st.expander("Supply Chain Risk (optional)", expanded=False):
            st.caption("Deterministic scoring → mitigations (only when used)")
            weight_fw = st.slider("Weight: Firmware integrity", 0.0, 1.0, 0.55, 0.05, key="weight_fw")

            uploaded_csv = st.file_uploader("Vendor CSV", type=["csv"], key="vendor_csv_uploader")
            st.caption(
                "Required columns: vendor_name, product_or_component, component_class, origin_jurisdiction, criticality, "
                "contains_ree_magnets, firmware_ota, firmware_signing, secure_boot_attestation, sbom_available, "
                "remote_admin_access, telemetry_logging, alt_supplier_available, known_single_point_of_failure"
            )

            if uploaded_csv is not None:
                try:
                    df = pd.read_csv(uploaded_csv)
                except Exception as e:
                    st.error(f"Unable to read CSV: {e}")
                    df = None

                if df is not None:
                    required_cols = {
                        "vendor_name",
                        "product_or_component",
                        "component_class",
                        "origin_jurisdiction",
                        "criticality",
                        "contains_ree_magnets",
                        "firmware_ota",
                        "firmware_signing",
                        "secure_boot_attestation",
                        "sbom_available",
                        "remote_admin_access",
                        "telemetry_logging",
                        "alt_supplier_available",
                        "known_single_point_of_failure",
                    }
                    missing = sorted(list(required_cols - set(df.columns)))
                    if missing:
                        st.error(f"CSV missing required columns: {missing}")
                    else:
                        def _score_row(r):
                            s = score_overall(r.to_dict(), weight_fw=weight_fw)
                            return pd.Series(s)

                        scores = df.apply(_score_row, axis=1)
                        out = pd.concat([df, scores], axis=1)
                        out["tier"] = out["overall_risk"].apply(tier_from_score)
                        st.dataframe(out.sort_values("overall_risk", ascending=False).head(10), use_container_width=True)

                        idx = st.number_input(
                            "Select row index (0-based)",
                            min_value=0,
                            max_value=max(0, len(out) - 1),
                            value=0,
                            step=1,
                            key="vendor_row_idx",
                        )
                        row = out.iloc[int(idx)].to_dict()

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

    # 🖨️ One-click PDF export of a Q&A session
    with st.expander("🖨️ Export Q&A (PDF)", expanded=False):
        if not REPORTLAB_OK:
            st.warning("PDF export unavailable (reportlab missing). Add `reportlab` to requirements.")
        else:
            export_title = st.text_input(
                "PDF title",
                value="Q&A — Dr. Stephen Dietrich-Kolokouris (Public-safe / Evidence-only)",
            )
            if st.button("Generate PDF from this session", type="primary"):
                # Only export user/assistant chat messages (skip tool/system artifacts)
                export_msgs = []
                for m in st.session_state.messages:
                    role = (m.get("role") or "").lower()
                    if role in ("user", "assistant"):
                        export_msgs.append({"role": role, "content": m.get("content") or ""})

                evidence = sorted(set(st.session_state.get("qa_evidence_files", []) or []))
                pdf_bytes = build_qa_pdf_bytes(export_title, export_msgs, evidence)

                fname = "QA_Transcript_Stephen_Dietrich_Kolokouris.pdf"
                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name=fname,
                    mime="application/pdf",
                )


# =========================
# Main UI
# =========================
st.markdown(
    """
<div class="dk-hero">
  <div class="dk-title">Strategic Technical Briefing</div>
  <div class="dk-subtitle">
    Cybersecurity • AI/ML Decision Support • Data Engineering • Strategic Systems Analysis
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

# Clean top controls
col1, col2 = st.columns([2.2, 1.0], gap="large")
with col1:
    st.markdown(
        """
<div class="dk-card">
  <b>Scope boundary</b>
  <hr class="dk-hr">
  <span class="small-muted">
    Public-safe, evidence-only responses. Filenames are appended as evidence. No external citations generated.
  </span>
</div>
""",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """
<div class="dk-card">
  <b>Mode</b>
  <hr class="dk-hr">
  <span class="small-muted">
    Personal Mode changes tone only (narrative vs technical). Evidence-only rule stays enforced.
  </span>
</div>
""",
        unsafe_allow_html=True,
    )

st.write("")

# Pin recruiter intake questions (first-run)
if st.session_state.pinned_opening and not st.session_state.messages:
    pinned = (
        "**Recruiter intake (pick one):**\n\n"
        "1) What role are you hiring for, and what environment (cloud / on-prem / restricted)?\n"
        "2) What’s the hardest problem you need solved in the next 90 days?\n"
        "3) Which domain matters most: **RAG/AI**, **network/security architecture**, **forensics/IR**, or **supply chain**?"
    )
    st.session_state.messages.append({"role": "assistant", "content": pinned})

# Show chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Ask about NAMECOMMS, WarSim, AI/ML systems, firmware risk, portfolio, or background…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # LLM init compatibility (api_key vs openai_api_key)
        try:
            llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=st.secrets["OPENAI_API_KEY"])
        except TypeError:
            llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])

        vendor_ctx = st.session_state.get("selected_vendor_context")

        vendor_block = ""
        if vendor_ctx:
            vendor_block = (
                "\n\nSelected Vendor Context (deterministic):\n"
                f"- Vendor: {vendor_ctx.get('vendor_name')}\n"
                f"- Component: {vendor_ctx.get('product_or_component')}\n"
                f"- Class: {vendor_ctx.get('component_class')}\n"
                f"- Origin/Jurisdiction: {vendor_ctx.get('origin_jurisdiction')}\n"
                f"- Criticality: {vendor_ctx.get('criticality')}\n"
                f"- Tier: {vendor_ctx.get('tier')}\n"
                f"- Scores: REE={vendor_ctx.get('ree_risk')}, FW={vendor_ctx.get('firmware_risk')}, Overall={vendor_ctx.get('overall_risk')}\n"
                "Mitigation priorities (deterministic):\n"
                + "\n".join([f"- {m}" for m in vendor_ctx.get("mitigations", [])])
                + "\n"
            )

        if st.session_state.personal_mode:
            tone_line = (
                "TONE MODE: Personal Mode.\n"
                "- You may include brief career context and lessons learned **only when supported** by {context}.\n"
                "- Keep it recruiter-friendly; avoid hype; keep it precise.\n"
            )
        else:
            tone_line = (
                "TONE MODE: Technical-only.\n"
                "- Direct, systems-focused, implementation-oriented.\n"
            )

        system_prompt = (
            "You are an evidence-only technical proxy representing Dr. Stephen Dietrich-Kolokouris.\n\n"
            "MANDATORY CONSTRAINTS:\n"
            "1) Use ONLY the retrieved corpus excerpts in {context}.\n"
            "2) If selected vendor context is present, you may use it as deterministic input.\n"
            "3) Do NOT invent facts, credentials, project details, or external citations.\n"
            "4) If asked for something not supported by {context}, say 'Not in corpus.' briefly and continue.\n"
            "5) Do NOT output bibliography-style citations. Evidence is appended automatically.\n\n"
            + tone_line +
            "\nRESPONSE RULE:\n"
            "- Default: coherent, recruiter-grade explanation. Use short headings/bullets where helpful.\n"
            "- Only produce 30/60/90 plans if explicitly requested.\n\n"
            "Retrieved Context:\n{context}"
            + vendor_block
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{question}"),
            ]
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
            input_key="query",
        )

        # Light prime if vendor context exists (no forced structure)
        if vendor_ctx:
            question_payload = (
                f"[Vendor={vendor_ctx.get('vendor_name')} Tier={vendor_ctx.get('tier')} "
                f"Overall={vendor_ctx.get('overall_risk')}] {user_input}"
            )
        else:
            question_payload = user_input

        result = qa.invoke({"query": question_payload})
        answer = (result.get("result", "") or "").strip()
        answer = enforce_no_external_refs(answer)

        sources = sorted({
            os.path.basename(d.metadata.get("source", ""))
            for d in result.get("source_documents", [])
            if d.metadata.get("source")
        })

        # Track evidence across the session for PDF export
        if sources:
            existing = set(st.session_state.get("qa_evidence_files", []) or [])
            st.session_state.qa_evidence_files = sorted(existing.union(set(sources)))
            answer += "\n\n**Evidence:** " + ", ".join(sources)

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
